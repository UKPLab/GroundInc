import glob
import json
import os

EXCLUDED_LABELS = {
    "Quantifier and Scope Distortions",
    "Epistemic Inflation",
    "Entity Disambiguation / Homonyms",
}


def seg_time_to_seconds(time_str):
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def infer_video_mode(data, video_mode):
    if video_mode == "auto":
        return "clip" if data.get("video_path") else "id"
    return video_mode


def resolve_video_path(data, video_mode, video_dir=None):
    video_mode = infer_video_mode(data, video_mode)
    if video_mode == "clip":
        return data.get("video_path")
    if not video_dir:
        return None
    return os.path.join(video_dir, f"{data['id']}.mp4")


def resolve_audio_path(data, audio_dir, audio_ext="mp3"):
    if data.get("audio_path") and os.path.exists(data["audio_path"]):
        return data["audio_path"]

    if not audio_dir:
        return None

    audio_ext = (audio_ext or "mp3").lstrip(".")
    vid_id = data["id"]
    select_seg = data["select_seg"]
    candidates = []

    if data.get("video_path"):
        video_stem = os.path.splitext(os.path.basename(data["video_path"]))[0]
        candidates.append(os.path.join(audio_dir, f"{video_stem}.{audio_ext}"))

    candidates.extend(
        [
            os.path.join(audio_dir, f"{vid_id}_seg{select_seg}_clip.{audio_ext}"),
            os.path.join(audio_dir, f"{vid_id}_seg{select_seg}.{audio_ext}"),
            os.path.join(audio_dir, f"{vid_id}.{audio_ext}"),
        ]
    )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    patterns = [
        os.path.join(audio_dir, f"{vid_id}_seg{select_seg}*.{audio_ext}"),
        os.path.join(audio_dir, f"{vid_id}*.{audio_ext}"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches) == 1:
            return matches[0]

    return None


def compute_ground_truth_span(data, video_mode):
    video_mode = infer_video_mode(data, video_mode)
    if video_mode == "clip":
        return data.get("relative_seg_gt", [])

    select_seg = data["select_seg"]
    seg = data["seg_time"][select_seg - 1]
    return [round(seg_time_to_seconds(seg[0]), 3), round(seg_time_to_seconds(seg[1]), 3)]


def _iter_legacy_manipulations(section):
    if not isinstance(section, dict):
        return

    for item in section.get("step2_execution", []):
        if not isinstance(item, dict):
            continue
        yield {
            "method": item.get("manipulation_method"),
            "entity_type": item.get("entity_type"),
            "false_caption": item.get("false_caption"),
            "type": item.get("type"),
        }


def build_caption_tasks(data, video_mode):
    gt_span = compute_ground_truth_span(data, video_mode)
    select_seg = data["select_seg"]
    fm = data.get("forensic_mentalization", {})

    visual_orig = data["video_captions"][select_seg - 1]
    audio_orig = data["audio_captions"][select_seg - 1]
    combined_orig = f"{visual_orig}. {audio_orig}"

    tasks = []

    for cap, cat in [
        (visual_orig, "visual"),
        (audio_orig, "audio"),
        (combined_orig, "combined"),
    ]:
        tasks.append(
            {
                "category": cat,
                "type": "original_true",
                "caption": cap,
                "manipulation_method_gt": "No Manipulation",
                "entity_type": None,
                "manipulated_modality": "none",
                "ground_truth_span": gt_span,
            }
        )

    for cap, cat in [
        (fm.get("true_visual"), "visual"),
        (fm.get("true_audio"), "audio"),
        (fm.get("true_combined"), "combined"),
    ]:
        if cap:
            tasks.append(
                {
                    "category": cat,
                    "type": "rephrased_true",
                    "caption": cap,
                    "manipulation_method_gt": "No Manipulation",
                    "entity_type": None,
                    "manipulated_modality": "none",
                    "ground_truth_span": gt_span,
                }
            )

    visual_manips = list(fm.get("visual_only_manipulations", []))
    audio_manips = list(fm.get("audio_only_manipulations", []))
    combined_manips = list(fm.get("combined_manipulations", []))

    # Backward compatibility for older forensic_mentalization records stored as:
    # fm["visual"|"audio"|"combined"]["step2_execution"].
    if not visual_manips and isinstance(fm.get("visual"), dict):
        visual_manips = list(_iter_legacy_manipulations(fm["visual"]))
    if not audio_manips and isinstance(fm.get("audio"), dict):
        audio_manips = list(_iter_legacy_manipulations(fm["audio"]))
    if not combined_manips and isinstance(fm.get("combined"), dict):
        combined_manips = list(_iter_legacy_manipulations(fm["combined"]))

    for manip_list, cat in [
        (visual_manips, "visual"),
        (audio_manips, "audio"),
    ]:
        for item in manip_list:
            method = item.get("method")
            cap = item.get("false_caption")
            if not method or not cap or method in EXCLUDED_LABELS:
                continue
            tasks.append(
                {
                    "category": cat,
                    "type": method,
                    "caption": cap,
                    "manipulation_method_gt": method,
                    "entity_type": item.get("entity_type"),
                    "manipulated_modality": cat,
                    "ground_truth_span": gt_span,
                    "refutation": item.get("refutation"),
                }
            )

    for item in combined_manips:
        method = item.get("method")
        cap = item.get("false_caption")
        if not method or not cap or method in EXCLUDED_LABELS:
            continue

        item_type = item.get("type") or ""
        if "manipulated_visual" in item_type:
            manip_mod = "visual"
        elif "manipulated_audio" in item_type:
            manip_mod = "audio"
        else:
            manip_mod = "combined"

        tasks.append(
            {
                "category": "combined",
                "type": method,
                "caption": cap,
                "manipulation_method_gt": method,
                "entity_type": item.get("entity_type"),
                "manipulated_modality": manip_mod,
                "ground_truth_span": gt_span,
                "refutation": item.get("refutation"),
            }
        )

    return tasks


def iter_input_records(input_file):
    with open(input_file, "r") as f_in:
        for line in f_in:
            if line.strip():
                yield json.loads(line)


def load_processed_keys(output_file):
    processed = set()
    if not os.path.exists(output_file):
        return processed

    with open(output_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                processed.add((data["id"], data["select_seg"]))
    return processed
