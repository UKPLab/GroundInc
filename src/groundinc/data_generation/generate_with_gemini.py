import json
import random
import argparse
import hashlib
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm
from google import genai
from google.genai import types

from groundinc.data.dataset import resolve_video_path


@dataclass
class MediaOptions:
    """Where the generator finds video, and how it is sampled."""

    video_dir: str = None
    video_fps: float = None
    poll_interval: float = 2.0
    upload_timeout: float = 900.0


def generate_with_retry(client, model_name, contents, config, retries=6, base_sleep=1.5):
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return resp.parsed
        except Exception as e:
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"\nAttempt {attempt + 1}/{retries} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(sleep_s)
            else:
                return None


_UPLOAD_LOCK = threading.Lock()
_UPLOAD_STATE = {}


def _await_active(client, file_obj, poll_interval, timeout):
    deadline = time.monotonic() + timeout
    while getattr(file_obj.state, "name", str(file_obj.state)) == "PROCESSING":
        if time.monotonic() > deadline:
            raise TimeoutError(f"Gemini file {file_obj.name} still PROCESSING after {timeout}s")
        time.sleep(poll_interval)
        file_obj = client.files.get(name=file_obj.name)

    state = getattr(file_obj.state, "name", str(file_obj.state))
    if state != "ACTIVE":
        raise RuntimeError(f"Gemini file {file_obj.name} entered state {state}")
    return file_obj


def upload_video_once(client, video_path, poll_interval=2.0, timeout=900):
    """Upload video_path at most once per process; concurrent callers share it."""
    with _UPLOAD_LOCK:
        entry = _UPLOAD_STATE.get(video_path)
        owner = entry is None
        if owner:
            entry = {"event": threading.Event(), "file": None, "error": None}
            _UPLOAD_STATE[video_path] = entry

    if owner:
        try:
            uploaded = client.files.upload(file=video_path)
            entry["file"] = _await_active(client, uploaded, poll_interval, timeout)
        except Exception as exc:
            entry["error"] = exc
        finally:
            entry["event"].set()
    elif not entry["event"].wait(timeout):
        raise TimeoutError(f"Timed out waiting for concurrent upload of {video_path}")

    if entry["error"]:
        raise entry["error"]
    return entry["file"]


def build_video_part(file_obj, fps=None):
    metadata = types.VideoMetadata(fps=float(fps)) if fps else None
    return types.Part(
        file_data=types.FileData(file_uri=file_obj.uri, mime_type=file_obj.mime_type),
        video_metadata=metadata,
    )


def resolve_generation_video(data, video_dir):
    """Return the video path for a record, or None when it cannot be resolved."""
    path = resolve_video_path(data, "auto", video_dir)
    return path if path and os.path.exists(path) else None


def format_timeline(captions, seg_time, seg_idx):
    """Render the timestamped caption timeline."""
    lines = []
    for i, caption in enumerate(captions):
        span = seg_time[i] if i < len(seg_time) and len(seg_time[i]) >= 2 else None
        stamp = f"[{span[0]} - {span[1]}] " if span else ""
        marker = "   <<< TARGET SEGMENT" if i == seg_idx else ""
        lines.append(f"[{i + 1}] {stamp}{caption}{marker}")
    return "\n".join(lines)


def format_inputs(data, seg_idx, modality):
    video_caps = data.get("video_captions", [])
    audio_caps = data.get("audio_captions", [])
    seg_time = data.get("seg_time") or []

    if modality == "visual":
        captions = video_caps
    elif modality == "audio":
        captions = audio_caps
    else:
        raise ValueError("Modality must be 'visual' or 'audio'")

    context = format_timeline(captions, seg_time, seg_idx)
    target = captions[seg_idx] if 0 <= seg_idx < len(captions) else "N/A"
    return context, target


def normalize_generated_item(item):
    if not isinstance(item, dict):
        return None

    method = item.get("method") or item.get("manipulation_method")
    false_caption = item.get("false_caption")
    refutation = item.get("refutation")
    if not method or not false_caption or not refutation:
        return None

    return {
        "method": method,
        "entity_type": item.get("entity_type", "n/a"),
        "false_caption": false_caption,
        "refutation": refutation,
    }


def process_modality_task(line_idx, line, api_key, p1_text, p2_text, mod, model_name, retries,
                          media=None):
    media = media or MediaOptions()
    data = json.loads(line)
    seg_idx = data.get("select_seg", 1) - 1
    client = genai.Client(api_key=api_key)

    step1_schema = types.Schema(
        type="OBJECT",
        properties={
            "rephrased_caption": {"type": "STRING"},
            "manipulation_plans": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={
                        "method": {"type": "STRING"},
                        "entity_type": {"type": "STRING"},
                        "rationale_steps": {"type": "STRING"},
                    },
                    required=["method", "entity_type", "rationale_steps"],
                ),
            ),
        },
        required=["rephrased_caption", "manipulation_plans"],
    )

    step2_schema = types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "method": {"type": "STRING"},
                "entity_type": {"type": "STRING"},
                "false_caption": {"type": "STRING"},
                "refutation": {"type": "STRING"},
            },
            required=["method", "entity_type", "false_caption", "refutation"],
        )
    )

    s1_config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=step1_schema, temperature=0.4)
    s2_config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=step2_schema, temperature=0.1)

    context, target = format_inputs(data, seg_idx, mod)

    video_path = resolve_generation_video(data, media.video_dir)
    if not video_path:
        print(f"Skipping line {line_idx} ({mod}): no video for id={data.get('id')!r}")
        return (line_idx, mod, None)
    try:
        file_obj = upload_video_once(client, video_path, media.poll_interval, media.upload_timeout)
    except Exception as exc:
        print(f"Skipping line {line_idx} ({mod}): video upload failed: {exc}")
        return (line_idx, mod, None)
    video_part = build_video_part(file_obj, fps=media.video_fps)

    def contents_for(prompt_text):
        return [video_part, prompt_text]

    p1_content = p1_text.replace("<CONTEXT_CAPTIONS>", context).replace("<TARGET_CAPTION>", target)
    step1_result = generate_with_retry(client, model_name, contents_for(p1_content), s1_config, retries)

    if not step1_result:
        return (line_idx, mod, None)

    rephrased = step1_result.get("rephrased_caption", target)
    all_plans = step1_result.get("manipulation_plans", [])

    audio_only_methods = ("Context Omission", "Affective/Framing")
    filtered_plans = [
        p
        for p in all_plans
        if not (mod == "visual" and any(m in p.get("method", "") for m in audio_only_methods))
    ]

    final_manipulations = []
    batch_size = 3
    for i in range(0, len(filtered_plans), batch_size):
        batch_plans = filtered_plans[i : i + batch_size]
        
        tasks = []
        for p in batch_plans:
            tasks.append({
                "source_caption": rephrased,
                "method": p.get("method"),
                "entity_type": p.get("entity_type"),
                "rationale_steps": p.get("rationale_steps")
            })

        p2_content = (
            p2_text.replace("<CONTEXT_CAPTIONS>", context)
            .replace("<TASKS_JSON>", json.dumps(tasks, indent=2))
        )
        
        batch_out = generate_with_retry(client, model_name, contents_for(p2_content), s2_config, retries)
        if batch_out:
            if isinstance(batch_out, dict):
                batch_out = [batch_out]
            final_manipulations.extend(
                normalized
                for item in batch_out
                if (normalized := normalize_generated_item(item))
            )

    return (line_idx, mod, {
        "rephrased": rephrased, 
        "manipulations": final_manipulations
    })


def combine_modalities(line_data, visual_res, audio_res):
    v_true = visual_res.get("rephrased", "")
    a_true = audio_res.get("rephrased", "")

    output = {
        "true_visual": v_true,
        "true_audio": a_true,
        "true_combined": f"{v_true} {a_true}".strip(),
        
        "visual_only_manipulations": visual_res.get("manipulations", []),
        "audio_only_manipulations": audio_res.get("manipulations", []),
        "combined_manipulations": []
    }

    for vm in visual_res.get("manipulations", []):
        output["combined_manipulations"].append({
            "type": "manipulated_visual_true_audio",
            "method": vm["method"],
            "entity_type": vm.get("entity_type", "n/a"),
            "false_caption": f"{vm['false_caption']} {a_true}".strip(),
            "refutation": vm["refutation"]
        })

    for am in audio_res.get("manipulations", []):
        output["combined_manipulations"].append({
            "type": "true_visual_manipulated_audio",
            "method": am["method"],
            "entity_type": am.get("entity_type", "n/a"),
            "false_caption": f"{v_true} {am['false_caption']}".strip(),
            "refutation": am["refutation"]
        })

    line_data["groundinc_annotations"] = output
    return line_data


def _line_hash(line):
    data = json.loads(line)
    data.pop("groundinc_annotations", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def generate_groundinc_annotations(input_jsonl, output_jsonl, api_key, p1_file, p2_file, model_name,
                               max_workers, retries, media=None):
    media = media or MediaOptions()
    with open(p1_file, "r", encoding="utf-8") as f:
        p1_text = f.read()
    with open(p2_file, "r", encoding="utf-8") as f:
        p2_text = f.read()
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    
    processed_hashes = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    processed_hashes.add(_line_hash(ln))
                except (json.JSONDecodeError, TypeError):
                    pass

    pending = [(i, ln) for i, ln in enumerate(lines) if _line_hash(ln) not in processed_hashes]
    results = defaultdict(dict)
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, ln in pending:
            for mod in ["visual", "audio"]:
                futures.append(executor.submit(process_modality_task, idx, ln, api_key, p1_text, p2_text, mod, model_name, retries, media))

        with open(output_jsonl, "a", encoding="utf-8", buffering=1) as outfile:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="GroundInc generation"):
                idx, mod, res = fut.result()
                results[idx][mod] = res
                
                if len(results[idx]) == 2:
                    if not results[idx]["visual"] or not results[idx]["audio"]:
                        print(f"Skipping line {idx}: failed generation for one or more modalities")
                        del results[idx]
                        continue
                    final_json = combine_modalities(
                        json.loads(lines[idx]),
                        results[idx]["visual"],
                        results[idx]["audio"],
                    )
                    outfile.write(json.dumps(final_json, ensure_ascii=False) + "\n")
                    del results[idx]


def read_api_key(api_key_file=None, api_key_env="GEMINI_API_KEY"):
    if api_key_file:
        with open(api_key_file, "r", encoding="utf-8") as fp:
            return fp.read().strip()
    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Gemini API key not found. Set {api_key_env}/GOOGLE_API_KEY or pass --api_key_file."
        )
    return api_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--prompt1_file", default="prompts/step1_prompt.md")
    parser.add_argument("--prompt2_file", default="prompts/step2_prompt.md")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)

    parser.add_argument("--video_dir", default=None,
                        help="Directory of {id}.mp4 files. Records carrying video_path use that instead.")
    parser.add_argument("--video_fps", type=float, default=None,
                        help="Frame sampling rate passed to Gemini. Default: server-side default.")
    args = parser.parse_args()

    if not args.video_dir:
        print("Note: --video_dir not set; only records carrying an explicit video_path will resolve.")

    api_key = read_api_key(args.api_key_file, args.api_key_env)
    generate_groundinc_annotations(
        args.input_jsonl,
        args.output_jsonl,
        api_key,
        args.prompt1_file,
        args.prompt2_file,
        args.model,
        args.max_workers,
        retries=args.retries,
        media=MediaOptions(
            video_dir=args.video_dir,
            video_fps=args.video_fps,
        ),
    )
