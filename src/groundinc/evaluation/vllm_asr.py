import argparse
import asyncio
import json
import os

from groundinc.evaluation.vllm_common import (
    GROUND_TRUTH_SNIPPET_MODES,
    SNIPPET_VIDEO_MODES,
    add_common_vllm_args,
    build_media_url,
    build_url_content_item,
    compute_snippet_span,
    cut_predicted_snippet,
    normalize_prompt_style,
    pad_and_clamp_span,
    prepare_video_path,
    print_prompt_styles,
    probe_video_metadata,
    process_records,
    resolve_predicted_span,
    run_prompt_style,
    validate_common_args,
)

from groundinc.data import (
    build_caption_tasks,
    iter_input_records,
)
from groundinc.data import prompts_asr


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def extract_asr_text(payload):
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("prompt", "asr_text", "text", "transcript"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        segments = payload.get("segments")
        if isinstance(segments, list):
            texts = [
                (segment.get("text") or "").strip()
                for segment in segments
                if isinstance(segment, dict) and (segment.get("text") or "").strip()
            ]
            return " ".join(texts)
    if isinstance(payload, list):
        texts = []
        for item in payload:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
            elif isinstance(item, dict) and (item.get("text") or "").strip():
                texts.append(item["text"].strip())
        return " ".join(texts)
    return ""


def load_asr_from_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    with open(path, "r", encoding="utf-8") as f:
        return extract_asr_text(json.load(f))


def load_asr_payload_from_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def snip_asr(payload, start, end):
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not segments:
        return extract_asr_text(payload)

    kept = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        if seg_start is None or seg_end is None:
            continue
        if seg_end >= start and seg_start <= end:
            text = (segment.get("text") or "").strip()
            if text:
                kept.append(text)
    return " ".join(kept) if kept else extract_asr_text(payload)


def iter_asr_candidates(data, asr_dir, asr_suffix):
    vid_id = data["id"]
    video_path = data.get("video_path")
    stems = [vid_id]
    if video_path:
        stems.append(os.path.splitext(os.path.basename(video_path))[0])

    seen = set()
    for stem in stems:
        for name in (
            f"{stem}{asr_suffix}",
            f"{stem}_asr.json",
            f"{stem}.json",
            f"{stem}.txt",
        ):
            path = os.path.join(asr_dir, name)
            if path not in seen:
                seen.add(path)
                yield path


def extract_asr_text_from_payload(payload, snippet_span=None):
    if snippet_span and len(snippet_span) >= 2:
        return snip_asr(payload, float(snippet_span[0]), float(snippet_span[1]))
    return extract_asr_text(payload)


def load_asr_text(data, asr_dir, asr_field, asr_suffix, snippet_span=None):
    if asr_field and asr_field in data:
        text = extract_asr_text_from_payload(data[asr_field], snippet_span)
        if text:
            return text

    for field in ("asr_text", "asr", "transcript"):
        if field != asr_field and field in data:
            text = extract_asr_text_from_payload(data[field], snippet_span)
            if text:
                return text

    if not asr_dir:
        return ""

    for candidate in iter_asr_candidates(data, asr_dir, asr_suffix):
        if os.path.exists(candidate):
            payload = load_asr_payload_from_file(candidate)
            return extract_asr_text_from_payload(payload, snippet_span)
    return ""


def load_asr_payload(data, asr_dir, asr_field, asr_suffix):
    """Raw ASR payload."""
    for field in (asr_field, "asr_text", "asr", "transcript"):
        if field and field in data and data[field]:
            return data[field]

    if not asr_dir:
        return None

    for candidate in iter_asr_candidates(data, asr_dir, asr_suffix):
        if os.path.exists(candidate):
            return load_asr_payload_from_file(candidate)
    return None


def build_asr_prompt(style, asr_text, caption, ground_truth, prompt_fps):
    fps_text = "1 FPS" if prompt_fps is None else f"{prompt_fps:g} FPS"
    kwargs = {
        "asr_content": asr_text,
        "caption": caption,
        "fps": fps_text,
    }
    if style == "separate":
        kwargs["query"] = caption
    if style == "ground_truth_window_inconsistency_veracity":
        if len(ground_truth) < 2:
            raise ValueError(f"Prompt style '{style}' requires a ground_truth_span")
        kwargs["start_time"] = ground_truth[0]
        kwargs["end_time"] = ground_truth[1]
    return prompts_asr.build_prompt(style, **kwargs)


async def resolve_predicted_evidence(
    client,
    model_id,
    media_items,
    asr_text,
    asr_payload,
    task,
    args,
    request_semaphore,
    context,
):
    """Ground on the full video, then cut the predicted interval."""
    grounding_prompt = build_asr_prompt(
        args.grounding_prompt_style,
        asr_text,
        task["caption"],
        task["ground_truth_span"],
        args.prompt_fps,
    )
    span, raw = await resolve_predicted_span(
        client,
        model_id,
        prompts_asr,
        grounding_prompt,
        media_items,
        context["duration"],
        args,
        request_semaphore,
    )
    extra = {
        "grounding": {"prediction_raw": raw, "prediction_parsed": span},
        "predicted_span": span,
        "crop": None,
        "snippet_path": None,
        "evidence_duration": None,
    }
    if span is None:
        extra["error"] = "grounding produced no usable interval"
        return media_items, asr_text, extra

    crop = pad_and_clamp_span(span, args.snippet_pad, context["duration"])
    if crop is None:
        extra["error"] = "predicted interval empty after padding and clamping"
        return media_items, asr_text, extra

    try:
        snippet_path = await cut_predicted_snippet(
            context["source_video_path"], crop, args, context["vid_id"], context["select_seg"]
        )
    except Exception as exc:
        extra["error"] = f"snippet cut failed: {exc}"
        return media_items, asr_text, extra

    if not snippet_path or not os.path.exists(snippet_path):
        extra["error"] = "snippet cut produced no file"
        return media_items, asr_text, extra

    cropped_media = [
        build_url_content_item(
            args.video_content_type,
            build_media_url(snippet_path, args.media_mode, "video"),
        )
    ]
    cropped_asr = asr_text
    if args.asr_align == "span" and asr_payload is not None:
        cropped_asr = snip_asr(asr_payload, crop[0], crop[1])

    extra.update({
        "crop": crop,
        "snippet_path": snippet_path,
        "evidence_duration": round(crop[1] - crop[0], 3),
    })
    return cropped_media, cropped_asr, extra


async def process_single_caption_entry(
    client,
    model_id,
    media_items,
    asr_text,
    task,
    args,
    request_semaphore,
    asr_payload=None,
    context=None,
):
    extra = {}
    if args.video_mode == "predicted_snippet":
        media_items, asr_text, extra = await resolve_predicted_evidence(
            client,
            model_id,
            media_items,
            asr_text,
            asr_payload,
            task,
            args,
            request_semaphore,
            context,
        )
        if extra.get("error"):
            return {
                "category": task["category"],
                "type": task["type"],
                "caption": task["caption"],
                "manipulation_method_gt": task["manipulation_method_gt"],
                "entity_type": task.get("entity_type"),
                "manipulated_modality": task["manipulated_modality"],
                "ground_truth_span": task["ground_truth_span"],
                "prompt_style": args.prompt_style,
                "terminal_failure": True,
                **extra,
            }

    prompt = build_asr_prompt(
        args.prompt_style,
        asr_text,
        task["caption"],
        task["ground_truth_span"],
        args.prompt_fps,
    )
    prediction = await run_prompt_style(
        client,
        model_id,
        prompts_asr,
        prompt,
        media_items,
        args,
        request_semaphore,
    )
    return {
        "category": task["category"],
        "type": task["type"],
        "caption": task["caption"],
        "manipulation_method_gt": task["manipulation_method_gt"],
        "entity_type": task.get("entity_type"),
        "manipulated_modality": task["manipulated_modality"],
        "ground_truth_span": task["ground_truth_span"],
        "prompt_style": args.prompt_style,
        **extra,
        **prediction,
    }


def resolve_video_duration(data, video_path):
    """Record duration if present, else probe the file. 0.0 when unknown."""
    duration = data.get("duration")
    if duration:
        return float(duration)
    meta = probe_video_metadata(video_path)
    return float(meta["duration"]) if meta else 0.0


async def process_video_line(client, model_id, data, args, request_semaphore):
    video_path, source_video_path = prepare_video_path(data, args)
    if not video_path or not os.path.exists(video_path):
        return None

    snippet_span = compute_snippet_span(data) if args.video_mode in GROUND_TRUTH_SNIPPET_MODES else None
    asr_text = load_asr_text(data, args.asr_dir, args.asr_field, args.asr_suffix, snippet_span)
    if args.require_asr and not asr_text:
        return None

    video_url = build_media_url(video_path, args.media_mode, "video")
    media_items = [build_url_content_item(args.video_content_type, video_url)]
    caption_semaphore = asyncio.Semaphore(max(1, args.max_concurrent_captions))

    asr_payload = None
    context = None
    if args.video_mode == "predicted_snippet":
        asr_payload = load_asr_payload(data, args.asr_dir, args.asr_field, args.asr_suffix)
        context = {
            "source_video_path": source_video_path or video_path,
            "duration": resolve_video_duration(data, video_path),
            "vid_id": data["id"],
            "select_seg": data["select_seg"],
        }

    async def run_caption_task(task):
        async with caption_semaphore:
            return await process_single_caption_entry(
                client,
                model_id,
                media_items,
                asr_text,
                task,
                args,
                request_semaphore,
                asr_payload=asr_payload,
                context=context,
            )

    caption_tasks = [
        run_caption_task(task)
        for task in build_caption_tasks(data, caption_task_video_mode(data, args.video_mode))
        if args.category_filter == "all" or task["category"] == args.category_filter
    ]
    if not caption_tasks:
        return data

    data["resolved_video_path"] = video_path
    if source_video_path:
        data["source_video_path"] = source_video_path
        if args.video_mode != "predicted_snippet":
            data["snippet_path"] = video_path
    data["asr_text"] = asr_text
    data["results"] = await asyncio.gather(*caption_tasks)
    return data


def caption_task_video_mode(data, video_mode):
    if video_mode not in SNIPPET_VIDEO_MODES:
        return video_mode
    return "id"


async def main():
    parser = argparse.ArgumentParser(description="Generic vLLM video+ASR evaluator.")
    add_common_vllm_args(parser, prompts_asr, DEFAULT_MODEL_ID)
    parser.add_argument("--asr_dir", "--whisperx_dir", dest="asr_dir", type=str, default=None)
    parser.add_argument("--asr_field", type=str, default="asr_text")
    parser.add_argument("--asr_suffix", type=str, default="_asr.json")
    parser.add_argument("--require_asr", action="store_true")
    args = parser.parse_args()

    if args.list_prompt_styles:
        print_prompt_styles(prompts_asr)
        return

    args.prompt_style = normalize_prompt_style(args.prompt_style, args.prompt_mode)
    validate_common_args(parser, args)

    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        parser.error("missing dependency: install/activate an environment with the 'openai' package")

    api_key = os.environ.get("VLLM_API_KEY", "vllm")
    client = AsyncOpenAI(base_url=args.vllm_url, api_key=api_key)
    await process_records(client, args, process_video_line, iter_input_records)


if __name__ == "__main__":
    asyncio.run(main())
