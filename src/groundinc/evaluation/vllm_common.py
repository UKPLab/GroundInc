import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import subprocess
from pathlib import Path


DEFAULT_PROMPT_STYLE = "temporal_inconsistency_veracity"
LEGACY_PROMPT_STYLE_ALIASES = {
    "combined": "temporal_inconsistency_veracity",
    "multi-turn": "multi_turn",
    "multi-turn-no-grounding": "multi_turn_no_grounding",
    "veracity_only": "temporal_veracity_only",
    "no_evidence": "inconsistency_veracity_only",
    "ground_truth_window_veracity": "ground_truth_window_inconsistency_veracity",
    "ground_truth_window_veracity_manipulation": "ground_truth_window_inconsistency_veracity",
}
LEGACY_PROMPT_MODE_MAP = {
    "combined": "temporal_inconsistency_veracity",
    "separate": "separate",
    "veracity_only": "temporal_veracity_only",
    "no_evidence": "inconsistency_veracity_only",
}

VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
}
AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
}
THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)

# Modes that cut a clip with ffmpeg and therefore need --snippet_dir/--video_dir.
# "snippet"/"masked_snippet" cut the annotated segment, which is known per record;
# "predicted_snippet" cuts the model's own predicted interval, which differs per
# caption task and so is resolved inside the per-task path, not prepare_video_path.
SNIPPET_VIDEO_MODES = ("snippet", "masked_snippet", "predicted_snippet")
GROUND_TRUTH_SNIPPET_MODES = ("snippet", "masked_snippet")

# ffmpeg must not inherit stdin (it reads it interactively and will block a
# non-tty run) and must not be able to hang a caption task indefinitely.
FFMPEG_TIMEOUT = float(os.environ.get("GROUNDINC_FFMPEG_TIMEOUT", "600"))


class VLLMConnectionError(RuntimeError):
    """Raised when the vLLM/OpenAI client cannot connect to the server."""


def is_connection_error_text(value):
    return isinstance(value, str) and "connection error" in value.lower()


def supported_prompt_style_choices(prompt_module):
    styles = set(prompt_module.PROMPT_STYLES)
    return tuple(sorted(styles | set(LEGACY_PROMPT_STYLE_ALIASES)))


def normalize_prompt_style(prompt_style, prompt_mode=None):
    if prompt_style:
        return LEGACY_PROMPT_STYLE_ALIASES.get(prompt_style, prompt_style)
    if prompt_mode:
        return LEGACY_PROMPT_MODE_MAP[prompt_mode]
    return DEFAULT_PROMPT_STYLE


def print_prompt_styles(prompt_module):
    for style in sorted(prompt_module.PROMPT_STYLES):
        spec = prompt_module.get_prompt_style_spec(style)
        family = spec.get("family", "prompt")
        outputs = ", ".join(spec.get("outputs", ()))
        print(f"{style}: family={family}. Outputs: {outputs}")
    if LEGACY_PROMPT_STYLE_ALIASES:
        print("\nLegacy aliases:")
        for alias, target in sorted(LEGACY_PROMPT_STYLE_ALIASES.items()):
            print(f"{alias} -> {target}")


def parse_json_object(value, arg_name):
    if not value:
        return {}
    raw = value
    if value.startswith("@"):
        with open(value[1:], "r") as f:
            raw = f.read()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{arg_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{arg_name} must decode to a JSON object")
    return parsed


def merge_dicts(base, update):
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_video_grid_thw(value):
    if value is None or value == "":
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--video_grid_thw must have exactly 3 comma-separated integers")
    return [int(part) for part in parts]


def parse_bool_arg(value):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("must be 'true' or 'false'")


def build_extra_body(args):
    extra_body = parse_json_object(args.extra_body_json, "--extra_body_json")
    mm_processor_kwargs = parse_json_object(
        args.mm_processor_kwargs_json,
        "--mm_processor_kwargs_json",
    )

    video_grid_thw = parse_video_grid_thw(args.video_grid_thw)
    if video_grid_thw is not None:
        mm_processor_kwargs["video_grid_thw"] = video_grid_thw
    if args.mm_fps is not None:
        mm_processor_kwargs["fps"] = args.mm_fps
    if args.num_frames is not None:
        mm_processor_kwargs["num_frames"] = args.num_frames
    if args.mm_min_pixels is not None:
        mm_processor_kwargs["min_pixels"] = args.mm_min_pixels
    if args.mm_max_pixels is not None:
        mm_processor_kwargs["max_pixels"] = args.mm_max_pixels
    if args.mm_total_pixels is not None:
        mm_processor_kwargs["max_pixels"] = args.mm_total_pixels
    if args.mm_do_resize is not None:
        mm_processor_kwargs["do_resize"] = args.mm_do_resize

    if mm_processor_kwargs:
        existing = extra_body.get("mm_processor_kwargs", {})
        extra_body["mm_processor_kwargs"] = merge_dicts(existing, mm_processor_kwargs)

    chat_template_kwargs = parse_json_object(
        args.chat_template_kwargs_json,
        "--chat_template_kwargs_json",
    )
    if chat_template_kwargs:
        existing = extra_body.get("chat_template_kwargs", {})
        extra_body["chat_template_kwargs"] = merge_dicts(existing, chat_template_kwargs)

    thinking_token_budget = getattr(args, "thinking_token_budget", None)
    if thinking_token_budget is not None:
        extra_body["thinking_token_budget"] = thinking_token_budget

    return extra_body


def encode_file_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime_type(path, media_kind):
    suffix = Path(path).suffix.lstrip(".").lower()
    if media_kind == "video":
        return VIDEO_MIME_TYPES.get(suffix) or mimetypes.guess_type(path)[0] or "video/mp4"
    if media_kind == "audio":
        return AUDIO_MIME_TYPES.get(suffix) or mimetypes.guess_type(path)[0] or "audio/mpeg"
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def build_media_url(path, media_mode, media_kind):
    if media_mode == "file_url":
        return Path(path).resolve().as_uri()
    mime_type = guess_mime_type(path, media_kind)
    return f"data:{mime_type};base64,{encode_file_base64(path)}"


def build_url_content_item(content_type, url):
    return {
        "type": content_type,
        content_type: {"url": url},
    }


def ensure_video_snippet(video_path, start, end, snippet_dir, vid_id, select_seg, resize_multiple=None):
    """Cut video[start:end] with ffmpeg and cache it. Returns the cached path."""
    os.makedirs(snippet_dir, exist_ok=True)
    resize_tag = f"_m{resize_multiple}" if resize_multiple else ""
    snippet_name = f"{vid_id}_seg{select_seg}_{start:.2f}_{end:.2f}{resize_tag}.mp4"
    snippet_path = os.path.join(snippet_dir, snippet_name)

    if os.path.exists(snippet_path) and os.path.getsize(snippet_path) > 0:
        return snippet_path

    duration = max(0.0, end - start)
    if duration <= 0:
        return None

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start}",
        "-i",
        video_path,
        "-t",
        f"{duration}",
    ]
    if resize_multiple:
        cmd.extend(
            [
                "-vf",
                build_pad_filter(resize_multiple),
            ]
        )
    cmd.extend(
        [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        snippet_path,
        ]
    )
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=FFMPEG_TIMEOUT)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if os.path.exists(snippet_path):
            os.remove(snippet_path)
        return None
    return snippet_path


def probe_video_metadata(video_path, timeout=None):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout if timeout is not None else FFMPEG_TIMEOUT,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None

    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    duration = float((payload.get("format") or {}).get("duration") or 0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = stream.get("avg_frame_rate") or "25"
    if fps in ("0/0", "0"):
        fps = "25"
    if duration <= 0 or width <= 0 or height <= 0:
        return None
    return {"duration": duration, "width": width, "height": height, "fps": fps}


def ensure_masked_video_snippet(
    video_path,
    start,
    end,
    snippet_dir,
    vid_id,
    select_seg,
    resize_multiple=None,
    timeout=None,
):
    """Cache a full-duration video where only video[start:end] remains visible."""
    os.makedirs(snippet_dir, exist_ok=True)
    resize_tag = f"_m{resize_multiple}" if resize_multiple else ""
    snippet_name = f"{vid_id}_seg{select_seg}_{start:.2f}_{end:.2f}_masked{resize_tag}.mp4"
    snippet_path = os.path.join(snippet_dir, snippet_name)

    if os.path.exists(snippet_path) and os.path.getsize(snippet_path) > 0:
        return snippet_path

    if end <= start:
        return None

    meta = probe_video_metadata(video_path, timeout=timeout)
    if not meta:
        return None
    duration = meta["duration"]
    start = max(0.0, min(float(start), duration))
    end = max(start, min(float(end), duration))
    span_duration = end - start
    if span_duration <= 0:
        return None

    filter_parts = [
        f"[1:v]setpts=PTS-STARTPTS+{start}/TB[fg]",
        f"[0:v][fg]overlay=shortest=0:eof_action=pass:enable='between(t,{start},{end})'[masked]",
    ]
    video_out = "[masked]"
    if resize_multiple:
        filter_parts.append(f"[masked]{build_pad_filter(resize_multiple)}[vout]")
        video_out = "[vout]"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={meta['width']}x{meta['height']}:r={meta['fps']}:d={duration}",
        "-ss",
        f"{start}",
        "-t",
        f"{span_duration}",
        "-i",
        video_path,
        "-nostdin",
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        video_out,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-shortest",
        snippet_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=timeout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if os.path.exists(snippet_path):
            os.remove(snippet_path)
        return None
    return snippet_path


def build_pad_filter(resize_multiple):
    return (
        f"pad=ceil(iw/{resize_multiple})*{resize_multiple}:"
        f"ceil(ih/{resize_multiple})*{resize_multiple}:"
        "(ow-iw)/2:(oh-ih)/2"
    )


def ensure_padded_video(video_path, video_cache_dir, resize_multiple):
    os.makedirs(video_cache_dir, exist_ok=True)
    source = Path(video_path)
    cache_name = f"{source.stem}_m{resize_multiple}.mp4"
    cache_path = os.path.join(video_cache_dir, cache_name)

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        build_pad_filter(resize_multiple),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        cache_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=FFMPEG_TIMEOUT)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None
    return cache_path


def seg_time_to_seconds(time_str):
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def compute_snippet_span(data):
    select_seg = data["select_seg"]
    seg = data["seg_time"][select_seg - 1]
    return [
        round(seg_time_to_seconds(seg[0]), 3),
        round(seg_time_to_seconds(seg[1]), 3),
    ]


def resolve_source_video_path(data, video_mode, video_dir=None):
    if video_mode == "clip":
        return data.get("video_path")
    if video_mode == "auto" and data.get("video_path"):
        return data.get("video_path")
    if not video_dir:
        return None
    return os.path.join(video_dir, f"{data['id']}.mp4")


def prepare_video_path(data, args):
    # predicted_snippet grounds on the full video first, so it resolves like the
    # non-snippet modes here; the crop happens per caption task afterwards.
    if args.video_mode == "predicted_snippet":
        source_video_path = resolve_source_video_path(data, "id", args.video_dir)
        if not source_video_path or not os.path.exists(source_video_path):
            return None, None
        return source_video_path, source_video_path

    if args.video_mode not in GROUND_TRUTH_SNIPPET_MODES:
        source_video_path = resolve_source_video_path(data, args.video_mode, args.video_dir)
        if not source_video_path:
            return None, None
        resize_multiple = getattr(args, "video_resize_multiple", 0) or 0
        if resize_multiple > 0:
            video_path = ensure_padded_video(
                source_video_path,
                args.video_cache_dir,
                resize_multiple,
            )
            return video_path, source_video_path
        return source_video_path, None

    source_video_path = resolve_source_video_path(data, "id", args.video_dir)
    if not source_video_path or not os.path.exists(source_video_path):
        return None, None

    snippet_dir = getattr(args, "snippet_dir", None)
    select_seg = data["select_seg"]
    gt_span = compute_snippet_span(data)

    if not gt_span or len(gt_span) < 2:
        return None, None

    snippet_fn = ensure_masked_video_snippet if args.video_mode == "masked_snippet" else ensure_video_snippet
    snippet_path = snippet_fn(
        source_video_path,
        float(gt_span[0]),
        float(gt_span[1]),
        snippet_dir,
        data["id"],
        select_seg,
        getattr(args, "snippet_resize_multiple", None),
    )
    return snippet_path, source_video_path


def build_content(prompt_text, media_items, media_order):
    text_item = {"type": "text", "text": prompt_text}
    if media_order == "text_first":
        return [text_item, *media_items]
    if media_order == "media_first":
        return [*media_items, text_item]
    raise ValueError(f"Unsupported media_order: {media_order}")


def extract_thinking(text):
    if not isinstance(text, str):
        return None
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else None


def strip_thinking(text):
    if not isinstance(text, str):
        return text
    return THINK_RE.sub("", text).strip()


def build_prediction(prompt_module, style, raw):
    if style in ("separate", "multi_turn", "multi_turn_no_grounding"):
        clean_raw = {key: strip_thinking(value) for key, value in raw.items()}
        reasoning_trace = {
            key: trace
            for key, value in raw.items()
            if (trace := extract_thinking(value))
        }
        parsed = prompt_module.parse_prompt_response(style, clean_raw)
        result = {"prediction_raw": raw}
        if reasoning_trace:
            result["reasoning_trace"] = reasoning_trace

        inconsistency = {
            "prediction_raw": raw.get("inconsistency"),
            "label": parsed.get("inconsistency", {}).get("label"),
            "reasoning": parsed.get("inconsistency", {}).get("reasoning"),
        }
        prediction = {
            **result,
            "inconsistency": inconsistency,
            "manipulation": dict(inconsistency),
            "veracity": {
                "prediction_raw": raw.get("veracity"),
                "label": parsed.get("veracity", {}).get("label"),
                "reasoning": parsed.get("veracity", {}).get("reasoning"),
            },
        }
        if "grounding" in raw:
            prediction["temporal_grounding"] = {
                "prediction_raw": raw.get("grounding"),
                "prediction_parsed": parsed.get("grounding", []),
            }
        return prediction

    reasoning_trace = extract_thinking(raw)
    clean_raw = strip_thinking(raw)
    parsed = prompt_module.parse_prompt_response(style, clean_raw)
    result = {"prediction_raw": raw}
    if reasoning_trace:
        result["reasoning_trace"] = reasoning_trace

    if not isinstance(parsed, dict):
        return result

    if parsed.get("temporal_grounding") is not None:
        tg_raw = parsed.get("temporal_grounding") or clean_raw
        result["temporal_grounding"] = {
            "prediction_raw": tg_raw,
            "prediction_parsed": prompt_module.parse_timestamps(tg_raw),
        }

    if parsed.get("inconsistency_label") is not None or parsed.get("inconsistency_reasoning") is not None:
        inconsistency = {
            "prediction_raw": raw,
            "label": parsed.get("inconsistency_label"),
            "reasoning": parsed.get("inconsistency_reasoning"),
        }
        if parsed.get("inconsistency_labels") is not None:
            inconsistency["labels"] = parsed.get("inconsistency_labels")
        result["inconsistency"] = inconsistency
        result["manipulation"] = dict(inconsistency)

    if parsed.get("veracity_label") is not None or parsed.get("veracity_reasoning") is not None:
        result["veracity"] = {
            "prediction_raw": raw,
            "label": parsed.get("veracity_label"),
            "reasoning": parsed.get("veracity_reasoning"),
        }

    if parsed.get("label") is not None or parsed.get("reasoning") is not None:
        outputs = prompt_module.get_prompt_style_spec(style).get("outputs", ())
        label_key = "inconsistency" if "inconsistency" in outputs else "veracity"
        prediction = {
            "prediction_raw": raw,
            "label": parsed.get("label"),
            "reasoning": parsed.get("reasoning"),
        }
        result[label_key] = prediction
        if label_key == "inconsistency":
            result["manipulation"] = dict(prediction)

    return result


async def run_chat_completion(client, model_id, content, args, request_semaphore):
    messages = [{"role": "user", "content": content}]
    return await run_chat_completion_messages(client, model_id, messages, args, request_semaphore)


async def run_chat_completion_messages(client, model_id, messages, args, request_semaphore):
    request_kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.top_p is not None:
        request_kwargs["top_p"] = args.top_p

    extra_body = getattr(args, "request_extra_body", None)
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    try:
        async with request_semaphore:
            response = await client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if is_connection_error_text(content):
            raise VLLMConnectionError(content)
        return content
    except Exception as exc:
        if isinstance(exc, VLLMConnectionError) or is_connection_error_text(str(exc)):
            raise VLLMConnectionError(str(exc)) from exc
        return f"Error: {exc}"


async def run_multi_turn_prompt(client, model_id, turns, media_items, args, request_semaphore):
    messages = []
    raw = {}

    for idx, (name, prompt_text) in enumerate(turns):
        content = (
            build_content(prompt_text, media_items, args.media_order)
            if idx == 0
            else [{"type": "text", "text": prompt_text}]
        )
        messages.append({"role": "user", "content": content})
        response_text = await run_chat_completion_messages(
            client,
            model_id,
            messages,
            args,
            request_semaphore,
        )
        raw[name] = response_text
        messages.append({"role": "assistant", "content": response_text})

    return raw


async def run_prompt_style(
    client,
    model_id,
    prompt_module,
    prompt,
    media_items,
    args,
    request_semaphore,
):
    if isinstance(prompt, dict):
        raw_items = await asyncio.gather(
            *[
                run_chat_completion(
                    client,
                    model_id,
                    build_content(prompt_text, media_items, args.media_order),
                    args,
                    request_semaphore,
                )
                for prompt_text in prompt.values()
            ]
        )
        raw = dict(zip(prompt.keys(), raw_items))
    elif isinstance(prompt, list):
        raw = await run_multi_turn_prompt(
            client,
            model_id,
            prompt,
            media_items,
            args,
            request_semaphore,
        )
    else:
        raw = await run_chat_completion(
            client,
            model_id,
            build_content(prompt, media_items, args.media_order),
            args,
            request_semaphore,
        )
    return build_prediction(prompt_module, args.prompt_style, raw)


def first_valid_interval(numbers, duration):
    """First ordered pair inside [0, duration] from a flat list of numbers."""
    for i in range(len(numbers) - 1):
        start, end = float(numbers[i]), float(numbers[i + 1])
        if end > start and 0.0 <= start < duration:
            return [max(0.0, start), min(end, float(duration))]
    return None


async def resolve_predicted_span(
    client,
    model_id,
    prompt_module,
    grounding_prompt,
    media_items,
    duration,
    args,
    request_semaphore,
):
    """Run the grounding pass and return (span, raw_response).

    span is None when the model's answer cannot be parsed into an ordered
    interval inside the video. Callers must treat that as a terminal failure
    rather than substituting the gold span or the full video.
    """
    raw = await run_chat_completion(
        client,
        model_id,
        build_content(grounding_prompt, media_items, args.media_order),
        args,
        request_semaphore,
    )
    if not duration or duration <= 0:
        return None, raw

    numbers = prompt_module.parse_timestamps(strip_thinking(raw))
    return first_valid_interval(numbers, duration), raw


def pad_and_clamp_span(span, pad, duration):
    start = max(0.0, float(span[0]) - float(pad))
    end = min(float(duration), float(span[1]) + float(pad))
    return [start, end] if end > start else None


async def cut_predicted_snippet(source_video_path, span, args, vid_id, select_seg):
    """Cut the predicted interval off the event loop; returns the cached path.

    ensure_video_snippet keys its cache on the interval bounds, so two different
    predicted spans for the same video cannot collide on one file.
    """
    return await asyncio.to_thread(
        ensure_video_snippet,
        source_video_path,
        float(span[0]),
        float(span[1]),
        args.snippet_dir,
        vid_id,
        select_seg,
        getattr(args, "snippet_resize_multiple", None),
    )


def has_connection_error(value):
    if isinstance(value, str):
        return is_connection_error_text(value)
    if isinstance(value, dict):
        return any(has_connection_error(child) for child in value.values())
    if isinstance(value, list):
        return any(has_connection_error(item) for item in value)
    return False


def has_error_prediction(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "prediction_raw" and isinstance(child, str) and child.startswith("Error:"):
                return True
            if has_error_prediction(child):
                return True
    elif isinstance(value, list):
        return any(has_error_prediction(item) for item in value)
    return False


def load_processed_keys(output_file, rerun_errors=False, key_fields=("id", "select_seg")):
    processed = set()
    if not output_file or not os.path.exists(output_file):
        return processed, 0

    retained_lines = []
    dropped_count = 0
    with open(output_file, "r") as f:
        for line in f:
            if not line.strip():
                retained_lines.append(line)
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                retained_lines.append(line)
                continue

            if has_connection_error(data) or (rerun_errors and has_error_prediction(data)):
                dropped_count += 1
                continue

            try:
                processed.add(tuple(data[field] for field in key_fields))
            except KeyError:
                retained_lines.append(line)
                continue
            retained_lines.append(line)

    if dropped_count:
        with open(output_file, "w") as f:
            f.writelines(retained_lines)
    return processed, dropped_count


def add_common_vllm_args(parser, prompt_module, default_model_id):
    # Stashed so validate_common_args can check prompt-style arguments without
    # re-plumbing the module through every caller.
    parser.set_defaults(prompt_module=prompt_module)
    parser.add_argument("--vllm_url", type=str, required=False)
    parser.add_argument("--model_id", type=str, default=default_model_id)
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=False)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument(
        "--video_mode",
        type=str,
        choices=["clip", "id", "auto", "snippet", "masked_snippet", "predicted_snippet"],
        default="auto",
    )
    parser.add_argument(
        "--grounding_prompt_style",
        type=str,
        default="grounding",
        help="Prompt used for the grounding pass of --video_mode predicted_snippet.",
    )
    parser.add_argument(
        "--snippet_pad",
        type=float,
        default=0.0,
        help="Seconds of padding added on each side of a predicted interval before cutting.",
    )
    parser.add_argument(
        "--asr_align",
        type=str,
        choices=["span", "full"],
        default="span",
        help="For predicted_snippet: restrict ASR to the predicted span, or keep the full transcript.",
    )
    parser.add_argument("--video_cache_dir", type=str, default=None)
    parser.add_argument(
        "--video_resize_multiple",
        type=int,
        default=0,
        help="Pad full videos so height/width are divisible by this value. Use 0 to disable.",
    )
    parser.add_argument("--snippet_dir", type=str, default=None)
    parser.add_argument(
        "--snippet_resize_multiple",
        type=int,
        default=32,
        help="Pad cached snippets so height/width are divisible by this value. Use 0 to disable.",
    )
    parser.add_argument("--prompt_mode", type=str, choices=sorted(LEGACY_PROMPT_MODE_MAP), default=None)
    parser.add_argument("--prompt_style", type=str, choices=supported_prompt_style_choices(prompt_module), default=None)
    parser.add_argument("--list_prompt_styles", action="store_true")
    parser.add_argument("--category_filter", type=str, choices=["all", "visual", "audio", "combined"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rerun_errors",
        action="store_true",
        help="Also rerun non-connection Error: rows. Connection error rows are always rerun.",
    )
    parser.add_argument("--max_async_jobs", type=int, default=1)
    parser.add_argument("--max_concurrent_captions", type=int, default=4)
    parser.add_argument(
        "--max_concurrent_requests",
        "--max_async_requests",
        dest="max_concurrent_requests",
        type=int,
        default=12,
    )
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--media_mode", type=str, choices=["file_url", "base64"], default="file_url")
    parser.add_argument("--media_order", type=str, choices=["text_first", "media_first"], default="text_first")
    parser.add_argument("--video_content_type", type=str, default="video_url")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--prompt_fps", "--fps", dest="prompt_fps", type=float, default=1.0)
    parser.add_argument("--mm_fps", type=float, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--mm_min_pixels", type=int, default=None)
    parser.add_argument("--mm_max_pixels", type=int, default=None)
    parser.add_argument(
        "--mm_total_pixels",
        type=int,
        default=None,
        help="Alias for Qwen-style total_pixels; forwarded to vLLM as mm_processor_kwargs.max_pixels.",
    )
    parser.add_argument(
        "--mm_do_resize",
        default=None,
        type=parse_bool_arg,
    )
    parser.add_argument("--video_grid_thw", type=str, default=None)
    parser.add_argument("--extra_body_json", type=str, default=None)
    parser.add_argument("--mm_processor_kwargs_json", type=str, default=None)
    parser.add_argument("--chat_template_kwargs_json", type=str, default=None)
    parser.add_argument("--thinking_token_budget", type=int, default=None,
                        help="Token budget for thinking models (adds thinking_token_budget to extra_body).")


def validate_common_args(parser, args):
    if args.list_prompt_styles:
        return
    missing = [
        name
        for name in ("vllm_url", "input_file", "output_file")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "the following arguments are required unless --list_prompt_styles is used: "
            + ", ".join("--" + name for name in missing)
        )
    if args.video_mode in SNIPPET_VIDEO_MODES and not args.snippet_dir:
        parser.error(f"--snippet_dir is required when --video_mode {args.video_mode} is used")
    if args.video_mode in SNIPPET_VIDEO_MODES and not args.video_dir:
        parser.error(f"--video_dir is required when --video_mode {args.video_mode} is used")
    if args.video_mode == "predicted_snippet":
        grounding_style = getattr(args, "grounding_prompt_style", None)
        if grounding_style not in supported_prompt_style_choices(args.prompt_module):
            parser.error(
                f"--grounding_prompt_style '{grounding_style}' is not a supported prompt style"
            )
    if args.video_resize_multiple and not args.video_cache_dir:
        parser.error("--video_cache_dir is required when --video_resize_multiple is nonzero")
    try:
        args.request_extra_body = build_extra_body(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))


async def process_records(client, args, process_record_fn, iter_input_records):
    processed = set()
    dropped_count = 0
    if not args.overwrite:
        processed, dropped_count = load_processed_keys(
            args.output_file,
            rerun_errors=args.rerun_errors,
        )
        if dropped_count:
            print(f"Found {dropped_count} incomplete/error output rows; rerunning them.")

    records = list(iter_input_records(args.input_file))
    if args.max_records is not None:
        records = records[: args.max_records]

    pending_records = [
        data for data in records if (data["id"], data["select_seg"]) not in processed
    ]
    request_semaphore = asyncio.Semaphore(max(1, args.max_concurrent_requests))
    work_queue = asyncio.Queue()
    for data in pending_records:
        work_queue.put_nowait(data)

    from tqdm.asyncio import tqdm

    open_mode = "w" if args.overwrite else "a"
    stop_event = asyncio.Event()
    connection_error = None

    with open(args.output_file, open_mode) as f_out:
        async def worker(pbar):
            nonlocal connection_error
            while True:
                if stop_event.is_set():
                    return
                try:
                    data = work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                try:
                    result_data = await process_record_fn(
                        client,
                        args.model_id,
                        data,
                        args,
                        request_semaphore,
                    )
                    if result_data and not stop_event.is_set():
                        f_out.write(json.dumps(result_data) + "\n")
                        f_out.flush()
                except VLLMConnectionError as exc:
                    connection_error = exc
                    stop_event.set()
                finally:
                    pbar.update(1)
                    work_queue.task_done()

        with tqdm(total=len(pending_records), desc="Processing Videos") as pbar:
            workers = [
                asyncio.create_task(worker(pbar))
                for _ in range(min(max(1, args.max_async_jobs), len(pending_records)))
            ]
            if workers:
                await asyncio.gather(*workers)

    if connection_error is not None:
        raise SystemExit(
            "Stopping without writing the failed record because vLLM returned a "
            f"connection error: {connection_error}"
        )
