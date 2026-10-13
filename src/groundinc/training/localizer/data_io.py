from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from .constants import (
    EXCLUDED_LABELS,
    INCONSISTENCY_LABELS,
    MODALITY_LABELS,
    VERACITY_LABELS,
)
from groundinc.training.prompts import validate_task_mode


logger = logging.getLogger(__name__)


def seg_time_to_seconds(time_str: str) -> float:
    hours, minutes, seconds = time_str.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def infer_video_mode(record: dict, video_mode: str) -> str:
    if video_mode == "auto":
        return "clip" if record.get("video_path") else "id"
    return video_mode


def resolve_video_path(record: dict, video_mode: str, video_dir: Optional[str]) -> str:
    resolved_mode = infer_video_mode(record, video_mode)
    if resolved_mode == "clip":
        video_path = record.get("video_path")
    else:
        video_path = os.path.join(video_dir, f"{record['id']}.mp4") if video_dir else None
    if not video_path:
        raise ValueError("Could not resolve video path. Set video_dir or use clip mode.")
    return video_path


def compute_ground_truth_span(record: dict, video_mode: str) -> List[float]:
    if infer_video_mode(record, video_mode) == "clip":
        return record.get("relative_seg_gt") or []

    select_seg = record["select_seg"]
    start, end = record["seg_time"][select_seg - 1]
    return [round(seg_time_to_seconds(start), 3), round(seg_time_to_seconds(end), 3)]


def iter_input_records(jsonl_path: str) -> Iterable[dict]:
    with open(jsonl_path, "r") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_legacy_manipulations(section):
    if not isinstance(section, dict):
        return
    for item in section.get("step2_execution", []):
        if isinstance(item, dict):
            yield {
                "method": item.get("manipulation_method"),
                "entity_type": item.get("entity_type"),
                "false_caption": item.get("false_caption"),
                "type": item.get("type"),
            }


def build_caption_tasks(record: dict, video_mode: str) -> List[dict]:
    gt_span = compute_ground_truth_span(record, video_mode)
    select_seg = record["select_seg"]
    fm = record.get("forensic_mentalization", {})

    visual_orig = record["video_captions"][select_seg - 1]
    audio_orig = record["audio_captions"][select_seg - 1]
    combined_orig = f"{visual_orig}. {audio_orig}"
    tasks = []

    for caption, category in [
        (visual_orig, "visual"),
        (audio_orig, "audio"),
        (combined_orig, "combined"),
    ]:
        tasks.append(
            {
                "category": category,
                "type": "original_true",
                "caption": caption,
                "manipulation_method_gt": "No Manipulation",
                "entity_type": None,
                "manipulated_modality": "none",
                "ground_truth_span": gt_span,
            }
        )

    for caption, category in [
        (fm.get("true_visual"), "visual"),
        (fm.get("true_audio"), "audio"),
        (fm.get("true_combined"), "combined"),
    ]:
        if caption:
            tasks.append(
                {
                    "category": category,
                    "type": "rephrased_true",
                    "caption": caption,
                    "manipulation_method_gt": "No Manipulation",
                    "entity_type": None,
                    "manipulated_modality": "none",
                    "ground_truth_span": gt_span,
                }
            )

    visual_manips = list(fm.get("visual_only_manipulations", []))
    audio_manips = list(fm.get("audio_only_manipulations", []))
    combined_manips = list(fm.get("combined_manipulations", []))

    if not visual_manips and isinstance(fm.get("visual"), dict):
        visual_manips = list(iter_legacy_manipulations(fm["visual"]))
    if not audio_manips and isinstance(fm.get("audio"), dict):
        audio_manips = list(iter_legacy_manipulations(fm["audio"]))
    if not combined_manips and isinstance(fm.get("combined"), dict):
        combined_manips = list(iter_legacy_manipulations(fm["combined"]))

    for manipulations, category in [(visual_manips, "visual"), (audio_manips, "audio")]:
        for item in manipulations:
            method = item.get("method")
            caption = item.get("false_caption")
            if not method or not caption or method in EXCLUDED_LABELS:
                continue
            tasks.append(
                {
                    "category": category,
                    "type": method,
                    "caption": caption,
                    "manipulation_method_gt": method,
                    "entity_type": item.get("entity_type"),
                    "manipulated_modality": category,
                    "ground_truth_span": gt_span,
                    "refutation": item.get("refutation"),
                }
            )

    for item in combined_manips:
        method = item.get("method")
        caption = item.get("false_caption")
        if not method or not caption or method in EXCLUDED_LABELS:
            continue

        item_type = item.get("type") or ""
        if "manipulated_visual" in item_type:
            manipulated_modality = "visual"
        elif "manipulated_audio" in item_type:
            manipulated_modality = "audio"
        else:
            manipulated_modality = "combined"

        tasks.append(
            {
                "category": "combined",
                "type": method,
                "caption": caption,
                "manipulation_method_gt": method,
                "entity_type": item.get("entity_type"),
                "manipulated_modality": manipulated_modality,
                "ground_truth_span": gt_span,
                "refutation": item.get("refutation"),
            }
        )

    return tasks


def load_asr_text(whisperx_dir: str, vid_id: str) -> str:
    asr_path = os.path.join(whisperx_dir, f"{vid_id}_asr.json")
    if not os.path.exists(asr_path):
        return ""
    try:
        with open(asr_path) as handle:
            return json.load(handle).get("prompt", "")
    except Exception as exc:
        logger.warning("Could not read ASR file %s: %s", asr_path, exc)
        return ""


def extract_frames_ffmpeg(
    video_path: str, fps: float, max_frames: Optional[int] = None
) -> List[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_pattern = os.path.join(tmp_dir, "frame_%04d.png")
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "vfr",
            output_pattern,
            "-y",
            "-loglevel",
            "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace")
            raise RuntimeError(f"FFmpeg failed for {video_path}: {stderr}") from exc

        frame_paths = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.png")))
        if not frame_paths:
            raise ValueError(f"FFmpeg extracted no frames from {video_path}")

        # Honor a frame cap with uniform subsampling so the fallback matches the
        # main decoder's temporal sampling policy (otherwise a decode failure
        # silently changes the model's input distribution).
        if max_frames is not None and len(frame_paths) > max_frames:
            idxs = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=np.int64)
            frame_paths = [frame_paths[i] for i in idxs]

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB"))
        return frames


def extract_frames_decord(
    video_path: str,
    fps: float,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Decord found no frames in {video_path}")

    native_fps = float(reader.get_avg_fps() or 0)
    if native_fps <= 0:
        native_fps = 30.0
    duration = total_frames / native_fps
    num_frames = int(duration * fps)
    if num_frames <= 0:
        raise ValueError(
            f"Video is too short to sample at {fps} FPS without forcing a minimum frame: {video_path}"
        )
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    num_frames = min(num_frames, total_frames)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)
    frames = reader.get_batch(indices).asnumpy()
    metadata = {
        "total_num_frames": total_frames,
        "fps": native_fps,
        "duration": duration,
        "width": int(frames.shape[2]) if frames.ndim == 4 else None,
        "height": int(frames.shape[1]) if frames.ndim == 4 else None,
        "video_backend": "decord",
        "frames_indices": indices.tolist(),
    }
    return frames, metadata


def map_modality(raw: Optional[str]) -> str:
    if not raw or raw == "none":
        return "none"
    if raw == "combined":
        return "both"
    return raw


def build_ground_truth_answer(
    task: dict,
    *,
    visual_evidence: str,
    audio_evidence: str,
    task_mode: str,
) -> str:
    validate_task_mode(task_mode)
    is_true = task["type"] in {"original_true", "rephrased_true"}
    veracity = "TRUE" if is_true else "FALSE"

    if task_mode == "veracity_only":
        return f"veracity: {veracity}"

    manipulated_modality = "none" if is_true else map_modality(task.get("manipulated_modality"))
    manipulation_type = "none" if is_true else task.get("manipulation_method_gt", "none")
    refutation = "N/A" if is_true else (task.get("refutation") or "N/A")

    if task_mode in {"inconsistency", "inconsistency_no_grounding"}:
        span = task.get("ground_truth_span") or []
        start = round(span[0], 1) if len(span) >= 2 else 0.0
        end = round(span[1], 1) if len(span) >= 2 else 0.0
        inconsistency_type = "No Inconsistency" if is_true else manipulation_type
        veracity_title = "True" if is_true else "False"
        if is_true:
            inconsistency_reasoning = (
                "The caption accurately reflects the relevant video segment and ASR evidence."
            )
            veracity_reasoning = (
                "Because no inconsistency is present, the caption is labeled True."
            )
        else:
            inconsistency_reasoning = refutation
            veracity_reasoning = (
                "Because the caption contains this inconsistency, it is labeled False."
            )
        output_lines = []
        if task_mode == "inconsistency":
            output_lines.append(
                f"Temporal Grounding: The event happens in {start} - {end} seconds"
            )
        output_lines.extend(
            [
                f"Inconsistency Type: {inconsistency_type}",
                f"Inconsistency Reasoning: {inconsistency_reasoning}",
                f"Veracity: {veracity_title}",
                f"Veracity Reasoning: {veracity_reasoning}",
            ]
        )
        return "\n".join(output_lines)

    if task_mode == "no_evidence":
        return (
            f"evidence_visual: {visual_evidence}\n"
            f"evidence_audio: {audio_evidence}\n"
            f"veracity: {veracity}\n"
            f"manipulated_modality: {manipulated_modality}\n"
            f"manipulation_type: {manipulation_type}\n"
            f"refutation: {refutation}"
        )

    span = task.get("ground_truth_span") or []
    start = round(span[0], 1) if len(span) >= 2 else 0.0
    end = round(span[1], 1) if len(span) >= 2 else 0.0
    return (
        f"evidence_start: {start}\n"
        f"evidence_end: {end}\n"
        f"evidence_visual: {visual_evidence}\n"
        f"evidence_audio: {audio_evidence}\n"
        f"veracity: {veracity}\n"
        f"manipulated_modality: {manipulated_modality}\n"
        f"manipulation_type: {manipulation_type}\n"
        f"refutation: {refutation}"
    )


def parse_model_output(text: str) -> dict:
    result = {
        "evidence_start": None,
        "evidence_end": None,
        "evidence_visual": None,
        "evidence_audio": None,
        "veracity": None,
        "manipulated_modality": None,
        "manipulation_type": None,
        "inconsistency_type": None,
        "inconsistency_reasoning": None,
        "veracity_reasoning": None,
        "refutation": None,
    }
    for line in text.strip().splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "Temporal Grounding":
            match = re.search(r"([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)", value)
            if match:
                result["evidence_start"] = float(match.group(1))
                result["evidence_end"] = float(match.group(2))
        elif key == "Inconsistency Type":
            result["inconsistency_type"] = value if value in INCONSISTENCY_LABELS else None
            result["manipulation_type"] = result["inconsistency_type"]
        elif key == "Inconsistency Reasoning":
            result["inconsistency_reasoning"] = value
            result["refutation"] = value
        elif key == "Veracity Reasoning":
            result["veracity_reasoning"] = value
        elif key == "Veracity":
            normalized = value.upper()
            result["veracity"] = normalized if normalized in VERACITY_LABELS else None
        elif key == "evidence_start":
            try:
                result[key] = float(value)
            except ValueError:
                pass
        elif key == "evidence_end":
            try:
                result[key] = float(value)
            except ValueError:
                pass
        elif key == "veracity":
            normalized = value.upper()
            result[key] = normalized if normalized in VERACITY_LABELS else None
        elif key == "manipulated_modality":
            normalized = value.lower()
            result[key] = normalized if normalized in MODALITY_LABELS else None
        elif key in result:
            result[key] = value
    return result
