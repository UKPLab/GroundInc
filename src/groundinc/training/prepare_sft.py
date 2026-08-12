"""Turn GroundInc annotations into a LLaMA-Factory SFT dataset."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from groundinc.data.dataset import (
    build_caption_tasks,
    iter_input_records,
    resolve_video_path,
)
from groundinc.training.constants import PROMPT_NAME_TO_INFERENCE_STYLE
from groundinc.training.prompts import prompt_names, render_prompt, resolve_prompt
from groundinc.training.targets import build_ground_truth_answer

VIDEO_TOKEN = "<video>"


def load_asr_text(record, asr_dir, asr_suffix="_asr.json"):
    """Resolve the ASR transcript for a record."""
    for field in ("asr_text", "asr", "transcript"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for key in ("prompt", "asr_text", "text", "transcript"):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

    if not asr_dir:
        return ""

    path = os.path.join(asr_dir, f"{record['id']}{asr_suffix}")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(payload, str):
        return payload.strip()
    for key in ("prompt", "asr_text", "text", "transcript"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    segments = payload.get("segments")
    if isinstance(segments, list):
        return " ".join(
            (seg.get("text") or "").strip()
            for seg in segments
            if isinstance(seg, dict) and (seg.get("text") or "").strip()
        )
    return ""


def build_samples(args, template, task_mode):
    stats = Counter()
    lengths = []
    samples = []

    for record in iter_input_records(args.input):
        video_path = resolve_video_path(record, args.video_mode, args.video_dir)
        if not video_path:
            stats["skipped_unresolved_video"] += 1
            continue
        if args.require_video and not os.path.exists(video_path):
            stats["skipped_missing_video"] += 1
            continue

        asr_text = load_asr_text(record, args.asr_dir, args.asr_suffix)
        if not asr_text:
            stats["records_without_asr"] += 1
            if args.require_asr:
                stats["skipped_missing_asr"] += 1
                continue

        select_seg = record["select_seg"]
        visual_evidence = record["video_captions"][select_seg - 1]
        audio_evidence = record["audio_captions"][select_seg - 1]
        stats["records_used"] += 1

        for task in build_caption_tasks(record, args.video_mode):
            prompt = render_prompt(
                template,
                fps=args.fps,
                asr_content=asr_text,
                caption=task["caption"],
            )
            target = build_ground_truth_answer(
                task,
                visual_evidence=visual_evidence,
                audio_evidence=audio_evidence,
                task_mode=task_mode,
            )
            samples.append(
                {
                    "messages": [
                        {"role": "user", "content": f"{VIDEO_TOKEN}{prompt}"},
                        {"role": "assistant", "content": target},
                    ],
                    "videos": [video_path],
                }
            )
            lengths.append(len(prompt) + len(target))
            stats["samples"] += 1
            stats[f"task::{'true' if task['type'] in ('original_true', 'rephrased_true') else 'manipulated'}"] += 1

    return samples, lengths, stats


def report(lengths, stats, args, task_mode):
    print(f"\nprompt_name={args.prompt_name}  task_mode={task_mode}  fps={args.fps}")
    print(f"  records used            {stats['records_used']}")
    print(f"  samples                 {stats['samples']}")
    print(f"  true / manipulated      {stats['task::true']} / {stats['task::manipulated']}")
    for key in ("skipped_unresolved_video", "skipped_missing_video",
                "skipped_missing_asr", "records_without_asr"):
        if stats[key]:
            print(f"  {key:<23} {stats[key]}")

    if not lengths:
        return
    lengths = sorted(lengths)
    def pct(p):
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    print("  chars per sample        "
          f"min={lengths[0]} p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={lengths[-1]}")
    approx = [n / 3.5 for n in lengths]
    print(f"  approx text tokens      p50={approx[len(approx)//2]:.0f} "
          f"p99={approx[int(0.99*len(approx))]:.0f} max={approx[-1]:.0f}")
    if args.cutoff_len:
        over = sum(1 for n in approx if n > args.cutoff_len)
        print(f"  OVER cutoff_len={args.cutoff_len}: {over} samples "
              f"({100 * over / len(approx):.1f}%) would be truncated by text alone, "
              "before video tokens are added")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="GroundInc annotation JSONL.")
    parser.add_argument("--output", help="Destination .json/.jsonl. Omit with --report_only.")
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--video_mode", default="id", choices=["id", "clip", "auto"])
    parser.add_argument("--asr_dir", default=None)
    parser.add_argument("--asr_suffix", default="_asr.json")
    parser.add_argument("--prompt_name", default="inconsistency_no_grounding",
                        help=f"One of: {prompt_names()}")
    parser.add_argument("--task_mode", default="auto")
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--prompt_template_path", default=None)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--cutoff_len", type=int, default=None,
                        help="Report how many samples exceed this token budget.")
    parser.add_argument("--require_video", action="store_true",
                        help="Skip records whose video file is absent from disk.")
    parser.add_argument("--require_asr", action="store_true")
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args()

    if not args.report_only and not args.output:
        parser.error("--output is required unless --report_only is used")

    template, task_mode = resolve_prompt(
        prompt_name=args.prompt_name,
        task_mode=args.task_mode,
        prompt_template=args.prompt_template,
        prompt_template_path=args.prompt_template_path,
    )

    samples, lengths, stats = build_samples(args, template, task_mode)
    report(lengths, stats, args, task_mode)

    if args.report_only:
        return 0
    if not samples:
        print("\nERROR: no samples produced.")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=1)

    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    inference_style = PROMPT_NAME_TO_INFERENCE_STYLE.get(args.prompt_name)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": os.path.abspath(args.input),
                "prompt_name": args.prompt_name,
                "task_mode": task_mode,
                "fps": args.fps,
                "video_mode": args.video_mode,
                "asr_dir": args.asr_dir,
                "samples": stats["samples"],
                "records_used": stats["records_used"],
                "inference_prompt_style": inference_style,
            },
            fh,
            indent=2,
        )

    print(f"\nwrote {len(samples)} samples -> {args.output}")
    print(f"      metadata            -> {meta_path}")
    if inference_style:
        print(f"      evaluate this checkpoint with --prompt_style {inference_style}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
