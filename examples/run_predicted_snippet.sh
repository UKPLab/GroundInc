#!/usr/bin/env bash
set -euo pipefail

# Two passes per caption task: ground on the full video, then verify on the
# model's own predicted interval with ASR windowed to the same span.
# Requires ffmpeg with libx264 on PATH.

python -m groundinc.evaluation.vllm_asr \
  --vllm_url http://localhost:8000/v1 \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --input_file data/groundinc_news.jsonl \
  --output_file outputs/qwen3_vl_predicted_snippet.jsonl \
  --video_mode predicted_snippet \
  --video_dir videos \
  --snippet_dir snippets/predicted \
  --asr_dir asr \
  --grounding_prompt_style grounding \
  --prompt_style inconsistency_veracity_only \
  --snippet_pad 0.0 \
  --asr_align span \
  --max_async_jobs 4 \
  --max_concurrent_requests 8
