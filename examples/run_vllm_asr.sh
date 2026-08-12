#!/usr/bin/env bash
set -euo pipefail

python -m groundinc.evaluation.vllm_asr \
  --vllm_url http://localhost:8000/v1 \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --input_file outputs/generated_groundinc.jsonl \
  --output_file outputs/qwen3_vl_asr.jsonl \
  --video_mode auto \
  --asr_dir asr \
  --prompt_style temporal_inconsistency_veracity \
  --max_async_jobs 4 \
  --max_concurrent_requests 8
