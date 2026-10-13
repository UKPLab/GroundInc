#!/usr/bin/env bash
set -euo pipefail

# Gemini sees the source video plus the timestamped caption timeline.

python -m groundinc.data_generation.generate_with_gemini \
  --input_jsonl examples/input_record.jsonl \
  --output_jsonl outputs/generated_groundinc.jsonl \
  --prompt1_file prompts/step1_prompt.md \
  --prompt2_file prompts/step2_prompt.md \
  --video_dir videos \
  --model gemini-2.5-flash \
  --max_workers 4
