#!/usr/bin/env bash
set -euo pipefail

# Build a LLaMA-Factory SFT dataset from GroundInc annotations.
# Run with --report_only first to size cutoff_len (see docs/TRAINING.md).

python -m groundinc.training.prepare_sft \
  --input data/groundinc_other.jsonl \
  --video_dir videos \
  --asr_dir asr \
  --prompt_name inconsistency_no_grounding \
  --require_video \
  --cutoff_len 8192 \
  --output sft/train.json

cp src/groundinc/training/dataset_info.json sft/
llamafactory-cli train src/groundinc/training/configs/qwen3vl_8b_lora.yaml
