# Finetuning

Finetuning uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). This
repository provides the data preparation step and the training configs.

## 1. Install

```bash
conda create -n groundinc python=3.10 -y
conda activate groundinc
pip install -r requirements.txt
pip install -e .
pip install -e ".[train]"
```

## 2. Build the dataset

```bash
python -m groundinc.training.prepare_sft \
  --input data/groundinc_other.jsonl \
  --video_dir videos \
  --asr_dir asr \
  --prompt_name inconsistency_no_grounding \
  --require_video \
  --output sft/train.json

cp src/groundinc/training/dataset_info.json sft/
```

## 3. Train

```bash
llamafactory-cli train src/groundinc/training/configs/qwen3vl_8b_lora.yaml
```

| Config | Model | cutoff_len | LR |
| --- | --- | --- | --- |
| `qwen3vl_8b_lora.yaml` | Qwen3-VL-8B-Instruct | 6144 | 2e-4 |
| `qwen3vl_32b_lora.yaml` | Qwen3-VL-32B-Instruct | 8192 | 2e-4 |
| `gemma_lora.yaml` | Gemma-4-E4B-it | 16384 | 2e-4 |

All three use LoRA, 2 epochs, per-device batch 1, and gradient accumulation 16,
with videos sampled at 1 FPS. The remaining settings are in the config files.

## 4. Prompt styles

Each `--prompt_name` used for training has a matching `--prompt_style` for
inference.

| `--prompt_name` | `--prompt_style` |
| --- | --- |
| `veracity` | `temporal_veracity_only` |
| `no_evidence` | `inconsistency_veracity_only` |
| `full` | `temporal_inconsistency_veracity` |
| `inconsistency` | `temporal_inconsistency_veracity` |
| `inconsistency_no_grounding` | `inconsistency_veracity_only` |
