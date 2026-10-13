# Finetuning

Training is delegated to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
This repository contributes only the preprocessing step: expanding annotations
into caption tasks, rendering the prompt, and building the supervision target.
There is no custom trainer, collator, or LoRA wiring to maintain.

```
annotations ──> groundinc.training.prepare_sft ──> ShareGPT JSON ──> llamafactory-cli train
```

## 1. Install

```bash
pip install -e ".[train]"
```

Pinned to `llamafactory==0.9.4`. The video keys used in the shipped configs
(`video_fps`, `video_maxlen`, `video_max_pixels`) and the template ids
(`qwen3_vl`, `gemma3`) have changed across releases — if you upgrade, check
these against `llamafactory-cli train --help` before trusting a run.

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

Each record becomes ~33 samples of the form

```json
{"messages": [{"role": "user", "content": "<video>...prompt..."},
              {"role": "assistant", "content": "...target..."}],
 "videos": ["videos/<id>.mp4"]}
```

`prepare_sft` reuses `groundinc.data.dataset.build_caption_tasks` — the same
function the evaluators call — so a training sample and an inference item are
the same object by construction.

### Check the length budget first

```bash
python -m groundinc.training.prepare_sft --input ... --cutoff_len 8192 --report_only
```

This prints the length distribution and how many samples exceed the budget on
text alone. Run it before committing to a `cutoff_len`: over-length samples are
truncated silently by the trainer, and a budget that looks fine can discard a
large fraction of supervision once video tokens are added.

## 3. Train

```bash
llamafactory-cli train src/groundinc/training/configs/qwen3vl_8b_lora.yaml
```

| Config | Model | cutoff_len | LR |
| --- | --- | --- | --- |
| `qwen3vl_8b_lora.yaml` | Qwen3-VL-8B-Instruct | 8192 | 2e-4 |
| `qwen3vl_32b_lora.yaml` | Qwen3-VL-32B-Instruct | 6144 | 5e-5 |
| `gemma_lora.yaml` | Gemma-4-E4B-it | 16384 | 2e-4 |

All three use LoRA r=32, α=64, dropout 0.05, rsLoRA, frozen vision tower,
2 epochs, cosine schedule, warmup 0.05, effective batch 16, bf16, gradient
checkpointing, DeepSpeed ZeRO-2, seed 42.

The Gemma config sets `video_maxlen: 96` explicitly. On the original Gemma path
`max_pixels` was silently ignored during frame extraction, so the frame cap is
the setting that actually bound the budget; making it explicit avoids
reintroducing that surprise.

## 4. Match the prompt at inference

Each training prompt has exactly one inference counterpart. Pairing a checkpoint
with a different `--prompt_style` produces output the evaluator's parser cannot
read — the failure looks like poor accuracy rather than a configuration error.

| `--prompt_name` | task_mode | evaluate with `--prompt_style` |
| --- | --- | --- |
| `veracity` | `veracity_only` | `temporal_veracity_only` |
| `no_evidence` | `no_evidence` | `inconsistency_veracity_only` |
| `full` | `full` | `temporal_inconsistency_veracity` |
| `inconsistency` | `inconsistency` | `temporal_inconsistency_veracity` |
| `inconsistency_no_grounding` | `inconsistency_no_grounding` | `inconsistency_veracity_only` |

`prepare_sft` writes the correct pairing into `<output>.meta.json` as
`inference_prompt_style` and prints it on completion, so the mapping travels
with the dataset rather than living in someone's memory.

## Custom prompts

```bash
python -m groundinc.training.prepare_sft ... \
  --prompt_template_path my_prompt.txt --task_mode inconsistency
```

Placeholders: `{fps}`, `{asr_content}`, `{caption}`, `{inconsistency_definitions}`.
`--task_mode` is required with a custom template, since it selects the target
format.

## Not included

Per-label F1 during training. Metrics are scoring code, which this release does
not ship — the runners emit predictions only.

---

## Interval-bank localizer (second training backend)

Alongside the LLaMA-Factory SFT path above, `groundinc.training.localizer` is a
**custom-trainer** backend for the categorical temporal localizer. It predicts a
single interval-ID token over a deterministic per-video candidate **bank** (widths
8/16/32/64 s), with three objectives selected in the config:

| Objective | `objective` | `lambda_pair` |
| --- | --- | --- |
| Grounding CE (control) | `categorical` | `0.0` |
| CE + paired Jensen–Shannon | `categorical` | `0.1` |
| Class-balanced sigmoid BCE | `sigmoid` | `0.0` |
| Sigmoid BCE + Bernoulli-JS | `sigmoid` | `0.1` |

Train:

```bash
python -m groundinc.training.localizer.train \
  --config src/groundinc/training/localizer/configs/qwen3vl_8b_ce.yaml
```

It reuses the shared caption-task construction, so a localizer training item and
an inference item cannot drift apart. Inference (the localizer selects a bank
interval on the full video) runs through:

```bash
python -m groundinc.evaluation.localizer_infer \
  --input_file <records>.jsonl --output_file outputs/localizer.jsonl \
  --vllm_url http://localhost:8000/v1 --model_id <served_model> \
  --bank_config <output_dir>/localizer_bank.json \
  --video_dir <video_dir> --whisperx_dir <asr_dir>
```

The selected span then feeds the snippet-verify path exactly like
`predicted_snippet`, so grounding quality transfers to the downstream verifier.

### Serving a Gemma localizer — merge the adapter first

On the Gemma path, serving a LoRA adapter through vLLM leaves it **silently
inert** (the adapter loads but is not applied, and a naive
`merge_and_unload()` + `save_pretrained()` drops tied `k_norm` tensors). Merge the
adapter into the base weights at the **safetensors level** and serve the merged
full model:

```bash
python -m groundinc.training.localizer.merge_adapter \
  --base_snapshot <hf_snapshot_dir> --adapter <checkpoint_dir> --out <merged_dir>
```

Qwen LoRA serving is unaffected and needs no merge. The sigmoid objectives tend to
overfit, so save frequently (`save_steps: 100`) and pick the early grounding peak.
