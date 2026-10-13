# GroundInc

Anonymous release code and annotations for GroundInc: generating manipulated
audio-visual captions, running VLM inference over them, and finetuning.

## Contents

- `data/`: the GroundInc annotations — single-span news (`groundinc_news.jsonl`) and
  cross-domain (`groundinc_other.jsonl`) sets, plus the compound multi-span /
  multi-label news set (`groundinc_news_compound.jsonl`).
- `prompts/`: the two-step Gemini prompts.
- `src/groundinc/data_generation/`: the annotation pipeline.
- `src/groundinc/evaluation/`: OpenAI-compatible vLLM inference runners, incl. the
  GroundSnip localizer runner (`localizer_infer.py`).
- `src/groundinc/training/`: SFT dataset preparation + LLaMA-Factory configs, and
  `training/localizer/`: the GroundSnip interval-bank localizer (custom trainer).
- `src/groundinc/data/`: JSONL loading, caption-task construction (single-span and
  `compound.py` for multi-span/label), prompt registry and parsers.
- `docs/`: data format, training, reproducibility.
- `scripts/`: `smoke_test.sh` (offline sanity check), `prepare_release_data.py`
  (single-span), `prepare_compound_data.py` (compound).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .          # generation + inference
pip install -e ".[train]" # adds LLaMA-Factory for finetuning
```

`ffmpeg`/`ffprobe` **built with libx264** are required for the snippet video
modes. A build without libx264 fails to cut clips.

## What is and isn't included

Included: annotations, the generation pipeline, inference runners, and
finetuning preparation.

**Not included: any scoring, metric, or analysis code.** The runners write
parsed predictions to JSONL and stop. Computing accuracy, macro-F1, or IoU is
left to you.

Also not included: videos, transcripts, and media-preprocessing scripts. See
[`docs/DATA.md`](docs/DATA.md) for what you must supply.

## Generate annotations

```bash
export GEMINI_API_KEY=...
python -m groundinc.data_generation.generate_with_gemini \
  --input_jsonl <base captions>.jsonl \
  --output_jsonl outputs/generated_groundinc.jsonl \
  --prompt1_file prompts/step1_prompt.md \
  --prompt2_file prompts/step2_prompt.md \
  --video_dir videos
```

The generator sees the source video together with the complete timestamped
caption timeline, the annotated target segment, and the manipulation taxonomy.
Per record and per modality, stage 1 rephrases the target caption and plans one
manipulation per applicable taxonomy method; stage 2 executes those plans into
false captions with refutations; the two modalities are then cross-stitched into
combined items sharing the original event's gold evidence span.

## Run inference

```bash
python -m groundinc.evaluation.vllm_asr \
  --vllm_url http://localhost:8000/v1 \
  --input_file data/groundinc_news.jsonl \
  --output_file outputs/model_asr.jsonl \
  --video_mode auto --asr_dir asr \
  --prompt_style temporal_inconsistency_veracity
```

`--list_prompt_styles` shows the registered styles. `vllm_audio` is the
video+audio counterpart.

### Video modes

| Mode | Evidence shown to the model |
| --- | --- |
| `auto` / `id` / `clip` | Whole video |
| `snippet` | Gold segment, cut out |
| `masked_snippet` | Full-length video with everything outside the gold segment blacked out |
| `predicted_snippet` | The model's **own** predicted interval |

`predicted_snippet` runs two passes per caption task: a grounding call on the
full video, then verification on the crop with ASR windowed to the same span
(`--asr_align span`). A task whose grounding cannot be parsed into an interval
inside the video is written as a terminal failure — the gold span is never
substituted in. Supported by the ASR runner only.

## Finetune

```bash
python -m groundinc.training.prepare_sft \
  --input data/groundinc_other.jsonl \
  --video_dir videos --asr_dir asr \
  --prompt_name inconsistency_no_grounding \
  --output sft/train.json
cp src/groundinc/training/dataset_info.json sft/
llamafactory-cli train src/groundinc/training/configs/qwen3vl_8b_lora.yaml
```

Preparation reuses the same `build_caption_tasks` the evaluators call, so a
training sample and an inference item cannot drift apart. See
[`docs/TRAINING.md`](docs/TRAINING.md), particularly the table mapping each
training prompt to the `--prompt_style` its checkpoint must be evaluated with.

## GroundSnip: interval-bank temporal localizer

GroundSnip trains a categorical **temporal localizer** that selects one evidence
interval from a deterministic per-video candidate bank, snips that interval, and
hands the crop to a verifier — so grounding quality transfers to the downstream
veracity / inconsistency tasks. It is a second training backend (a custom trainer,
separate from the LLaMA-Factory SFT above) under `src/groundinc/training/localizer/`.

**Train** (objectives: `categorical` CE, CE + paired Jensen–Shannon via
`lambda_pair: 0.1`, or class-balanced `sigmoid` BCE / Bernoulli-JS):

```bash
python -m groundinc.training.localizer.train \
  --config src/groundinc/training/localizer/configs/qwen3vl_8b_ce.yaml
```

Edit the config's `<placeholder>` paths first (train JSONL, `video_dir`, `asr_dir`,
`output_dir`). Training writes `localizer_bank.json` (bank + single-token ID vocab)
next to the checkpoint. It reuses `build_caption_tasks`, so a localizer training item
and an inference item cannot drift apart.

**Infer** (the localizer picks a bank interval on the full video; the span then
feeds the same snippet-verify path as `predicted_snippet`):

```bash
python -m groundinc.evaluation.localizer_infer \
  --input_file data/groundinc_news.jsonl --output_file outputs/localizer.jsonl \
  --vllm_url http://localhost:8000/v1 --model_id <served_model> \
  --bank_config <output_dir>/localizer_bank.json \
  --video_dir videos --whisperx_dir asr
```

**Serving a Gemma localizer — merge first.** vLLM leaves a Gemma LoRA adapter
silently inert, so merge it into the base weights at the safetensors level and serve
the merged model (Qwen LoRA serving is unaffected):

```bash
python -m groundinc.training.localizer.merge_adapter \
  --base_snapshot <hf_snapshot_dir> --adapter <checkpoint_dir> --out <merged_dir>
```

The sigmoid objectives overfit; save often (`save_steps: 100`) and early-stop at the
grounding peak. See [`docs/TRAINING.md`](docs/TRAINING.md) for the full notes.

## Compound set (multi-span / multi-label)

`data/groundinc_news_compound.jsonl` holds the compound news set: each record's
`multi_inconsistency` carries three subsets — `multi_span` (≥2 regions, 1 label),
`multi_label` (1 region, ≥2 labels), and `multi_span_label` — with list-valued gold
`ground_truth_spans` and `methods`. Expand it with the dedicated builder:

```python
from groundinc.data.compound import build_multi_caption_tasks, SUBSETS
import json
records = [json.loads(l) for l in open("data/groundinc_news_compound.jsonl")]
for subset in SUBSETS:                      # multi_span, multi_label, multi_span_label
    tasks = build_multi_caption_tasks(records[0], subset)
```

Rebuild the release file from the working dataset with
`scripts/prepare_compound_data.py`. See [`docs/DATA.md`](docs/DATA.md) for the record
shape.

## Notes for anonymous release

Excludes raw media, model checkpoints, cluster paths, API keys, git metadata,
and cache directories.
