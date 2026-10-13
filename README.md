# GroundInc

Anonymous release code and annotations for GroundInc: generating manipulated
audio-visual captions, running VLM inference over them, and finetuning.

## Contents

- `data/`: the GroundInc annotations (news test set and cross-domain set).
- `prompts/`: the two-step Gemini prompts.
- `src/groundinc/data_generation/`: the annotation pipeline.
- `src/groundinc/evaluation/`: OpenAI-compatible vLLM inference runners.
- `src/groundinc/training/`: SFT dataset preparation and LLaMA-Factory configs.
- `src/groundinc/data/`: JSONL loading, caption-task construction, prompt registry and parsers.
- `docs/`: data format, training, reproducibility.
- `scripts/smoke_test.sh`: offline sanity check needing no data, media, or server.

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

## Notes for anonymous release

Excludes raw media, model checkpoints, cluster paths, API keys, git metadata,
and cache directories.
