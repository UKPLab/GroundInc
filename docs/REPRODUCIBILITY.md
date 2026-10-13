# Reproducibility

Four independent stages. Only the first runs without media.

## 0. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"
bash scripts/smoke_test.sh
```

The smoke test compiles the package, checks the prompt registry and video
modes, asserts the timestamp parser handles clock formats, and verifies each
training prompt has a target format and an inference counterpart. It needs no
data, media, or server.

## 1. Annotations

The shipped files under `data/` are the annotations used in the paper; nothing
needs regenerating to run stages 2–4.

To regenerate from base captions:

```bash
export GEMINI_API_KEY=...
bash examples/run_generation.sh
```

This stage needs the videos (`--video_dir`) but no ASR. A record whose video
cannot be resolved or uploaded is skipped.

## 2. Media

Not shipped and not scripted. You need:

- `videos/{id}.mp4` — fetched by YouTube id.
- `asr/{id}_asr.json` — a `"prompt"` string, or `"segments"` with
  `start`/`end`/`text`. Segment timings are required for `--asr_align span`.
- `audio/{id}.wav` — only for the video+audio runner.

Transcripts may instead be inlined per record as `asr_text`, which removes the
`--asr_dir` requirement entirely.

For `--video_mode masked_snippet`, precompute the masked clips once:

```bash
python -m groundinc.evaluation.precompute_masked_snippets \
  --input_file data/groundinc_news.jsonl \
  --video_dir videos --snippet_dir masked_snippets
```

## 3. Inference

```bash
bash examples/run_vllm_asr.sh          # video + ASR
bash examples/run_vllm_audio.sh        # video + audio
bash examples/run_predicted_snippet.sh # predicted-interval crop
```

All expect an OpenAI-compatible vLLM server at `--vllm_url`. Runs are
resumable: rows already present in the output file are skipped, and connection-
error rows are always retried.

Output is parsed predictions, not scores. **No metric code ships with this
release.**

## 4. Finetuning

```bash
bash examples/run_prepare_sft.sh
```

Run `prepare_sft` with `--report_only` first to size `cutoff_len` against the
printed length distribution. Then train with `llamafactory-cli`. Details and
the prompt/style pairing table are in [`TRAINING.md`](TRAINING.md).

## Determinism

Inference defaults to `--temperature 0.01`, near-greedy but not bit-exact
across servers or batch compositions. Generation uses temperature 0.4 for
planning and 0.1 for execution, so regenerated annotations will differ in
wording from the shipped ones.
