# Reproducibility

## 1. Install

```bash
conda create -n groundinc python=3.10 -y
conda activate groundinc
pip install -r requirements.txt
pip install -e .
```

## 2. Annotations

The files under `data/` are the annotations used in the paper. To regenerate
them from base captions:

```bash
export GEMINI_API_KEY=...
bash examples/run_generation.sh
```

## 3. Media

Videos, transcripts, and audio are not included. You need:

- `videos/{id}.mp4`
- `asr/{id}_asr.json` with a `"prompt"` string or `"segments"` (`start`/`end`/`text`)
- `audio/{id}.wav` for the video and audio runner

For `masked_snippet`, precompute the masked clips first:

```bash
python -m groundinc.evaluation.precompute_masked_snippets \
  --input_file data/groundinc_news.jsonl \
  --video_dir videos --snippet_dir masked_snippets
```

## 4. Inference

```bash
bash examples/run_vllm_asr.sh
bash examples/run_vllm_audio.sh
bash examples/run_predicted_snippet.sh
```

All expect a vLLM server exposing a chat-completions endpoint at `--vllm_url`.

## 5. Finetuning

```bash
bash examples/run_prepare_sft.sh
```

See [`TRAINING.md`](TRAINING.md).
