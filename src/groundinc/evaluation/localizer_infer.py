"""vLLM news-eval client for the paired localizer — emits paper-format records.

For each news record + caption task, this builds the trained localizer prompt
(candidate bank from the video duration + the checkpoint's ID vocabulary), sends
the video + prompt to a vLLM-served model (base or a LoRA adapter selected via
`--model_id`), and lets the model emit `Evidence interval: <label>` greedily
(faithful to training — no guided decoding). The label maps back to a candidate
interval → predicted span.

Output is written in the schema `analysis_v2/dataloader.py` consumes: each record
is the original input record + `asr_text` + a `results` list whose entries carry
the task fields plus `temporal_grounding.prediction_parsed = [start, end]`. Grounding
is scored independently by the paper analysis; `veracity`/`inconsistency` are left as
empty blocks (this experiment measures grounding — a verifier pass can fill them later).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys


from groundinc.training.localizer.data_io import build_caption_tasks, load_asr_text, resolve_video_path  # noqa: E402
from groundinc.training.localizer.id_tokens import DEFAULT_PREFIX, IdVocabulary  # noqa: E402
from groundinc.training.localizer.interval_bank import build_bank  # noqa: E402
from groundinc.training.localizer.prompt import build_localizer_prompt  # noqa: E402

_LABEL_RE = re.compile(r"Evidence interval:\s*([A-Za-z]{1,3})")


def load_vocab(bank_config_path):
    with open(bank_config_path) as h:
        cfg = json.load(h)
    vocab = IdVocabulary(
        prefix=cfg.get("id_prefix", DEFAULT_PREFIX),
        symbols=cfg["id_symbols"],
        token_ids=cfg["id_token_ids"],
        display_labels=cfg["id_display_labels"],
    )
    bank_kw = dict(
        widths=tuple(float(w) for w in cfg.get("widths", [8, 16, 32, 64])),
        stride_frac=cfg.get("stride_frac", 0.5),
        max_candidates=cfg.get("max_candidates", 128),
        min_len=cfg.get("min_len", 1.0),
    )
    return vocab, bank_kw


def parse_label_to_index(text, vocab):
    # Base model emits "Evidence interval: M"; trained adapters emit the bare "M".
    if not text or text.startswith("error"):
        return None
    m = _LABEL_RE.search(text)
    cand = m.group(1) if m else (text.strip().split() or [""])[0]
    if cand in vocab.display_labels:
        return vocab.display_labels.index(cand)
    for lab in sorted(vocab.display_labels, key=len, reverse=True):  # "Ab" before "A"
        if cand.startswith(lab):
            return vocab.display_labels.index(lab)
    return None


async def run(args):
    from openai import AsyncOpenAI

    vocab, bank_kw = load_vocab(args.bank_config)
    client = AsyncOpenAI(base_url=args.vllm_url, api_key="vllm")
    sem = asyncio.Semaphore(args.concurrency)

    with open(args.input_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]

    async def predict(video_path, caption, asr, duration):
        bank = build_bank(duration, **bank_kw)
        prompt = build_localizer_prompt(
            caption=caption, asr_content=asr, fps=args.fps, bank=bank, vocab=vocab
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "file://" + os.path.realpath(video_path)}},
                {"type": "text", "text": prompt},
            ],
        }]
        # Match training preprocessing EXACTLY: nframes = min(int(dur*fps), cap).
        # News videos average ~220s, so the 32-frame cap binds — relying on fps here
        # would feed ~110 frames and break parity with the trained localizer.
        if args.model_family == "gemma4":
            # Gemma uses num_frames/do_sample_frames (set in the server config); do NOT
            # pass Qwen's nframes/max_pixels (no-ops / wrong keys on the Gemma path).
            extra = {}
        else:
            nframes = max(2, int(duration * args.fps)) if duration > 0 else args.max_video_frames
            nframes = min(nframes, args.max_video_frames)
            extra = {"mm_processor_kwargs": {"nframes": nframes, "max_pixels": args.max_pixels}}
        # select_mode:
        #   generate           -> unchanged free (unconstrained) greedy generation.
        #   candidate_restricted -> restrict the first (single) token to this video's
        #     candidate ID tokens and take max_tokens=1 (candidates are single tokens
        #     by construction), i.e. argmax over candidate logits through the serving
        #     path. Applied IDENTICALLY to every arm; separates grounding quality from
        #     surface-form label emission. No change to models, prompts, or bank.
        max_tok = 8
        if args.select_mode == "candidate_restricted":
            extra["allowed_token_ids"] = vocab.token_ids[: len(bank)]
            max_tok = 1
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=args.model_id, messages=messages,
                    temperature=0.0, max_tokens=max_tok, extra_body=extra,
                )
                raw = resp.choices[0].message.content
            except Exception as exc:
                return None, f"error: {exc}", bank
        idx = parse_label_to_index(raw, vocab)
        span = list(bank.intervals[idx]) if (idx is not None and idx < len(bank)) else None
        return span, raw, bank

    out = open(args.output_file, "w")
    n_done = 0
    for record in records:
        vid = record["id"]
        duration = float(record.get("duration") or 0)
        try:
            video_path = resolve_video_path(record, args.video_mode, args.video_dir)
        except Exception:
            continue
        if not os.path.exists(video_path):
            continue
        asr = load_asr_text(args.whisperx_dir, vid) if args.whisperx_dir else ""
        tasks = build_caption_tasks(record, args.video_mode)
        preds = await asyncio.gather(*[predict(video_path, t["caption"], asr, duration) for t in tasks])

        results = []
        for task, (span, raw, _bank) in zip(tasks, preds):
            results.append({
                "category": task["category"],
                "type": task["type"],
                "caption": task["caption"],
                "manipulation_method_gt": task.get("manipulation_method_gt"),
                "entity_type": task.get("entity_type"),
                "manipulated_modality": task.get("manipulated_modality"),
                "ground_truth_span": task.get("ground_truth_span"),
                "prompt_style": args.variant,
                "prediction_raw": {"grounding": raw},
                "temporal_grounding": {"prediction_raw": raw, "prediction_parsed": span},
                # grounding-only: veracity/type intentionally empty (paper dataloader tolerates)
                "veracity": {},
                "inconsistency": {},
            })
        rec_out = dict(record)
        rec_out["asr_text"] = asr
        rec_out["results"] = results
        out.write(json.dumps(rec_out) + "\n")
        out.flush()
        n_done += 1
        if n_done % 25 == 0:
            print(f"[{args.variant}] {n_done}/{len(records)} records", flush=True)
    out.close()
    print(f"[{args.variant}] DONE {n_done} records -> {args.output_file}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--vllm_url", default="http://localhost:8602/v1")
    p.add_argument("--model_id", required=True, help="served model name: base id, or lora name (ce_32b/js_32b/...)")
    p.add_argument("--variant", required=True, help="prompt_style label written into results")
    p.add_argument("--bank_config", required=True, help="localizer_bank.json from a trained ckpt (bank+vocab)")
    p.add_argument("--video_dir", default=None)
    p.add_argument("--video_mode", default="id", choices=["id", "clip"])
    p.add_argument("--whisperx_dir", default=None)
    p.add_argument("--fps", type=float, default=0.5)
    p.add_argument("--max_pixels", type=int, default=50176)
    p.add_argument("--max_video_frames", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model_family", default="qwen3vl", choices=["qwen3vl", "gemma4"],
                   help="gemma4: skip Qwen nframes/max_pixels; server config sets num_frames/fps")
    p.add_argument("--select_mode", default="generate",
                   choices=["generate", "candidate_restricted"],
                   help="generate = unconstrained greedy (default, unchanged); "
                        "candidate_restricted = first token restricted to candidate IDs")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
