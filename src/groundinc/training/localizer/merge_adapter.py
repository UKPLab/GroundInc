#!/usr/bin/env python3
"""Merge a LoRA adapter into Gemma-4 at the SAFETENSORS level — robust against the
transformers 5.8.1 Gemma-4-E4B model-save bug that drops tied k_norm tensors
(merge_and_unload().state_dict() is missing k_norm for ~half the layers).

We read the base weights straight from its safetensors (ALL keys intact — every k_norm,
vision tower, embeddings), add the scaled LoRA delta W' = W + (alpha/r)*B@A to only the
adapted modules, and write the full tensor set back out. Nothing is loaded into a
transformers model, so nothing can be dropped or re-tied. The result is a plain model dir
vLLM serves with no LoRA.

Usage: merge_gemma_lora_safetensors.py --base_snapshot <dir> --adapter <ckpt> --out <dir>
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_snapshot", required=True, help="HF cache snapshot dir of google/gemma-4-E4B-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
    r = acfg["r"]; alpha = acfg["lora_alpha"]; use_rslora = acfg.get("use_rslora", False)
    scaling = (alpha / (r ** 0.5)) if use_rslora else (alpha / r)
    print(f"r={r} alpha={alpha} use_rslora={use_rslora} -> scaling={scaling}", flush=True)

    # --- load base tensors (all of them) ---
    base = {}
    for f in sorted(glob.glob(os.path.join(args.base_snapshot, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                base[k] = h.get_tensor(k)
    print(f"base tensors: {len(base)} | k_norm={sum('k_norm' in k for k in base)} q_norm={sum('q_norm' in k for k in base)}", flush=True)

    # --- load adapter A/B, grouped by target base weight key ---
    adp = {}
    for f in sorted(glob.glob(os.path.join(args.adapter, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                adp[k] = h.get_tensor(k)
    # adapter key: base_model.model.<MODULE>.lora_{A,B}.weight ; base weight: <MODULE>.weight
    mods = {}
    for k in adp:
        if ".lora_A.weight" in k:
            m = k[len("base_model.model."):].replace(".lora_A.weight", "")
            mods.setdefault(m, {})["A"] = adp[k]
        elif ".lora_B.weight" in k:
            m = k[len("base_model.model."):].replace(".lora_B.weight", "")
            mods.setdefault(m, {})["B"] = adp[k]

    applied = missing = 0
    for m, ab in mods.items():
        wkey = m + ".weight"
        if wkey not in base:
            print("  WARN: base weight not found for", wkey); missing += 1; continue
        A = ab["A"].float(); B = ab["B"].float()           # A:[r,in]  B:[out,r]
        delta = scaling * (B @ A)                            # [out,in]
        W = base[wkey]
        if delta.shape != W.shape:
            print(f"  WARN shape mismatch {wkey}: W{tuple(W.shape)} delta{tuple(delta.shape)}"); missing += 1; continue
        base[wkey] = (W.float() + delta).to(W.dtype)
        applied += 1
    print(f"applied LoRA to {applied} modules; problems={missing}", flush=True)

    # --- write out: all base keys (modified where adapted), single shard ---
    os.makedirs(args.out, exist_ok=True)
    base = {k: v.contiguous() for k, v in base.items()}
    save_file(base, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    # copy config/tokenizer/processor/chat-template from base snapshot
    for fn in os.listdir(args.base_snapshot):
        if fn.endswith(".safetensors") or fn.endswith(".index.json"):
            continue
        src = os.path.join(args.base_snapshot, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, fn))
    # verify
    saved = set()
    with safe_open(os.path.join(args.out, "model.safetensors"), framework="pt") as h:
        saved.update(h.keys())
    print(f"SAVED-CHECK total={len(saved)} k_norm={sum('k_norm' in k for k in saved)} q_norm={sum('q_norm' in k for k in saved)}", flush=True)
    print("SAFETENSORS MERGE DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
