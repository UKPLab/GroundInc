"""CE / paired-JS / sigmoid categorical interval-bank temporal localizer.

A custom-trainer backend (separate from the LLaMA-Factory SFT path): it predicts
a single interval-ID token over a deterministic per-video candidate bank. Train
with ``python -m groundinc.training.localizer.train --config <config.yaml>``.
"""
