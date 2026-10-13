"""Entrypoint for training the paired categorical localizer.

Two variants share this script and differ only in ``lambda_pair``:
  * ``paired_ce``: ``lambda_pair = 0`` (ordinary categorical grounding CE).
  * ``paired_js``: ``lambda_pair > 0`` (adds the JS pair term).

The sampler draws one manipulated partner per (event, modality) per epoch and
rotates partners across ``num_sampler_epochs``. The rotated sequence is expanded
once and trained for a single HF epoch, so both variants use an identical sampler,
order, and step count.

Usage:
    export PYTHONPATH=.../training/final_train:.../training/paired_grounding:$PYTHONPATH
    python train_paired.py config_paired_ce.yaml
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed

_HERE = os.path.dirname(os.path.abspath(__file__))

from . import model_utils as _qwen_mu  # noqa: E402  (Qwen3-VL loader)
from . import gemma_model_utils as _gemma_mu  # noqa: E402  (Gemma loader)
from .training_args import ModelArguments  # noqa: E402


def _is_gemma(model_args) -> bool:
    fam = (getattr(model_args, "model_family", "auto") or "auto").lower()
    if fam in ("gemma4", "gemma"):
        return True
    if fam == "auto":
        return "gemma" in model_args.model_name_or_path.lower()
    return False


def load_model_and_processor(model_args):
    mu = _gemma_mu if _is_gemma(model_args) else _qwen_mu
    return mu.load_model_and_processor(model_args)


def maybe_freeze_vision_tower(model, enabled, model_args):
    mu = _gemma_mu if _is_gemma(model_args) else _qwen_mu
    mu.maybe_freeze_vision_tower(model, enabled)


def maybe_apply_lora(model, model_args):
    """Qwen: unchanged. Gemma: resolve LoRA targets under language_model and honor
    `lora_use_rslora` (the Gemma SFT util hardcodes rsLoRA=True, but the localizer's
    WORKING recipe is rsLoRA OFF — the pilot collapse was rsLoRA on × high lr)."""
    if not _is_gemma(model_args):
        return _qwen_mu.maybe_apply_lora(model, model_args)
    from peft import LoraConfig, get_peft_model

    targets = _gemma_mu.resolve_lora_target_modules(model, model_args.lora_target_modules)
    use_rslora = bool(getattr(model_args, "lora_use_rslora", False))
    lora = LoraConfig(
        r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
        target_modules=targets, lora_dropout=model_args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", use_rslora=use_rslora,
    )
    model = get_peft_model(model, lora)
    scale = (model_args.lora_alpha / (model_args.lora_r ** 0.5)) if use_rslora else (model_args.lora_alpha / model_args.lora_r)
    logger.info("Gemma LoRA: %d targets, rslora=%s (scale=%.2f)", len(targets), use_rslora, scale)
    return model

from .id_tokens import DEFAULT_PREFIX, build_id_vocabulary  # noqa: E402
from .dataset import PairedLocalizerCollator, PairedLocalizerDataset  # noqa: E402
from .trainer import PairedLocalizerTrainer  # noqa: E402
from .pairing import (  # noqa: E402
    PairSampler,
    assert_video_disjoint,
    load_pairs,
    video_disjoint_split,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class PairedDataArguments:
    train_data_path: str = field(metadata={"help": "Input JSONL (news_data_mt_final / other_class_final)."})
    whisperx_dir: Optional[str] = None
    video_dir: Optional[str] = None
    video_mode: str = "id"
    fps: float = 0.5
    max_pixels: int = 50176
    max_video_frames: Optional[int] = None
    max_length: int = 6144
    enable_video: bool = True


@dataclass
class PairedArguments:
    lambda_pair: float = 0.0
    # Grounding objective: "categorical" (CE / CE+JS, the existing default) or
    # "sigmoid" (class-balanced BCE / BCE + Bernoulli JS).
    objective: str = "categorical"
    # Target construction. Only "existing_hard" (the single argmax-IoU gold) is
    # implemented; "interval_multi_positive" is the deferred §7 extension.
    target_mode: str = "existing_hard"
    # Number of pairs for the sigmoid bias-init probe (section 4).
    bias_init_pairs: int = 8
    bank_widths: str = "8,16,32,64"
    bank_stride_frac: float = 0.5
    max_candidates: int = 128
    bank_min_len: float = 1.0
    iou_mode: str = "hull"
    num_sampler_epochs: int = 2
    dev_frac: float = 0.1
    split_seed: int = 42
    pair_report_path: Optional[str] = None


def parse():
    parser = HfArgumentParser((ModelArguments, PairedDataArguments, TrainingArguments, PairedArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        return parser.parse_yaml_file(str(Path(sys.argv[1]).resolve()))
    return parser.parse_args_into_dataclasses()


def _existing_pairs(pairs, data_args):
    """Drop pairs whose source video cannot be resolved, so __getitem__ never fails."""
    from .data_io import resolve_video_path

    kept, missing = [], 0
    for pair in pairs:
        try:
            record_like = {"id": pair.video_id, "video_path": pair.video_path}
            path = resolve_video_path(record_like, data_args.video_mode, data_args.video_dir)
            if os.path.exists(path):
                kept.append(pair)
            else:
                missing += 1
        except Exception:
            missing += 1
    if missing:
        logger.warning("Dropped %d pairs with missing/unresolvable videos", missing)
    return kept


def main():
    model_args, data_args, train_args, paired_args = parse()
    set_seed(train_args.seed)
    train_args.remove_unused_columns = False
    train_args.label_names = []  # we compute our own loss; no HF "labels"

    # Objective / target-mode validation (fail early on unimplemented combos).
    if paired_args.objective not in ("categorical", "sigmoid"):
        raise ValueError(f"objective must be 'categorical' or 'sigmoid', got {paired_args.objective!r}")
    if paired_args.target_mode == "interval_multi_positive":
        raise NotImplementedError(
            "target_mode='interval_multi_positive' (the §7 multi-positive extension) "
            "is deferred; use target_mode='existing_hard'."
        )
    if paired_args.target_mode != "existing_hard":
        raise ValueError(f"Unknown target_mode {paired_args.target_mode!r}")

    logger.info("Loading model/processor: %s", model_args.model_name_or_path)
    model, processor = load_model_and_processor(model_args)
    model.config.use_cache = False
    if train_args.gradient_checkpointing:
        model.enable_input_require_grads()
    maybe_freeze_vision_tower(model, model_args.freeze_vision_tower, model_args)
    model = maybe_apply_lora(model, model_args)

    # For the sigmoid objective, register the shared scalar bias BEFORE the
    # Trainer/optimizer/DeepSpeed are built (its value is fitted by the probe at
    # on_train_begin, on the device-placed model). No-op for the categorical path.
    if paired_args.objective == "sigmoid":
        from sigmoid_head import attach_bias

        attach_bias(model, 0.0)
        logger.info("Attached sigmoid localizer bias (fitted by probe at train start).")

    widths = tuple(float(w) for w in paired_args.bank_widths.split(",") if w.strip())
    vocab = build_id_vocabulary(
        processor.tokenizer, num_needed=paired_args.max_candidates, prefix=DEFAULT_PREFIX
    )
    logger.info("ID vocabulary: %d single-token symbols (labels %s..%s)",
                len(vocab), vocab.display_labels[0], vocab.display_labels[-1])

    all_pairs, excluded = load_pairs(data_args.train_data_path, data_args.video_mode)
    all_pairs = _existing_pairs(all_pairs, data_args)
    train_pairs, dev_pairs = video_disjoint_split(
        all_pairs, dev_frac=paired_args.dev_frac, seed=paired_args.split_seed
    )
    assert_video_disjoint(train_pairs, dev_pairs)
    logger.info("Pairs: %d train / %d dev (dev is video-disjoint); %d excluded",
                len(train_pairs), len(dev_pairs), len(excluded))

    sampler = PairSampler(train_pairs, seed=paired_args.split_seed)
    expanded: List = []
    for epoch in range(paired_args.num_sampler_epochs):
        expanded.extend(sampler.epoch_pairs(epoch))
    logger.info("Expanded %d events x %d sampler-epochs = %d training pairs",
                sampler.num_events(), paired_args.num_sampler_epochs, len(expanded))
    # We pre-expanded the rotation; run exactly one HF epoch over it.
    train_args.num_train_epochs = 1

    dataset = PairedLocalizerDataset(
        pairs=expanded,
        processor=processor,
        vocab=vocab,
        whisperx_dir=data_args.whisperx_dir,
        video_dir=data_args.video_dir,
        video_mode=data_args.video_mode,
        fps=data_args.fps,
        max_pixels=data_args.max_pixels,
        max_video_frames=data_args.max_video_frames,
        max_length=data_args.max_length,
        bank_widths=widths,
        bank_stride_frac=paired_args.bank_stride_frac,
        max_candidates=paired_args.max_candidates,
        bank_min_len=paired_args.bank_min_len,
        iou_mode=paired_args.iou_mode,
        enable_video=data_args.enable_video,
        model_family="gemma4" if _is_gemma(model_args) else "qwen3vl",
    )

    collator = PairedLocalizerCollator(pad_token_id=processor.tokenizer.pad_token_id)
    head_metadata = {
        "objective": paired_args.objective,
        "target_mode": paired_args.target_mode,
        "lambda_pair": paired_args.lambda_pair,
    }
    trainer = PairedLocalizerTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        lambda_pair=paired_args.lambda_pair,
        objective=paired_args.objective,
        head_metadata=head_metadata,
    )

    # Resume detection: reuse the DeepSpeed-native checkpoint if one exists.
    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = None
    if os.path.isdir(train_args.output_dir):
        last_ckpt = get_last_checkpoint(train_args.output_dir)
    resuming = last_ckpt is not None

    # Bias-init probe (section 4): run it BEFORE trainer.train() so the fitted
    # value is in the model param when DeepSpeed snapshots the optimizer's fp32
    # master (deepspeed_init runs inside train(), before on_train_begin). Skipped
    # on resume — deepspeed_load_checkpoint restores the bias (module + master +
    # optimizer state) from the native checkpoint.
    if paired_args.objective == "sigmoid":
        from sigmoid_head import get_bias, initialize_bias_from_probe

        if resuming:
            logger.info("Resuming from %s; skipping bias probe (restored from checkpoint).", last_ckpt)
        else:
            model.to(train_args.device)  # probe needs the model on the training device
            record = initialize_bias_from_probe(
                model, dataset, collator, device=train_args.device,
                n_pairs=paired_args.bias_init_pairs,
            )
            trainer.head_metadata["bias_probe"] = record
            logger.info("Bias before init=0.0; after probe fit=%.5f (prevalence %.4f->%.4f)",
                        record["fitted_bias"], record["weighted_prevalence_before"],
                        record["weighted_prevalence_after"])
            logger.info("BIAS_PARAM_VALUE_AFTER_FIT %.6f", float(get_bias(model).detach().reshape(-1)[0]))

    # Check-only, env-gated early stop: interrupt after N optimizer steps while
    # keeping the FULL scheduler horizon (max_steps), so the resume-equivalence
    # check compares identical LR schedules. No effect in normal runs (env unset).
    _stop_after = os.environ.get("PRODCHECK_STOP_AFTER")
    if _stop_after:
        from transformers import TrainerCallback

        class _StopAfter(TrainerCallback):
            def __init__(self, n):
                self.n = int(n)

            def on_step_end(self, a, s, c, **kw):
                if s.global_step >= self.n:
                    c.should_training_stop = True
                return c

        trainer.add_callback(_StopAfter(_stop_after))
        logger.info("PRODCHECK_STOP_AFTER=%s (check-only early stop; schedule horizon unchanged)", _stop_after)

    variant = {
        ("categorical", False): "paired_ce",
        ("categorical", True): "paired_js",
        ("sigmoid", False): "sigmoid_bce",
        ("sigmoid", True): "sigmoid_bce_js",
    }[(paired_args.objective, bool(paired_args.lambda_pair))]
    logger.info("Starting training: objective=%s lambda_pair=%s variant=%s (resume=%s)",
                paired_args.objective, paired_args.lambda_pair, variant, last_ckpt)
    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(train_args.output_dir)
    processor.save_pretrained(train_args.output_dir)
    # Persist the exact bank/vocab configuration used, for reproducible inference.
    import json

    with open(os.path.join(train_args.output_dir, "localizer_bank.json"), "w") as handle:
        json.dump(
            {
                "widths": list(widths),
                "stride_frac": paired_args.bank_stride_frac,
                "max_candidates": paired_args.max_candidates,
                "min_len": paired_args.bank_min_len,
                "iou_mode": paired_args.iou_mode,
                "id_prefix": DEFAULT_PREFIX,
                "id_symbols": vocab.symbols,
                "id_display_labels": vocab.display_labels,
                "id_token_ids": vocab.token_ids,
                "lambda_pair": paired_args.lambda_pair,
                "objective": paired_args.objective,
                "target_mode": paired_args.target_mode,
                "bias_probe": getattr(trainer, "head_metadata", {}).get("bias_probe"),
            },
            handle,
            indent=2,
        )
    logger.info("Saved model, processor, and localizer_bank.json to %s", train_args.output_dir)


if __name__ == "__main__":
    main()
