"""Shared scalar bias for the sigmoid localizer objectives, plus its
initialization probe and checkpoint persistence.

The categorical CE objective is invariant to a common offset of all candidate
logits; sigmoid BCE is not. Applying sigmoid to CE-style LM logits can therefore
start with every candidate saturated. We add ONE learned scalar ``b`` shared
across every candidate, example, and faithful/manipulated view (section 4). The
score is ``s_j = z_j + b`` with the existing scoring path's multiplier fixed at 1
(there is no temperature in this repo's raw-logit path). ``b`` is a genuine
trainable parameter (localizer lr, zero weight decay).

Because span selection is ``argmax`` over the candidate logits and argmax is
invariant to a common offset, ``b`` does NOT change inference span selection and
is not needed by the vLLM inference path. It matters only for (a) training,
(b) reproducible resume, and (c) reporting sigmoid RELEVANCE scores offline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import torch

from .losses import class_balanced_weights, fit_bias_bisection

logger = logging.getLogger(__name__)

BIAS_ATTR = "localizer_sigmoid_bias"
HEAD_FILE = "localizer_head.pt"

_MODEL_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values_videos",
    "video_grid_thw",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
)


def attach_bias(model, init_value: float = 0.0) -> torch.nn.Parameter:
    """Register the shared scalar bias on ``model`` as a trainable float32 param.

    Registered at the top level so it appears in ``model.named_parameters()`` and
    is picked up by the optimizer, DeepSpeed partitioning/grad-sync, and the
    DeepSpeed-native checkpoint. Must be called BEFORE the Trainer/optimizer and
    distributed init. Idempotent: returns the existing param if already attached.
    """
    existing = getattr(model, BIAS_ATTR, None)
    if isinstance(existing, torch.nn.Parameter):
        return existing
    param = torch.nn.Parameter(torch.tensor([float(init_value)], dtype=torch.float32))
    setattr(model, BIAS_ATTR, param)
    param.requires_grad_(True)
    return param


def get_bias(model) -> Optional[torch.nn.Parameter]:
    param = getattr(model, BIAS_ATTR, None)
    return param if isinstance(param, torch.nn.Parameter) else None


def _unwrap(model):
    # PeftModel / DDP / DeepSpeedEngine wrappers all expose the bias attribute at
    # the level it was registered; walk common wrapper attributes to find it.
    for attr in (BIAS_ATTR,):
        if getattr(model, attr, None) is not None:
            return model
    for wrapper_attr in ("module", "model"):
        inner = getattr(model, wrapper_attr, None)
        if inner is not None and getattr(inner, BIAS_ATTR, None) is not None:
            return inner
    return model


@torch.no_grad()
def initialize_bias_from_probe(
    model,
    dataset,
    collator,
    *,
    device,
    n_pairs: int = 8,
    target_prevalence: float = 0.5,
) -> Dict:
    """Fit the initial bias from a small deterministic TRAINING-only probe (section 4).

    Uses the first ``n_pairs`` items of the (already seeded, pre-expanded) dataset
    WITHOUT consuming the training iterator. Disables dropout and grads, runs the
    same forward + gather + class-balanced weighting as training, then fits ``b``
    by bounded bisection so the weighted mean ``sigmoid(z + b)`` equals the target
    positive prevalence (0.5 with balanced weights). Restores model mode and RNG
    afterward. Never rerun on resume.

    Returns a record (probe ids, prevalence, fitted bias, saturation stats).
    """
    bias = get_bias(model)
    if bias is None:
        raise RuntimeError("initialize_bias_from_probe: bias not attached to model")

    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.eval()

    logits_list: List[torch.Tensor] = []
    weights_list: List[torch.Tensor] = []
    probe_ids: List[str] = []
    sat_hi = sat_lo = total = 0

    try:
        n = min(n_pairs, len(dataset))
        for i in range(n):
            item = dataset[i]
            batch = collator([item])
            model_inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
                if k in _MODEL_KEYS
            }
            out = model(**model_inputs)
            logits = out.logits  # [2, T, V]
            pred_pos = batch["pred_positions"].to(device)
            rows = torch.arange(logits.shape[0], device=device)
            logits_at = logits[rows, pred_pos]
            plus, minus = logits_at[0::2], logits_at[1::2]
            cand = batch["cand_token_ids"].to(device)
            mask = batch["cand_mask"].to(device)
            gold = batch["gold_index"].to(device)
            lp = torch.gather(plus, -1, cand).float()
            lq = torch.gather(minus, -1, cand).float()
            w, _y = class_balanced_weights(mask, gold)
            for z in (lp, lq):
                logits_list.append(z.reshape(-1).cpu())
                weights_list.append(w.reshape(-1).cpu())
                sat_hi += int((z[mask] > 15).sum())
                sat_lo += int((z[mask] < -15).sum())
                total += int(mask.sum())
            probe_ids.extend(m["pair_id"] for m in batch["meta"])
    finally:
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    z_all = torch.cat(logits_list)
    w_all = torch.cat(weights_list)
    fitted = fit_bias_bisection(z_all, w_all, target_prevalence=target_prevalence)
    bias.data.fill_(float(fitted))

    prevalence_before = float((w_all * torch.sigmoid(z_all)).sum() / w_all.sum())
    prevalence_after = float((w_all * torch.sigmoid(z_all + fitted)).sum() / w_all.sum())
    record = {
        "probe_pairs": probe_ids,
        "n_pairs": len(probe_ids),
        "target_prevalence": target_prevalence,
        "fitted_bias": float(fitted),
        "weighted_prevalence_before": prevalence_before,
        "weighted_prevalence_after": prevalence_after,
        "saturation_hi_frac": (sat_hi / total) if total else 0.0,
        "saturation_lo_frac": (sat_lo / total) if total else 0.0,
    }
    logger.info("Bias probe: fitted b=%.4f (prevalence %.4f -> %.4f); sat hi=%.3f lo=%.3f over %d probe captions",
                fitted, prevalence_before, prevalence_after,
                record["saturation_hi_frac"], record["saturation_lo_frac"], len(probe_ids))
    return record


def save_localizer_head(output_dir: str, model, metadata: Optional[Dict] = None) -> Optional[str]:
    """Persist the bias value + objective metadata next to the adapter.

    PEFT ``save_pretrained`` writes only adapter weights, so the bias would be
    silently dropped without this. Written into every resumable checkpoint and the
    final adapter dir. (DeepSpeed's native checkpoint also carries the bias weight
    and optimizer state for in-run resume; this file is the portable, human-
    readable copy for reload/scoring.)
    """
    bias = get_bias(_unwrap(model))
    if bias is None:
        return None
    payload = {"localizer_sigmoid_bias": bias.detach().float().cpu()}
    if metadata:
        payload["metadata"] = metadata
    path = os.path.join(output_dir, HEAD_FILE)
    torch.save(payload, path)
    return path


def load_localizer_head(model, load_dir: str, *, required: bool = True) -> Optional[float]:
    """Restore the bias from ``load_dir`` into ``model``.

    Raises if ``required`` and the head file is absent (a sigmoid checkpoint must
    not silently load with a zero bias). Returns the restored bias value or None.
    """
    path = os.path.join(load_dir, HEAD_FILE)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Sigmoid checkpoint missing required {HEAD_FILE} in {load_dir}"
            )
        return None
    payload = torch.load(path, map_location="cpu")
    value = float(payload["localizer_sigmoid_bias"].reshape(-1)[0])
    target = _unwrap(model)
    bias = get_bias(target)
    if bias is None:
        bias = attach_bias(target, value)
    else:
        bias.data.fill_(value)
    return value


def head_metadata_path(load_dir: str) -> str:
    return os.path.join(load_dir, HEAD_FILE)


# NOTE on init ordering (do NOT move the probe into an on_train_begin callback):
# HF Trainer runs `_prepare_for_training` -> `deepspeed_init` (which snapshots the
# optimizer's authoritative fp32 master from the model params) BEFORE firing
# `on_train_begin`. A callback that fits the bias at on_train_begin would update
# only the module param, not the fp32 master, so the first optimizer step would
# overwrite the fitted bias back toward its init. The probe therefore runs in the
# training entrypoint BEFORE `trainer.train()` (before deepspeed_init), so the
# fitted value is already in the param when the master is snapshotted. On resume,
# `deepspeed_load_checkpoint` restores the bias (module + master + optimizer
# state) from the native checkpoint, so the probe must be skipped.
