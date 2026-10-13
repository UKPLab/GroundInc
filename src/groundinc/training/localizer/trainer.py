"""Trainer for the paired categorical / sigmoid localizer.

Overrides ``compute_loss`` to:
  1. run one forward over the flattened ``2B`` batch (faithful/manipulated),
  2. read the next-token logits at each sequence's fixed prediction position,
  3. gather the valid candidate-ID logits into per-candidate scores,
  4. apply the selected grounding objective:
       * ``categorical`` (default): CE + lambda * categorical JS (the existing
         ``paired_ce`` / ``paired_js`` behaviour). At ``lambda_pair == 0`` this is
         exactly ordinary categorical grounding CE.
       * ``sigmoid``: class-balanced BCE on ``z + b`` + lambda * class-balanced
         Bernoulli JS (``sigmoid_bce`` / ``sigmoid_bce_js``). ``b`` is the shared
         scalar bias attached via ``sigmoid_head.attach_bias``. At
         ``lambda_pair == 0`` this is BCE-only.

Only ``lambda_pair`` distinguishes an objective's regularized variant from its
control, so each (objective, lambda) pair is a clean matched arm.
"""

from __future__ import annotations

import logging

import torch
from transformers import Trainer

from .losses import paired_grounding_loss, paired_sigmoid_loss
from .sigmoid_head import get_bias, save_localizer_head

logger = logging.getLogger(__name__)

_META_KEYS = ("pred_positions", "gold_index", "cand_token_ids", "cand_mask", "meta")
# Keys the model forward accepts.
_MODEL_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values_videos",
    "video_grid_thw",
    "video_position_ids",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
)


class PairedLocalizerTrainer(Trainer):
    def __init__(
        self,
        *args,
        lambda_pair: float = 0.0,
        objective: str = "categorical",
        head_metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_pair = float(lambda_pair)
        self.objective = str(objective)
        if self.objective not in ("categorical", "sigmoid"):
            raise ValueError(f"Unknown objective {self.objective!r}")
        self.head_metadata = dict(head_metadata or {})
        if self.objective == "categorical":
            self._pair_log = {"l_ground": 0.0, "l_pair": 0.0, "acc": 0.0, "n": 0}
        else:
            self._pair_log = {
                "l_bce": 0.0, "l_js": 0.0, "l_js_weighted": 0.0, "js_unweighted": 0.0,
                "acc": 0.0, "bias": 0.0, "sat": 0.0, "n": 0, "n_skipped": 0,
            }

    # -- loss ----------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pred_positions = inputs["pred_positions"]
        gold_index = inputs["gold_index"]
        cand_token_ids = inputs["cand_token_ids"]
        cand_mask = inputs["cand_mask"]

        model_inputs = {k: v for k, v in inputs.items() if k in _MODEL_KEYS}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [2B, T, V]

        rows = torch.arange(logits.shape[0], device=logits.device)
        logits_at = logits[rows, pred_positions]  # [2B, V]

        plus = logits_at[0::2]   # [B, V]
        minus = logits_at[1::2]  # [B, V]

        # Gather valid candidate-ID logits (padded slots gather id 0, masked out).
        logits_plus = torch.gather(plus, -1, cand_token_ids)   # [B, Kmax]
        logits_minus = torch.gather(minus, -1, cand_token_ids)  # [B, Kmax]

        # Default to the categorical path when the objective attribute is absent
        # (e.g. trainers built via __new__ in legacy tests) — the feature is off.
        objective = getattr(self, "objective", "categorical")
        if objective == "categorical":
            result = paired_grounding_loss(
                logits_plus, logits_minus, cand_mask, gold_index, self.lambda_pair
            )
            loss = result["loss"]
            self._accumulate_categorical(logits_plus, logits_minus, cand_mask, gold_index, result)
        elif objective == "sigmoid":
            # Under DeepSpeed/DDP the passed `model` may be a wrapper; the bias is
            # registered on the underlying PeftModel (self.model), the same leaf
            # tensor the optimizer tracks.
            bias = get_bias(model)
            if bias is None:
                bias = get_bias(self.model)
            if bias is None:
                raise RuntimeError("sigmoid objective requires an attached bias (sigmoid_head.attach_bias)")
            result = paired_sigmoid_loss(
                logits_plus, logits_minus, cand_mask, gold_index, self.lambda_pair, bias=bias
            )
            loss = result["loss"]
            self._accumulate_sigmoid(logits_plus, logits_minus, cand_mask, gold_index, result, bias)
        else:
            raise ValueError(f"Unknown objective {objective!r}")

        if return_outputs:
            return loss, outputs
        return loss

    # -- optimizer -----------------------------------------------------------
    def create_optimizer(self):
        """Build the optimizer with an explicit param group for the sigmoid bias.

        The bias trains at the localizer learning rate with ZERO weight decay, in
        its own group (verified here, not inferred from its name). LoRA params keep
        the configured decay. For the categorical objective this defers to the
        standard HF logic. The DeepSpeed config carries no ``optimizer`` block, so
        this optimizer (and thus these groups) is the one DeepSpeed wraps.
        """
        if self.objective != "sigmoid" or self.optimizer is not None:
            return super().create_optimizer()

        bias = get_bias(self.model)
        if bias is None:
            return super().create_optimizer()

        decay_names = set(self.get_decay_parameter_names(self.model))
        decay_params, no_decay_params, bias_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param is bias:
                bias_params.append(param)
            elif name in decay_names:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        assert bias_params, "sigmoid bias parameter not found among trainable params"
        wd = self.args.weight_decay
        lr = self.args.learning_rate
        groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": bias_params, "weight_decay": 0.0, "lr": lr},
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        # Post-condition: the bias lives in exactly one group, with wd == 0.
        bias_groups = [g for g in self.optimizer.param_groups if any(p is bias for p in g["params"])]
        assert len(bias_groups) == 1 and bias_groups[0]["weight_decay"] == 0.0, (
            "sigmoid bias must be in exactly one optimizer group with weight_decay=0"
        )
        logger.info("Optimizer: %d decay + %d no-decay + 1 bias param(s); bias lr=%s wd=0",
                    len(decay_params), len(no_decay_params), bias_groups[0].get("lr", lr))
        return self.optimizer

    # -- checkpointing -------------------------------------------------------
    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if self.objective == "sigmoid":
            out = output_dir if output_dir is not None else self.args.output_dir
            meta = dict(self.head_metadata)
            meta.update({"objective": self.objective, "lambda_pair": self.lambda_pair})
            path = save_localizer_head(out, self.model, metadata=meta)
            if path:
                logger.info("Saved localizer head (bias + metadata) -> %s", path)

    # -- diagnostics ---------------------------------------------------------
    @torch.no_grad()
    def _accumulate_categorical(self, logits_plus, logits_minus, cand_mask, gold_index, result):
        neg_inf = torch.finfo(logits_plus.dtype).min
        masked_plus = logits_plus.masked_fill(~cand_mask, neg_inf)
        masked_minus = logits_minus.masked_fill(~cand_mask, neg_inf)
        pred_plus = masked_plus.argmax(dim=-1)
        pred_minus = masked_minus.argmax(dim=-1)
        acc = 0.5 * (
            (pred_plus == gold_index).float().mean() + (pred_minus == gold_index).float().mean()
        )
        b = gold_index.shape[0]
        self._pair_log["l_ground"] += float(result["l_ground"]) * b
        self._pair_log["l_pair"] += float(result["l_pair"]) * b
        self._pair_log["acc"] += float(acc) * b
        self._pair_log["n"] += b

    @torch.no_grad()
    def _accumulate_sigmoid(self, logits_plus, logits_minus, cand_mask, gold_index, result, bias):
        neg_inf = torch.finfo(logits_plus.dtype).min
        pred_plus = logits_plus.masked_fill(~cand_mask, neg_inf).argmax(dim=-1)
        pred_minus = logits_minus.masked_fill(~cand_mask, neg_inf).argmax(dim=-1)
        acc = 0.5 * (
            (pred_plus == gold_index).float().mean() + (pred_minus == gold_index).float().mean()
        )
        # Saturation rate of the relevance scores over valid candidates.
        s = (logits_plus.float() + bias).masked_select(cand_mask)
        sat = float(((s.abs() > 15).float().mean())) if s.numel() else 0.0
        b = gold_index.shape[0]
        self._pair_log["l_bce"] += float(result["l_bce"]) * b
        self._pair_log["l_js"] += float(result["l_js"]) * b
        self._pair_log["l_js_weighted"] += float(result["l_js_weighted"]) * b
        self._pair_log["js_unweighted"] += float(result["js_unweighted"]) * b
        self._pair_log["acc"] += float(acc) * b
        self._pair_log["bias"] += float(bias.detach().reshape(-1)[0]) * b
        self._pair_log["sat"] += sat * b
        self._pair_log["n"] += b
        self._pair_log["n_skipped"] += int(result["n_skipped"])

    def log(self, logs, *args, **kwargs):
        if self._pair_log["n"] > 0 and "loss" in logs:
            n = self._pair_log["n"]
            if self.objective == "categorical":
                logs["l_ground"] = round(self._pair_log["l_ground"] / n, 5)
                logs["l_pair"] = round(self._pair_log["l_pair"] / n, 5)
                logs["ground_acc"] = round(self._pair_log["acc"] / n, 5)
                self._pair_log = {"l_ground": 0.0, "l_pair": 0.0, "acc": 0.0, "n": 0}
            else:
                logs["l_bce"] = round(self._pair_log["l_bce"] / n, 5)
                logs["l_js"] = round(self._pair_log["l_js"] / n, 5)
                logs["l_js_weighted"] = round(self._pair_log["l_js_weighted"] / n, 6)
                logs["js_unweighted"] = round(self._pair_log["js_unweighted"] / n, 6)
                logs["ground_acc"] = round(self._pair_log["acc"] / n, 5)
                logs["bias"] = round(self._pair_log["bias"] / n, 5)
                logs["saturation"] = round(self._pair_log["sat"] / n, 5)
                logs["skipped_rows"] = self._pair_log["n_skipped"]
                self._pair_log = {
                    "l_bce": 0.0, "l_js": 0.0, "l_js_weighted": 0.0, "js_unweighted": 0.0,
                    "acc": 0.0, "bias": 0.0, "sat": 0.0, "n": 0, "n_skipped": 0,
                }
        super().log(logs, *args, **kwargs)
