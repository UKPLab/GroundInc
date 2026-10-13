"""Paired grounding objective: grounding cross-entropy + Jensen-Shannon pair loss.

For a paired faithful caption ``c+`` and manipulated caption ``c-`` with
distributions ``p+`` and ``p-`` over their *shared* candidate bank and shared
gold target ``j*``:

    L_ground = 0.5 * (CE(p+, j*) + CE(p-, j*))
    m        = 0.5 * (p+ + p-)
    L_pair   = 0.5 * KL(p+ || m) + 0.5 * KL(p- || m)      # Jensen-Shannon
    L        = L_ground + lambda_pair * L_pair

Both branches receive gradients; neither is a fixed teacher. At ``lambda_pair=0``
the objective is exactly the ordinary categorical grounding cross-entropy on both
captions. Divergences are accumulated in float32 for stability under mixed
precision, with log-space math and safe clamping so extreme logits do not NaN.

Never align verdict or inconsistency-type distributions between the two captions;
this module only couples the *localization* distributions.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_NEG_INF = float("-inf")
_EPS = 1e-8


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over valid candidates only.

    Args:
        logits: ``[..., K]`` raw logits for candidate IDs.
        mask: ``[..., K]`` boolean/0-1 tensor; True = valid candidate.

    Invalid positions are set to ``-inf`` before normalization so they receive
    exactly zero probability. Computed in float32.
    """
    logits = logits.float()
    mask_bool = mask.bool()
    masked = logits.masked_fill(~mask_bool, _NEG_INF)
    return F.log_softmax(masked, dim=-1)


def js_divergence(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two categorical distributions.

    Args:
        log_p, log_q: ``[..., K]`` log-probabilities (already masked/normalized).
        mask: ``[..., K]`` validity mask.

    Returns per-row JS ``[...]`` in nats, computed in float32. JS is symmetric,
    bounded in ``[0, log 2]``, and zero iff ``p == q``.
    """
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = (m + _EPS).log()

    # KL(p || m) = sum_j p_j (log p_j - log m_j), only over valid, non-zero p.
    mask_bool = mask.bool()

    def _kl(log_a: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Make log_a finite *before* the multiply. Padded/zero positions have
        # log_a = -inf; `a * (-inf)` with a==0 is NaN in the forward and, even
        # after masking the value, contaminates the backward (0 * -inf grad).
        # Substituting a finite log where a==0 keeps the product exactly 0 there.
        log_a_safe = torch.where(a > 0, log_a, torch.zeros_like(log_a))
        term = a * (log_a_safe - log_m)  # a==0 -> 0 * finite = 0
        term = torch.where(mask_bool & (a > 0), term, torch.zeros_like(term))
        return term.sum(dim=-1)

    js = 0.5 * _kl(log_p, p) + 0.5 * _kl(log_q, q)
    return js.clamp_min(0.0)


def paired_grounding_loss(
    logits_plus: torch.Tensor,
    logits_minus: torch.Tensor,
    mask: torch.Tensor,
    gold_index: torch.Tensor,
    lambda_pair: float,
) -> Dict[str, torch.Tensor]:
    """Compute the paired grounding loss over a batch of pairs.

    Args:
        logits_plus: ``[B, K]`` candidate logits for faithful captions.
        logits_minus: ``[B, K]`` candidate logits for manipulated captions.
        mask: ``[B, K]`` validity mask (shared per pair).
        gold_index: ``[B]`` index of the gold target candidate within the K slots.
        lambda_pair: weight on the JS pair term.

    Returns a dict with ``loss``, ``l_ground``, ``l_pair`` (all scalar tensors),
    averaged over pairs.
    """
    # Every pair must have at least one valid candidate, else the softmax is all
    # -inf and the loss is undefined.
    if not bool(mask.bool().any(dim=-1).all()):
        raise ValueError("Every row must have >=1 valid candidate (cand_mask all-False row)")

    log_p = masked_log_softmax(logits_plus, mask)
    log_q = masked_log_softmax(logits_minus, mask)

    gather_idx = gold_index.long().unsqueeze(-1)
    ce_plus = -log_p.gather(-1, gather_idx).squeeze(-1)  # [B]
    ce_minus = -log_q.gather(-1, gather_idx).squeeze(-1)  # [B]
    l_ground = 0.5 * (ce_plus + ce_minus).mean()

    if lambda_pair != 0.0:
        js = js_divergence(log_p, log_q, mask)  # [B]
        l_pair = js.mean()
    else:
        # Exact zero, in-graph, device/dtype-correct. NOTE: do NOT use
        # `log_p.sum() * 0` — padded candidates are -inf, and -inf * 0 = NaN,
        # which contaminates the loss once a batch has >1 pair with variable K.
        l_pair = torch.zeros((), device=l_ground.device, dtype=l_ground.dtype)

    loss = l_ground + lambda_pair * l_pair
    return {"loss": loss, "l_ground": l_ground.detach(), "l_pair": l_pair.detach()}


# ===========================================================================
# Sigmoid objectives (SigLIP-inspired): class-balanced BCE + Bernoulli JS.
#
# These are the sigmoid counterparts of the categorical CE / CE+JS objectives
# above. They operate on exactly the same gathered candidate logits ``z[B, K]``,
# the same validity ``mask``, and the same single hard gold index ``j*`` used by
# the categorical path (``target_mode = existing_hard``), so the only thing that
# changes between a categorical arm and its sigmoid counterpart is the loss.
#
# Score:  s_j = z_j + b   (one shared scalar bias b per interval head; added by
#                          the caller — pass ``bias`` here).
# Target: y_j = 1[j == j*]  over valid candidates.
# BCE:    class-balanced weighted binary cross-entropy on ``s`` (section 5).
# JS:     class-balanced mean of per-bucket Bernoulli JS divergences (section 6),
#         NOT categorical JS on a sigmoid vector.
#
# Weighted sigmoid outputs are RELEVANCE SCORES, not calibrated probabilities.
# ===========================================================================


def class_balanced_weights(
    mask: torch.Tensor, gold_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row class-balanced candidate weights and binary targets (section 5).

    Args:
        mask: ``[B, K]`` validity mask (True = supervised candidate).
        gold_index: ``[B]`` index of the single positive candidate (``existing_hard``).

    Returns ``(w, y)`` each ``[B, K]`` float32:
      * ``y`` is the binary target (1 at the valid gold slot, else 0).
      * ``w`` sums to 1 over every supervised row: positives share total weight
        1/2 (``1/(2|P|)`` each), negatives share 1/2 (``1/(2|N|)`` each). If a row
        has only one class present, that class gets total weight 1 spread
        uniformly. Padded/invalid slots and rows with no valid candidate get 0.

    No additional ``pos_weight`` is applied on top of these weights.
    """
    mask_b = mask.bool()
    b, k = mask_b.shape
    device = mask_b.device
    rows = torch.arange(b, device=device)

    pos = torch.zeros(b, k, dtype=torch.bool, device=device)
    gi = gold_index.long().clamp(0, k - 1)
    pos[rows, gi] = True
    pos = pos & mask_b  # a positive must itself be a valid candidate
    neg = mask_b & ~pos

    n_pos = pos.sum(dim=-1)  # [B]
    n_neg = neg.sum(dim=-1)  # [B]

    both = (n_pos > 0) & (n_neg > 0)
    only_pos = (n_pos > 0) & (n_neg == 0)
    only_neg = (n_pos == 0) & (n_neg > 0)

    w_pos = torch.zeros(b, device=device)
    w_neg = torch.zeros(b, device=device)
    w_pos[both] = 1.0 / (2.0 * n_pos[both].float())
    w_neg[both] = 1.0 / (2.0 * n_neg[both].float())
    w_pos[only_pos] = 1.0 / n_pos[only_pos].float()
    w_neg[only_neg] = 1.0 / n_neg[only_neg].float()

    w = pos.float() * w_pos.unsqueeze(-1) + neg.float() * w_neg.unsqueeze(-1)
    return w.float(), pos.float()


def bernoulli_js_per_bucket(s_plus: torch.Tensor, s_minus: torch.Tensor) -> torch.Tensor:
    """Per-bucket Bernoulli Jensen-Shannon divergence (section 6).

    For each bucket ``j`` the two logits define two-outcome distributions
    ``P_j = [sigmoid(s_j), 1 - sigmoid(s_j)]``; returns ``d_j`` in nats, bounded
    in ``[0, log 2]``, symmetric, zero iff the two logits agree. Computed with
    log-space math (``logsigmoid`` / ``logaddexp``) for stability at extreme
    logits; both branches receive gradients (no detached teacher).

    Preserves the input dtype: callers running under mixed precision pass float32
    (the production path does so via ``paired_sigmoid_loss``), while numerical-
    reference / gradcheck tests pass float64 and get full precision.
    """
    log_p = torch.stack([F.logsigmoid(s_plus), F.logsigmoid(-s_plus)], dim=-1)
    log_q = torch.stack([F.logsigmoid(s_minus), F.logsigmoid(-s_minus)], dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    d = 0.5 * (
        (log_p.exp() * (log_p - log_m)).sum(dim=-1)
        + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
    )
    return d.clamp_min(0.0)


def paired_sigmoid_loss(
    logits_plus: torch.Tensor,
    logits_minus: torch.Tensor,
    mask: torch.Tensor,
    gold_index: torch.Tensor,
    lambda_pair: float,
    bias: "float | torch.Tensor" = 0.0,
) -> Dict[str, torch.Tensor]:
    """Paired sigmoid grounding loss: class-balanced BCE + Bernoulli JS.

    ``L_view = sum_j w_j * BCE(z_j + b, y_j)``; ``L_BCE = 0.5*(L+ + L-)`` per pair;
    ``L = L_BCE + lambda_pair * L_BernJS`` with the SAME per-pair balanced weights
    on the Bernoulli-JS term. Averaged over supervised pairs (not padded elements).
    At ``lambda_pair == 0`` the JS computation is bypassed entirely so the sigmoid
    BCE-only arm matches the paired sigmoid arm's ``lambda=0`` gradients exactly.

    Masked/padded candidates are replaced with finite placeholders BEFORE BCE/JS
    and then given zero weight, so a NaN can never be created and multiplied by
    zero. Nonfinite logits at *valid* candidates are an explicit error, not a
    silently dropped entry. Returns raw BCE, the (balanced) JS term, and the
    weighted JS contribution separately for logging.
    """
    if not bool(mask.bool().any(dim=-1).all()):
        # Callers must not pass all-invalid rows; the collator guarantees >=1
        # valid candidate per pair. (Multi-positive coverage-miss skipping, when
        # implemented, drops such rows upstream and keeps them in eval denominators.)
        raise ValueError("Every row must have >=1 valid candidate (cand_mask all-False row)")

    mask_b = mask.bool()
    valid = mask_b
    if not bool(torch.isfinite(logits_plus[valid]).all() and torch.isfinite(logits_minus[valid]).all()):
        raise ValueError("Nonfinite logit at a valid supervised candidate")

    w, y = class_balanced_weights(mask_b, gold_index)  # [B, K]
    supervised = mask_b.any(dim=-1)  # [B]

    zeros = torch.zeros_like(logits_plus, dtype=torch.float32)
    s_plus = torch.where(mask_b, logits_plus.float() + bias, zeros)
    s_minus = torch.where(mask_b, logits_minus.float() + bias, zeros)
    y = torch.where(mask_b, y, torch.zeros_like(y))

    bce_plus = F.binary_cross_entropy_with_logits(s_plus, y, reduction="none")
    bce_minus = F.binary_cross_entropy_with_logits(s_minus, y, reduction="none")
    l_bce_plus = (w * bce_plus).sum(dim=-1)   # [B]
    l_bce_minus = (w * bce_minus).sum(dim=-1)  # [B]
    l_bce_pair = 0.5 * (l_bce_plus + l_bce_minus)  # [B]
    l_bce = l_bce_pair[supervised].mean()

    if lambda_pair != 0.0:
        d = bernoulli_js_per_bucket(s_plus, s_minus)  # [B, K]
        l_js_row = (w * d).sum(dim=-1)  # [B]
        l_js = l_js_row[supervised].mean()
        # Unweighted per-bucket JS mean over valid candidates: a scale diagnostic.
        d_valid = torch.where(mask_b, d, torch.zeros_like(d))
        js_unweighted = (d_valid.sum(dim=-1) / mask_b.float().sum(dim=-1).clamp(min=1))[supervised].mean()
    else:
        l_js = torch.zeros((), device=l_bce.device, dtype=l_bce.dtype)
        js_unweighted = torch.zeros((), device=l_bce.device, dtype=l_bce.dtype)

    loss = l_bce + lambda_pair * l_js
    return {
        "loss": loss,
        "l_bce": l_bce.detach(),
        "l_js": l_js.detach(),                       # balanced Bernoulli-JS term
        "l_js_weighted": (lambda_pair * l_js).detach(),  # its contribution to loss
        "js_unweighted": js_unweighted.detach(),     # scale diagnostic
        "n_supervised": int(supervised.sum()),
        "n_skipped": int((~supervised).sum()),
    }


def fit_bias_bisection(
    logits: torch.Tensor,
    weights: torch.Tensor,
    target_prevalence: float = 0.5,
    iters: int = 100,
    pad: float = 20.0,
) -> float:
    """Fit a scalar bias ``b`` so the weighted mean of ``sigmoid(z + b)`` equals
    ``target_prevalence`` (section 4 initialization).

    ``sigmoid(z + b)`` is monotone increasing in ``b``, so a bounded bisection on
    a finite bracket derived from the logit range converges. Only supervised
    (nonzero-weight) entries contribute. Deterministic; used at init only, never
    training or validation-set calibration. Raises on nonfinite supervised logits
    or an out-of-range target.
    """
    z = logits.detach().float().reshape(-1)
    w = weights.detach().float().reshape(-1)
    keep = w > 0
    z = z[keep]
    w = w[keep]
    if z.numel() == 0:
        raise ValueError("fit_bias_bisection: no supervised entries")
    if not torch.isfinite(z).all():
        raise ValueError("fit_bias_bisection: nonfinite supervised logit")
    if not (0.0 < target_prevalence < 1.0):
        raise ValueError(f"target_prevalence must be in (0,1), got {target_prevalence}")
    w_sum = float(w.sum())

    def mean_prev(b: float) -> float:
        return float((w * torch.sigmoid(z + b)).sum() / w_sum)

    # Finite bracket: b so low that every sigmoid ~0, so high that every ~1.
    zmax = float(z.max())
    zmin = float(z.min())
    lo = -zmax - pad
    hi = -zmin + pad
    f_lo = mean_prev(lo) - target_prevalence
    f_hi = mean_prev(hi) - target_prevalence
    if f_lo > 0 or f_hi < 0:
        # Bracket did not span the target (should not happen for target in (0,1));
        # widen once defensively.
        lo -= 10.0 * pad
        hi += 10.0 * pad
    mid = 0.5 * (lo + hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_prev(mid) - target_prevalence > 0:
            hi = mid
        else:
            lo = mid
    return float(mid)
