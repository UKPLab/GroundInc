"""Deterministic candidate temporal-interval bank for paired grounding regularization.

The bank is a *caption-independent, gold-independent* set of candidate evidence
intervals derived only from the video duration (and, optionally, a segmentation
that is available at inference time). It is the categorical support over which the
localizer predicts, and it must be identical for both captions in a pair and for
both experimental variants (`paired_ce` / `paired_js`).

Design rules enforced here:
  * The bank never uses gold evidence boundaries, gold types, the source caption
    timeline, or the faithful counterpart of a test caption. It is built from the
    video extent alone.
  * Intervals are half-open ``[start, end)`` seconds stored as floats.
  * Construction is deterministic: fixed window widths, fixed strides, clip to the
    video extent, guaranteed tail coverage, dedup, and a deterministic cap-reduction
    that preserves coverage across scales and temporal positions.
  * During *training only*, the target candidate is the one with maximum IoU
    against the gold span, with a deterministic tie-break. The gold interval is
    never inserted into the bank.

IoU convention: to keep oracle-recall numbers directly comparable with the
project's grounding metric, the default IoU uses the "hull" union
``max(a_end, b_end) - min(a_start, b_start)``. A standard
intersection-over-union (``len(a)+len(b)-inter``) is also provided.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Default rounding used for dedup / storage of interval boundaries (seconds).
_ROUND = 3


def temporal_iou(a: Sequence[float], b: Sequence[float], mode: str = "hull") -> float:
    """IoU between two ``[start, end)`` intervals in seconds.

    ``mode="hull"`` (default) matches the project's grounding metric, using the
    convex-hull span as the union. ``mode="standard"`` uses the classic
    length(a)+length(b)-intersection union.
    """
    a_start, a_end = float(a[0]), float(a[1])
    b_start, b_end = float(b[0]), float(b[1])
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    if mode == "standard":
        union = (a_end - a_start) + (b_end - b_start) - inter
    elif mode == "hull":
        union = max(a_end, b_end) - min(a_start, b_start)
    else:
        raise ValueError(f"Unknown IoU mode: {mode!r}")
    return inter / union if union > 0 else 0.0


@dataclass
class IntervalBank:
    """A deterministic candidate bank for a single video.

    Attributes:
        intervals: sorted, deduplicated list of ``(start, end)`` float tuples.
        duration: the video duration (seconds) the bank was built for.
        config: the construction parameters (for provenance / logging).
        version_hash: sha256 over the config and intervals; identical config +
            duration always yields the same hash, so the same bank version is
            used for both captions of a pair and both variants.
    """

    intervals: List[Tuple[float, float]]
    duration: float
    config: dict = field(default_factory=dict)
    version_hash: str = ""

    def __post_init__(self):
        if not self.version_hash:
            self.version_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = {
            "config": self.config,
            "intervals": [[round(s, _ROUND), round(e, _ROUND)] for s, e in self.intervals],
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.intervals)

    def best_candidate(
        self, gold_span: Sequence[float], iou_mode: str = "hull"
    ) -> Tuple[int, float]:
        """Return ``(index, iou)`` of the max-IoU candidate against ``gold_span``.

        Training-only. Deterministic tie-break: highest IoU, then the tightest
        (shortest) interval, then the earliest start, then the smallest index.
        Returns ``(-1, 0.0)`` if ``gold_span`` is not a valid 2-tuple.
        """
        if not gold_span or len(gold_span) < 2:
            return -1, 0.0
        gs, ge = float(gold_span[0]), float(gold_span[1])
        if ge <= gs:
            return -1, 0.0

        best_idx = -1
        best_key = None
        for idx, (start, end) in enumerate(self.intervals):
            iou = temporal_iou((start, end), (gs, ge), mode=iou_mode)
            # Sort key: maximize iou, then minimize length, then minimize start,
            # then minimize index. Negate the "maximize" terms so smaller wins.
            key = (-iou, (end - start), start, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        best_iou = -best_key[0] if best_key is not None else 0.0
        return best_idx, best_iou

    def to_id_map(self) -> dict:
        """Deterministic mapping ``candidate_index -> [start, end]`` for logging."""
        return {i: [round(s, _ROUND), round(e, _ROUND)] for i, (s, e) in enumerate(self.intervals)}


def _tail_extended_starts(duration: float, width: float, stride: float) -> List[float]:
    """Sliding-window starts covering ``[0, duration)`` including the tail.

    Windows start at ``0, stride, 2*stride, ...`` while the start is inside the
    video, and a final window flush against the tail is always included so the
    end of the video is covered even when the stride does not divide it evenly.
    """
    starts: List[float] = []
    if duration <= 0 or width <= 0 or stride <= 0:
        return starts
    s = 0.0
    # Step while the window still begins before the end of the video.
    while s < duration - 1e-9:
        starts.append(s)
        s += stride
    # Guarantee tail coverage with a window flush to the end of the video.
    tail_start = max(0.0, duration - width)
    starts.append(tail_start)
    return starts


def _windows_for_width(
    duration: float, width: float, stride: float, min_len: float
) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    for start in _tail_extended_starts(duration, width, stride):
        s = max(0.0, start)
        e = min(duration, s + width)
        if e - s >= min_len:
            windows.append((round(s, _ROUND), round(e, _ROUND)))
    return windows


def _downselect_by_width(
    windows_by_width: List[List[Tuple[float, float]]], cap: int
) -> List[Tuple[float, float]]:
    """Deterministically reduce a set of per-width windows to at most ``cap``.

    The cap is split across the scales (widths) as evenly as possible, and within
    each scale windows are subsampled at evenly-spaced temporal positions. This
    preserves coverage across both scale and time and never consults gold
    annotations. Any leftover budget (from scales with fewer windows than their
    share) is redistributed to the remaining scales.
    """
    n_scales = len(windows_by_width)
    if n_scales == 0:
        return []
    # Round-robin budget allocation so the split is deterministic and balanced.
    budgets = [cap // n_scales] * n_scales
    for i in range(cap % n_scales):
        budgets[i] += 1

    # Redistribute unused budget from scales that have fewer windows than allotted.
    changed = True
    while changed:
        changed = False
        deficit = 0
        hungry = []
        for i, wins in enumerate(windows_by_width):
            if budgets[i] > len(wins):
                deficit += budgets[i] - len(wins)
                budgets[i] = len(wins)
            elif budgets[i] < len(wins):
                hungry.append(i)
        while deficit > 0 and hungry:
            for i in list(hungry):
                if deficit <= 0:
                    break
                if budgets[i] < len(windows_by_width[i]):
                    budgets[i] += 1
                    deficit -= 1
                    changed = True
                if budgets[i] >= len(windows_by_width[i]):
                    hungry.remove(i)

    selected: List[Tuple[float, float]] = []
    for wins, budget in zip(windows_by_width, budgets):
        if budget <= 0:
            continue
        if len(wins) <= budget:
            selected.extend(wins)
            continue
        # Evenly-spaced indices across the (time-sorted) windows of this scale.
        n = len(wins)
        idxs = sorted({round(k * (n - 1) / (budget - 1)) if budget > 1 else 0 for k in range(budget)})
        # Fill in case rounding collapsed some indices.
        i = 0
        while len(idxs) < budget and i < n:
            if i not in idxs:
                idxs.append(i)
            i += 1
        for j in sorted(idxs)[:budget]:
            selected.append(wins[j])
    return selected


def build_bank(
    duration: float,
    *,
    widths: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
    stride_frac: float = 0.5,
    max_candidates: int = 128,
    min_len: float = 1.0,
) -> IntervalBank:
    """Build a deterministic candidate interval bank from the video duration only.

    Args:
        duration: video length in seconds (may be non-integer).
        widths: candidate window widths in seconds.
        stride_frac: stride as a fraction of each width (0.5 => 50% overlap).
        max_candidates: hard cap on the number of candidates.
        min_len: drop clipped windows shorter than this (seconds). Short videos
            still yield at least one candidate (the whole video).

    The bank is independent of any caption, gold span, or label.
    """
    duration = float(duration)
    config = {
        "widths": list(widths),
        "stride_frac": stride_frac,
        "max_candidates": max_candidates,
        "min_len": min_len,
    }

    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    # Effective widths never exceed the video; drop duplicates while preserving order.
    eff_widths: List[float] = []
    for w in widths:
        w = min(float(w), duration)
        if w > 0 and w not in eff_widths:
            eff_widths.append(w)
    if not eff_widths:
        eff_widths = [duration]

    windows_by_width: List[List[Tuple[float, float]]] = []
    for w in eff_widths:
        stride = max(w * stride_frac, _round_up(min_len))
        wins = _windows_for_width(duration, w, stride, min_len)
        if wins:
            windows_by_width.append(wins)

    # Always guarantee a whole-video candidate so short/edge videos are covered.
    whole = (0.0, round(duration, _ROUND))
    windows_by_width.append([whole])

    # Dedup across scales while remembering which scale each interval came from,
    # so the cap-reduction can still balance across scales.
    seen = set()
    deduped_by_width: List[List[Tuple[float, float]]] = []
    for wins in windows_by_width:
        kept = []
        for iv in wins:
            key = (round(iv[0], _ROUND), round(iv[1], _ROUND))
            if key not in seen and key[1] > key[0]:
                seen.add(key)
                kept.append(key)
        if kept:
            deduped_by_width.append(kept)

    total = sum(len(w) for w in deduped_by_width)
    if total > max_candidates:
        intervals = _downselect_by_width(deduped_by_width, max_candidates)
    else:
        intervals = [iv for wins in deduped_by_width for iv in wins]

    # Final deterministic ordering: by (start, end).
    intervals = sorted(set(intervals), key=lambda iv: (iv[0], iv[1]))
    return IntervalBank(intervals=intervals, duration=duration, config=config)


def _round_up(x: float) -> float:
    return max(x, 1e-6)


def oracle_recall(
    best_ious: Sequence[float], thresholds: Sequence[float] = (0.3, 0.5)
) -> dict:
    """Fraction of examples whose best candidate meets each IoU threshold.

    ``best_ious`` is the per-example best achievable IoU of the bank against the
    gold span. Reporting this makes the proposal ceiling visible. Examples below
    the reporting threshold are *not* filtered out here — pass every example.
    """
    n = len(best_ious)
    out = {"n": n, "mean_best_iou": (sum(best_ious) / n) if n else 0.0}
    for t in thresholds:
        key = f"oracle_recall@{t}"
        out[key] = (sum(1 for v in best_ious if v >= t) / n) if n else 0.0
    return out
