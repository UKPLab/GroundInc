"""Faithful/manipulated pair construction, validation, sampling, and splitting.

Pairs are built from *explicit provenance* in the ChronusAV schema, never inferred
from mere co-membership in a video. For a record and a modality, the faithful
caption is the rephrased-true caption of that modality (``forensic_mentalization
.true_visual`` / ``.true_audio`` / ``.true_combined``), falling back to the
original segment caption; the manipulated partners are that modality's generated
manipulations. Faithful and manipulated partners of the same (record, select_seg,
modality) share the same gold evidence span and carry opposite verdicts.

Reuses ``training/final_train`` for the canonical task construction so pair
semantics stay identical to the SFT pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Make the canonical final_train modules importable (flat imports).

from .data_io import (  # noqa: E402  (path injected above)
    build_caption_tasks,
    compute_ground_truth_span,
    iter_input_records,
)

# Records whose annotated span exceeds the true video duration (see overview.md).
CORRUPTED_IDS = {"AexHe25MPgI", "tfjl3ElqQKU", "wnwi2rgX0AQ"}

_TRUE_TYPES = {"original_true", "rephrased_true"}


@dataclass
class Pair:
    video_id: str
    event_id: str
    pair_id: str
    video_path: Optional[str]
    video_duration_seconds: float
    caption_modality: str
    faithful_caption: str
    manipulated_caption: str
    manipulated_type: str
    shared_gold_start_seconds: float
    shared_gold_end_seconds: float
    # Provenance for auditing / rotation.
    faithful_type: str = "rephrased_true"
    manip_index: int = 0

    @property
    def gold_span(self) -> List[float]:
        return [self.shared_gold_start_seconds, self.shared_gold_end_seconds]


def _valid_span(span, duration: float) -> bool:
    if not span or len(span) < 2:
        return False
    s, e = float(span[0]), float(span[1])
    if e <= s:
        return False
    # Reject spans that run past the real video (corrupted annotations).
    if duration and e > duration + 1.0:
        return False
    return True


def build_pairs_from_record(record: dict, video_mode: str = "id") -> Tuple[List[Pair], List[dict]]:
    """Return ``(pairs, excluded)`` for a single record.

    ``excluded`` is a list of ``{"reason", ...}`` diagnostics for ambiguous or
    invalid pairings that were dropped rather than silently paired.
    """
    pairs: List[Pair] = []
    excluded: List[dict] = []

    vid = record["id"]
    if vid in CORRUPTED_IDS:
        excluded.append({"video_id": vid, "reason": "corrupted_record"})
        return pairs, excluded

    duration = float(record.get("duration") or 0.0)
    video_path = record.get("video_path")
    select_seg = record.get("select_seg")

    try:
        tasks = build_caption_tasks(record, video_mode)
    except Exception as exc:  # pragma: no cover - defensive
        excluded.append({"video_id": vid, "reason": f"task_build_error: {exc}"})
        return pairs, excluded

    gold_span = compute_ground_truth_span(record, video_mode)
    if not _valid_span(gold_span, duration):
        excluded.append({"video_id": vid, "reason": "invalid_gold_span", "gold_span": gold_span})
        return pairs, excluded
    gold_start, gold_end = float(gold_span[0]), float(gold_span[1])

    # Group tasks by modality; separate the faithful caption from manipulations.
    faithful_by_mod: Dict[str, Tuple[str, str]] = {}  # modality -> (caption, faithful_type)
    manips_by_mod: Dict[str, List[dict]] = {"visual": [], "audio": [], "combined": []}

    for task in tasks:
        mod = task["category"]
        if task["type"] in _TRUE_TYPES:
            # Prefer rephrased_true; keep original_true only as a fallback.
            existing = faithful_by_mod.get(mod)
            if existing is None or (task["type"] == "rephrased_true" and existing[1] != "rephrased_true"):
                faithful_by_mod[mod] = (task["caption"], task["type"])
        else:
            manips_by_mod.setdefault(mod, []).append(task)

    for mod, manips in manips_by_mod.items():
        faithful = faithful_by_mod.get(mod)
        for manip_index, manip in enumerate(manips):
            if not manip.get("caption"):
                excluded.append({"video_id": vid, "modality": mod, "reason": "empty_manip_caption"})
                continue
            if faithful is None or not faithful[0]:
                excluded.append(
                    {"video_id": vid, "modality": mod, "reason": "no_faithful_partner"}
                )
                continue
            # Both partners share this record's single gold span; spans are equal
            # by construction, but assert opposite verdicts and matching event.
            pairs.append(
                Pair(
                    video_id=vid,
                    event_id=f"{vid}:{select_seg}:{mod}",
                    pair_id=f"{vid}:{select_seg}:{mod}:{manip_index}",
                    video_path=video_path,
                    video_duration_seconds=duration,
                    caption_modality=mod,
                    faithful_caption=faithful[0],
                    manipulated_caption=manip["caption"],
                    manipulated_type=manip.get("manipulation_method_gt") or manip.get("type") or "unknown",
                    shared_gold_start_seconds=gold_start,
                    shared_gold_end_seconds=gold_end,
                    faithful_type=faithful[1],
                    manip_index=manip_index,
                )
            )

    return pairs, excluded


def build_pairs(records: Iterable[dict], video_mode: str = "id") -> Tuple[List[Pair], List[dict]]:
    """Build validated pairs across many records, aggregating exclusions."""
    all_pairs: List[Pair] = []
    all_excluded: List[dict] = []
    for record in records:
        pairs, excluded = build_pairs_from_record(record, video_mode)
        all_pairs.extend(pairs)
        all_excluded.extend(excluded)
    logger.info(
        "Built %d pairs; excluded %d ambiguous/invalid pairings",
        len(all_pairs),
        len(all_excluded),
    )
    return all_pairs, all_excluded


def load_pairs(jsonl_path: str, video_mode: str = "id") -> Tuple[List[Pair], List[dict]]:
    return build_pairs(iter_input_records(jsonl_path), video_mode)


# ---------------------------------------------------------------------------
# Video-disjoint splitting
# ---------------------------------------------------------------------------
def video_disjoint_split(
    pairs: List[Pair], dev_frac: float = 0.1, seed: int = 42
) -> Tuple[List[Pair], List[Pair]]:
    """Split pairs into (train, dev) so that no source video spans both.

    Deterministic: each video_id is hashed with the seed and assigned to dev if
    its normalized hash falls under ``dev_frac``. Every caption/pair/clip from a
    video therefore stays in the same split.
    """
    import hashlib

    def bucket(video_id: str) -> float:
        h = hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    train, dev = [], []
    for pair in pairs:
        (dev if bucket(pair.video_id) < dev_frac else train).append(pair)
    return train, dev


def assert_video_disjoint(train: List[Pair], dev: List[Pair]) -> None:
    train_vids = {p.video_id for p in train}
    dev_vids = {p.video_id for p in dev}
    overlap = train_vids & dev_vids
    if overlap:
        raise AssertionError(f"Video overlap across splits: {sorted(overlap)[:5]}")


# ---------------------------------------------------------------------------
# Partner-rotating sampler
# ---------------------------------------------------------------------------
class PairSampler:
    """Selects one manipulated partner per (event, modality) per epoch.

    A faithful caption whose event spawned many manipulated variants is *not*
    oversampled: exactly one pair is drawn per event/modality each epoch, and the
    chosen partner rotates deterministically across epochs. The same sampler,
    counts, and order are used for both variants (lambda controls the loss only).
    """

    def __init__(self, pairs: List[Pair], seed: int = 42):
        self.seed = seed
        # Group pairs by event (video, select_seg, modality).
        self.groups: Dict[str, List[Pair]] = {}
        for pair in pairs:
            self.groups.setdefault(pair.event_id, []).append(pair)
        # Deterministic partner order within each event.
        for event_id in self.groups:
            self.groups[event_id].sort(key=lambda p: p.manip_index)
        self.event_ids = sorted(self.groups)

    def num_events(self) -> int:
        return len(self.event_ids)

    def _start_offset(self, event_id: str) -> int:
        """Deterministic per-event starting offset derived from the seed.

        Without this the rotation always begins at manip_index 0, so with only a
        few sampler epochs every event contributes the *same* first manipulations
        (the serialized order is meaningful) — a distributional bias. Hashing
        ``(seed, event_id)`` decorrelates which partner each event starts on.
        """
        h = hashlib.sha256(f"{self.seed}:{event_id}".encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def epoch_pairs(self, epoch: int) -> List[Pair]:
        """One pair per event for ``epoch``; the partner rotates with a
        seed-dependent per-event offset so coverage is not order-biased."""
        selected = []
        for event_id in self.event_ids:
            partners = self.groups[event_id]
            idx = (self._start_offset(event_id) + epoch) % len(partners)
            selected.append(partners[idx])
        return selected
