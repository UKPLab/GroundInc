"""Task builder for the multi-span / multi-label ChronusAV subset.

The multi-generation pipeline (ChronusAV/scripts/multi_gen.py) stores manipulations
under ``data["multi_inconsistency"]`` in three subsets, keyed by:

    multi_span        -> multi_span_manipulations        (>=2 regions, 1 label)
    multi_label       -> multi_label_manipulations       (1 region, >=2 labels)
    multi_span_label  -> multi_span_label_manipulations  (>=2 regions, >=2 labels)

Each item carries a controlled true/false caption pair plus list-valued gold
``ground_truth_spans`` and ``methods``. This is a different shape from the
single-span ``forensic_mentalization`` records handled by
``methods_v2.dataloader.build_caption_tasks`` (which expects a scalar
``ground_truth_span``), so the multi subset needs its own task builder.
"""

SUBSET_KEYS = {
    "multi_span": "multi_span_manipulations",
    "multi_label": "multi_label_manipulations",
    "multi_span_label": "multi_span_label_manipulations",
}

SUBSETS = tuple(SUBSET_KEYS)


def _normalize_spans(spans):
    """Return a list of [start, end] float pairs, dropping malformed entries."""
    normalized = []
    if not isinstance(spans, list):
        return normalized
    for span in spans:
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            try:
                normalized.append([float(span[0]), float(span[1])])
            except (TypeError, ValueError):
                continue
    return normalized


def _normalize_methods(methods):
    if not isinstance(methods, list):
        return []
    return [m for m in methods if isinstance(m, str) and m.strip()]


def _make_task(item, subset, item_index, caption, task_type, spans, methods, veracity):
    return {
        "subset": subset,
        "set": item.get("set"),
        "item_index": item_index,
        "type": task_type,
        "caption": caption,
        "ground_truth_spans": spans,
        "methods_gt": methods,
        "veracity_gt": veracity,
        "num_spans_gt": len(spans),
        "num_labels_gt": len(methods),
        "manipulated_modality": item.get("evidence_scope"),
        "evidence_scope": item.get("evidence_scope"),
        "generation_scope": item.get("generation_scope"),
        "temporal_scope": item.get("temporal_scope"),
        "entity_type": item.get("entity_type"),
        "narrative": item.get("narrative"),
        "refutation": item.get("refutation"),
        "inconsistencies": item.get("inconsistencies"),
        "true_caption": item.get("true_caption"),
        "false_caption": item.get("false_caption"),
    }


def build_multi_caption_tasks(data, subset, include_true=True):
    """Build caption evaluation tasks for one multi subset of a record.

    Emits one task per manipulation item (the ``false_caption``) and, when
    ``include_true`` is set, one matched control task per item (the paired
    ``true_caption``, gold veracity True / methods "No Inconsistency"). Both
    captions share the same list-valued gold spans.
    """
    if subset not in SUBSET_KEYS:
        raise ValueError(
            f"Unknown multi subset '{subset}'. Valid: {', '.join(SUBSETS)}"
        )

    mi = data.get("multi_inconsistency") or {}
    items = mi.get(SUBSET_KEYS[subset]) or []

    tasks = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        spans = _normalize_spans(item.get("ground_truth_spans"))
        methods = _normalize_methods(item.get("methods"))
        false_caption = item.get("false_caption")
        if not false_caption or not spans or not methods:
            continue

        tasks.append(
            _make_task(
                item, subset, idx, false_caption, "manipulated",
                spans, methods, "False",
            )
        )

        if include_true:
            true_caption = item.get("true_caption")
            if true_caption:
                tasks.append(
                    _make_task(
                        item, subset, idx, true_caption, "controlled_true",
                        spans, ["No Inconsistency"], "True",
                    )
                )

    return tasks
