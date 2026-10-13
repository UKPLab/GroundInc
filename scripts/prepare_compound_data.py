#!/usr/bin/env python3
"""Build the released GroundInc compound (multi-span / multi-label) annotation file.

Keeps only what ``groundinc.data.compound.build_multi_caption_tasks`` reads: the
record identity/timing, the caption timelines, and the three evaluation subsets of
``multi_inconsistency``. Generation scratch (per-modality planning, anchor
rephrasings, rejected manipulations) and the inherited multi-turn QA / positional
bookkeeping are dropped. Shipping this script keeps the transformation auditable.

    python scripts/prepare_compound_data.py \
      --input  /path/to/news_data_multi.jsonl \
      --output data/groundinc_news_compound.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from groundinc.data.compound import SUBSET_KEYS, build_multi_caption_tasks  # noqa: E402

# Multi-turn QA + positional bookkeeping inherited from the upstream corpus; no
# GroundInc code reads them.
QA_FIELDS = tuple(
    f"{a}2{b}_{kind}"
    for a, b in (("v", "a"), ("a", "v"), ("v", "t"), ("t", "v"), ("a", "t"), ("t", "a"))
    for kind in ("question", "answer")
)
DROP_FIELDS = QA_FIELDS + ("index", "segment_num")

# Record-level fields the loader / bank construction need.
KEEP_RECORD = (
    "id", "duration", "category", "select_seg", "seg_time",
    "video_captions", "audio_captions",
)

# The only multi_inconsistency subkeys any GroundInc code reads (the eval subsets).
KEEP_MI = tuple(SUBSET_KEYS.values())


def strip_record(record):
    out = {k: record[k] for k in KEEP_RECORD if k in record}
    mi = record.get("multi_inconsistency") or {}
    out["multi_inconsistency"] = {k: mi[k] for k in KEEP_MI if k in mi}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = 0
    tasks = Counter()
    kept_items = Counter()
    out_records = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows += 1
            stripped = strip_record(rec)
            for subset in SUBSET_KEYS:
                n = len(stripped["multi_inconsistency"].get(SUBSET_KEYS[subset]) or [])
                kept_items[subset] += n
                tasks[subset] += len(build_multi_caption_tasks(stripped, subset))
            out_records.append(stripped)

    print(f"records: {rows}")
    for subset in SUBSET_KEYS:
        print(f"  {subset:16s}: {kept_items[subset]:5d} items -> {tasks[subset]:5d} caption tasks")

    if args.dry_run:
        print("dry run; nothing written")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for rec in out_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_records)} records -> {args.output}")


if __name__ == "__main__":
    main()
