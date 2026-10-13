#!/usr/bin/env python3
"""Build the released GroundInc annotation files from the working dataset.

Drops fields that no GroundInc code reads, then reports what the release
actually contains. Shipping this script rather than hand-edited JSONL keeps the
transformation auditable.

    python scripts/prepare_release_data.py \
      --input  /path/to/news_data_mt_final.jsonl \
      --output data/groundinc_news.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from groundinc.data.dataset import EXCLUDED_LABELS, build_caption_tasks  # noqa: E402

# Multi-turn QA inherited from the upstream caption source. No GroundInc or
# training code reads these.
QA_FIELDS = tuple(
    f"{a}2{b}_{kind}"
    for a, b in (("v", "a"), ("a", "v"), ("v", "t"), ("t", "v"), ("a", "t"), ("t", "a"))
    for kind in ("question", "answer")
)

# Positional bookkeeping from the source corpus; zero references anywhere.
BOOKKEEPING_FIELDS = ("index", "segment_num")

DROP_FIELDS = QA_FIELDS + BOOKKEEPING_FIELDS

# Read by groundinc.data.dataset / the evaluators / SFT preparation.
REQUIRED_FIELDS = (
    "id",
    "select_seg",
    "seg_time",
    "video_captions",
    "audio_captions",
    "duration",
    "forensic_mentalization",
)


def strip_record(record):
    return {k: v for k, v in record.items() if k not in DROP_FIELDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    rows, tasks_total = 0, 0
    methods, categories, task_types = Counter(), Counter(), Counter()
    missing = Counter()
    dropped_keys = Counter()
    out_records = []

    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1

            for field in DROP_FIELDS:
                if field in record:
                    dropped_keys[field] += 1
            for field in REQUIRED_FIELDS:
                if field not in record:
                    missing[field] += 1

            stripped = strip_record(record)
            out_records.append(stripped)

            categories[stripped.get("category", "<none>")] += 1
            # video_mode "id" mirrors how the evaluators read these files.
            for task in build_caption_tasks(stripped, "id"):
                tasks_total += 1
                methods[task["manipulation_method_gt"]] += 1
                task_types[task["type"] if task["type"] in ("original_true", "rephrased_true")
                           else "manipulation"] += 1

    if missing:
        print("ERROR: required fields missing from some records:", dict(missing))
        return 1

    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for record in out_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    src_mb = os.path.getsize(args.input) / 1e6
    print(f"\n{os.path.basename(args.input)} -> {os.path.basename(args.output)}")
    print(f"  records           {rows}")
    print(f"  caption tasks     {tasks_total}  ({tasks_total / rows:.1f} per record)")
    print(f"  fields dropped    {len(dropped_keys)} distinct "
          f"({sum(1 for f in dropped_keys if f in QA_FIELDS)} QA, "
          f"{sum(1 for f in dropped_keys if f in BOOKKEEPING_FIELDS)} bookkeeping)")
    print(f"  video categories  {dict(categories)}")
    print(f"  task mix          {dict(task_types)}")
    if not args.dry_run:
        out_mb = os.path.getsize(args.output) / 1e6
        print(f"  size              {src_mb:.1f} MB -> {out_mb:.1f} MB "
              f"({100 * (1 - out_mb / src_mb):.1f}% smaller)")

    print("  manipulation methods:")
    for method, count in methods.most_common():
        marker = "  [EXCLUDED at load]" if method in EXCLUDED_LABELS else ""
        print(f"    {count:6d}  {method}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
