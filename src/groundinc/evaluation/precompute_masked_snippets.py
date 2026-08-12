import argparse
import concurrent.futures
import os
import time

from groundinc.evaluation.vllm_common import (
    compute_snippet_span,
    ensure_masked_video_snippet,
    resolve_source_video_path,
)


from groundinc.data import iter_input_records


def build_task(data, video_dir, snippet_dir, resize_multiple):
    source_video_path = resolve_source_video_path(data, "id", video_dir)
    if not source_video_path or not os.path.exists(source_video_path):
        return {
            "status": "missing_video",
            "id": data.get("id"),
            "select_seg": data.get("select_seg"),
            "source_video_path": source_video_path,
        }

    gt_span = compute_snippet_span(data)
    if not gt_span or len(gt_span) < 2:
        return {
            "status": "missing_span",
            "id": data.get("id"),
            "select_seg": data.get("select_seg"),
            "source_video_path": source_video_path,
        }

    return {
        "status": "pending",
        "id": data["id"],
        "select_seg": data["select_seg"],
        "source_video_path": source_video_path,
        "start": float(gt_span[0]),
        "end": float(gt_span[1]),
        "snippet_dir": snippet_dir,
        "resize_multiple": resize_multiple,
    }


def run_task(task, overwrite=False, ffmpeg_timeout=None):
    if task["status"] != "pending":
        return task

    resize_tag = f"_m{task['resize_multiple']}" if task["resize_multiple"] else ""
    output_name = (
        f"{task['id']}_seg{task['select_seg']}_"
        f"{task['start']:.2f}_{task['end']:.2f}_masked{resize_tag}.mp4"
    )
    output_path = os.path.join(task["snippet_dir"], output_name)
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return {**task, "status": "cached", "output_path": output_path}

    created = ensure_masked_video_snippet(
        task["source_video_path"],
        task["start"],
        task["end"],
        task["snippet_dir"],
        task["id"],
        task["select_seg"],
        task["resize_multiple"],
        timeout=ffmpeg_timeout,
    )
    if created and os.path.exists(created) and os.path.getsize(created) > 0:
        return {**task, "status": "created", "output_path": created}
    return {**task, "status": "failed", "output_path": output_path}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute masked_snippet videos for groundinc.evaluation.vllm_asr.",
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--snippet_dir", required=True)
    parser.add_argument("--snippet_resize_multiple", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--ffmpeg_timeout",
        type=float,
        default=900,
        help="Seconds before one ffmpeg/ffprobe task is treated as failed. Use 0 to disable.",
    )
    parser.add_argument(
        "--status_interval",
        type=float,
        default=30,
        help="Seconds between status messages while waiting for workers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build the work list and print counts without running ffmpeg.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.snippet_dir, exist_ok=True)

    records = list(iter_input_records(args.input_file))
    if args.max_records is not None:
        records = records[: args.max_records]

    tasks = [
        build_task(
            data,
            args.video_dir,
            args.snippet_dir,
            args.snippet_resize_multiple,
        )
        for data in records
    ]

    deduped = []
    seen = set()
    for task in tasks:
        if task["status"] != "pending":
            deduped.append(task)
            continue
        key = (
            task["source_video_path"],
            task["id"],
            task["select_seg"],
            task["start"],
            task["end"],
            task["resize_multiple"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)

    pending = [task for task in deduped if task["status"] == "pending"]
    skipped = [task for task in deduped if task["status"] != "pending"]
    print(
        f"Records: {len(records)} | unique tasks: {len(deduped)} | "
        f"pending: {len(pending)} | skipped before ffmpeg: {len(skipped)}"
    )

    if args.dry_run:
        return

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = None

    counts = {}
    results = list(skipped)
    ffmpeg_timeout = args.ffmpeg_timeout if args.ffmpeg_timeout and args.ffmpeg_timeout > 0 else None
    last_status_time = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = {
            executor.submit(run_task, task, args.overwrite, ffmpeg_timeout): task
            for task in pending
        }
        if tqdm is not None:
            pbar = tqdm(total=len(futures), desc="Masked snippets")
        else:
            pbar = None

        remaining = set(futures)
        while remaining:
            done, remaining = concurrent.futures.wait(
                remaining,
                timeout=max(1.0, args.status_interval),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                result = future.result()
                results.append(result)
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if pbar is not None:
                    pbar.update(1)

            now = time.monotonic()
            if remaining and now - last_status_time >= args.status_interval:
                newest = []
                for future in list(remaining)[:5]:
                    task = futures[future]
                    newest.append(f"{task['id']}_seg{task['select_seg']}")
                print(
                    f"Still waiting: done={len(futures) - len(remaining)}/{len(futures)} "
                    f"running={len(remaining)} created={counts.get('created', 0)} "
                    f"cached={counts.get('cached', 0)} failed={counts.get('failed', 0)} "
                    f"examples={', '.join(newest)}",
                    flush=True,
                )
                last_status_time = now

        if pbar is not None:
            pbar.close()

    for task in skipped:
        counts[task["status"]] = counts.get(task["status"], 0) + 1

    print("Done.")
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")

    failures = [result for result in results if result["status"] in {"failed", "missing_video", "missing_span"}]
    if failures:
        print("\nFirst failures/skips:")
        for result in failures[:10]:
            print(
                f"{result['status']}: id={result.get('id')} "
                f"select_seg={result.get('select_seg')} source={result.get('source_video_path')}"
            )
        if any(result["status"] == "failed" for result in failures):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
