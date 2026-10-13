import argparse
import asyncio
import os

from groundinc.evaluation.vllm_common import (
    add_common_vllm_args,
    build_media_url,
    build_url_content_item,
    normalize_prompt_style,
    prepare_video_path,
    print_prompt_styles,
    process_records,
    run_prompt_style,
    validate_common_args,
)

from groundinc.data import (
    build_caption_tasks,
    iter_input_records,
    resolve_audio_path,
)
from groundinc.data import prompts_audio


DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def build_audio_prompt(style, caption, ground_truth, prompt_fps):
    fps_text = "1 FPS" if prompt_fps is None else f"{prompt_fps:g} FPS"
    kwargs = {
        "caption": caption,
        "fps": fps_text,
    }
    if style == "separate":
        kwargs["query"] = caption
    if style == "ground_truth_window_inconsistency_veracity":
        if len(ground_truth) < 2:
            raise ValueError(f"Prompt style '{style}' requires a ground_truth_span")
        kwargs["start_time"] = ground_truth[0]
        kwargs["end_time"] = ground_truth[1]
    return prompts_audio.build_prompt(style, **kwargs)


async def process_single_caption_entry(
    client,
    model_id,
    media_items,
    task,
    args,
    request_semaphore,
):
    prompt = build_audio_prompt(
        args.prompt_style,
        task["caption"],
        task["ground_truth_span"],
        args.prompt_fps,
    )
    prediction = await run_prompt_style(
        client,
        model_id,
        prompts_audio,
        prompt,
        media_items,
        args,
        request_semaphore,
    )
    return {
        "category": task["category"],
        "type": task["type"],
        "caption": task["caption"],
        "manipulation_method_gt": task["manipulation_method_gt"],
        "entity_type": task.get("entity_type"),
        "manipulated_modality": task["manipulated_modality"],
        "ground_truth_span": task["ground_truth_span"],
        "prompt_style": args.prompt_style,
        **prediction,
    }


async def process_video_line(client, model_id, data, args, request_semaphore):
    video_path, source_video_path = prepare_video_path(data, args)
    if not video_path or not os.path.exists(video_path):
        return None

    audio_path = resolve_audio_path(data, args.audio_dir, args.audio_ext)
    if not audio_path or not os.path.exists(audio_path):
        return None

    video_url = build_media_url(video_path, args.media_mode, "video")
    audio_url = build_media_url(audio_path, args.media_mode, "audio")
    media_items = [
        build_url_content_item(args.video_content_type, video_url),
        build_url_content_item(args.audio_content_type, audio_url),
    ]
    caption_semaphore = asyncio.Semaphore(max(1, args.max_concurrent_captions))

    async def run_caption_task(task):
        async with caption_semaphore:
            return await process_single_caption_entry(
                client,
                model_id,
                media_items,
                task,
                args,
                request_semaphore,
            )

    caption_tasks = [
        run_caption_task(task)
        for task in build_caption_tasks(data, caption_task_video_mode(data, args.video_mode))
        if args.category_filter == "all" or task["category"] == args.category_filter
    ]
    if not caption_tasks:
        return data

    data["resolved_video_path"] = video_path
    if source_video_path:
        data["source_video_path"] = source_video_path
        data["snippet_path"] = video_path
    data["resolved_audio_path"] = audio_path
    data["results"] = await asyncio.gather(*caption_tasks)
    return data


def caption_task_video_mode(data, video_mode):
    if video_mode not in ("snippet", "masked_snippet"):
        return video_mode
    return "id"


async def main():
    parser = argparse.ArgumentParser(description="Generic vLLM video+audio evaluator.")
    add_common_vllm_args(parser, prompts_audio, DEFAULT_MODEL_ID)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_ext", type=str, default="mp3")
    parser.add_argument("--audio_content_type", type=str, default="audio_url")
    args = parser.parse_args()

    if args.list_prompt_styles:
        print_prompt_styles(prompts_audio)
        return

    args.prompt_style = normalize_prompt_style(args.prompt_style, args.prompt_mode)
    validate_common_args(parser, args)

    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        parser.error("missing dependency: install/activate an environment with the 'openai' package")

    api_key = os.environ.get("OPENAI_API_KEY", "vllm")
    client = AsyncOpenAI(base_url=args.vllm_url, api_key=api_key)
    await process_records(client, args, process_video_line, iter_input_records)


if __name__ == "__main__":
    asyncio.run(main())
