from .prompts_asr import (
    INCONSISTENCY_DEFINITIONS,
    PROMPT_STYLE_PARSERS,
    PROMPT_STYLE_SPECS,
    build_combined_prediction,
    get_prompt_style_spec,
    parse_prompt_response,
    parse_timestamps,
)


GROUNDING_PROMPT = """You are an expert video analyst. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds"""

INCONSISTENCY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the audio.

Given the list of inconsistency types below, identify which type occurs in the caption. If the caption accurately reflects the video and audio, use "No Inconsistency".

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>"""

VERACITY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the audio.

Use the following labels:
- True: the caption faithfully describes the relevant video segment.
- False: the caption contains any inconsistency with the relevant video segment or audio.

Respond strictly in this format:

Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

COMBINED_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

2. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the audio.

3. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the audio.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds
Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

MULTI_INSTANCE_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

2. Multi-instance inconsistency identification:
Determine whether the caption contains any inconsistencies with what is seen in the relevant video segment or what is heard/stated in the audio.

Given the list of inconsistency types below, identify up to three inconsistency types that occur in the caption, ranked from highest confidence to lowest confidence. If the caption accurately reflects the video and audio, use only "No Inconsistency".

3. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the audio.

If the inconsistency types list contains only "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds
Inconsistency Types: [<highest confidence type>, <second highest confidence type>, <third highest confidence type>]
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

GT_WINDOW_INCONSISTENCY_VERACITY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps}, its audio track, a caption to evaluate, and the relevant temporal segment in the video.

Caption to evaluate: "{caption}"

Relevant temporal segment: The event happens in {start_time} - {end_time} seconds

Complete the following tasks for only the relevant temporal segment:

1. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the segment or what is heard/stated in the audio during the segment.

2. Veracity:
Determine whether the caption faithfully describes the segment.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

INCONSISTENCY_VERACITY_ONLY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Complete the following tasks. Do not perform temporal grounding.

1. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the video or what is heard/stated in the audio.

2. Veracity:
Determine whether the caption faithfully describes the video and audio evidence.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

TEMPORAL_VERACITY_ONLY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

2. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the audio.

Use the following labels:
- True: the caption faithfully describes the relevant video segment.
- False: the caption contains any inconsistency with the relevant video segment or audio.

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

MULTI_TURN_GROUNDING_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

First, perform temporal grounding.
Identify the start and end times of the video segment that is most relevant to the caption.

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds"""

MULTI_TURN_INCONSISTENCY_PROMPT = """Now perform inconsistency identification using the same video, audio track, caption, and the temporal grounding you just provided.

Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the audio.

Given the list of inconsistency types below, identify which type occurs in the caption. If the caption accurately reflects the video and audio, use "No Inconsistency".

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>"""

MULTI_TURN_VERACITY_PROMPT = """Finally, determine veracity using the same video, audio track, caption, and your previous temporal-grounding and inconsistency-identification answers.

Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the audio.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Respond strictly in this format:

Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

MULTI_TURN_NO_GROUNDING_INCONSISTENCY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and its audio track.

Caption to evaluate: "{caption}"

First, perform inconsistency identification. Do not perform temporal grounding.

Determine whether the caption contains any inconsistency with what is seen in the video or what is heard/stated in the audio.

Given the list of inconsistency types below, identify which type occurs in the caption. If the caption accurately reflects the video and audio, use "No Inconsistency".

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>"""

MULTI_TURN_NO_GROUNDING_VERACITY_PROMPT = """Finally, determine veracity using the same video, audio track, caption, and your previous inconsistency-identification answer.

Determine whether the caption faithfully describes the video and audio evidence.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Respond strictly in this format:

Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""


def grounding_prompt(caption, fps="1 FPS", query=None):
    return GROUNDING_PROMPT.format(fps=fps, caption=caption if query is None else query)


def inconsistency_prompt(caption, fps="1 FPS"):
    return INCONSISTENCY_PROMPT.format(
        fps=fps,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def veracity_prompt(caption, fps="1 FPS"):
    return VERACITY_PROMPT.format(fps=fps, caption=caption)


def separate_prompt_bundle(caption, fps="1 FPS", query=None):
    return {
        "grounding": grounding_prompt(caption, fps=fps, query=query),
        "inconsistency": inconsistency_prompt(caption, fps=fps),
        "veracity": veracity_prompt(caption, fps=fps),
    }


def combined_audio_prompt(caption, fps="1 FPS"):
    return COMBINED_PROMPT.format(
        fps=fps,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def multi_instance_prompt(caption, fps="1 FPS"):
    return MULTI_INSTANCE_PROMPT.format(
        fps=fps,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def ground_truth_window_inconsistency_veracity_prompt(caption, start_time, end_time, fps="1 FPS"):
    return GT_WINDOW_INCONSISTENCY_VERACITY_PROMPT.format(
        fps=fps,
        caption=caption,
        start_time=start_time,
        end_time=end_time,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def inconsistency_veracity_only_prompt(caption, fps="1 FPS"):
    return INCONSISTENCY_VERACITY_ONLY_PROMPT.format(
        fps=fps,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def temporal_veracity_only_prompt(caption, fps="1 FPS"):
    return TEMPORAL_VERACITY_ONLY_PROMPT.format(fps=fps, caption=caption)


def multi_turn_prompt_bundle(caption, fps="1 FPS"):
    return [
        (
            "grounding",
            MULTI_TURN_GROUNDING_PROMPT.format(
                fps=fps,
                caption=caption,
            ),
        ),
        (
            "inconsistency",
            MULTI_TURN_INCONSISTENCY_PROMPT.format(
                inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
            ),
        ),
        ("veracity", MULTI_TURN_VERACITY_PROMPT),
    ]


def multi_turn_no_grounding_prompt_bundle(caption, fps="1 FPS"):
    return [
        (
            "inconsistency",
            MULTI_TURN_NO_GROUNDING_INCONSISTENCY_PROMPT.format(
                fps=fps,
                caption=caption,
                inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
            ),
        ),
        ("veracity", MULTI_TURN_NO_GROUNDING_VERACITY_PROMPT),
    ]


PROMPT_STYLES = {
    "separate": separate_prompt_bundle,
    "temporal_inconsistency_veracity": combined_audio_prompt,
    "multi_turn": multi_turn_prompt_bundle,
    "multi_turn_no_grounding": multi_turn_no_grounding_prompt_bundle,
    "multi_instance": multi_instance_prompt,
    "ground_truth_window_inconsistency_veracity": ground_truth_window_inconsistency_veracity_prompt,
    "inconsistency_veracity_only": inconsistency_veracity_only_prompt,
    "temporal_veracity_only": temporal_veracity_only_prompt,
}


def build_prompt(style, **kwargs):
    try:
        return PROMPT_STYLES[style](**kwargs)
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"Unknown prompt style '{style}'. Valid styles: {valid}") from exc


video_audio_grounding_prompt = grounding_prompt
video_audio_inconsistency_prompt = inconsistency_prompt
video_audio_manipulation_prompt = inconsistency_prompt
video_audio_veracity_prompt = veracity_prompt
