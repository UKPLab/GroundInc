import re


INCONSISTENCY_DEFINITIONS = """* Negation/Polarity Flip: Directly invert the truth of the statement, such as replacing a visible or audible attribute with its exact opposite, or inserting words like "not" or "never".
* Comparative/Scalar Flip: Reverse the direction of a comparison, such as changing "more than" to "less than", "higher" to "lower", or "faster" to "slower".
* Numeric Distortion: Change a specific quantity, measurement, or number to a different, incorrect number (e.g., changing a stated or visible count from 3 to 5).
* Entity Substitution: Replace the name or description of a person, object, or place with a completely different one that contradicts the provided evidence.
* Predicate/Role Substitution: Reverse the roles of the subjects and objects, such as swapping who is performing an action with who is receiving it.
* Temporal/Spatial Reframing: Change the explicitly stated time, duration, or location of the event to directly contradict the provided timeline.
* Context Omission / Quote Mining: Splice the caption by explicitly deleting a crucial qualifying clause, condition, or context marker, so that the remaining text creates a false or misleading impression.
* Causal Overreach: Claim that one event explicitly caused an outcome, when the timeline only shows them happening sequentially or incidentally.
* Affective/Framing Distortions: Insert loaded language or evaluative adjectives to explicitly misrepresent the tone, emotion, or sentiment of the visual/audio evidence.
* Exaggeration / Minimisation: Add intensifiers or remove down-toners to inflate or deflate the severity or scale of an event described in the evidence.
* No Inconsistency: The caption does not contain any of the above inconsistencies and accurately reflects the video and ASR evidence.
"""

GROUNDING_PROMPT = """You are an expert video analyst. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds"""

INCONSISTENCY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the ASR.

Given the list of inconsistency types below, identify which type occurs in the caption. If the caption accurately reflects the video and ASR, use "No Inconsistency".

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>"""

VERACITY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the ASR.

Use the following labels:
- True: the caption faithfully describes the relevant video segment.
- False: the caption contains any inconsistency with the relevant video segment or ASR.

Respond strictly in this format:

Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

COMBINED_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

2. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the ASR.

3. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the ASR.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds
Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

MULTI_INSTANCE_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

2. Multi-instance inconsistency identification:
Determine whether the caption contains any inconsistencies with what is seen in the relevant video segment or what is heard/stated in the ASR.

Given the list of inconsistency types below, identify up to three inconsistency types that occur in the caption, ranked from highest confidence to lowest confidence. If the caption accurately reflects the video and ASR, use only "No Inconsistency".

3. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the ASR.

If the inconsistency types list contains only "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds
Inconsistency Types: [<highest confidence type>, <second highest confidence type>, <third highest confidence type>]
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

GT_WINDOW_INCONSISTENCY_VERACITY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps}, a transcribed ASR text, a caption to evaluate, and the relevant temporal segment in the video.

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Relevant temporal segment: The event happens in {start_time} - {end_time} seconds

Complete the following tasks for only the relevant temporal segment:

1. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the segment or what is heard/stated in the ASR during the segment.

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

INCONSISTENCY_VERACITY_ONLY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Complete the following tasks. Do not perform temporal grounding.

1. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the video or what is heard/stated in the ASR.

2. Veracity:
Determine whether the caption faithfully describes the video and ASR evidence.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

TEMPORAL_VERACITY_ONLY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Temporal grounding:
Identify the start and end times of the video segment that is most relevant to the caption.

2. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the ASR.

Use the following labels:
- True: the caption faithfully describes the relevant video segment.
- False: the caption contains any inconsistency with the relevant video segment or ASR.

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

MULTI_TURN_GROUNDING_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

First, perform temporal grounding.
Identify the start and end times of the video segment that is most relevant to the caption.

Respond strictly in this format:

Temporal Grounding: The event happens in <start time> - <end time> seconds"""

MULTI_TURN_INCONSISTENCY_PROMPT = """Now perform inconsistency identification using the same video, ASR, caption, and the temporal grounding you just provided.

Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the ASR.

Given the list of inconsistency types below, identify which type occurs in the caption. If the caption accurately reflects the video and ASR, use "No Inconsistency".

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>"""

MULTI_TURN_VERACITY_PROMPT = """Finally, determine veracity using the same video, ASR, caption, and your previous temporal-grounding and inconsistency-identification answers.

Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the ASR.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Respond strictly in this format:

Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""

MULTI_TURN_NO_GROUNDING_INCONSISTENCY_PROMPT = """You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

First, perform inconsistency identification. Do not perform temporal grounding.

Determine whether the caption contains any inconsistency with what is seen in the video or what is heard/stated in the ASR.

Given the list of inconsistency types below, identify which type occurs in the caption. If the caption accurately reflects the video and ASR, use "No Inconsistency".

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>"""

MULTI_TURN_NO_GROUNDING_VERACITY_PROMPT = """Finally, determine veracity using the same video, ASR, caption, and your previous inconsistency-identification answer.

Determine whether the caption faithfully describes the video and ASR evidence.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Respond strictly in this format:

Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>"""


def grounding_prompt(asr_content, caption, fps="1 FPS", query=None):
    return GROUNDING_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption if query is None else query,
    )


def inconsistency_prompt(asr_content, caption, fps="1 FPS"):
    return INCONSISTENCY_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def veracity_prompt(asr_content, caption, fps="1 FPS"):
    return VERACITY_PROMPT.format(fps=fps, asr_content=asr_content, caption=caption)


def separate_prompt_bundle(asr_content, caption, fps="1 FPS", query=None):
    return {
        "grounding": grounding_prompt(asr_content, caption, fps=fps, query=query),
        "inconsistency": inconsistency_prompt(asr_content, caption, fps=fps),
        "veracity": veracity_prompt(asr_content, caption, fps=fps),
    }


def combined_prompt(asr_content, caption, fps="1 FPS"):
    return COMBINED_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def multi_instance_prompt(asr_content, caption, fps="1 FPS"):
    return MULTI_INSTANCE_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def ground_truth_window_inconsistency_veracity_prompt(
    asr_content,
    caption,
    start_time,
    end_time,
    fps="1 FPS",
):
    return GT_WINDOW_INCONSISTENCY_VERACITY_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
        start_time=start_time,
        end_time=end_time,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def inconsistency_veracity_only_prompt(asr_content, caption, fps="1 FPS"):
    return INCONSISTENCY_VERACITY_ONLY_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
        inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
    )


def temporal_veracity_only_prompt(asr_content, caption, fps="1 FPS"):
    return TEMPORAL_VERACITY_ONLY_PROMPT.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
    )


def multi_turn_prompt_bundle(asr_content, caption, fps="1 FPS"):
    return [
        (
            "grounding",
            MULTI_TURN_GROUNDING_PROMPT.format(
                fps=fps,
                asr_content=asr_content,
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


def multi_turn_no_grounding_prompt_bundle(asr_content, caption, fps="1 FPS"):
    return [
        (
            "inconsistency",
            MULTI_TURN_NO_GROUNDING_INCONSISTENCY_PROMPT.format(
                fps=fps,
                asr_content=asr_content,
                caption=caption,
                inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
            ),
        ),
        ("veracity", MULTI_TURN_NO_GROUNDING_VERACITY_PROMPT),
    ]


PROMPT_STYLES = {
    "grounding": grounding_prompt,
    "separate": separate_prompt_bundle,
    "temporal_inconsistency_veracity": combined_prompt,
    "multi_turn": multi_turn_prompt_bundle,
    "multi_turn_no_grounding": multi_turn_no_grounding_prompt_bundle,
    "multi_instance": multi_instance_prompt,
    "ground_truth_window_inconsistency_veracity": ground_truth_window_inconsistency_veracity_prompt,
    "inconsistency_veracity_only": inconsistency_veracity_only_prompt,
    "temporal_veracity_only": temporal_veracity_only_prompt,
}

PROMPT_STYLE_SPECS = {
    "grounding": {
        "family": "prompt",
        "outputs": ("temporal_grounding",),
    },
    "separate": {
        "family": "separate",
        "outputs": ("temporal_grounding", "inconsistency", "veracity"),
    },
    "temporal_inconsistency_veracity": {
        "family": "combined",
        "outputs": ("temporal_grounding", "inconsistency", "veracity"),
    },
    "multi_turn": {
        "family": "multi_turn",
        "outputs": ("temporal_grounding", "inconsistency", "veracity"),
    },
    "multi_turn_no_grounding": {
        "family": "multi_turn",
        "outputs": ("inconsistency", "veracity"),
    },
    "multi_instance": {
        "family": "combined",
        "outputs": ("temporal_grounding", "inconsistency", "veracity"),
    },
    "ground_truth_window_inconsistency_veracity": {
        "family": "oracle_like",
        "outputs": ("inconsistency", "veracity"),
    },
    "inconsistency_veracity_only": {
        "family": "combined",
        "outputs": ("inconsistency", "veracity"),
    },
    "temporal_veracity_only": {
        "family": "combined",
        "outputs": ("temporal_grounding", "veracity"),
    },
}


def build_prompt(style, **kwargs):
    try:
        return PROMPT_STYLES[style](**kwargs)
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"Unknown prompt style '{style}'. Valid styles: {valid}") from exc


def get_prompt_style_spec(style):
    try:
        return PROMPT_STYLE_SPECS[style]
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_STYLE_SPECS))
        raise ValueError(f"Unknown prompt style '{style}'. Valid styles: {valid}") from exc


def parse_timestamps(text):
    if not text:
        return []
    values = []
    for token in re.findall(r"\d{1,2}:\d{2}(?::\d{2})?|[-+]?\d*\.\d+|\d+", text):
        if ":" in token:
            seconds = 0.0
            for part in token.split(":"):
                seconds = seconds * 60 + int(part)
            values.append(seconds)
        else:
            values.append(float(token))
    return values


def parse_label_reasoning_response(text, label_key="Veracity"):
    patterns = {
        "label": rf"{re.escape(label_key)}:\s*(.+?)(?=\n(?:Reasoning:|Veracity Reasoning:|$))",
        "reasoning": r"(?:Reasoning|Veracity Reasoning):\s*(.+?)(?=$)",
    }
    return {
        key: match.group(1).strip() if (match := re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)) else None
        for key, pattern in patterns.items()
    }


def parse_inconsistency_response(text):
    patterns = {
        "label": r"Inconsistency Type:\s*(.+?)(?=\n(?:Inconsistency Reasoning:|Veracity:|$))",
        "reasoning": r"Inconsistency Reasoning:\s*(.+?)(?=\n(?:Veracity:|$)|$)",
    }
    return {
        key: match.group(1).strip() if (match := re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)) else None
        for key, pattern in patterns.items()
    }


def parse_inconsistency_veracity_response(text):
    patterns = {
        "inconsistency_label": r"Inconsistency Type:\s*(.+?)(?=\n(?:Inconsistency Reasoning:|Veracity:|$))",
        "inconsistency_reasoning": r"Inconsistency Reasoning:\s*(.+?)(?=\n(?:Veracity:|$)|$)",
        "veracity_label": r"Veracity:\s*(.+?)(?=\n(?:Veracity Reasoning:|$))",
        "veracity_reasoning": r"Veracity Reasoning:\s*(.+?)(?=$)",
    }
    return {
        key: match.group(1).strip() if (match := re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)) else None
        for key, pattern in patterns.items()
    }


def parse_inconsistency_types(value):
    if not value:
        return []
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    labels = []
    for item in re.split(r"\s*,\s*", value):
        item = item.strip().strip("\"'")
        if item:
            labels.append(item)
    return labels


def parse_multi_instance_inconsistency_veracity_response(text):
    patterns = {
        "inconsistency_labels": r"Inconsistency Types:\s*(.+?)(?=\n(?:Inconsistency Reasoning:|Veracity:|$))",
        "inconsistency_reasoning": r"Inconsistency Reasoning:\s*(.+?)(?=\n(?:Veracity:|$)|$)",
        "veracity_label": r"Veracity:\s*(.+?)(?=\n(?:Veracity Reasoning:|$))",
        "veracity_reasoning": r"Veracity Reasoning:\s*(.+?)(?=$)",
    }
    parsed = {
        key: match.group(1).strip() if (match := re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)) else None
        for key, pattern in patterns.items()
    }
    parsed["inconsistency_labels"] = parse_inconsistency_types(parsed["inconsistency_labels"])
    parsed["inconsistency_label"] = parsed["inconsistency_labels"]
    return parsed


def parse_temporal_inconsistency_veracity_response(text):
    parsed = parse_inconsistency_veracity_response(text)
    match = re.search(
        r"Temporal Grounding:\s*(.+?)(?=\n(?:Inconsistency Type:|Veracity:|$))",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    parsed["temporal_grounding"] = match.group(1).strip() if match else None
    return parsed


def parse_temporal_multi_instance_response(text):
    parsed = parse_multi_instance_inconsistency_veracity_response(text)
    match = re.search(
        r"Temporal Grounding:\s*(.+?)(?=\n(?:Inconsistency Types:|Veracity:|$))",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    parsed["temporal_grounding"] = match.group(1).strip() if match else None
    return parsed


def parse_temporal_veracity_response(text):
    parsed = parse_label_reasoning_response(text, "Veracity")
    match = re.search(
        r"Temporal Grounding:\s*(.+?)(?=\n(?:Veracity:|$))",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        "temporal_grounding": match.group(1).strip() if match else None,
        "veracity_label": parsed["label"],
        "veracity_reasoning": parsed["reasoning"],
    }


def parse_grounding_response(text):
    return {"temporal_grounding": parse_timestamps(text)}


PROMPT_STYLE_PARSERS = {
    "grounding": parse_grounding_response,
    "separate": None,
    "multi_turn": None,
    "multi_turn_no_grounding": None,
    "temporal_inconsistency_veracity": parse_temporal_inconsistency_veracity_response,
    "multi_instance": parse_temporal_multi_instance_response,
    "ground_truth_window_inconsistency_veracity": parse_inconsistency_veracity_response,
    "inconsistency_veracity_only": parse_inconsistency_veracity_response,
    "temporal_veracity_only": parse_temporal_veracity_response,
}


def parse_prompt_response(style, text):
    if style in ("separate", "multi_turn", "multi_turn_no_grounding"):
        if not isinstance(text, dict):
            raise ValueError(f"Prompt style '{style}' expects a dict with keys: grounding, inconsistency, veracity")
        return {
            "grounding": parse_timestamps(text.get("grounding", "")),
            "inconsistency": parse_inconsistency_response(text.get("inconsistency", "")),
            "veracity": parse_label_reasoning_response(text.get("veracity", ""), "Veracity"),
        }

    try:
        return PROMPT_STYLE_PARSERS[style](text)
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_STYLE_PARSERS))
        raise ValueError(f"Unknown prompt style '{style}'. Valid styles: {valid}") from exc


def build_combined_prediction(raw):
    parsed = parse_temporal_inconsistency_veracity_response(raw)
    tg_raw = parsed.get("temporal_grounding") or raw
    return {
        "prediction_raw": raw,
        "temporal_grounding": {
            "prediction_raw": tg_raw,
            "prediction_parsed": parse_timestamps(tg_raw),
        },
        "inconsistency": {
            "prediction_raw": raw,
            "label": parsed.get("inconsistency_label"),
            "reasoning": parsed.get("inconsistency_reasoning"),
        },
        "manipulation": {
            "prediction_raw": raw,
            "label": parsed.get("inconsistency_label"),
            "reasoning": parsed.get("inconsistency_reasoning"),
        },
        "veracity": {
            "prediction_raw": raw,
            "label": parsed.get("veracity_label"),
            "reasoning": parsed.get("veracity_reasoning"),
        },
    }


video_asr_grounding_prompt = grounding_prompt
video_asr_inconsistency_prompt = inconsistency_prompt
video_asr_manipulation_prompt = inconsistency_prompt
video_asr_veracity_prompt = veracity_prompt
ground_truth_window_veracity_prompt = ground_truth_window_inconsistency_veracity_prompt
trimmed_segment_inconsistency_veracity_prompt = inconsistency_veracity_only_prompt
ground_truth_window_veracity_manipulation_prompt = ground_truth_window_inconsistency_veracity_prompt

QWEN3_VL_PROMPT_STYLES = PROMPT_STYLES
QWEN3_VL_PROMPT_STYLE_SPECS = PROMPT_STYLE_SPECS
QWEN3_VL_PROMPT_STYLE_PARSERS = PROMPT_STYLE_PARSERS
build_qwen3_vl_prompt = build_prompt
get_qwen3_vl_prompt_style_spec = get_prompt_style_spec
parse_qwen3_vl_prompt_response = parse_prompt_response
