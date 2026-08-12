from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from groundinc.training.constants import TASK_MODES


VERACITY_PROMPT = '''You are an expert audiovisual fact-checker.

You are given a video (sampled at {fps} FPS), its audio transcript, and a caption. Determine whether the caption is fully supported by the audiovisual evidence.

Rules:
- Use only evidence from the video and audio transcript.
- Do not rely on outside knowledge.
- Reply with exactly one line.
- Use TRUE only if the caption is fully supported. Otherwise use FALSE.

Audio transcript: "{asr_content}"
Caption to evaluate: "{caption}"

Return the answer in exactly this format:

veracity: <TRUE | FALSE>'''


NO_EVIDENCE_PROMPT = '''You are an expert audiovisual fact-checker.

You are given a video (sampled at {fps} FPS), its audio transcript, and a caption. Verify whether the caption is supported by the video and audio evidence.

Rules:
- Use only evidence from the video and audio transcript.
- Do not rely on outside knowledge.
- If any part of the caption is contradicted, the caption is FALSE.
- If the caption is TRUE, there is no manipulation.
- Be concise and use the exact output keys below.

Manipulation types:
* Negation/Polarity Flip: Directly invert the truth of the statement, such as replacing a visible or audible attribute with its exact opposite, or inserting words like "not" or "never".
* Comparative/Scalar Flip: Reverse the direction of a comparison, such as changing "more than" to "less than", "higher" to "lower", or "faster" to "slower".
* Numeric Distortion: Change a specific quantity, measurement, or number to a different, incorrect number (e.g., changing a stated or visible count from 3 to 5).
* Entity Substitution: Replace the name or description of a person, object, or place with a completely different one that contradicts the provided evidence.
* Predicate/Role Substitution: Reverse the roles of the subjects and objects, such as swapping who is performing an action with who is receiving it.
* Temporal/Spatial Reframing: Change the explicitly stated time, duration, or location of the event to directly contradict the provided timeline. Constraint: You must only change times or locations that are explicitly established in the provided captions (e.g., changing "morning" to "night"). Do NOT invent external historical context (e.g., "before his appointment"). If the text lacks clear temporal or spatial anchors, do not select this method.
* Context Omission / Quote Mining: Splice the caption by explicitly deleting a crucial qualifying clause, condition, or context marker from within the text, stitching the remaining beginning and end together. Do NOT just extract a middle sentence and drop the rest. The deletion must fundamentally distort the original truth or intent (e.g., deleting an "if" condition) so that the remaining text creates a false or misleading impression.
* Causal Overreach: Claim that one event explicitly caused an outcome, when the timeline only shows them happening sequentially or incidentally.
* Affective/Framing Distortions: Insert loaded language or evaluative adjectives to explicitly misrepresent the tone, emotion, or sentiment of the visual/audio evidence.
* Exaggeration / Minimisation: Add intensifiers or remove down-toners to inflate or deflate the severity or scale of an event described in the evidence.
* No Inconsistency: The caption does not contain any of the above inconsistencies and accurately reflects the video and ASR evidence.

Audio transcript: "{asr_content}"
Caption to evaluate: "{caption}"

Return the answer in exactly this format:


veracity: <TRUE | FALSE>
manipulated_modality: <visual | audio | both | none>
manipulation_type: <type | none>
refutation: <short explanation if FALSE, else N/A>'''


FULL_PROMPT = '''You are an expert audiovisual fact-checker.

You are given a video (sampled at {fps} FPS), its audio transcript, and a caption. Your task is to verify whether the caption is supported by the video and audio evidence.

Follow this exact reasoning process:
1. Identify the relevant time span in the video where the event described in the caption occurs.
2. Extract the key visual evidence from that span.
3. Extract the key audio evidence from that span.
4. Compare the caption to the visual and audio evidence.
5. If any part of the caption is contradicted, the caption is FALSE.
6. Identify which modality (visual, audio, or both) is manipulated if FALSE.
7. Identify the manipulation type.
8. Provide a short grounded refutation if the caption is FALSE.

Rules:
- Use only evidence from the video and audio transcript.
- Do not rely on outside knowledge.
- Be concise and factual.
- If the caption is TRUE, there is no manipulation.

Manipulation types (use the exact name when identifying step 7):
* Negation/Polarity Flip: Directly invert the truth of the statement, such as replacing a visible or audible attribute with its exact opposite, or inserting words like "not" or "never".
* Comparative/Scalar Flip: Reverse the direction of a comparison, such as changing "more than" to "less than", "higher" to "lower", or "faster" to "slower".
* Numeric Distortion: Change a specific quantity, measurement, or number to a different, incorrect number (e.g., changing a stated or visible count from 3 to 5).
* Entity Substitution: Replace the name or description of a person, object, or place with a completely different one that contradicts the provided evidence.
* Predicate/Role Substitution: Reverse the roles of the subjects and objects, such as swapping who is performing an action with who is receiving it.
* Temporal/Spatial Reframing: Change the explicitly stated time, duration, or location of the event to directly contradict the provided timeline.
* Context Omission / Quote Mining: Splice the caption by explicitly deleting a crucial qualifying clause, condition, or context marker, so that the remaining text creates a false or misleading impression.
* Causal Overreach: Claim that one event explicitly caused an outcome, when the timeline only shows them happening sequentially or incidentally.
* Affective/Framing Distortions: Insert loaded language or evaluative adjectives to explicitly misrepresent the tone, emotion, or sentiment of the visual/audio evidence.
* Exaggeration / Minimisation: Add intensifiers or remove down-toners to inflate or deflate the severity or scale of an event described in the evidence.

Audio transcript: "{asr_content}"
Caption to evaluate: "{caption}"

Return the answer in exactly this format:

evidence_start: <start_time_in_seconds>
evidence_end: <end_time_in_seconds>
evidence_visual: <concise visual evidence>
evidence_audio: <concise audio evidence>
veracity: <TRUE | FALSE>
manipulated_modality: <visual | audio | both | none>
manipulation_type: <type | none>
refutation: <short explanation if FALSE, else N/A>'''


INCONSISTENCY_DEFINITIONS = '''* Negation/Polarity Flip: Directly invert the truth of the statement, such as replacing a visible or audible attribute with its exact opposite, or inserting words like "not" or "never".
* Comparative/Scalar Flip: Reverse the direction of a comparison, such as changing "more than" to "less than", "higher" to "lower", or "faster" to "slower".
* Numeric Distortion: Change a specific quantity, measurement, or number to a different, incorrect number (e.g., changing a stated or visible count from 3 to 5).
* Entity Substitution: Replace the name or description of a person, object, or place with a completely different one that contradicts the provided evidence.
* Predicate/Role Substitution: Reverse the roles of the subjects and objects, such as swapping who is performing an action with who is receiving it.
* Temporal/Spatial Reframing: Change the explicitly stated time, duration, or location of the event to directly contradict the provided timeline. Constraint: You must only change times or locations that are explicitly established in the provided captions (e.g., changing "morning" to "night"). Do NOT invent external historical context (e.g., "before his appointment"). If the text lacks clear temporal or spatial anchors, do not select this method.
* Context Omission / Quote Mining: Splice the caption by explicitly deleting a crucial qualifying clause, condition, or context marker from within the text, stitching the remaining beginning and end together. Do NOT just extract a middle sentence and drop the rest. The deletion must fundamentally distort the original truth or intent (e.g., deleting an "if" condition) so that the remaining text creates a false or misleading impression.
* Causal Overreach: Claim that one event explicitly caused an outcome, when the timeline only shows them happening sequentially or incidentally.
* Affective/Framing Distortions: Insert loaded language or evaluative adjectives to explicitly misrepresent the tone, emotion, or sentiment of the visual/audio evidence.
* Exaggeration / Minimisation: Add intensifiers or remove down-toners to inflate or deflate the severity or scale of an event described in the evidence.
* No Inconsistency: The caption does not contain any of the above inconsistencies and accurately reflects the video and ASR evidence.'''


INCONSISTENCY_PROMPT = '''You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

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
Veracity Reasoning: <1-2 sentences>'''


INCONSISTENCY_NO_GROUNDING_PROMPT = '''You are an expert video analyst and fact-checker. You are provided with a video sampled at {fps} and a transcribed ASR text:

ASR: "{asr_content}"

Caption to evaluate: "{caption}"

Complete the following tasks in order:

1. Inconsistency identification:
Determine whether the caption contains any inconsistency with what is seen in the relevant video segment or what is heard/stated in the ASR.

2. Veracity:
Determine whether the caption faithfully describes what is seen in the relevant video segment and what is heard/stated in the ASR.

If the inconsistency type is "No Inconsistency", the caption should be labeled True. Otherwise, it should be labeled False.

Inconsistency Types:
{inconsistency_definitions}

Respond strictly in this format:

Inconsistency Type: <type>
Inconsistency Reasoning: <1-2 sentences>
Veracity: <True | False>
Veracity Reasoning: <1-2 sentences>'''


NO_EVIDENCE_WITH_EVIDENCE_PROMPT = '''You are an expert audiovisual fact-checker.

You are given a video (sampled at {fps} FPS), its audio transcript, and a caption. Verify whether the caption is supported by the video and audio evidence.

Rules:
- Use only evidence from the video and audio transcript.
- Do not rely on outside knowledge.
- If any part of the caption is contradicted, the caption is FALSE.
- If the caption is TRUE, there is no manipulation.
- Include concise visual and audio evidence.
- Do not output timestamps.
- Be concise and use the exact output keys below.

Manipulation types:
{inconsistency_definitions}

Audio transcript: "{asr_content}"
Caption to evaluate: "{caption}"

Return the answer in exactly this format:

evidence_visual: <concise visual evidence>
evidence_audio: <concise audio evidence>
veracity: <TRUE | FALSE>
manipulated_modality: <visual | audio | both | none>
manipulation_type: <type | none>
refutation: <short explanation if FALSE, else N/A>'''


PROMPTS = {
    "veracity": ("veracity_only", VERACITY_PROMPT),
    "no_evidence": ("no_evidence", NO_EVIDENCE_WITH_EVIDENCE_PROMPT),
    "full": ("full", FULL_PROMPT),
    "inconsistency": ("inconsistency", INCONSISTENCY_PROMPT),
    "inconsistency_no_grounding": (
        "inconsistency_no_grounding",
        INCONSISTENCY_NO_GROUNDING_PROMPT,
    ),
}


def prompt_names() -> str:
    return ", ".join(sorted(PROMPTS))


def validate_task_mode(task_mode: str) -> str:
    if task_mode not in TASK_MODES:
        raise ValueError(
            f"Unsupported task_mode='{task_mode}'. Expected one of: {sorted(TASK_MODES)}"
        )
    return task_mode


def resolve_prompt(
    *,
    prompt_name: str,
    task_mode: str,
    prompt_template: Optional[str],
    prompt_template_path: Optional[str],
) -> Tuple[str, str]:
    if prompt_template and prompt_template_path:
        raise ValueError("Use either prompt_template or prompt_template_path, not both.")

    selected = PROMPTS.get(prompt_name)
    if selected is None and not (prompt_template or prompt_template_path):
        raise ValueError(
            f"Unknown prompt_name='{prompt_name}'. Available prompts: {prompt_names()}"
        )

    if prompt_template_path:
        template = Path(prompt_template_path).expanduser().read_text()
    elif prompt_template:
        template = prompt_template
    else:
        template = selected[1]

    if task_mode == "auto":
        if selected is None:
            raise ValueError(
                "task_mode='auto' requires a known prompt_name. For a custom prompt, "
                "set task_mode to veracity_only, no_evidence, full, inconsistency, "
                "or inconsistency_no_grounding."
            )
        task_mode = selected[0]

    return template, validate_task_mode(task_mode)


def render_prompt(template: str, *, fps: float, asr_content: str, caption: str) -> str:
    try:
        return template.format(
            fps=fps,
            asr_content=asr_content,
            caption=caption,
            inconsistency_definitions=INCONSISTENCY_DEFINITIONS,
        )
    except KeyError as exc:
        raise ValueError(
            "Unknown prompt placeholder "
            f"'{exc.args[0]}'. Supported placeholders: {{fps}}, {{asr_content}}, "
            "{{caption}}, {{inconsistency_definitions}}."
        ) from exc
