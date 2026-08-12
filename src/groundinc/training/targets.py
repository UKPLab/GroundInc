"""Supervision targets for GroundInc SFT."""
from __future__ import annotations

from typing import Optional

from groundinc.training.prompts import validate_task_mode


def map_modality(raw: Optional[str]) -> str:
    if not raw or raw == "none":
        return "none"
    if raw == "combined":
        return "both"
    return raw


def build_ground_truth_answer(
    task: dict,
    *,
    visual_evidence: str,
    audio_evidence: str,
    task_mode: str,
) -> str:
    validate_task_mode(task_mode)
    is_true = task["type"] in {"original_true", "rephrased_true"}
    veracity = "TRUE" if is_true else "FALSE"

    if task_mode == "veracity_only":
        return f"veracity: {veracity}"

    manipulated_modality = "none" if is_true else map_modality(task.get("manipulated_modality"))
    manipulation_type = "none" if is_true else task.get("manipulation_method_gt", "none")
    refutation = "N/A" if is_true else (task.get("refutation") or "N/A")

    if task_mode in {"inconsistency", "inconsistency_no_grounding"}:
        span = task.get("ground_truth_span") or []
        start = round(span[0], 1) if len(span) >= 2 else 0.0
        end = round(span[1], 1) if len(span) >= 2 else 0.0
        inconsistency_type = "No Inconsistency" if is_true else manipulation_type
        veracity_title = "True" if is_true else "False"
        if is_true:
            inconsistency_reasoning = (
                "The caption accurately reflects the relevant video segment and ASR evidence."
            )
            veracity_reasoning = (
                "Because no inconsistency is present, the caption is labeled True."
            )
        else:
            inconsistency_reasoning = refutation
            veracity_reasoning = (
                "Because the caption contains this inconsistency, it is labeled False."
            )
        output_lines = []
        if task_mode == "inconsistency":
            output_lines.append(
                f"Temporal Grounding: The event happens in {start} - {end} seconds"
            )
        output_lines.extend(
            [
                f"Inconsistency Type: {inconsistency_type}",
                f"Inconsistency Reasoning: {inconsistency_reasoning}",
                f"Veracity: {veracity_title}",
                f"Veracity Reasoning: {veracity_reasoning}",
            ]
        )
        return "\n".join(output_lines)

    if task_mode == "no_evidence":
        return (
            f"evidence_visual: {visual_evidence}\n"
            f"evidence_audio: {audio_evidence}\n"
            f"veracity: {veracity}\n"
            f"manipulated_modality: {manipulated_modality}\n"
            f"manipulation_type: {manipulation_type}\n"
            f"refutation: {refutation}"
        )

    span = task.get("ground_truth_span") or []
    start = round(span[0], 1) if len(span) >= 2 else 0.0
    end = round(span[1], 1) if len(span) >= 2 else 0.0
    return (
        f"evidence_start: {start}\n"
        f"evidence_end: {end}\n"
        f"evidence_visual: {visual_evidence}\n"
        f"evidence_audio: {audio_evidence}\n"
        f"veracity: {veracity}\n"
        f"manipulated_modality: {manipulated_modality}\n"
        f"manipulation_type: {manipulation_type}\n"
        f"refutation: {refutation}"
    )
