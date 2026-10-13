"""Prompt construction for the categorical temporal localizer.

The localizer sees the video, the audio transcript, the caption, and a numbered
list of candidate intervals (the deterministic bank). It must answer with exactly
one candidate ID after the fixed prefix ``Evidence interval:``. The IDs shown are
the ``display_labels`` of the single-token ID vocabulary, so the model's emitted
token maps back to a bank index unambiguously.
"""

from __future__ import annotations

from typing import Tuple

from .id_tokens import DEFAULT_PREFIX, IdVocabulary
from .interval_bank import IntervalBank

_LOCALIZER_TEMPLATE = '''You are an expert video temporal-grounding model. You are given frames sampled uniformly across the full video, and its audio transcript.

ASR: "{asr_content}"
Caption: "{caption}"

Below is a numbered list of candidate time intervals (in seconds) within the video. Exactly one of them best contains the evidence for the caption. Consider both the visual content and the transcript.

Candidates:
{candidate_lines}

Respond with exactly one line, choosing a single candidate ID from the list above:
{prefix} <ID>'''


def format_candidate_lines(bank: IntervalBank, vocab: IdVocabulary) -> str:
    lines = []
    for idx, (start, end) in enumerate(bank.intervals):
        label = vocab.display_labels[idx]
        lines.append(f"  {label}: {start:.1f}-{end:.1f}s")
    return "\n".join(lines)


def build_localizer_prompt(
    *,
    caption: str,
    asr_content: str,
    fps: float,
    bank: IntervalBank,
    vocab: IdVocabulary,
) -> str:
    """Return the user-turn text for the localizer."""
    if len(vocab) < len(bank):
        raise ValueError(
            f"ID vocabulary ({len(vocab)}) smaller than bank ({len(bank)}); "
            "build the vocabulary with num_needed >= len(bank)."
        )
    return _LOCALIZER_TEMPLATE.format(
        fps=fps,
        asr_content=asr_content,
        caption=caption,
        candidate_lines=format_candidate_lines(bank, vocab),
        prefix=DEFAULT_PREFIX,
    )


def assistant_prefix() -> str:
    """The fixed assistant-side prefix; the ID token is predicted right after it."""
    return DEFAULT_PREFIX
