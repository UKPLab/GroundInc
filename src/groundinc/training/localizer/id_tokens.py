"""Single-token interval-ID vocabulary for the categorical localizer.

The localizer emits exactly one interval ID immediately after a fixed output
prefix (``"Evidence interval:"``). We use preverified *single-token* ID symbols so
the next-token logits at one fixed position define a categorical distribution over
candidates, with no new head.

The plan warns that tokenization is model- and context-dependent. Empirically, for
the Qwen3-VL tokenizer, ``" 12"`` after ``":"`` is *not* a single token (the space
splits off and multi-digit numbers tokenize digit-by-digit), but ``" A"``..``" Z"``,
``" a"``..``" z"`` and space+two-letter symbols like ``" Ab"`` *are* single tokens.
So symbols are drawn from a deterministic space+letter pool and each is verified to
append as exactly one token without changing the prefix tokens. The display label
shown in the prompt is the symbol with its leading space stripped (``"A"``,
``"Ab"``, ...).

Because byte-level BPE pre-tokenizes on whitespace, a space-led symbol tokenizes
the same whether it follows the short prefix or a long chat-template context
(verified in tests), so the vocabulary is built once against ``DEFAULT_PREFIX``.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from typing import Iterator, List

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "Evidence interval:"


@dataclass
class IdVocabulary:
    """A verified mapping from bank index to a single-token ID symbol."""

    prefix: str
    symbols: List[str]         # bank_index -> appended text (e.g. " A", " Ab")
    token_ids: List[int]       # bank_index -> single token id
    display_labels: List[str]  # bank_index -> label shown in the prompt (e.g. "A")

    def __len__(self) -> int:
        return len(self.token_ids)

    def index_to_symbol(self, index: int) -> str:
        return self.symbols[index]

    def token_id_for_index(self, index: int) -> int:
        return self.token_ids[index]


def _encode(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def single_token_append(tokenizer, prefix_ids: List[int], prefix: str, symbol: str):
    """Check that appending ``symbol`` to ``prefix`` adds exactly one token.

    Returns ``(ok, token_id)``. ``ok`` is True only when the combined encoding is
    the prefix encoding followed by exactly one new token id (prefix tokens
    unchanged, ID is a single token).
    """
    combined = _encode(tokenizer, prefix + symbol)
    if len(combined) != len(prefix_ids) + 1:
        return False, None
    if combined[: len(prefix_ids)] != prefix_ids:
        return False, None
    return True, combined[-1]


def _symbol_pool() -> Iterator[str]:
    """Deterministic, readable pool of candidate single-token symbols.

    Order: single uppercase, single lowercase, then space+two-letter combos.
    Every yielded symbol carries a leading space.
    """
    letters = string.ascii_uppercase + string.ascii_lowercase
    for c in string.ascii_uppercase:
        yield f" {c}"
    for c in string.ascii_lowercase:
        yield f" {c}"
    for a in letters:
        for b in string.ascii_lowercase:
            yield f" {a}{b}"


def build_id_vocabulary(
    tokenizer,
    num_needed: int,
    *,
    prefix: str = DEFAULT_PREFIX,
) -> IdVocabulary:
    """Build a single-token ID vocabulary of at least ``num_needed`` symbols.

    Args:
        tokenizer: a HF tokenizer (the localizer's own).
        num_needed: number of candidate slots the bank needs.
        prefix: the fixed output prefix that precedes every ID.

    Raises ``RuntimeError`` if the pool cannot supply ``num_needed`` distinct
    single-token symbols (lower ``max_candidates``).
    """
    prefix_ids = _encode(tokenizer, prefix)
    symbols: List[str] = []
    token_ids: List[int] = []
    display_labels: List[str] = []
    used = set()

    for symbol in _symbol_pool():
        if len(token_ids) >= num_needed:
            break
        ok, token_id = single_token_append(tokenizer, prefix_ids, prefix, symbol)
        if not ok or token_id in used:
            continue
        symbols.append(symbol)
        token_ids.append(token_id)
        display_labels.append(symbol.strip())
        used.add(token_id)

    if len(token_ids) < num_needed:
        raise RuntimeError(
            f"Only found {len(token_ids)} single-token ID symbols for prefix "
            f"{prefix!r}; need {num_needed}. Lower max_candidates."
        )
    return IdVocabulary(
        prefix=prefix,
        symbols=symbols,
        token_ids=token_ids,
        display_labels=display_labels,
    )
