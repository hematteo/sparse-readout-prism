"""Display-string helpers for token / text rendering.

These are formatting utilities for the paper figures: they convert raw
token-ids and decoded strings into compact, plot-safe display strings. They
are not part of the runtime decomposition API.
"""

from __future__ import annotations

from typing import Any


def clean_token(
    tokenizer: Any,
    token_id: int,
    *,
    max_len: int = 24,
    skip_special_tokens: bool = False,
) -> str:
    """Decode a token id and produce a compact display string.

    The vocab-aware variant: returns ``<extra:N>`` for ids outside the
    tokenizer's vocabulary, ``<space>`` for the literal single-space token,
    and ``<empty>`` if the decoded text is empty. Escapes control characters,
    keeps a single leading space, and truncates to ``max_len`` for the
    printable case.
    """
    if int(token_id) < 0 or int(token_id) >= len(tokenizer):
        return f"<extra:{int(token_id)}>"
    text = tokenizer.decode([int(token_id)], skip_special_tokens=skip_special_tokens)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace("�", "")
    if text == " ":
        text = "<space>"
    elif text.startswith(" "):
        text = " " + text.strip()
    else:
        text = text.strip()
    if not text:
        text = "<empty>"
    if len(text) > max_len:
        text = text[: max_len - 1] + "..."
    return text


def shorten(text: str, max_len: int, *, ellipsis: str = "...") -> str:
    """Collapse internal whitespace and truncate ``text`` to ``max_len`` chars.

    Appends ``ellipsis`` when truncated (the slice keeps ``max_len - 1`` chars,
    matching the figure-label convention). Unlike ``clean_token`` (which escapes
    control characters and preserves a leading space for token display),
    ``shorten`` is for free-text annotations (prompts, top-token summaries).
    """
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + ellipsis


def is_display_token(text: str) -> bool:
    """True if ``text`` is a reasonable display label (printable, non-trivial).

    Filters out empty strings, single punctuation, single whitespace, and
    pure non-printable strings — what paper figures should not show.
    """
    s = text.strip()
    if not s:
        return False
    if len(s) == 1 and not s.isalnum():
        return False
    if not any(c.isprintable() and not c.isspace() for c in s):
        return False
    return True
