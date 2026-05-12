"""Text chunking.

Markdown mode: split on top-level headings (^#, ^##, ...). Sections > 1500
chars are further split via sliding window (1000 char window, 200 overlap).

Plain text / source mode: pure sliding window (1000 char window, 200 overlap).

Empty / whitespace-only chunks are skipped. Each chunk is capped at MAX_CHUNK
characters so chunk_meta.text fits comfortably.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

WINDOW = 1000
OVERLAP = 200
SECTION_SOFT_MAX = 1500
MAX_CHUNK = 4000

_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


def _sliding(text: str, base_offset: int = 0) -> Iterable[Tuple[int, str]]:
    """Yield (offset, chunk) pairs from sliding-window over `text`.

    `base_offset` is added to each yielded offset so callers can translate
    chunk-local offsets back to file-level offsets.
    """
    if not text:
        return
    n = len(text)
    if n <= WINDOW:
        if text.strip():
            yield (base_offset, text)
        return
    step = WINDOW - OVERLAP
    i = 0
    while i < n:
        piece = text[i : i + WINDOW]
        if piece.strip():
            yield (base_offset + i, piece[:MAX_CHUNK])
        if i + WINDOW >= n:
            break
        i += step


def _split_markdown(text: str) -> Iterable[Tuple[int, str]]:
    """Yield (offset, section) pairs split on heading boundaries.

    The heading line itself is included with the section that follows it.
    Sections larger than SECTION_SOFT_MAX are run through `_sliding`.
    """
    if not text:
        return
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        # No headings — fall back to sliding window over whole doc.
        yield from _sliding(text)
        return
    # Pre-heading prelude (if any).
    first = matches[0].start()
    if first > 0:
        prelude = text[:first]
        if prelude.strip():
            if len(prelude) > SECTION_SOFT_MAX:
                yield from _sliding(prelude, base_offset=0)
            else:
                yield (0, prelude[:MAX_CHUNK])
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end]
        if not section.strip():
            continue
        if len(section) > SECTION_SOFT_MAX:
            yield from _sliding(section, base_offset=start)
        else:
            yield (start, section[:MAX_CHUNK])


def chunk_text(text: str, mode: str = "text") -> List[Tuple[int, int, str]]:
    """Return list of (chunk_idx, offset, chunk_text).

    mode='md': markdown heading-aware split (with overflow sliding).
    mode='text': pure sliding window.
    """
    if mode not in ("md", "text"):
        raise ValueError(f"unknown chunk mode: {mode}")
    iterator = _split_markdown(text) if mode == "md" else _sliding(text)
    out: List[Tuple[int, int, str]] = []
    for idx, (offset, piece) in enumerate(iterator):
        out.append((idx, offset, piece))
    return out


def detect_mode(path: str) -> str:
    """Heuristic: .md / .markdown → md; everything else → text."""
    p = path.lower()
    if p.endswith(".md") or p.endswith(".markdown"):
        return "md"
    return "text"
