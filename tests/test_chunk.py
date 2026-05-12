"""Chunking tests — pure-python, no model required."""

from __future__ import annotations

from eichi.chunk import chunk_text, detect_mode


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", mode="text")
    assert len(chunks) == 1
    assert chunks[0] == (0, 0, "hello world")


def test_empty_text_no_chunks():
    assert chunk_text("", mode="text") == []
    assert chunk_text("    \n\t  ", mode="text") == []


def test_sliding_window_long_text():
    text = "a" * 2500
    chunks = chunk_text(text, mode="text")
    # WINDOW=1000, OVERLAP=200, step=800 → offsets 0, 800, 1600 (last covers 1600..2500).
    assert len(chunks) == 3
    assert chunks[0][1] == 0
    assert chunks[1][1] == 800
    assert chunks[2][1] == 1600
    # Each chunk's idx is its position in the list.
    assert [c[0] for c in chunks] == [0, 1, 2]
    # Overlap: the last 200 chars of chunk 0 == the first 200 chars of chunk 1.
    assert chunks[0][2][-200:] == chunks[1][2][:200]


def test_markdown_split_on_headings():
    text = (
        "# Title\nintro line\n\n"
        "## Section A\nbody A\n\n"
        "## Section B\nbody B has lots of content\n"
    )
    chunks = chunk_text(text, mode="md")
    # Three sections (the leading "# Title" line groups with its prelude/body).
    assert len(chunks) == 3
    # Heading text is preserved at the front of each chunk.
    assert chunks[0][2].startswith("# Title")
    assert chunks[1][2].startswith("## Section A")
    assert chunks[2][2].startswith("## Section B")
    # Offsets monotonically increase.
    offsets = [c[1] for c in chunks]
    assert offsets == sorted(offsets)


def test_markdown_oversized_section_splits():
    big_body = "x" * 2000
    text = f"# Title\n\n## Big\n{big_body}\n"
    chunks = chunk_text(text, mode="md")
    # The "Big" section is > SECTION_SOFT_MAX (1500), so it gets sliding-window'd.
    assert len(chunks) >= 3  # title + at least 2 sub-windows of Big.


def test_detect_mode():
    assert detect_mode("README.md") == "md"
    assert detect_mode("notes.markdown") == "md"
    assert detect_mode("script.py") == "text"
    assert detect_mode("no-ext") == "text"


def test_chunk_max_length_cap():
    # If a heading "section" somehow got huge, the per-chunk text is still capped.
    big = "y" * 50000
    chunks = chunk_text(big, mode="text")
    for _, _, piece in chunks:
        assert len(piece) <= 4000


def test_invalid_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", mode="weirdmode")
