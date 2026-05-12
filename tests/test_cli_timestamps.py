"""Tests for the timestamp-formatting helpers added to eichi query output.

These exercise ``_pick_hit_timestamp``, ``_human_age``,
``_format_hit_timestamp``, and ``_print_hit`` against synthetic
``SearchHit`` objects — no embedding model load, no DB.

The end-to-end "search returns timestamps" path is exercised via the
existing test_store.py KNN path plus a small integration check below
that adds rows with explicit mtimes and walks the SearchHit shape.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from eichi import EMBEDDING_DIM
from eichi.cli import (
    _format_hit_timestamp,
    _human_age,
    _pick_hit_timestamp,
    _print_hit,
)
from eichi.store import SearchHit, add_chunks, open_db, search


def _hit(*, mtime=0.0, indexed_at_unix=0.0, source="memory"):
    return SearchHit(
        rowid=1,
        score=0.42,
        source=source,
        path=f"/x/{source}.md",
        chunk_idx=0,
        offset=0,
        text="example chunk text",
        mtime=mtime,
        indexed_at_unix=indexed_at_unix,
    )


def test_pick_prefers_mtime_over_indexed():
    h = _hit(mtime=1_700_000_000.0, indexed_at_unix=1_777_000_000.0)
    ts, kind = _pick_hit_timestamp(h)
    assert ts == 1_700_000_000.0
    assert kind == "mtime"


def test_pick_falls_back_to_indexed_when_mtime_zero():
    """navidrome-artists style: upstream timestamp absent, ingest time present."""
    h = _hit(mtime=0.0, indexed_at_unix=1_777_000_000.0)
    ts, kind = _pick_hit_timestamp(h)
    assert ts == 1_777_000_000.0
    assert kind == "indexed"


def test_pick_returns_none_when_neither_populated():
    h = _hit(mtime=0.0, indexed_at_unix=0.0)
    ts, kind = _pick_hit_timestamp(h)
    assert ts is None
    assert kind == ""


def test_human_age_buckets():
    assert _human_age(10) == "now"
    assert _human_age(120) == "2m ago"
    assert _human_age(3 * 60 * 60) == "3h ago"
    assert _human_age(2 * 24 * 60 * 60) == "2d ago"
    assert _human_age(10 * 24 * 60 * 60) == "1w ago"
    assert _human_age(60 * 60 * 24 * 400) == "1y ago"


def test_human_age_negative_is_in_future():
    assert _human_age(-3 * 60 * 60) == "in 3h"
    assert _human_age(-30) == "in <1m"


def test_format_hit_timestamp_includes_ET_and_age():
    """Spot-check the ET-formatted absolute and the relative suffix."""
    # 1735689600 = 2025-01-01 00:00:00 UTC = 2024-12-31 19:00 ET
    h = _hit(mtime=1_735_689_600.0)
    out = _format_hit_timestamp(h)
    assert "ET" in out
    # Date portion: month-day in 2024 or 2025 depending on TZ. Just
    # require a YYYY-MM-DD HH:MM prefix inside the brackets.
    assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} ET\]", out), out
    # Relative age must be present (the test runs years after this date).
    assert "(" in out and ")" in out


def test_format_hit_timestamp_empty_when_no_ts():
    h = _hit()
    assert _format_hit_timestamp(h) == ""


def test_print_hit_text_includes_timestamp_block():
    h = _hit(mtime=time.time() - 3600)  # 1h ago
    line = _print_hit(h, json_out=False)
    # Format: "<path>:<offset>\t<score>\t[<source>]\t[<ET-stamp>] (<age>)\t<snippet>"
    parts = line.split("\t")
    assert len(parts) == 5, parts
    assert parts[0] == "/x/memory.md:0"
    assert parts[2] == "[memory]"
    assert "ET" in parts[3]
    assert "ago" in parts[3] or "now" in parts[3]
    assert parts[4] == "example chunk text"


def test_print_hit_text_omits_timestamp_block_when_no_ts():
    h = _hit()  # both 0 → no timestamp segment
    line = _print_hit(h, json_out=False)
    parts = line.split("\t")
    # Legacy 4-column layout: <path>\t<score>\t[<source>]\t<snippet>
    assert len(parts) == 4, parts


def test_print_hit_json_carries_timestamp_fields():
    now = time.time()
    h = _hit(mtime=now - 7200, indexed_at_unix=now - 1800)
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["mtime"] == pytest.approx(now - 7200, abs=1.0)
    assert rec["indexed_at_unix"] == pytest.approx(now - 1800, abs=1.0)
    assert rec["ts_kind"] == "mtime"
    assert rec["ts"] == pytest.approx(now - 7200, abs=1.0)
    assert "ET" in rec["ts_human"]
    # ts_human must NOT carry the literal "[" / "]" wrapper from the CLI
    # text format — easier for downstream consumers to inline as-is.
    assert "[" not in rec["ts_human"]
    assert "]" not in rec["ts_human"]


def test_print_hit_json_no_ts_field_when_unpopulated():
    h = _hit()  # mtime=0, indexed_at_unix=0
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["ts"] is None
    assert rec["ts_kind"] == ""
    assert rec["ts_human"] == ""
    assert rec["mtime"] == 0.0
    assert rec["indexed_at_unix"] == 0.0


def test_print_hit_json_carries_date_aware_fields():
    """library_added_at_unix + release_year must round-trip in JSON."""
    h = SearchHit(
        rowid=1,
        score=0.42,
        source="embiguity-content",
        path="embiguity:content:show:1",
        chunk_idx=0,
        offset=0,
        text="One Piece",
        library_added_at=1_650_000_000.0,
        release_year=1999,
    )
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["library_added_at_unix"] == pytest.approx(1_650_000_000.0)
    assert rec["release_year"] == 1999


def test_print_hit_json_date_fields_zero_when_unset():
    """Sentinel: library_added_at_unix == 0.0 and release_year == 0 means
    'unknown' — frontend / CLI must not surface a fake date for legacy rows."""
    h = _hit()  # default SearchHit — no date-aware fields populated
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["library_added_at_unix"] == 0.0
    assert rec["release_year"] == 0


def _stub_embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def test_search_populates_mtime_and_indexed_at(tmp_path):
    """End-to-end: adding a row with an explicit mtime → search() echoes it back."""
    db = open_db(tmp_path / "ts.db")
    embs = _stub_embeddings(2)
    upstream_mtime = 1_700_000_000.0
    add_chunks(
        db,
        source="chat",
        path="chat:dm:alice:1",
        mtime=upstream_mtime,
        file_hash="hashy",
        chunks=[(0, 0, "alpha"), (1, 5, "beta")],
        embeddings=embs,
    )
    hits = search(db, embs[0], k=2)
    assert hits, "expected search to return at least one hit"
    h = hits[0]
    assert h.mtime == pytest.approx(upstream_mtime)
    # indexed_at is CURRENT_TIMESTAMP at insert; must be a positive recent unix-ts.
    assert h.indexed_at_unix > 1_700_000_000.0
    assert h.indexed_at_unix <= time.time() + 1.0


def test_search_falls_back_when_files_row_absent(tmp_path):
    """If chunk_meta exists but files row is absent (LEFT JOIN miss),
    SearchHit must still construct cleanly with mtime=0.0 / indexed=0.0."""
    db = open_db(tmp_path / "noflz.db")
    embs = _stub_embeddings(1)
    add_chunks(
        db,
        source="x",
        path="/x/a.md",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "z")],
        embeddings=embs,
    )
    # Manually wipe the files row to simulate the edge case stats_by_source
    # already documents (chunk_meta orphan).
    db.execute("DELETE FROM files WHERE path = ?", ("/x/a.md",))
    db.commit()
    hits = search(db, embs[0], k=1)
    assert hits and hits[0].mtime == 0.0
    assert hits[0].indexed_at_unix == 0.0
