"""Tests for `eichi index-stream`.

We monkeypatch `eichi.embed.encode` so the test does not require the
sentence-transformers model on disk.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eichi import EMBEDDING_DIM
from eichi.cli import build_parser
from eichi.store import open_db, stats


def _stub_encode(texts, batch_size=32):
    n = len(texts)
    if n == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return (arr / norms).astype(np.float32)


@pytest.fixture
def db_path():
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "stream.db"
    yield p
    for f in p.parent.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    try:
        p.parent.rmdir()
    except OSError:
        pass


def _run(argv, stdin_text, db_path):
    parser = build_parser()
    args = parser.parse_args(["--db", str(db_path)] + argv)
    with patch("eichi.embed.encode", side_effect=_stub_encode), \
         patch("sys.stdin", io.StringIO(stdin_text)):
        return args.func(args)


def test_index_stream_basic(db_path):
    payload = "\n".join(
        json.dumps(d)
        for d in [
            {
                "source": "chat",
                "doc_id": "chat:dm:alice:1771652256042",
                "text": "yo this is a test message",
                "mtime": 1771652256.042,
            },
            {
                "source": "chat",
                "doc_id": "chat:dm:alice:1771652256043",
                "text": "another message in the conversation",
                "mtime": 1771652256.043,
            },
        ]
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    s = stats(conn)
    assert s["chunk_count"] == 2
    assert s["file_count"] == 2
    assert "chat" in s["sources"]


def test_index_stream_idempotent(db_path):
    payload = json.dumps(
        {
            "source": "chat",
            "doc_id": "x:1",
            "text": "hello world",
            "mtime": 1.0,
        }
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    # Second run should not have added new chunks.
    conn = open_db(db_path)
    s = stats(conn)
    assert s["chunk_count"] == 1


def test_index_stream_default_source(db_path):
    payload = json.dumps({"doc_id": "y:1", "text": "no source on line"})
    rc = _run(["index-stream", "--source", "fallback"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    s = stats(conn)
    assert "fallback" in s["sources"]


def test_index_stream_missing_source_errors(db_path):
    payload = json.dumps({"doc_id": "z:1", "text": "no source anywhere"})
    rc = _run(["index-stream"], payload, db_path)
    assert rc != 0


def test_index_stream_bad_json_continues(db_path):
    payload = "not json\n" + json.dumps(
        {"source": "x", "doc_id": "ok:1", "text": "after bad line"}
    )
    rc = _run(["index-stream"], payload, db_path)
    # rc is non-zero (errors > 0) but the good line should still index.
    conn = open_db(db_path)
    s = stats(conn)
    assert s["chunk_count"] == 1
    assert rc != 0


def test_index_stream_persists_date_aware_fields(db_path):
    """Connectors emit library_added_at + release_year nested under
    ``metadata`` (preferred) or top-level (legacy). Both shapes must
    flow through to the files-table columns and back out via search().
    """
    from eichi.store import search
    payload = "\n".join(
        json.dumps(d)
        for d in [
            {
                "source": "media-content",
                "doc_id": "media:content:show:1",
                "text": "Show: One Piece (1999)",
                "mtime": 1_777_000_000.0,
                # Nested-shape: connector base.Document.metadata.
                "metadata": {
                    "library_added_at": 1_650_000_000.0,
                    "release_year": 1999,
                },
            },
            {
                "source": "media-content",
                "doc_id": "media:content:movie:2",
                "text": "Movie: Spirited Away (2001)",
                "mtime": 1_777_000_001.0,
                # Top-level shape (legacy fallback).
                "library_added_at": 1_700_000_000.0,
                "release_year": 2001,
            },
        ]
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT path, library_added_at, release_year FROM files ORDER BY path"
    ).fetchall()
    by_path = {r[0]: (r[1], r[2]) for r in rows}
    assert by_path["media:content:show:1"] == (1_650_000_000.0, 1999)
    assert by_path["media:content:movie:2"] == (1_700_000_000.0, 2001)


def test_index_stream_skips_bad_release_year(db_path):
    """A non-numeric release_year must NOT poison the row — left NULL."""
    payload = json.dumps(
        {
            "source": "calibre",
            "doc_id": "calibre:99",
            "text": "Book: Foo",
            "metadata": {
                "library_added_at": 1_650_000_000.0,
                "release_year": "not-a-year",
            },
        }
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    row = conn.execute(
        "SELECT library_added_at, release_year FROM files WHERE path = ?",
        ("calibre:99",),
    ).fetchone()
    assert row[0] == 1_650_000_000.0  # the good field still landed
    assert row[1] is None  # bad year → NULL


def test_index_stream_dry_run(db_path):
    payload = json.dumps(
        {"source": "x", "doc_id": "dry:1", "text": "should not be persisted"}
    )
    rc = _run(["index-stream", "-n"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    s = stats(conn)
    assert s["chunk_count"] == 0
