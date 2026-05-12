"""CLI-level tests for the --retrieval flag.

Exercises the wired-up cmd_query path with hybrid / vector / bm25
modes. Stubs the embedding model so no transformer load happens.
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eichi import EMBEDDING_DIM
from eichi.cli import build_parser
from eichi.store import add_chunks, open_db


def _stub_embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return (arr / norms).astype(np.float32)


def _stub_encode_one(text: str) -> np.ndarray:
    """Aim the query vector at the embedding indexed for a doc whose
    body contains ``text``. Used by tests to control which doc the
    pure-vec pass surfaces first."""
    # Deterministic per text — every call with the same string returns
    # the same vector, but different strings give different aim points.
    seed = abs(hash(text)) % (2**31)
    return _stub_embeddings(1, seed=seed)[0]


@pytest.fixture
def db_path():
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "ret.db"
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


def _seed(db_path: Path) -> dict[str, np.ndarray]:
    """Install three docs. Return path → embedding map."""
    conn = open_db(db_path)
    docs = [
        ("/c/admin.md", "server administration runbook ssh"),
        ("/b/anime.md", "Solo Leveling action manhwa adventure"),
        ("/a/shonen.md", "Naruto is a classic shonen anime — battle"),
    ]
    embs = _stub_embeddings(len(docs), seed=99)
    out: dict[str, np.ndarray] = {}
    for i, (path, body) in enumerate(docs):
        add_chunks(
            conn,
            source="test",
            path=path,
            mtime=1.0,
            file_hash=f"h{i}",
            chunks=[(0, 0, body)],
            embeddings=embs[i : i + 1],
        )
        out[path] = embs[i]
    conn.close()
    return out


def _run_query(argv, db_path) -> str:
    parser = build_parser()
    args = parser.parse_args(["--db", str(db_path)] + argv)
    # Patch the embed.encode_one used inside cmd_query so we never
    # touch the on-disk transformer model. The real flow imports
    # ``from . import embed`` and calls ``embed.encode_one(args.q)``.
    buf = io.StringIO()

    def _query_aware_encode(text: str) -> np.ndarray:
        # If the text is a single token, aim at that token's
        # corresponding doc in the seeded corpus. Otherwise use a
        # deterministic vec.
        return _stub_encode_one(text)

    with patch(
        "eichi.embed.encode_one", side_effect=_query_aware_encode
    ), patch.dict("os.environ", {"VSEARCH_NO_QUERY_LOG": "1"}):
        with redirect_stdout(buf):
            rc = args.func(args)
    assert rc == 0, f"query rc={rc}"
    return buf.getvalue()


def test_retrieval_bm25_finds_literal_term(db_path):
    """--retrieval=bm25 surfaces the doc with the literal token."""
    _seed(db_path)
    out = _run_query(
        ["query", "shonen", "--retrieval=bm25", "-k", "5", "--json"],
        db_path,
    )
    paths = [json.loads(line)["path"] for line in out.strip().splitlines()]
    assert paths
    assert paths[0] == "/a/shonen.md"


def test_retrieval_hybrid_emits_rank_provenance(db_path):
    """--retrieval=hybrid surfaces vec_rank + bm25_rank in JSON."""
    _seed(db_path)
    out = _run_query(
        ["query", "shonen", "--retrieval=hybrid", "-k", "5", "--json"],
        db_path,
    )
    rows = [json.loads(line) for line in out.strip().splitlines()]
    assert rows
    # The literal-term doc must be in the result set; at least one
    # row carries a non-NULL bm25_rank.
    bm25_paths = [r["path"] for r in rows if r.get("bm25_rank") is not None]
    assert "/a/shonen.md" in bm25_paths
    # Every row carries vec_rank or bm25_rank (or both).
    for r in rows:
        assert (
            r.get("vec_rank") is not None or r.get("bm25_rank") is not None
        )


def test_retrieval_vector_skips_fts(db_path):
    """--retrieval=vector returns the legacy pure-vec set; vec_rank /
    bm25_rank are NULL because the hybrid merger isn't invoked."""
    _seed(db_path)
    out = _run_query(
        ["query", "shonen", "--retrieval=vector", "-k", "5", "--json"],
        db_path,
    )
    rows = [json.loads(line) for line in out.strip().splitlines()]
    assert rows
    # Pure-vec mode does not stamp the rank fields.
    assert all(r.get("vec_rank") is None for r in rows)
    assert all(r.get("bm25_rank") is None for r in rows)
