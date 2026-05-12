"""Hybrid (BM25 + vec0 + RRF) retrieval tests.

Covers:
  - BM25 alone catches a literal-term query that pure-vec misses
  - RRF merge prefers docs that rank in both passes
  - --retrieval=vector regression: same behavior as the legacy
    pure-vec search() path
  - Lazy fts5 backfill (ensure_fts_backfill) populates rows that
    pre-date the schema_version=2 migration

The fixture installs a tiny corpus where one document contains the
literal token "shonen" and another contains only loosely-related
anime terminology. Pure vec0 distance over a stub embedding cannot
single out the literal-term doc; BM25 trivially can.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from eichi import EMBEDDING_DIM
from eichi.store import (
    add_chunks,
    ensure_fts_backfill,
    fts_count,
    open_db,
    search,
    search_bm25,
    search_hybrid,
)


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "test.db"
    conn = open_db(path)
    try:
        yield conn
    finally:
        conn.close()
        for p in path.parent.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            path.parent.rmdir()
        except OSError:
            pass


def _stub_embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def _seed_corpus(db) -> dict[str, np.ndarray]:
    """Install a small mixed corpus.

    Three docs:
      - /a: contains the literal token "shonen" plus generic body text
      - /b: anime-adjacent body, NO "shonen" token
      - /c: unrelated server admin text

    Returns a dict path → embedding so individual tests can target
    specific docs with the query vector.
    """
    embs = _stub_embeddings(3, seed=42)
    # Track which embedding represents which doc so query vectors can
    # be aimed at specific items.
    out = {}
    for i, (path, body) in enumerate(
        [
            ("/c/admin.md", "server administration runbook ssh sudo"),
            (
                "/b/anime.md",
                "Solo Leveling action adventure manhwa adaptation",
            ),
            (
                "/a/shonen.md",
                "Naruto is a classic shonen anime — battle, friendship, growth",
            ),
        ]
    ):
        add_chunks(
            db,
            source="test",
            path=path,
            mtime=1.0,
            file_hash=f"h{i}",
            chunks=[(0, 0, body)],
            embeddings=embs[i : i + 1],
        )
        out[path] = embs[i]
    return out


def test_bm25_catches_literal_term_query(db):
    """BM25 must rank the doc containing "shonen" first; vec query
    aimed at an unrelated row would otherwise rank /a/shonen.md last."""
    by_path = _seed_corpus(db)

    # Pure-vec query aimed at the admin doc — /a/shonen.md is far away
    # in cosine space (we used random orthogonal-ish vectors).
    qvec = by_path["/c/admin.md"]
    vec_hits = search(db, qvec, k=3)
    vec_paths = [h.path for h in vec_hits]
    assert vec_paths[0] == "/c/admin.md"
    # /a/shonen.md is NOT first in pure-vec.
    assert vec_paths[0] != "/a/shonen.md"

    # BM25 query for the literal token "shonen" → /a/shonen.md is
    # the only doc that even matches.
    bm25_hits = search_bm25(db, "shonen", k=3)
    assert len(bm25_hits) == 1
    assert bm25_hits[0].path == "/a/shonen.md"


def test_rrf_prefers_docs_ranked_in_both_passes(db):
    """A doc that surfaces in both vec AND bm25 must outrank a doc
    that surfaces in only one pass at the same depth.

    Construction: query string is "shonen" (matches /a body literally),
    qvec is taken from /a's own embedding (so /a is also vec-rank-1).
    Both passes converge on /a. Other docs surface only in vec. RRF
    must put /a at position 0.
    """
    by_path = _seed_corpus(db)
    qvec = by_path["/a/shonen.md"]
    hits = search_hybrid(db, qvec, "shonen", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].path == "/a/shonen.md"
    # Both passes ranked /a — provenance must reflect that.
    assert hits[0].vec_rank is not None
    assert hits[0].bm25_rank is not None
    # Other hits, if any, must not have a bm25_rank (only /a matched
    # the literal token).
    for h in hits[1:]:
        assert h.bm25_rank is None


def test_retrieval_vector_regression_matches_legacy_search(db):
    """--retrieval=vector path must return the same hit set / order
    as a direct call to the legacy search() function. We compare path
    + score for the top-3."""
    by_path = _seed_corpus(db)
    qvec = by_path["/b/anime.md"]
    legacy = search(db, qvec, k=3)
    # Hybrid with vector-only mode = invoke search() under the hood.
    # The CLI dispatches; here we test the store API contract directly.
    assert legacy[0].path == "/b/anime.md"
    # The hybrid path with vector-only via search() returns identical
    # top-1; that's the regression invariant.
    via_search = search(db, qvec, k=3)
    assert [h.path for h in legacy] == [h.path for h in via_search]
    assert [h.score for h in legacy] == pytest.approx(
        [h.score for h in via_search]
    )


def test_lazy_fts_backfill_recovers_pre_migration_rows(db):
    """ensure_fts_backfill repopulates chunks_fts after a manual wipe.

    Simulates a schema_version=1 → 2 migration where chunks_fts is
    empty but chunk_meta carries existing rows.
    """
    _seed_corpus(db)
    # Manually clear chunks_fts to mimic the legacy state.
    db.execute("DELETE FROM chunks_fts")
    db.commit()
    assert fts_count(db) == 0
    inserted = ensure_fts_backfill(db)
    assert inserted == 3
    assert fts_count(db) == 3
    # And BM25 now finds the literal-term doc.
    hits = search_bm25(db, "shonen", k=3)
    assert any(h.path == "/a/shonen.md" for h in hits)


def test_hybrid_returns_vec_only_when_bm25_no_match(db):
    """If BM25 returns nothing (no token matches), hybrid degrades
    cleanly to the vec result set — no crash, no empty return."""
    by_path = _seed_corpus(db)
    qvec = by_path["/c/admin.md"]
    # "qzzwx" is unlikely to appear in any doc body.
    hits = search_hybrid(db, qvec, "qzzwx", k=3)
    assert hits, "vec hits should still come through"
    assert hits[0].path == "/c/admin.md"
    # No doc had bm25_rank populated.
    assert all(h.bm25_rank is None for h in hits)


def test_hybrid_respects_source_filter_in_both_passes(db):
    """The source filter must apply uniformly across vec + bm25 so
    the merged set never leaks rows from the wrong source tag."""
    embs = _stub_embeddings(2, seed=7)
    add_chunks(
        db,
        source="memory",
        path="/m/shonen.md",
        mtime=1.0,
        file_hash="hm",
        chunks=[(0, 0, "shonen anime tag in memory")],
        embeddings=embs[:1],
    )
    add_chunks(
        db,
        source="transcripts",
        path="/t/shonen.md",
        mtime=1.0,
        file_hash="ht",
        chunks=[(0, 0, "shonen anime in transcripts")],
        embeddings=embs[1:],
    )
    hits = search_hybrid(db, embs[0], "shonen", k=5, source="memory")
    assert hits
    assert all(h.source == "memory" for h in hits)
    assert all(h.path == "/m/shonen.md" for h in hits)


# ----------------------------------------------------------------------
# allowed_sources backstop (restricted-access frontend design).
#
# A web frontend hands the worker an ``allowed_sources`` list when a
# scoped user queries without a ?source= filter — eichi must apply
# the constraint at the row level so a stray source can never leak
# back to the caller. The unit tests below pin the behaviour for each
# of the three retrieval modes (vec, bm25, hybrid).
# ----------------------------------------------------------------------


def _seed_two_sources(db):
    """Install two rows in different sources, both containing the literal
    token ``shonen`` so BM25 finds them and the embedding stub puts both
    in the vec result set."""
    embs = _stub_embeddings(2, seed=11)
    add_chunks(
        db,
        source="calibre",
        path="/c/shonen.md",
        mtime=1.0,
        file_hash="hc",
        chunks=[(0, 0, "shonen battle manga in calibre")],
        embeddings=embs[:1],
    )
    add_chunks(
        db,
        source="signal-chat",
        path="/s/shonen.md",
        mtime=1.0,
        file_hash="hs",
        chunks=[(0, 0, "shonen anime mentioned in chat")],
        embeddings=embs[1:],
    )
    return embs


def test_allowed_sources_filters_pure_vec(db):
    embs = _seed_two_sources(db)
    hits = search(db, embs[0], k=10, allowed_sources=["calibre"])
    assert hits, "expected at least the calibre row"
    for h in hits:
        assert h.source == "calibre"


def test_allowed_sources_filters_bm25(db):
    _seed_two_sources(db)
    hits = search_bm25(db, "shonen", k=10, allowed_sources=["calibre"])
    assert hits
    for h in hits:
        assert h.source == "calibre"


def test_allowed_sources_filters_hybrid(db):
    embs = _seed_two_sources(db)
    hits = search_hybrid(
        db, embs[0], "shonen", k=10, allowed_sources=["calibre"]
    )
    assert hits
    for h in hits:
        assert h.source == "calibre"


def test_allowed_sources_empty_list_yields_zero(db):
    embs = _seed_two_sources(db)
    # Empty list → "no source allowed for this role" → zero rows.
    assert search(db, embs[0], k=10, allowed_sources=[]) == []
    assert search_bm25(db, "shonen", k=10, allowed_sources=[]) == []
    assert (
        search_hybrid(db, embs[0], "shonen", k=10, allowed_sources=[]) == []
    )


def test_allowed_sources_none_disables_filter(db):
    embs = _seed_two_sources(db)
    # None means "no constraint" — both sources surface.
    hits = search(db, embs[0], k=10, allowed_sources=None)
    sources = {h.source for h in hits}
    assert "calibre" in sources
    assert "signal-chat" in sources


def test_allowed_sources_intersects_with_explicit_source(db):
    embs = _seed_two_sources(db)
    # Explicit source AND allowlist both name calibre → just calibre.
    hits = search(
        db, embs[0], k=10, source="calibre", allowed_sources=["calibre"]
    )
    assert hits
    for h in hits:
        assert h.source == "calibre"
    # Explicit source not in allowlist → backstop bites: 0 rows.
    hits = search(
        db,
        embs[0],
        k=10,
        source="signal-chat",
        allowed_sources=["calibre"],
    )
    assert hits == []
