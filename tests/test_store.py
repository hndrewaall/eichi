"""Store tests — uses sqlite-vec but stubs embeddings (no model load)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eichi import EMBEDDING_DIM
from eichi.store import (
    add_chunks,
    file_record,
    infer_source,
    list_files,
    needs_reindex,
    open_db,
    remove_path,
    search,
    sha256_file,
    sha256_text,
    stats,
    stats_by_source,
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
        # tmp dir cleanup best-effort.
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
    """Deterministic unit-norm pseudo-embeddings for tests."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    # Normalize.
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def test_open_creates_schema(db):
    from eichi import SCHEMA_VERSION

    s = stats(db)
    assert s["chunk_count"] == 0
    assert s["file_count"] == 0
    assert s["embedding_model"]
    assert s["schema_version"] == SCHEMA_VERSION


def test_add_and_search(db):
    chunks = [(0, 0, "alpha"), (1, 100, "bravo"), (2, 200, "charlie")]
    embs = _stub_embeddings(3)
    n = add_chunks(
        db,
        source="test",
        path="/tmp/x.txt",
        mtime=1.0,
        file_hash="deadbeef",
        chunks=chunks,
        embeddings=embs,
    )
    assert n == 3
    # Search for the closest match to chunk 0's own embedding.
    hits = search(db, embs[0], k=2)
    assert len(hits) == 2
    assert hits[0].path == "/tmp/x.txt"
    assert hits[0].text == "alpha"
    assert hits[0].score < hits[1].score  # closer first


def test_add_replaces_existing(db):
    embs = _stub_embeddings(2)
    add_chunks(
        db,
        source="test",
        path="/tmp/y.txt",
        mtime=1.0,
        file_hash="hash1",
        chunks=[(0, 0, "first"), (1, 10, "second")],
        embeddings=embs,
    )
    assert stats(db)["chunk_count"] == 2
    # Re-add with different chunks → old ones gone.
    embs2 = _stub_embeddings(1, seed=1)
    add_chunks(
        db,
        source="test",
        path="/tmp/y.txt",
        mtime=2.0,
        file_hash="hash2",
        chunks=[(0, 0, "replacement")],
        embeddings=embs2,
    )
    s = stats(db)
    assert s["chunk_count"] == 1
    assert s["file_count"] == 1
    rec = file_record(db, "/tmp/y.txt")
    assert rec == (2.0, "hash2")


def test_remove_path(db):
    embs = _stub_embeddings(2)
    add_chunks(
        db,
        source="test",
        path="/tmp/z.txt",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "one"), (1, 10, "two")],
        embeddings=embs,
    )
    assert stats(db)["chunk_count"] == 2
    removed = remove_path(db, "/tmp/z.txt")
    assert removed == 2
    assert stats(db)["chunk_count"] == 0
    assert stats(db)["file_count"] == 0


def test_remove_path_prefix(db):
    add_chunks(
        db,
        source="test",
        path="/tmp/dir/a.txt",
        mtime=1.0,
        file_hash="ha",
        chunks=[(0, 0, "a")],
        embeddings=_stub_embeddings(1),
    )
    add_chunks(
        db,
        source="test",
        path="/tmp/dir/sub/b.txt",
        mtime=1.0,
        file_hash="hb",
        chunks=[(0, 0, "b")],
        embeddings=_stub_embeddings(1, seed=2),
    )
    add_chunks(
        db,
        source="test",
        path="/tmp/other.txt",
        mtime=1.0,
        file_hash="hc",
        chunks=[(0, 0, "c")],
        embeddings=_stub_embeddings(1, seed=3),
    )
    # Removing /tmp/dir should drop the two under-dir files but leave /tmp/other.txt.
    n = remove_path(db, "/tmp/dir")
    assert n == 2
    s = stats(db)
    assert s["chunk_count"] == 1
    assert s["file_count"] == 1


def test_needs_reindex(db):
    add_chunks(
        db,
        source="test",
        path="/tmp/r.txt",
        mtime=1.0,
        file_hash="h1",
        chunks=[(0, 0, "x")],
        embeddings=_stub_embeddings(1),
    )
    # Same mtime + hash → no reindex needed.
    assert not needs_reindex(db, "/tmp/r.txt", 1.0, "h1")
    # Different mtime → reindex.
    assert needs_reindex(db, "/tmp/r.txt", 2.0, "h1")
    # Different hash → reindex.
    assert needs_reindex(db, "/tmp/r.txt", 1.0, "h2")
    # Unknown path → reindex.
    assert needs_reindex(db, "/tmp/never-seen.txt", 1.0, "h1")


def test_list_files_filters_by_source(db):
    add_chunks(
        db,
        source="memory",
        path="/m/a.md",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "m1")],
        embeddings=_stub_embeddings(1),
    )
    add_chunks(
        db,
        source="transcripts",
        path="/t/b.jsonl",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "t1")],
        embeddings=_stub_embeddings(1, seed=2),
    )
    all_rows = list_files(db)
    assert len(all_rows) == 2
    mem_rows = list_files(db, source="memory")
    assert len(mem_rows) == 1
    assert mem_rows[0][0] == "/m/a.md"


def test_stats_by_source(db):
    add_chunks(
        db,
        source="memory",
        path="/m/a.md",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "m1"), (1, 5, "m2")],
        embeddings=_stub_embeddings(2),
    )
    add_chunks(
        db,
        source="memory",
        path="/m/b.md",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "mb")],
        embeddings=_stub_embeddings(1, seed=3),
    )
    add_chunks(
        db,
        source="transcripts",
        path="/t/x.jsonl",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "t1")],
        embeddings=_stub_embeddings(1, seed=2),
    )
    rows = stats_by_source(db)
    by_src = {r["source"]: r for r in rows}
    assert by_src["memory"]["file_count"] == 2
    assert by_src["memory"]["chunk_count"] == 3
    assert by_src["transcripts"]["file_count"] == 1
    assert by_src["transcripts"]["chunk_count"] == 1
    # last_indexed_unix is the unix-time of the CURRENT_TIMESTAMP write —
    # must be a positive number, not 0.
    assert by_src["memory"]["last_indexed_unix"] > 0


def test_infer_source():
    assert infer_source("/home/x/.claude/projects/foo/bar.jsonl") == "transcripts"
    assert infer_source("/srv/Notes/2026-01-01.md") == "obsidian"
    assert infer_source("/srv/repos/foo/memory/bar.md") == "memory"
    assert infer_source("/etc/hosts") == "file"


def test_sha256_helpers(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello world")
    h_file = sha256_file(str(p))
    h_text = sha256_text("hello world")
    assert h_file == h_text  # sha256 of bytes vs sha256 of utf8 of same string.


def test_dimension_mismatch_raises(db):
    bad = np.zeros((1, EMBEDDING_DIM - 1), dtype=np.float32)
    with pytest.raises(ValueError):
        add_chunks(
            db,
            source="t",
            path="/tmp/bad.txt",
            mtime=1.0,
            file_hash="h",
            chunks=[(0, 0, "x")],
            embeddings=bad,
        )


def test_search_source_filter(db):
    add_chunks(
        db,
        source="memory",
        path="/m/a.md",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "memory chunk")],
        embeddings=_stub_embeddings(1, seed=10),
    )
    add_chunks(
        db,
        source="transcripts",
        path="/t/b.jsonl",
        mtime=1.0,
        file_hash="h",
        chunks=[(0, 0, "transcript chunk")],
        embeddings=_stub_embeddings(1, seed=11),
    )
    qvec = _stub_embeddings(1, seed=10)[0]
    hits = search(db, qvec, k=5, source="memory")
    assert all(h.source == "memory" for h in hits)
    assert any(h.path == "/m/a.md" for h in hits)


# --- date-aware fields ------------------------------------------------------
#
# Coverage:
#   - migration adds library_added_at + release_year columns
#   - add_chunks persists both fields and search() echoes them back
#   - --sort=added re-orders by library_added_at descending
#   - --year-min / --year-max filter on release_year (NULLs excluded)


def test_migration_adds_date_aware_columns(db):
    """Schema must carry library_added_at + release_year on the files table."""
    cols = {
        r[1] for r in db.execute("PRAGMA table_info(files)").fetchall()
    }
    assert "library_added_at" in cols, cols
    assert "release_year" in cols, cols


def test_add_chunks_persists_date_aware_fields(db):
    """add_chunks(library_added_at=..., release_year=...) → search() returns them."""
    embs = _stub_embeddings(1)
    add_chunks(
        db,
        source="media-content",
        path="media:content:show:42",
        mtime=1_700_000_000.0,
        file_hash="h42",
        chunks=[(0, 0, "Show: One Piece (1999) [Continuing]")],
        embeddings=embs,
        library_added_at=1_650_000_000.0,
        release_year=1999,
    )
    hits = search(db, embs[0], k=1)
    assert hits, "expected one hit"
    h = hits[0]
    assert h.library_added_at == pytest.approx(1_650_000_000.0)
    assert h.release_year == 1999


def test_search_sort_added_orders_by_library_added_descending(db):
    """--sort=added: most recently added row surfaces first regardless of relevance.

    All three rows share the same content so vec0 distance is identical;
    we use library_added_at to break the tie. The middle row (newest
    library_added_at) must come first.
    """
    embs = _stub_embeddings(3)
    paths = [
        ("media:content:show:1", 1_600_000_000.0, 2010),
        ("media:content:show:2", 1_750_000_000.0, 2020),  # newest add
        ("media:content:show:3", 1_700_000_000.0, 2015),
    ]
    for i, (p, added, year) in enumerate(paths):
        add_chunks(
            db,
            source="media-content",
            path=p,
            mtime=1_777_000_000.0,
            file_hash=f"h{i}",
            chunks=[(0, 0, "shared anime show body")],
            embeddings=embs[i : i + 1],
            library_added_at=added,
            release_year=year,
        )
    # Query with the first row's embedding so all three are candidates.
    hits = search(db, embs[0], k=3, sort="added")
    assert len(hits) == 3
    # Ordered: 1_750_000_000 → 1_700_000_000 → 1_600_000_000
    assert hits[0].path == "media:content:show:2"
    assert hits[1].path == "media:content:show:3"
    assert hits[2].path == "media:content:show:1"


def test_search_year_filter_excludes_out_of_range_and_nulls(db):
    """--year-min / --year-max bound release_year; rows with NULL year drop."""
    embs = _stub_embeddings(4)
    rows = [
        ("/x/a", 2005, "old"),
        ("/x/b", 2018, "mid"),
        ("/x/c", 2025, "new"),
        ("/x/d", None, "no-year"),
    ]
    for i, (p, year, body) in enumerate(rows):
        add_chunks(
            db,
            source="memory",
            path=p,
            mtime=1.0,
            file_hash=f"h{i}",
            chunks=[(0, 0, body)],
            embeddings=embs[i : i + 1],
            release_year=year,
        )
    qvec = embs[0]
    # Lower bound only.
    hits = search(db, qvec, k=10, year_min=2015)
    assert {h.path for h in hits} == {"/x/b", "/x/c"}
    # Upper bound only.
    hits = search(db, qvec, k=10, year_max=2010)
    assert {h.path for h in hits} == {"/x/a"}
    # Range — must exclude both /x/a (out) and /x/d (NULL year).
    hits = search(db, qvec, k=10, year_min=2010, year_max=2020)
    assert {h.path for h in hits} == {"/x/b"}


def test_search_added_since_excludes_nulls(db):
    """--added-since filter drops rows with NULL library_added_at."""
    import time as _time
    embs = _stub_embeddings(3)
    now = _time.time()
    rows = [
        ("/y/recent", now - 24 * 3600),     # 1 day ago — within window
        ("/y/old", now - 100 * 24 * 3600),  # 100 days ago — outside
        ("/y/no-add", None),                 # NULL → excluded
    ]
    for i, (p, added) in enumerate(rows):
        add_chunks(
            db,
            source="memory",
            path=p,
            mtime=1.0,
            file_hash=f"h{i}",
            chunks=[(0, 0, f"row {i}")],
            embeddings=embs[i : i + 1],
            library_added_at=added,
        )
    cutoff = now - 7 * 24 * 3600  # last 7 days
    hits = search(db, embs[0], k=10, added_since_unix=cutoff)
    assert {h.path for h in hits} == {"/y/recent"}


def test_embedder_migration_wipes_on_model_change(monkeypatch):
    """Switching ``EMBEDDING_MODEL`` must auto-wipe stale rows + rebuild vec0
    at the new dim — operators shouldn't have to manually rm the DB file.

    Simulates an upgrade by:
      1. Opening a DB with the current model + dim, adding rows.
      2. Patching the constants to a new model + dim.
      3. Reopening the DB — migration path must drop chunks/chunk_meta/files
         and recreate the vec0 table with the new dim.
      4. Adding new-dim rows must succeed, and old rows must be gone.
    """
    import eichi
    from eichi import store as _store

    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "migrate.db"
    try:
        # Phase 1: write rows under the original model/dim.
        conn = open_db(path)
        old_dim = eichi.EMBEDDING_DIM
        embs = _stub_embeddings(2)
        add_chunks(
            conn,
            source="memory",
            path="/m/old1",
            mtime=1.0,
            file_hash="h1",
            chunks=[(0, 0, "old row 1")],
            embeddings=embs[0:1],
        )
        add_chunks(
            conn,
            source="memory",
            path="/m/old2",
            mtime=1.0,
            file_hash="h2",
            chunks=[(0, 0, "old row 2")],
            embeddings=embs[1:2],
        )
        s_before = stats(conn)
        assert s_before["chunk_count"] == 2
        assert s_before["file_count"] == 2
        conn.close()

        # Phase 2: simulate an embedder swap. Pick a different dim so the
        # vec0 schema must actually change; the model name change alone is
        # enough to trigger _maybe_migrate_embedder, but we want to also
        # exercise the new-dim insert path.
        new_dim = old_dim * 2  # arbitrary; just must differ
        new_model = "test/synthetic-bigger-embedder"
        monkeypatch.setattr(eichi, "EMBEDDING_MODEL", new_model)
        monkeypatch.setattr(eichi, "EMBEDDING_DIM", new_dim)
        monkeypatch.setattr(_store, "EMBEDDING_MODEL", new_model)
        monkeypatch.setattr(_store, "EMBEDDING_DIM", new_dim)

        # Phase 3: reopen — migration must wipe old rows + rebuild vec0
        # at the new dim.
        conn = open_db(path)
        s_after = stats(conn)
        assert s_after["chunk_count"] == 0, "old chunks should be wiped"
        assert s_after["file_count"] == 0, "old files should be wiped"
        assert s_after["embedding_model"] == new_model

        # Phase 4: insert at the new dim must succeed.
        rng = np.random.default_rng(99)
        new_emb = rng.standard_normal((1, new_dim)).astype(np.float32)
        new_emb /= np.linalg.norm(new_emb, axis=1, keepdims=True) + 1e-9
        add_chunks(
            conn,
            source="memory",
            path="/m/new1",
            mtime=1.0,
            file_hash="h3",
            chunks=[(0, 0, "new row 1")],
            embeddings=new_emb,
        )
        s_final = stats(conn)
        assert s_final["chunk_count"] == 1
        assert s_final["file_count"] == 1
        conn.close()
    finally:
        # Best-effort cleanup; tmp files are small but be tidy.
        for p in path.parent.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            path.parent.rmdir()
        except OSError:
            pass
