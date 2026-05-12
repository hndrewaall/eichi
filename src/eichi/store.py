"""sqlite-vec backed storage for eichi.

Schema (see CLI task spec):
    chunks         vec0 virtual table, embedding float[EMBEDDING_DIM]
    chunk_meta     rowid, source, path, chunk_idx, offset, text, content_hash, indexed_at
    files          path PK, source, mtime, file_hash, indexed_at
    meta           k, v key/value (embedding_model, schema_version)

Source tagging is path-inferred (memory / obsidian / transcripts / file). Each
file is fully replaced (delete rows then insert) when its mtime or sha256
changes — no half-state on partial reindex.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import EMBEDDING_DIM, EMBEDDING_MODEL, SCHEMA_VERSION

def _default_db_path() -> Path:
    """Resolve the on-disk index database path.

    Precedence:
      1. ``EICHI_DB`` env var (absolute path to the .db file).
      2. ``$XDG_DATA_HOME/eichi/index.db`` if XDG_DATA_HOME is set.
      3. ``~/.local/share/eichi/index.db`` (default).
    """
    override = os.environ.get("EICHI_DB")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "eichi" / "index.db"


DEFAULT_DB_PATH = _default_db_path()


@dataclass
class SearchHit:
    rowid: int
    score: float  # smaller = closer (vec0 distance)
    source: str
    path: str
    chunk_idx: int
    offset: int
    text: str
    # Upstream / per-connector "relevance" timestamp. For file-based
    # connectors this is the file mtime; for stream connectors (chat logs,
    # queue logs, etc.) it's the doc's send/event time.
    # 0.0 means "no upstream timestamp" — fall back to indexed_at_unix.
    mtime: float = 0.0
    # When eichi ingested the row (unix epoch). Always present.
    indexed_at_unix: float = 0.0
    # Document kind — "msg" / "cluster" / "" (legacy / unset). Carried so
    # the CLI / web UI can render cluster rows differently and so the
    # query path can dedupe a cluster hit against its constituent msg
    # hits.
    kind: str = ""
    # Owning cluster id when kind="msg" and the connector grouped this
    # message into a cluster. For kind="cluster" this is the cluster's
    # own id. Empty otherwise.
    cluster_id: str = ""
    # Cluster end-time (unix epoch). Only meaningful when kind="cluster".
    mtime_end: float = 0.0
    # When the underlying CONTENT (not the eichi row) landed in the
    # source library. Distinct from mtime (which is "upstream relevance",
    # typically updated_at) and indexed_at_unix (when eichi ingested).
    # 0.0 = unknown / unset.
    library_added_at: float = 0.0
    # Canonical release year for the content (show.year / movie.year /
    # book pubdate / album year). 0 = unknown / unset. Stored as int.
    release_year: int = 0
    # --- hybrid retrieval bookkeeping (NULL when not in that pass) ---
    # 1-based rank of this doc in the vec0 / fts5 retrieval pass. None
    # when the doc did not surface in that pass. The hybrid query path
    # populates these so callers can show "vec_rank=3, bm25_rank=NULL"
    # style provenance per hit. Plain ``search()`` and
    # ``search_bm25()`` leave them at None — only ``search_hybrid()``
    # writes them.
    vec_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


def _serialize_f32(vec: np.ndarray) -> bytes:
    """sqlite-vec accepts a packed little-endian float32 blob."""
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    return vec.tobytes(order="C")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def open_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (or create+initialize) the eichi sqlite-vec database."""
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_path.parent.chmod(0o700)
    except OSError:
        pass
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    import sqlite_vec  # type: ignore

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _maybe_migrate_embedder(conn)
    _init_schema(conn)
    return conn


def _maybe_migrate_embedder(conn: sqlite3.Connection) -> None:
    """Detect an embedder change vs the on-disk DB and wipe to force rebuild.

    The vec0 ``chunks`` table hard-codes ``float[N]`` at creation time, so a
    model swap (e.g. MiniLM-L6 384d → mpnet 768d) leaves us unable to
    INSERT new vectors into a stale table. ``CREATE VIRTUAL TABLE IF NOT
    EXISTS chunks USING vec0(embedding float[NEW_DIM])`` is a no-op when
    the table already exists, so we have to actively drop the prior table.

    We also clear ``chunk_meta`` / ``chunks_fts`` / ``files`` because all
    the row content is now stale (it was embedded with the old model and
    the rowids are about to be reused by the new vec0 chunks table).

    No-op when the on-disk model matches the code constant, or when the
    DB is fresh (no ``meta`` table yet — first-run path).
    """
    cur = conn.cursor()
    # ``meta`` may not exist yet on a brand-new DB. ``IF NOT EXISTS`` is
    # cheap and identical to the schema-init definition.
    cur.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    row = cur.execute(
        "SELECT v FROM meta WHERE k = 'embedding_model'"
    ).fetchone()
    if row is None:
        # Fresh DB — nothing to migrate.
        return
    on_disk = row[0]
    if on_disk == EMBEDDING_MODEL:
        return
    # Model change detected — print to stderr so cron / interactive runs
    # both see it, then wipe every embedder-dependent artifact.
    import sys as _sys
    print(
        f"eichi: embedder change detected ({on_disk} → {EMBEDDING_MODEL}); "
        "wiping vec0 + chunk_meta + chunks_fts + files (cursors must be "
        "reset by the connector driver to re-emit).",
        file=_sys.stderr,
    )
    # Order matters: the vec0 virtual table backs onto multiple shadow
    # tables (chunks_chunks / chunks_rowids / chunks_vector_chunks00).
    # ``DROP TABLE chunks`` cleans them up via the vec0 destructor.
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("DROP TABLE IF EXISTS chunks_fts")
    cur.execute("DELETE FROM chunk_meta") if _table_exists(cur, "chunk_meta") else None
    cur.execute("DELETE FROM files") if _table_exists(cur, "files") else None
    # Clear meta keys we'll re-write in _init_schema. Leave any other
    # operator-set keys alone.
    cur.execute("DELETE FROM meta WHERE k IN ('embedding_model', 'schema_version')")
    conn.commit()


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(embedding float[{EMBEDDING_DIM}])"
    )
    # BM25 (full-text) sibling of `chunks`. Uses fts5's "external content"
    # mode: rows are explicitly INSERT/DELETE'd in lockstep with chunk_meta
    # so the rowid is the same across `chunks`, `chunk_meta`, and
    # `chunks_fts`. Tokenizer is `unicode61 remove_diacritics 2` (default
    # SQLite-fts5 behavior plus accent folding) — keeps "résumé" and
    # "resume" both findable, and handles Japanese kana / Han runs by
    # splitting on character-class boundaries (good enough for our intent
    # term hits; not a full CJK tokenizer but neither is the embedding).
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text,
          tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_meta (
          rowid INTEGER PRIMARY KEY,
          source TEXT NOT NULL,
          path TEXT NOT NULL,
          chunk_idx INTEGER NOT NULL,
          offset INTEGER,
          text TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meta_path ON chunk_meta(path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meta_source ON chunk_meta(source)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          mtime REAL NOT NULL,
          file_hash TEXT NOT NULL,
          indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Additive columns for clustering (chat-style connectors etc.) — nullable so all
    # existing rows / connectors work without a migration. `kind` is one of
    # "msg" / "cluster" (or NULL for legacy/file connectors); `cluster_id`
    # links a "msg" row to its owning "cluster" doc, or carries a
    # "cluster" doc's own id; `mtime_end` is the last-message timestamp
    # for cluster rows.
    for col_def in (
        "ALTER TABLE files ADD COLUMN kind TEXT",
        "ALTER TABLE files ADD COLUMN cluster_id TEXT",
        "ALTER TABLE files ADD COLUMN mtime_end REAL",
        # Date-aware fields. Nullable; per-connector best-effort populated
        # (NULL when unknown — never fabricated). Used by --sort=added /
        # --sort=year and --year-min / --year-max / --added-since filters.
        "ALTER TABLE files ADD COLUMN library_added_at REAL",
        "ALTER TABLE files ADD COLUMN release_year INTEGER",
    ):
        try:
            cur.execute(col_def)
        except sqlite3.OperationalError:
            # Column already exists — sqlite has no IF NOT EXISTS for ALTER.
            pass
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_files_cluster_id ON files(cluster_id)"
    )
    # Index on library_added_at to keep --sort=added / --added-since fast
    # without a full-files scan once we have many rows.
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_files_library_added_at "
        "ON files(library_added_at)"
    )
    cur.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    cur.execute(
        "INSERT OR IGNORE INTO meta(k, v) VALUES (?, ?)",
        ("embedding_model", EMBEDDING_MODEL),
    )
    cur.execute(
        "INSERT OR IGNORE INTO meta(k, v) VALUES (?, ?)",
        ("schema_version", SCHEMA_VERSION),
    )
    # Migration bookkeeping. When the on-disk meta says we're behind
    # the current code's schema_version, advance it. This is the
    # only path that mutates meta.schema_version after creation —
    # the additive ALTER TABLE / CREATE VIRTUAL TABLE statements
    # above are idempotent so the migration is safe to re-run.
    current = cur.execute(
        "SELECT v FROM meta WHERE k = 'schema_version'"
    ).fetchone()
    if current and current[0] != SCHEMA_VERSION:
        cur.execute(
            "UPDATE meta SET v = ? WHERE k = 'schema_version'",
            (SCHEMA_VERSION,),
        )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, value),
    )
    conn.commit()


def file_record(conn: sqlite3.Connection, path: str) -> Optional[Tuple[float, str]]:
    row = conn.execute(
        "SELECT mtime, file_hash FROM files WHERE path = ?", (path,)
    ).fetchone()
    return (row[0], row[1]) if row else None


def needs_reindex(
    conn: sqlite3.Connection, path: str, mtime: float, file_hash: str
) -> bool:
    rec = file_record(conn, path)
    if rec is None:
        return True
    return rec[0] != mtime or rec[1] != file_hash


def remove_path(conn: sqlite3.Connection, path: str) -> int:
    """Delete all chunks + file row for `path`. Returns chunk count removed.

    Accepts either a single file path or a directory prefix (anything starting
    with the prefix is removed).
    """
    cur = conn.cursor()
    rowids = [
        r[0]
        for r in cur.execute(
            "SELECT rowid FROM chunk_meta WHERE path = ? OR path LIKE ?",
            (path, path.rstrip("/") + "/%"),
        ).fetchall()
    ]
    if rowids:
        # vec0 virtual table requires individual row delete via rowid match.
        cur.executemany("DELETE FROM chunks WHERE rowid = ?", [(r,) for r in rowids])
        cur.executemany(
            "DELETE FROM chunk_meta WHERE rowid = ?", [(r,) for r in rowids]
        )
        # fts5: same rowid contract — drop matching rows. The fts5 table
        # may be empty for migrated DBs that haven't been re-indexed yet
        # (lazy backfill — see _ensure_fts_for_rowid in add_chunks);
        # DELETE on a missing rowid is a no-op so this is safe either way.
        cur.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(r,) for r in rowids])
    cur.execute(
        "DELETE FROM files WHERE path = ? OR path LIKE ?",
        (path, path.rstrip("/") + "/%"),
    )
    conn.commit()
    return len(rowids)


def add_chunks(
    conn: sqlite3.Connection,
    *,
    source: str,
    path: str,
    mtime: float,
    file_hash: str,
    chunks: Sequence[Tuple[int, int, str]],
    embeddings: np.ndarray,
    kind: Optional[str] = None,
    cluster_id: Optional[str] = None,
    mtime_end: Optional[float] = None,
    library_added_at: Optional[float] = None,
    release_year: Optional[int] = None,
) -> int:
    """Replace existing rows for `path` and insert new (chunks, embeddings).

    embeddings is shape (N, EMBEDDING_DIM); chunks is the parallel metadata.

    Optional `kind` / `cluster_id` / `mtime_end` populate the corresponding
    columns on the `files` row. They are NULL when omitted.

    `library_added_at` (unix epoch float) and `release_year` (int) are the
    date-aware fields used by ``--sort=added``, ``--sort=year``,
    ``--year-min/--year-max``, and ``--added-since`` at query time. NULL
    when omitted — connectors populate best-effort.
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks/embeddings length mismatch: {len(chunks)} vs {embeddings.shape[0]}"
        )
    if embeddings.shape and embeddings.shape[0] > 0 and embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"embedding dim mismatch: got {embeddings.shape[1]}, want {EMBEDDING_DIM}"
        )
    # Wipe prior rows for this exact path (re-index in place).
    remove_path(conn, path)
    cur = conn.cursor()
    for (idx, offset, piece), vec in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO chunk_meta(source, path, chunk_idx, offset, text, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source, path, idx, offset, piece, sha256_text(piece)),
        )
        rowid = cur.lastrowid
        cur.execute(
            "INSERT INTO chunks(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_f32(vec)),
        )
        # BM25 sibling. Same rowid as chunks / chunk_meta — the hybrid
        # query path joins by rowid. We index the raw chunk body; fts5
        # owns its own tokenization (see CREATE VIRTUAL TABLE in
        # _init_schema). Lazy backfill for legacy rows happens in
        # _ensure_fts_backfill on first hybrid query.
        cur.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            (rowid, piece),
        )
    # Coerce release_year to int when given a non-None numeric/string
    # value. Bad input → None (i.e. left NULL on disk).
    if release_year is not None:
        try:
            release_year = int(release_year)
        except (TypeError, ValueError):
            release_year = None
    if library_added_at is not None:
        try:
            library_added_at = float(library_added_at)
        except (TypeError, ValueError):
            library_added_at = None
    cur.execute(
        """
        INSERT INTO files(path, source, mtime, file_hash, indexed_at,
                          kind, cluster_id, mtime_end,
                          library_added_at, release_year)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            source=excluded.source,
            mtime=excluded.mtime,
            file_hash=excluded.file_hash,
            indexed_at=CURRENT_TIMESTAMP,
            kind=excluded.kind,
            cluster_id=excluded.cluster_id,
            mtime_end=excluded.mtime_end,
            library_added_at=excluded.library_added_at,
            release_year=excluded.release_year
        """,
        (path, source, mtime, file_hash, kind, cluster_id, mtime_end,
         library_added_at, release_year),
    )
    conn.commit()
    return len(chunks)


def fts_count(conn: sqlite3.Connection) -> int:
    """Number of rows currently indexed in the BM25 (fts5) sibling table.

    Cheap diagnostic — used by the hybrid query path to detect a stale /
    not-yet-backfilled fts table on legacy DBs (schema_version=1 →
    schema_version=2 migration).
    """
    try:
        return int(
            conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        )
    except sqlite3.OperationalError:
        # Table missing entirely — not even initialised.
        return 0


def ensure_fts_backfill(
    conn: sqlite3.Connection, *, batch: int = 5000
) -> int:
    """Lazy backfill of chunks_fts from chunk_meta when migrating from
    schema_version=1.

    The migration path doesn't eager-rebuild the BM25 index — that
    would block the first eichi invocation post-upgrade for minutes
    on a 100k-row DB. Instead, the hybrid query path calls this once
    and amortizes the cost over a single startup. Subsequent calls
    are essentially free: we compare COUNT(chunks_fts) to
    COUNT(chunk_meta); if they match we return immediately. Only when
    counts diverge do we run the (expensive) NOT EXISTS scan to fill
    the gap.

    Returns the number of rows inserted. 0 when the table is already
    in sync — the common steady-state case.
    """
    cur = conn.cursor()
    try:
        meta_n = int(cur.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()[0])
        fts_n = int(cur.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    if fts_n >= meta_n:
        # Steady-state — no work to do. fts_n > meta_n shouldn't happen
        # in practice (we always delete from both in lockstep), but if
        # it does it doesn't hurt the BM25 result quality.
        return 0
    inserted = 0
    while True:
        rows = cur.execute(
            """
            SELECT m.rowid, m.text
            FROM chunk_meta m
            WHERE NOT EXISTS (
              SELECT 1 FROM chunks_fts f WHERE f.rowid = m.rowid
            )
            LIMIT ?
            """,
            (batch,),
        ).fetchall()
        if not rows:
            break
        cur.executemany(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            [(r[0], r[1]) for r in rows],
        )
        inserted += len(rows)
        conn.commit()
        if len(rows) < batch:
            break
    return inserted


def _quote_fts_term(term: str) -> str:
    """Escape a single token for fts5 MATCH.

    fts5 MATCH syntax interprets a number of characters specially (`-`,
    `*`, `(`, `)`, `:`, `"`, AND/OR/NOT keywords, etc.). The cheapest
    safe way to pass user-typed text through is to wrap each token in
    double quotes (which makes it a literal phrase) and escape any
    embedded `"`.
    """
    return '"' + term.replace('"', '""') + '"'


def _build_fts_match(query: str) -> str:
    """Tokenize free-form user text into an fts5 MATCH expression.

    Strategy: split on whitespace, drop empty tokens, quote each token
    individually, join with the implicit AND (space). This means
    multi-word queries require all terms — same default as a Google-
    style search. fts5 returns nothing for a missing term, which fits
    the "BM25 catches literal-term hits" use case.

    Punctuation inside a token (e.g. ``foo-bar``) is preserved by the
    quoting; fts5's tokenizer handles segmentation at index time.
    """
    tokens = [t for t in query.split() if t.strip()]
    if not tokens:
        return ""
    return " ".join(_quote_fts_term(t) for t in tokens)


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    source: Optional[str] = None,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    added_since_unix: Optional[float] = None,
    allowed_sources: Optional[List[str]] = None,
) -> List[SearchHit]:
    """BM25 (fts5) search; returns top-k SearchHit ordered by ascending
    bm25() score (smaller = better in fts5's bm25() rank function).

    Mirrors :func:`search`'s post-filter set (source / year range /
    added-since / allowed_sources) so the hybrid path can apply the
    SAME filter rules to each retrieval pass.

    On migrated DBs where the fts5 table is empty for legacy rows,
    callers should ensure :func:`ensure_fts_backfill` has run first.
    Uncommon on a freshly-indexed DB; see the hybrid path for the
    one-shot trigger.
    """
    match = _build_fts_match(query)
    if not match:
        return []
    if allowed_sources is not None and len(allowed_sources) == 0:
        return []
    allowed_set = set(allowed_sources) if allowed_sources else None
    has_post_filter = bool(
        source
        or year_min is not None
        or year_max is not None
        or added_since_unix is not None
        or allowed_set
    )
    fetch_k = max(k * 5, 50) if has_post_filter else k
    try:
        rows = conn.execute(
            """
            SELECT chunks_fts.rowid, bm25(chunks_fts) AS score,
                   m.source, m.path, m.chunk_idx, m.offset, m.text,
                   COALESCE(f.mtime, 0) AS f_mtime,
                   COALESCE(strftime('%s', f.indexed_at), 0) AS f_indexed_at_unix,
                   f.kind, f.cluster_id, COALESCE(f.mtime_end, 0) AS f_mtime_end,
                   f.library_added_at, f.release_year
            FROM chunks_fts
            JOIN chunk_meta m ON m.rowid = chunks_fts.rowid
            LEFT JOIN files f ON f.path = m.path
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (match, fetch_k),
        ).fetchall()
    except sqlite3.OperationalError:
        # Bad MATCH expression (e.g. all tokens stripped to nothing).
        # Be forgiving — return empty, let the caller fall back to vec.
        return []

    hits: List[SearchHit] = []
    for (rowid, score, src, path, idx, offset, text,
         f_mtime, f_indexed_at_unix, f_kind, f_cluster_id, f_mtime_end,
         f_library_added_at, f_release_year) in rows:
        if source and src != source:
            continue
        if allowed_set is not None and src not in allowed_set:
            continue
        try:
            mtime_f = float(f_mtime) if f_mtime is not None else 0.0
        except (TypeError, ValueError):
            mtime_f = 0.0
        try:
            indexed_at_f = float(f_indexed_at_unix) if f_indexed_at_unix is not None else 0.0
        except (TypeError, ValueError):
            indexed_at_f = 0.0
        try:
            mtime_end_f = float(f_mtime_end) if f_mtime_end is not None else 0.0
        except (TypeError, ValueError):
            mtime_end_f = 0.0
        added_known = f_library_added_at is not None
        try:
            added_f = float(f_library_added_at) if added_known else 0.0
        except (TypeError, ValueError):
            added_f = 0.0
            added_known = False
        year_known = f_release_year is not None
        try:
            year_i = int(f_release_year) if year_known else 0
        except (TypeError, ValueError):
            year_i = 0
            year_known = False

        if year_min is not None or year_max is not None:
            if not year_known:
                continue
            if year_min is not None and year_i < int(year_min):
                continue
            if year_max is not None and year_i > int(year_max):
                continue
        if added_since_unix is not None:
            if not added_known or added_f < float(added_since_unix):
                continue

        hits.append(
            SearchHit(
                rowid=rowid,
                score=float(score),
                source=src,
                path=path,
                chunk_idx=idx,
                offset=offset or 0,
                text=text,
                mtime=mtime_f,
                indexed_at_unix=indexed_at_f,
                kind=f_kind or "",
                cluster_id=f_cluster_id or "",
                mtime_end=mtime_end_f,
                library_added_at=added_f,
                release_year=year_i,
            )
        )
        if len(hits) >= k:
            break
    return hits


def search(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int = 10,
    source: Optional[str] = None,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    added_since_unix: Optional[float] = None,
    sort: str = "relevance",
    allowed_sources: Optional[List[str]] = None,
) -> List[SearchHit]:
    """KNN search; returns top-k SearchHit.

    Default ordering is by ascending vec0 distance ("relevance"). Set
    ``sort="added"`` to re-order the post-filter result set by
    ``library_added_at`` descending, or ``sort="year"`` to re-order by
    ``release_year`` descending. Within either alternate sort, ties (or
    NULLs) keep the original relevance order — so a year-tied newer
    library-add still surfaces above an older one when both have the
    same release year.

    ``year_min`` / ``year_max`` filter on ``release_year`` (rows with
    NULL release_year are excluded when EITHER bound is set —
    "filtering by year" implies the field must be known).

    ``added_since_unix`` filters on ``library_added_at`` (rows with NULL
    library_added_at are excluded when this filter is set).

    ``allowed_sources`` is a defense-in-depth list filter. When non-None
    and non-empty, only rows whose ``source`` is in the list survive;
    when None, no constraint is applied (admin / unrestricted callers).
    Empty list yields zero rows. Combines with ``source`` as an
    intersection (both filters apply when both are set), but typical
    usage sets only one of the two — search-minisite hands a single
    explicit source through ``source`` AND the broader role-allowlist
    through ``allowed_sources`` only on no-filter queries.
    """
    blob = _serialize_f32(query_vec)
    # Defense-in-depth: when ``allowed_sources`` is an explicit empty
    # list the caller has signalled "no source allowed for this role" —
    # short-circuit to zero rows rather than burn a vec0 KNN query.
    if allowed_sources is not None and len(allowed_sources) == 0:
        return []
    allowed_set = (
        set(allowed_sources) if allowed_sources else None
    )
    # vec0 supports MATCH-with-blob KNN; we then JOIN to chunk_meta.
    # We also LEFT JOIN files to surface the per-doc upstream `mtime` and
    # ingest `indexed_at` timestamps so the CLI / web UI can show a
    # human-readable date alongside each hit. LEFT JOIN because edge-case
    # connectors may have written chunk_meta rows without a files row
    # (the same edge case stats_by_source compensates for).
    # We over-fetch when filtering by source / year / added-since to
    # avoid empty results — a strict filter can prune many candidates.
    has_post_filter = bool(
        source or year_min is not None or year_max is not None
        or added_since_unix is not None or sort != "relevance"
        or allowed_set
    )
    fetch_k = max(k * 5, 50) if has_post_filter else k
    rows = conn.execute(
        """
        SELECT chunks.rowid, chunks.distance, m.source, m.path, m.chunk_idx,
               m.offset, m.text,
               COALESCE(f.mtime, 0) AS f_mtime,
               COALESCE(strftime('%s', f.indexed_at), 0) AS f_indexed_at_unix,
               f.kind, f.cluster_id, COALESCE(f.mtime_end, 0) AS f_mtime_end,
               f.library_added_at, f.release_year
        FROM chunks
        JOIN chunk_meta m ON m.rowid = chunks.rowid
        LEFT JOIN files f ON f.path = m.path
        WHERE chunks.embedding MATCH ? AND k = ?
        ORDER BY chunks.distance ASC
        """,
        (blob, fetch_k),
    ).fetchall()
    hits: List[SearchHit] = []
    for (rowid, dist, src, path, idx, offset, text,
         f_mtime, f_indexed_at_unix, f_kind, f_cluster_id, f_mtime_end,
         f_library_added_at, f_release_year) in rows:
        if source and src != source:
            continue
        if allowed_set is not None and src not in allowed_set:
            continue
        try:
            mtime_f = float(f_mtime) if f_mtime is not None else 0.0
        except (TypeError, ValueError):
            mtime_f = 0.0
        try:
            indexed_at_f = float(f_indexed_at_unix) if f_indexed_at_unix is not None else 0.0
        except (TypeError, ValueError):
            indexed_at_f = 0.0
        try:
            mtime_end_f = float(f_mtime_end) if f_mtime_end is not None else 0.0
        except (TypeError, ValueError):
            mtime_end_f = 0.0
        # library_added_at / release_year may be NULL on disk → 0 sentinel
        # in the SearchHit. Track presence as a bool for the filter pass.
        added_known = f_library_added_at is not None
        try:
            added_f = float(f_library_added_at) if added_known else 0.0
        except (TypeError, ValueError):
            added_f = 0.0
            added_known = False
        year_known = f_release_year is not None
        try:
            year_i = int(f_release_year) if year_known else 0
        except (TypeError, ValueError):
            year_i = 0
            year_known = False

        # --- date-aware filters ---
        if year_min is not None or year_max is not None:
            if not year_known:
                continue
            if year_min is not None and year_i < int(year_min):
                continue
            if year_max is not None and year_i > int(year_max):
                continue
        if added_since_unix is not None:
            if not added_known or added_f < float(added_since_unix):
                continue

        hits.append(
            SearchHit(
                rowid=rowid,
                score=float(dist),
                source=src,
                path=path,
                chunk_idx=idx,
                offset=offset or 0,
                text=text,
                mtime=mtime_f,
                indexed_at_unix=indexed_at_f,
                kind=f_kind or "",
                cluster_id=f_cluster_id or "",
                mtime_end=mtime_end_f,
                library_added_at=added_f,
                release_year=year_i,
            )
        )
        # Don't trim to k yet if we'll re-sort: alternate sorts may
        # promote a row that was outside the relevance top-k.
        if sort == "relevance" and len(hits) >= k:
            break

    # Alternate sorts: stable-sort by the requested column descending.
    # Python's sorted() is stable, so within a tie we keep the original
    # relevance order — exactly what we want.
    if sort == "added":
        hits.sort(key=lambda h: -float(h.library_added_at or 0.0))
    elif sort == "year":
        hits.sort(key=lambda h: -int(h.release_year or 0))

    return hits[:k]


# Reciprocal Rank Fusion (RRF) constant. The standard from Cormack/Clarke/
# Buettcher 2009 is k=60 — chosen so that rank 1 carries ~1/61, rank 60
# ~1/120, and the long tail decays gently. We expose it as a default
# argument so tests can dial it down for synthetic fixtures.
RRF_K = 60


def search_hybrid(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    query_text: str,
    k: int = 10,
    source: Optional[str] = None,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    added_since_unix: Optional[float] = None,
    sort: str = "relevance",
    rrf_k: int = RRF_K,
    fetch_per_pass: Optional[int] = None,
    allowed_sources: Optional[List[str]] = None,
) -> List[SearchHit]:
    """Hybrid retrieval: union of vec0 KNN + fts5 BM25 merged by RRF.

    Reciprocal Rank Fusion::

        score(doc) = sum( 1 / (rrf_k + rank_in_pass) ) for each pass that
                     surfaced the doc

    A doc that ranks highly in EITHER pass surfaces; a doc ranked in
    BOTH outranks any single-pass hit at the same depth. No score
    normalization needed — RRF only consumes ranks.

    The two passes apply the same source / year / added-since filters
    so the merged set has consistent provenance.

    Returns up to ``k`` SearchHits, ordered by descending RRF score.
    Each hit carries the original vec0 distance in ``score`` (preserving
    the legacy display) plus 1-based ``vec_rank`` / ``bm25_rank``
    fields (None when the doc did not surface in that pass).

    ``sort != "relevance"`` re-orders the merged set the same way the
    pure-vector path does, by the requested column descending.
    """
    # Over-fetch each pass to give RRF more material — a doc ranked
    # 12th in vec but 3rd in bm25 needs both passes to see it. The
    # default heuristic is `max(k * 5, 50)` per pass, mirroring the
    # post-filter over-fetch already used in search().
    per = (
        fetch_per_pass if fetch_per_pass is not None else max(k * 5, 50)
    )

    vec_hits = search(
        conn,
        query_vec,
        k=per,
        source=source,
        year_min=year_min,
        year_max=year_max,
        added_since_unix=added_since_unix,
        sort="relevance",  # rank by raw distance — sort applied post-merge
        allowed_sources=allowed_sources,
    )
    bm25_hits = search_bm25(
        conn,
        query_text,
        k=per,
        source=source,
        year_min=year_min,
        year_max=year_max,
        added_since_unix=added_since_unix,
        allowed_sources=allowed_sources,
    )

    # Build the merged map keyed by rowid (the unique chunk id).
    # Each entry tracks the underlying SearchHit, the per-pass ranks,
    # and the accumulating RRF score.
    merged: dict[int, dict] = {}

    for rank, hit in enumerate(vec_hits, start=1):
        entry = merged.setdefault(
            hit.rowid,
            {"hit": hit, "vec_rank": None, "bm25_rank": None, "rrf": 0.0},
        )
        entry["vec_rank"] = rank
        entry["rrf"] += 1.0 / (rrf_k + rank)
        # Prefer the vec hit's metadata when present — vec carries the
        # distance score that callers historically display.
        entry["hit"] = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        entry = merged.setdefault(
            hit.rowid,
            {"hit": hit, "vec_rank": None, "bm25_rank": None, "rrf": 0.0},
        )
        entry["bm25_rank"] = rank
        entry["rrf"] += 1.0 / (rrf_k + rank)
        # If we only saw this row in BM25, the BM25 hit is canonical.
        if entry["vec_rank"] is None:
            entry["hit"] = hit

    # Sort by descending RRF score; ties broken by ascending vec_rank
    # then ascending bm25_rank (a doc that ranked higher in either
    # pass wins the tie).
    def _rank_or_inf(r):
        return r if r is not None else float("inf")

    ordered = sorted(
        merged.values(),
        key=lambda e: (
            -e["rrf"],
            _rank_or_inf(e["vec_rank"]),
            _rank_or_inf(e["bm25_rank"]),
        ),
    )

    out: List[SearchHit] = []
    for entry in ordered:
        h = entry["hit"]
        # Stamp the per-pass ranks back onto the hit dataclass so the
        # CLI / web layer can render them. We replace() rather than
        # mutate so two hybrid calls don't cross-contaminate cached
        # SearchHit instances.
        h.vec_rank = entry["vec_rank"]
        h.bm25_rank = entry["bm25_rank"]
        out.append(h)

    if sort == "added":
        out.sort(key=lambda h: -float(h.library_added_at or 0.0))
    elif sort == "year":
        out.sort(key=lambda h: -int(h.release_year or 0))

    return out[:k]


def stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    chunk_count = cur.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()[0]
    file_count = cur.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    sources = [
        r[0] for r in cur.execute("SELECT DISTINCT source FROM chunk_meta").fetchall()
    ]
    last_indexed = cur.execute(
        "SELECT MAX(indexed_at) FROM files"
    ).fetchone()[0]
    db_path = None
    for r in cur.execute("PRAGMA database_list").fetchall():
        if r[1] == "main":
            db_path = r[2]
            break
    db_size = os.path.getsize(db_path) if db_path and os.path.exists(db_path) else 0
    return {
        "chunk_count": chunk_count,
        "file_count": file_count,
        "sources": sources,
        "last_indexed": last_indexed,
        "db_path": db_path,
        "db_size_bytes": db_size,
        "embedding_model": get_meta(conn, "embedding_model"),
        "schema_version": get_meta(conn, "schema_version"),
    }


def stats_by_source(conn: sqlite3.Connection) -> list:
    """Return per-source metric rows: file_count, chunk_count, last_indexed_unix.

    `last_indexed_unix` is the unix-time of the most recent `indexed_at`
    timestamp across the connector's files (NULL → 0). Used by the metrics
    exporter to drive eichi_index_last_indexed_seconds{connector=...}.

    Schema: list of dicts:
        {"source": str, "file_count": int, "chunk_count": int,
         "last_indexed_unix": float}
    """
    # SQLite stores indexed_at as a TEXT/timestamp via CURRENT_TIMESTAMP, so
    # we coerce with strftime('%s', ...) which handles both ISO format and
    # the SQLite default 'YYYY-MM-DD HH:MM:SS' UTC form.
    rows = conn.execute(
        """
        SELECT
            f.source,
            COUNT(DISTINCT f.path) AS file_count,
            COALESCE((SELECT COUNT(*) FROM chunk_meta m
                      WHERE m.source = f.source), 0) AS chunk_count,
            COALESCE(MAX(strftime('%s', f.indexed_at)), 0) AS last_indexed_unix
        FROM files f
        GROUP BY f.source
        ORDER BY f.source
        """
    ).fetchall()
    out = []
    for source, fc, cc, last in rows:
        try:
            last_unix = float(last) if last is not None else 0.0
        except (TypeError, ValueError):
            last_unix = 0.0
        out.append({
            "source": source,
            "file_count": int(fc),
            "chunk_count": int(cc),
            "last_indexed_unix": last_unix,
        })
    # Also pick up sources that have chunk_meta rows but no files row (edge
    # case: connector wrote chunks then files row got pruned). Rare, but the
    # exporter should still see them.
    seen = {r["source"] for r in out}
    extra = conn.execute(
        "SELECT source, COUNT(*) FROM chunk_meta WHERE source NOT IN (SELECT source FROM files) GROUP BY source"
    ).fetchall()
    for source, cc in extra:
        if source not in seen:
            out.append({
                "source": source,
                "file_count": 0,
                "chunk_count": int(cc),
                "last_indexed_unix": 0.0,
            })
    return out


def list_files(
    conn: sqlite3.Connection, source: Optional[str] = None
) -> List[Tuple[str, str, int]]:
    """Return [(path, source, chunk_count), ...] sorted by path."""
    if source:
        rows = conn.execute(
            """
            SELECT f.path, f.source, COUNT(m.rowid)
            FROM files f
            LEFT JOIN chunk_meta m ON m.path = f.path
            WHERE f.source = ?
            GROUP BY f.path
            ORDER BY f.path
            """,
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT f.path, f.source, COUNT(m.rowid)
            FROM files f
            LEFT JOIN chunk_meta m ON m.path = f.path
            GROUP BY f.path
            ORDER BY f.path
            """
        ).fetchall()
    return [(p, s, c) for (p, s, c) in rows]


def infer_source(path: str) -> str:
    """Path-based source tag for the core indexer.

    Connectors will override this with explicit source tags later.
    """
    p = path.replace(os.sep, "/")
    if "/.claude/projects/" in p:
        return "transcripts"
    if "/Notes/" in p:
        return "obsidian"
    if "/memory/" in p:
        return "memory"
    return "file"


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise
