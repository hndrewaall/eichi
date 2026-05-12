"""eichi CLI dispatch.

Subcommands: index | query | reindex | stats | ls | rm. Each subcommand
supports --json. Index/reindex/rm support -n (dry run) and -v (verbose).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional

from . import EMBEDDING_DIM, EMBEDDING_MODEL, __version__
from .chunk import chunk_text, detect_mode
from .store import (
    DEFAULT_DB_PATH,
    add_chunks,
    ensure_fts_backfill,
    fts_count,
    infer_source,
    list_files,
    needs_reindex,
    open_db,
    remove_path,
    search,
    search_bm25,
    search_hybrid,
    sha256_file,
    stats,
    stats_by_source,
)

# File extensions the core indexer will read. Connectors will own their own
# pipelines; this is the default catch-all.
TEXT_EXTS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".log",
    ".py",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".zsh",
    ".bash",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".sql",
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".rb",
    ".lua",
    ".jsonl",
}

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB cap


def _walk(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip dotdirs that are noisy (.git, .venv, __pycache__).
        dirnames[:] = [
            d for d in dirnames if d not in {".git", ".venv", "__pycache__", "node_modules"}
        ]
        for name in filenames:
            yield Path(dirpath) / name


def _eligible(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() not in TEXT_EXTS:
        return False
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def _read(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _pick_hit_timestamp(h) -> tuple[Optional[float], str]:
    """Return ``(unix_ts, kind)`` — the timestamp we want to surface for
    this hit, plus a one-word label describing which column it came from.

    Preference: ``files.mtime`` (upstream / per-connector relevance time)
    when populated, else ``files.indexed_at`` (when eichi ingested it).
    Some connectors carry no upstream timestamp — those gracefully fall
    back to the ingest time. Returns ``(None, "")`` if neither is populated.
    """
    mtime = getattr(h, "mtime", 0.0) or 0.0
    indexed = getattr(h, "indexed_at_unix", 0.0) or 0.0
    if mtime > 0:
        return float(mtime), "mtime"
    if indexed > 0:
        return float(indexed), "indexed"
    return None, ""


def _human_age(seconds: float) -> str:
    """Compact ``(Nh ago)`` / ``(2d ago)`` / ``(in 5m)`` style label.

    Negative seconds (timestamp in the future, clock skew) collapse to
    ``"in <abs>"``; tiny intervals (<60s) round to ``"now"``.
    """
    if seconds is None:
        return ""
    s = float(seconds)
    sign = ""
    if s < 0:
        sign = "in "
        s = -s
    if s < 60:
        return "now" if not sign else "in <1m"
    units = (
        (60 * 60 * 24 * 365, "y"),
        (60 * 60 * 24 * 30, "mo"),
        (60 * 60 * 24 * 7, "w"),
        (60 * 60 * 24, "d"),
        (60 * 60, "h"),
        (60, "m"),
    )
    for size, label in units:
        if s >= size:
            n = int(s // size)
            if sign:
                return f"in {n}{label}"
            return f"{n}{label} ago"
    return ""


def _format_hit_timestamp(h) -> str:
    """Build the human-readable timestamp suffix for a CLI line.

    Format: ``[YYYY-MM-DD HH:MM ET] (Nh ago)`` if a timestamp is known.
    Empty string if neither ``mtime`` nor ``indexed_at_unix`` is populated.
    The kind label (``mtime`` vs ``indexed``) is omitted from the visible
    output but kept in the JSON shape (``ts_kind``) so downstream callers
    can disambiguate when they care.

    For cluster rows (``kind="cluster"``) the absolute portion is rendered
    as a range ``[YYYY-MM-DD HH:MM–HH:MM ET]`` when ``mtime_end`` is
    populated and lands on the same day; otherwise the start-time form is
    used.
    """
    import time as _time
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except Exception:  # pragma: no cover — zoneinfo always available on 3.12+
        et = None

    ts, _kind = _pick_hit_timestamp(h)
    if ts is None:
        return ""
    if et is not None:
        dt = datetime.fromtimestamp(ts, tz=et)
        stamp = dt.strftime("%Y-%m-%d %H:%M ET")
    else:
        dt = datetime.utcfromtimestamp(ts)
        stamp = dt.strftime("%Y-%m-%d %H:%M UTC")

    # Cluster rendering: append "–HH:MM" suffix when mtime_end is on the
    # same day as the start, otherwise the full end-stamp.
    cluster_kind = getattr(h, "kind", "") or ""
    mtime_end = getattr(h, "mtime_end", 0.0) or 0.0
    if cluster_kind == "cluster" and mtime_end > 0:
        if et is not None:
            dt_end = datetime.fromtimestamp(mtime_end, tz=et)
        else:
            dt_end = datetime.utcfromtimestamp(mtime_end)
        if dt_end.date() == dt.date():
            stamp = (
                f"{dt.strftime('%Y-%m-%d %H:%M')}"
                f"–{dt_end.strftime('%H:%M')} "
                f"{'ET' if et is not None else 'UTC'}"
            )
        else:
            stamp = (
                f"{dt.strftime('%Y-%m-%d %H:%M')}"
                f"–{dt_end.strftime('%Y-%m-%d %H:%M')} "
                f"{'ET' if et is not None else 'UTC'}"
            )

    age = _human_age(_time.time() - ts)
    if age:
        return f"[{stamp}] ({age})"
    return f"[{stamp}]"


# --- similarity / relevance band ---------------------------------------------
#
# vec0 (sqlite-vec) returns L2 distance — smaller = closer. The raw number
# (typically 0.6–1.4 for normalized sentence-transformer embeddings) is opaque
# on the CLI and inside a web frontend. We project it into a 0–1 similarity
# score and a named band for human-readable surfaces.
#
# Empirical thresholds (mixed chat / notes / library-metadata corpus):
# ``≤0.7`` strong, ``0.8–1.05`` moderate, ``1.05–1.20`` weak,
# ``>1.20`` no-relationship. The transform pegs distance=0 → 1.0 and
# distance=THRESHOLD → 0.0 with linear interpolation between, then clamps
# both ends.
#
# THRESHOLD sits just above the empirical "no relationship" line — anything
# weaker than that should land at similarity 0 even if the raw distance is
# 1.5 / 1.8 / 2.0. Exposed as a constant so downstream callers (a web
# frontend, a custom CLI wrapper) can import + reuse.
SIMILARITY_THRESHOLD: float = 1.4

# Band cutoffs are expressed in raw distance space (NOT similarity space)
# because the source signal lives there. ``BAND_CUTOFFS`` is the list of
# upper bounds for each band, paired with ``BAND_LABELS`` 1:1. Walking
# them in order yields the first band whose cutoff is >= the distance.
# Distance > final cutoff → ``distant`` (the catch-all).
#
# Tunable: bumping the moderate cutoff to 1.10 would broaden the moderate
# band; tightening strong to 0.6 would narrow the green band. Keep the
# CLI + frontend in sync by editing here only.
BAND_LABELS: tuple[str, ...] = ("strong", "moderate", "weak", "distant")
BAND_CUTOFFS: tuple[float, ...] = (0.7, 1.05, 1.20)  # one fewer than labels


def similarity_from_distance(distance: float) -> float:
    """Project a vec0 L2 distance to a 0–1 similarity score.

    ``similarity = max(0.0, min(1.0, (THRESHOLD - distance) / THRESHOLD))``

    distance=0   → similarity=1.0
    distance=THRESHOLD → similarity=0.0
    distance>THRESHOLD → clamps at 0.0 (no negative similarity)

    Negative distances (input came from a BM25-only hit in hybrid mode —
    fts5's bm25() returns negative scores where smaller is better) are
    NOT real L2 distances and can't be projected meaningfully. Treat as
    "unknown" → similarity=0.0 so the chip at least doesn't paint a
    misleadingly-high green strong-match. The frontend still has access
    to the raw `score` for diagnostics. Vec0 distances are always >= 0.

    Monotonic on the valid input range — closer distances always produce
    higher similarity. Does NOT change ranking order; both raw distance
    and similarity sort the same way (just opposite directions).
    """
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if d < 0.0:
        # BM25-only hit (or other non-distance scoring) — bail out.
        return 0.0
    sim = (SIMILARITY_THRESHOLD - d) / SIMILARITY_THRESHOLD
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def relevance_band(distance: float) -> str:
    """Categorize a raw vec0 distance into a named band.

    Walks ``BAND_CUTOFFS`` in order; returns the label for the first
    cutoff the distance is at-or-below. Falls through to the final
    ``BAND_LABELS`` entry (``distant``) when no cutoff matches.

    Negative distances (BM25-only hit from hybrid retrieval — see
    :func:`similarity_from_distance`) are not real L2 distances and
    can't be banded; collapse to ``distant`` rather than mis-painting
    them as ``strong``. Defensive: garbage / None input also returns
    ``distant``.
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return BAND_LABELS[-1]
    if d < 0.0:
        # BM25-only hit — no meaningful band.
        return BAND_LABELS[-1]
    for cutoff, label in zip(BAND_CUTOFFS, BAND_LABELS):
        if d <= cutoff:
            return label
    return BAND_LABELS[-1]


def _cluster_size_from_text(text: str) -> int:
    """Heuristic: count the number of ``[<thread>] <sender>: …`` headers
    embedded in a cluster doc's text.

    Cluster rows are joined with ``\n``-separated headers; each leading
    header line starts with ``[``. We use this for the human-readable
    ``[cluster, N msgs, ...]`` prefix without needing an extra column.
    Returns ``1`` if no headers are found (defensive — should never happen
    for a properly-formatted cluster doc).
    """
    if not text:
        return 1
    n = sum(1 for line in text.split("\n") if line.startswith("["))
    return max(1, n)


def _print_hit(h, json_out: bool, raw_score: bool = False) -> str:
    snippet = h.text.replace("\n", " ").strip()
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    cluster_kind = getattr(h, "kind", "") or ""
    cluster_id = getattr(h, "cluster_id", "") or ""
    cluster_size = (
        _cluster_size_from_text(h.text) if cluster_kind == "cluster" else 0
    )
    mtime_end_val = float(getattr(h, "mtime_end", 0.0) or 0.0)
    # Similarity / band — derived from the raw vec0 distance so every
    # caller gets the same projection (CLI text, CLI JSON, web JSON).
    similarity_val = similarity_from_distance(h.score)
    band_val = relevance_band(h.score)
    if json_out:
        ts, kind = _pick_hit_timestamp(h)
        # Strip the outer "[...]" wrapper but keep the optional "(Nh ago)"
        # suffix in the JSON ts_human field — easier for downstream
        # tools to inline without re-formatting.
        if ts is not None:
            full = _format_hit_timestamp(h)
            # full looks like "[YYYY-MM-DD HH:MM ET] (3h ago)" or "[…]"
            close_idx = full.find("]")
            if close_idx >= 0:
                ts_human = (full[1:close_idx] + full[close_idx + 1 :]).strip()
            else:
                ts_human = full
        else:
            ts_human = ""
        # Date-aware fields. Surfaced as ``library_added_at_unix`` /
        # ``release_year``. 0 sentinel means "unknown / unset" — never
        # fabricate from mtime.
        library_added_at_val = float(
            getattr(h, "library_added_at", 0.0) or 0.0
        )
        release_year_val = int(getattr(h, "release_year", 0) or 0)
        return json.dumps(
            {
                # ``score`` is the raw vec0 L2 distance — kept for backwards
                # compat (older clients sort / filter on this). ``similarity``
                # is the projected 0–1 score (smaller-distance == higher
                # similarity). ``relevance_band`` is the named band derived
                # from the same distance.
                "score": h.score,
                "similarity": similarity_val,
                "relevance_band": band_val,
                "source": h.source,
                "path": h.path,
                "chunk_idx": h.chunk_idx,
                "offset": h.offset,
                "snippet": snippet,
                # Human-friendly + machine-readable: emit BOTH the picked
                # timestamp + which column it came from, plus the raw
                # mtime / indexed_at_unix so callers can re-derive
                # whatever they want without losing precision.
                "mtime": float(getattr(h, "mtime", 0.0) or 0.0),
                "indexed_at_unix": float(getattr(h, "indexed_at_unix", 0.0) or 0.0),
                "ts": ts,
                "ts_kind": kind,
                "ts_human": ts_human,
                # Cluster fields — always emitted; defaults are empty/0.
                "kind": cluster_kind,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "mtime_end": mtime_end_val,
                # Date-aware fields. 0 sentinel for unknown/unset.
                "library_added_at_unix": library_added_at_val,
                "release_year": release_year_val,
                # Hybrid retrieval provenance. NULL when this doc did
                # not surface in that pass — only set by the hybrid
                # query path (vector- or bm25-only modes leave both
                # NULL). 1-based ranks.
                "vec_rank": getattr(h, "vec_rank", None),
                "bm25_rank": getattr(h, "bm25_rank", None),
            }
        )
    ts_str = _format_hit_timestamp(h)
    # Source/kind tag column. For cluster rows we expand it from
    # "[signal-chat]" to "[signal-chat cluster, 5 msgs]" so the line
    # is self-describing at a glance.
    if cluster_kind == "cluster":
        kind_tag = f"[{h.source} cluster, {cluster_size} msgs]"
    elif cluster_kind == "msg":
        kind_tag = f"[{h.source} msg]"
    else:
        kind_tag = f"[{h.source}]"
    # Default text rendering: similarity + band ("0.61 [moderate]"). The
    # raw vec0 distance is opaque — readers can't tell whether 1.157 is
    # close or far without context. ``--raw-score`` re-exposes the raw
    # distance for diagnostics / regression tests.
    if raw_score:
        score_field = f"{h.score:.4f}"
    else:
        score_field = f"{similarity_val:.2f} [{band_val}]"
    if ts_str:
        return (
            f"{h.path}:{h.offset}\t{score_field}\t{kind_tag}\t"
            f"{ts_str}\t{snippet}"
        )
    return f"{h.path}:{h.offset}\t{score_field}\t{kind_tag}\t{snippet}"


# --- subcommand handlers -----------------------------------------------------


def _connector_state_path() -> Path:
    """Where the connector incremental cursor lives.

    One JSON file, keyed by connector name. Per-connector slices keep
    re-runs idempotent (only re-emit changed content). Configurable via
    ``$EICHI_CONNECTOR_STATE``.
    """
    override = os.environ.get("EICHI_CONNECTOR_STATE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "eichi" / "connector-state.json"


def _load_connector_state(path: Path, key: str) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    slot = data.get(key)
    return slot if isinstance(slot, dict) else {}


def _save_connector_state(path: Path, key: str, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                full = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(full, dict):
                    full = {}
            except (OSError, json.JSONDecodeError):
                full = {}
        else:
            full = {}
        full[key] = state
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(full, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _load_corpus_config(name: str) -> dict:
    """Read the [[corpus]] block for ``name`` from eichi.toml. Returns {}
    when the file or the named block is missing.
    """
    try:
        from .config import load as _load_cfg
    except Exception:
        return {}
    try:
        cfg = _load_cfg()
    except Exception:
        return {}
    for corpus in cfg.corpora:
        if corpus.name == name:
            return {
                "path": str(corpus.path) if corpus.path else None,
                "extensions": corpus.extensions,
            }
    return {}


def _cmd_index_corpus(args) -> int:
    """Run a named connector and pipe its JSONL output into the index.

    Connectors live under :mod:`eichi.connectors` and are registered in
    that package's ``REGISTRY`` dict. Each connector exposes
    ``iter_documents(state, config) -> Iterator[dict]``.
    """
    from . import embed
    from .connectors import REGISTRY
    from .store import add_chunks, sha256_text

    name = args.corpus
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        print(
            f"eichi: unknown corpus {name!r}. Known: {known}",
            file=sys.stderr,
        )
        return 2

    state_path = _connector_state_path()
    state = _load_connector_state(state_path, name)
    config = _load_corpus_config(name)

    iter_docs = REGISTRY[name]
    conn = open_db(args.db)
    cur = conn.cursor()

    indexed = 0
    skipped = 0
    new_chunks = 0
    errors = 0

    for doc in iter_docs(state=state, config=config):
        doc_id = doc.get("doc_id")
        text = doc.get("text")
        source = doc.get("source") or name
        if not doc_id or not text:
            errors += 1
            continue
        mtime = float(doc.get("mtime") or 0.0)
        meta = doc.get("metadata") or {}
        kind = meta.get("kind") or doc.get("kind")
        cluster_id = meta.get("cluster_id") or doc.get("cluster_id")
        mtime_end_raw = meta.get("mtime_end", doc.get("mtime_end"))
        try:
            mtime_end_val = (
                float(mtime_end_raw) if mtime_end_raw is not None else None
            )
        except (TypeError, ValueError):
            mtime_end_val = None
        library_added_at_raw = meta.get(
            "library_added_at", doc.get("library_added_at")
        )
        try:
            library_added_at_val = (
                float(library_added_at_raw)
                if library_added_at_raw is not None
                else None
            )
        except (TypeError, ValueError):
            library_added_at_val = None

        file_hash = sha256_text(text)

        if not getattr(args, "force", False):
            existing = cur.execute(
                "SELECT file_hash FROM files WHERE path = ?", (doc_id,)
            ).fetchone()
            if existing and existing[0] == file_hash:
                skipped += 1
                if args.verbose:
                    print(f"skip (clean): {doc_id}", file=sys.stderr)
                continue

        if args.dry_run:
            indexed += 1
            if args.verbose:
                print(f"[dry-run] would index: {doc_id}", file=sys.stderr)
            continue

        mode = "md" if doc.get("md") else "text"
        chunks = chunk_text(text, mode=mode)
        if not chunks:
            skipped += 1
            continue
        texts = [c[2] for c in chunks]
        embeddings = embed.encode(texts, batch_size=32)
        n = add_chunks(
            conn,
            source=source,
            path=doc_id,
            mtime=mtime,
            file_hash=file_hash,
            chunks=chunks,
            embeddings=embeddings,
            kind=kind,
            cluster_id=cluster_id,
            mtime_end=mtime_end_val,
            library_added_at=library_added_at_val,
        )
        new_chunks += n
        indexed += 1
        if args.verbose:
            print(f"indexed: {doc_id} ({n} chunks)", file=sys.stderr)

    if not args.dry_run:
        _save_connector_state(state_path, name, state)

    payload = {
        "corpus": name,
        "docs_indexed": indexed,
        "docs_skipped_clean": skipped,
        "chunks_added": new_chunks,
        "errors": errors,
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"corpus={name}: indexed {indexed} docs "
            f"({new_chunks} chunks), skipped {skipped} clean, "
            f"{errors} errors"
            + (" [DRY RUN]" if args.dry_run else "")
        )
    return 0 if errors == 0 else 1


def cmd_index(args) -> int:
    from . import embed

    if getattr(args, "corpus", None):
        return _cmd_index_corpus(args)

    if not args.path:
        print(
            "eichi: provide a path or --corpus <name>",
            file=sys.stderr,
        )
        return 2

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"eichi: path not found: {root}", file=sys.stderr)
        return 2
    conn = open_db(args.db)
    candidates = [p for p in _walk(root) if _eligible(p)]
    if args.verbose:
        print(f"scan: {len(candidates)} eligible files under {root}", file=sys.stderr)
    indexed = 0
    skipped = 0
    new_chunks = 0
    for p in candidates:
        path_str = str(p)
        try:
            st = p.stat()
        except OSError:
            continue
        text = _read(p)
        if text is None:
            continue
        file_hash = sha256_file(path_str)
        if not needs_reindex(conn, path_str, st.st_mtime, file_hash):
            skipped += 1
            if args.verbose:
                print(f"skip (clean): {path_str}", file=sys.stderr)
            continue
        if args.dry_run:
            indexed += 1
            print(f"[dry-run] would index: {path_str}", file=sys.stderr)
            continue
        mode = detect_mode(path_str)
        chunks = chunk_text(text, mode=mode)
        if not chunks:
            skipped += 1
            continue
        texts = [c[2] for c in chunks]
        embeddings = embed.encode(texts, batch_size=32)
        source = infer_source(path_str)
        n = add_chunks(
            conn,
            source=source,
            path=path_str,
            mtime=st.st_mtime,
            file_hash=file_hash,
            chunks=chunks,
            embeddings=embeddings,
        )
        new_chunks += n
        indexed += 1
        if args.verbose:
            print(f"indexed: {path_str} ({n} chunks)", file=sys.stderr)
    payload = {
        "root": str(root),
        "files_indexed": indexed,
        "files_skipped_clean": skipped,
        "chunks_added": new_chunks,
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"indexed {indexed} files ({new_chunks} chunks), "
            f"skipped {skipped} clean files"
            + (" [DRY RUN]" if args.dry_run else "")
        )
    return 0


def cmd_index_stream(args) -> int:
    """Index synthetic documents streamed as JSONL on stdin.

    Each line: {"source": str, "doc_id": str, "text": str,
                "mtime": float (optional), "metadata": object (optional)}

    The doc_id becomes the `path` column (unique identifier) — connectors
    are responsible for namespacing their ids (e.g. "chat:dm:alice:<msg_ts>").

    Source on the line wins; --source is a default for lines that omit it.

    Idempotent: a doc_id whose content_hash matches the stored chunk-0 hash
    is skipped (no re-embed). Use --force to ignore the dedup check.
    """
    from . import embed
    from .store import sha256_text

    conn = open_db(args.db)
    default_source = args.source

    indexed = 0
    skipped = 0
    new_chunks = 0
    errors = 0
    line_no = 0

    cur = conn.cursor()

    # Stream JSONL lines.
    for line in sys.stdin:
        line_no += 1
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as e:
            errors += 1
            print(f"eichi: line {line_no}: bad JSON: {e}", file=sys.stderr)
            continue
        text = doc.get("text")
        doc_id = doc.get("doc_id")
        if not text or not doc_id:
            errors += 1
            if args.verbose:
                print(
                    f"eichi: line {line_no}: missing text or doc_id",
                    file=sys.stderr,
                )
            continue
        source = doc.get("source") or default_source
        if not source:
            errors += 1
            print(
                f"eichi: line {line_no}: no source (provide --source or per-line)",
                file=sys.stderr,
            )
            continue
        mtime = float(doc.get("mtime", 0.0))
        # Optional clustering metadata. May arrive at the top level (legacy)
        # or nested under "metadata" (preferred shape, mirrors connector
        # base.Document.metadata).
        meta_block = doc.get("metadata") or {}
        kind = doc.get("kind") or meta_block.get("kind")
        cluster_id = doc.get("cluster_id") or meta_block.get("cluster_id")
        mtime_end_raw = doc.get("mtime_end", meta_block.get("mtime_end"))
        try:
            mtime_end_val = (
                float(mtime_end_raw) if mtime_end_raw is not None else None
            )
        except (TypeError, ValueError):
            mtime_end_val = None
        # Date-aware fields. Same dual-shape rule as kind/cluster_id —
        # accepted at top level OR nested under "metadata". NULL when
        # absent so connectors that don't know either value don't pollute
        # downstream filters. ``release_year`` is normalized to int (or
        # None on bad input).
        library_added_at_raw = doc.get(
            "library_added_at", meta_block.get("library_added_at")
        )
        try:
            library_added_at_val = (
                float(library_added_at_raw)
                if library_added_at_raw is not None
                else None
            )
        except (TypeError, ValueError):
            library_added_at_val = None
        release_year_raw = doc.get(
            "release_year", meta_block.get("release_year")
        )
        try:
            release_year_val = (
                int(release_year_raw)
                if release_year_raw is not None and str(release_year_raw).strip() != ""
                else None
            )
        except (TypeError, ValueError):
            release_year_val = None
        # Use sha256 of text as the file_hash for dedup.
        file_hash = sha256_text(text)

        # Dedup: if doc_id is already present with matching hash, skip.
        if not args.force:
            existing = cur.execute(
                "SELECT file_hash FROM files WHERE path = ?", (doc_id,)
            ).fetchone()
            if existing and existing[0] == file_hash:
                skipped += 1
                if args.verbose:
                    print(f"skip (clean): {doc_id}", file=sys.stderr)
                continue

        if args.dry_run:
            indexed += 1
            print(f"[dry-run] would index: {doc_id}", file=sys.stderr)
            continue

        # Chunk + embed. Chat messages tend to be short — text mode is fine.
        # Connectors that want md-mode should mark text with leading headings.
        mode = "md" if doc.get("md") else "text"
        chunks = chunk_text(text, mode=mode)
        if not chunks:
            skipped += 1
            continue
        texts = [c[2] for c in chunks]
        embeddings = embed.encode(texts, batch_size=32)
        n = add_chunks(
            conn,
            source=source,
            path=doc_id,
            mtime=mtime,
            file_hash=file_hash,
            chunks=chunks,
            embeddings=embeddings,
            kind=kind,
            cluster_id=cluster_id,
            mtime_end=mtime_end_val,
            library_added_at=library_added_at_val,
            release_year=release_year_val,
        )
        new_chunks += n
        indexed += 1
        if args.verbose:
            print(f"indexed: {doc_id} ({n} chunks)", file=sys.stderr)

    payload = {
        "docs_indexed": indexed,
        "docs_skipped_clean": skipped,
        "chunks_added": new_chunks,
        "errors": errors,
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"indexed {indexed} docs ({new_chunks} chunks), "
            f"skipped {skipped} clean, {errors} errors"
            + (" [DRY RUN]" if args.dry_run else "")
        )
    return 0 if errors == 0 else 1


def _query_log_path() -> Path:
    """Append-only JSONL log of eichi queries — local-only telemetry used by an
    optional metrics exporter to populate the RTT histogram. Each line:
    {"ts": float, "elapsed_ms": float, "k": int, "hits": int,
    "source": str|null, "encode_ms": float, "search_ms": float,
    "caller": str, "user_uid": str|null, "role": str|null}. Path is
    configurable via ``EICHI_QUERY_LOG``. Set ``EICHI_NO_QUERY_LOG=1`` to
    disable entirely. The file never leaves the local machine.

    ``caller`` (always present): one of ``cli`` / ``web`` / ``agent`` /
    ``mainloop`` / ``_other``. Identifies which interface produced the
    query so the exporter can split metrics across CLI / web / agent /
    mainloop traffic. ``user_uid`` + ``role`` are only set when
    ``caller=web`` — forwarded through from an auth-gate's
    ``X-Auth-Uid`` / ``X-Auth-Role`` headers — and are NULL for the
    bot-driven callers (cli/agent/mainloop).
    """
    override = os.environ.get("EICHI_QUERY_LOG") or os.environ.get("VSEARCH_QUERY_LOG")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "eichi" / "query.log"


# Allowed caller values. Anything outside this set is collapsed to ``_other``
# at log-write time (defense in depth — the exporter does the same on read).
ALLOWED_CALLERS = {"cli", "web", "agent", "mainloop"}


def _resolve_caller(args) -> str:
    """Pick the caller label for this query.

    Resolution order:
      1. ``--caller=<value>`` flag on the query subcommand (explicit wins).
      2. ``$EICHI_CALLER`` (or legacy ``$VSEARCH_CALLER``) env var, set by
         agent harnesses / wrapper programs.
      3. Default ``cli`` — anything reaching the CLI without an explicit
         caller is, by definition, an interactive CLI invocation.
    Anything outside :data:`ALLOWED_CALLERS` collapses to ``_other``.
    """
    val = (
        getattr(args, "caller", None)
        or os.environ.get("EICHI_CALLER")
        or os.environ.get("VSEARCH_CALLER")
        or "cli"
    )
    val = str(val).strip().lower()
    if val not in ALLOWED_CALLERS:
        return "_other"
    return val


def _log_query(record: dict) -> None:
    """Best-effort append to the RTT log. Never raise — query path must not
    fail because metrics logging hit a disk error / permission issue."""
    try:
        path = _query_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _dedupe_cluster_overlap(hits, *, score_window: float = 0.05):
    """Drop msg hits already represented by a higher-or-equally-ranked cluster.

    Walks the hit list in score order (vec0 distance: smaller = closer).
    For each cluster hit seen, remember its ``cluster_id``. A subsequent
    msg hit whose ``cluster_id`` matches one of those cluster ids is
    suppressed when the cluster's score is at most ``score_window``
    fractional units worse than the msg (i.e. the cluster ranks at or
    near the same relevance — already represented). Other msg hits and
    msgs from clusters not in the result set pass through unchanged.

    ``score_window`` is the multiplicative slack — 0.05 = 5%. With vec0
    distances typically in [0, 2], a 5% window is a tight ~0.025 band.
    """
    cluster_scores: dict[str, float] = {}
    for h in hits:
        if (getattr(h, "kind", "") or "") == "cluster":
            cid = getattr(h, "cluster_id", "") or ""
            if cid and cid not in cluster_scores:
                cluster_scores[cid] = float(h.score)
    out = []
    for h in hits:
        kind = getattr(h, "kind", "") or ""
        if kind == "msg":
            cid = getattr(h, "cluster_id", "") or ""
            if cid and cid in cluster_scores:
                cluster_score = cluster_scores[cid]
                msg_score = float(h.score)
                # Suppress when cluster ranked higher OR within `score_window`.
                # vec0 distance: smaller = closer, so cluster wins iff
                # cluster_score <= msg_score * (1 + score_window).
                if cluster_score <= msg_score * (1 + score_window):
                    continue
        out.append(h)
    return out


def _parse_duration(spec: Optional[str]) -> Optional[float]:
    """Parse ``30d`` / ``6m`` / ``1y`` / ``12h`` / ``45min`` style durations
    into seconds. Returns None for empty/None input. Raises ``ValueError``
    for invalid input.

    Suffixes (case-insensitive): ``s`` seconds, ``min`` minutes, ``m``
    months (~30 days), ``h`` hours, ``d`` days, ``w`` weeks, ``y`` years.
    Note: bare ``m`` means MONTHS (matches the ``--added-since 6m`` UX
    expectation) — use ``min`` for minutes. We never expect a ``--added-
    since`` shorter than days in practice.
    """
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if not s:
        return None
    # Order matters: check 'min' before 'm'.
    units: list[tuple[str, float]] = [
        ("min", 60.0),
        ("s", 1.0),
        ("h", 60.0 * 60.0),
        ("d", 60.0 * 60.0 * 24.0),
        ("w", 60.0 * 60.0 * 24.0 * 7.0),
        ("mo", 60.0 * 60.0 * 24.0 * 30.0),
        ("m", 60.0 * 60.0 * 24.0 * 30.0),
        ("y", 60.0 * 60.0 * 24.0 * 365.0),
    ]
    for suffix, mult in units:
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            if not num:
                raise ValueError(f"missing number in duration: {spec!r}")
            try:
                return float(num) * mult
            except ValueError as e:
                raise ValueError(
                    f"bad duration {spec!r}: {e}"
                ) from None
    # No suffix → treat as bare seconds.
    try:
        return float(s)
    except ValueError:
        raise ValueError(
            f"unrecognized duration {spec!r} "
            f"(use Nd / Nh / Nw / Nmo / Ny / Nmin)"
        ) from None


def cmd_query(args) -> int:
    import time as _time
    from . import embed

    conn = open_db(args.db)

    # Resolve --added-since DUR → unix epoch cutoff.
    added_since_unix: Optional[float] = None
    if getattr(args, "added_since", None):
        try:
            secs = _parse_duration(args.added_since)
        except ValueError as e:
            print(f"eichi: {e}", file=sys.stderr)
            return 2
        if secs is not None:
            added_since_unix = _time.time() - secs

    # Validate --year-min / --year-max.
    year_min = getattr(args, "year_min", None)
    year_max = getattr(args, "year_max", None)
    if year_min is not None and year_max is not None and year_min > year_max:
        print(
            f"eichi: --year-min {year_min} > --year-max {year_max}",
            file=sys.stderr,
        )
        return 2

    sort = getattr(args, "sort", "relevance") or "relevance"
    retrieval = getattr(args, "retrieval", "hybrid") or "hybrid"
    if retrieval not in ("hybrid", "vector", "bm25"):
        print(
            f"eichi: --retrieval must be one of hybrid|vector|bm25 "
            f"(got {retrieval!r})",
            file=sys.stderr,
        )
        return 2

    # Over-fetch when dedupe is on, so we don't return < k after suppression.
    fetch_k = args.k * 3 if not args.granular else args.k

    t0 = _time.monotonic()
    # bm25-only mode skips the embedding round-trip entirely — useful
    # for benchmarking + for the FTS-catches-literal-term test cases.
    if retrieval == "bm25":
        qvec = None  # type: ignore[assignment]
        t1 = _time.monotonic()
        # Lazy backfill — populate fts5 for any chunk_meta rows that
        # don't yet have an fts entry. Cheap no-op once steady-state.
        ensure_fts_backfill(conn)
        hits = search_bm25(
            conn,
            args.q,
            k=fetch_k,
            source=args.source,
            year_min=year_min,
            year_max=year_max,
            added_since_unix=added_since_unix,
        )
    else:
        qvec = embed.encode_one(args.q)
        t1 = _time.monotonic()
        if retrieval == "vector":
            hits = search(
                conn,
                qvec,
                k=fetch_k,
                source=args.source,
                year_min=year_min,
                year_max=year_max,
                added_since_unix=added_since_unix,
                sort=sort,
            )
        else:  # hybrid
            if fts_count(conn) == 0:
                ensure_fts_backfill(conn)
            hits = search_hybrid(
                conn,
                qvec,
                args.q,
                k=fetch_k,
                source=args.source,
                year_min=year_min,
                year_max=year_max,
                added_since_unix=added_since_unix,
                sort=sort,
            )
    if not args.granular:
        hits = _dedupe_cluster_overlap(hits)
    hits = hits[: args.k]
    t2 = _time.monotonic()

    encode_ms = (t1 - t0) * 1000.0
    search_ms = (t2 - t1) * 1000.0
    elapsed_ms = (t2 - t0) * 1000.0

    if not (os.environ.get("EICHI_NO_QUERY_LOG") or os.environ.get("VSEARCH_NO_QUERY_LOG")):
        _log_query({
            "ts": _time.time(),
            "elapsed_ms": round(elapsed_ms, 3),
            "encode_ms": round(encode_ms, 3),
            "search_ms": round(search_ms, 3),
            "k": int(args.k),
            "hits": len(hits),
            "source": args.source,
            "caller": _resolve_caller(args),
        })

    raw_score = bool(getattr(args, "raw_score", False))
    if args.json:
        for h in hits:
            print(_print_hit(h, json_out=True, raw_score=raw_score))
    else:
        if not hits:
            print("(no results)")
        for h in hits:
            print(_print_hit(h, json_out=False, raw_score=raw_score))
    return 0


def cmd_reindex(args) -> int:
    conn = open_db(args.db)
    if args.path:
        target = str(Path(args.path).expanduser().resolve())
        if args.dry_run:
            print(f"[dry-run] would wipe rows under: {target}", file=sys.stderr)
            removed = 0
        else:
            removed = remove_path(conn, target)
        # Re-run index over target.
        # Build args for cmd_index — preserve --json/--verbose.
        sub = argparse.Namespace(
            path=target,
            json=args.json,
            verbose=args.verbose,
            dry_run=args.dry_run,
            db=args.db,
        )
        rc = cmd_index(sub)
        if not args.json:
            print(f"reindex: removed {removed} prior chunks under {target}")
        return rc
    # Full wipe — drop everything and rebuild only the existing files set.
    if args.dry_run:
        print("[dry-run] would wipe all chunks/files", file=sys.stderr)
        return 0
    paths = [r[0] for r in list_files(conn)]
    for p in paths:
        remove_path(conn, p)
    if args.json:
        print(json.dumps({"wiped_files": len(paths)}))
    else:
        print(f"wiped {len(paths)} files; re-run `eichi index <path>` to rebuild")
    return 0


def cmd_stats(args) -> int:
    conn = open_db(args.db)
    s = stats(conn)
    if args.by_source:
        rows = stats_by_source(conn)
        if args.json:
            print(json.dumps({**s, "by_source": rows}, default=str))
            return 0
        print(f"{'source':<24} {'files':>8} {'chunks':>10} {'last_indexed_unix':>20}")
        print("-" * 64)
        for r in rows:
            print(f"{r['source']:<24} {r['file_count']:>8} {r['chunk_count']:>10} {r['last_indexed_unix']:>20.0f}")
        return 0
    if args.json:
        print(json.dumps(s, default=str))
        return 0
    print(f"db:               {s['db_path']}")
    print(f"db size:          {s['db_size_bytes']} bytes")
    print(f"embedding model:  {s['embedding_model']} (dim={EMBEDDING_DIM})")
    print(f"schema version:   {s['schema_version']}")
    print(f"files indexed:    {s['file_count']}")
    print(f"chunks total:     {s['chunk_count']}")
    print(f"sources:          {', '.join(s['sources']) or '(none)'}")
    print(f"last indexed at:  {s['last_indexed']}")
    return 0


def cmd_ls(args) -> int:
    conn = open_db(args.db)
    rows = list_files(conn, source=args.source)
    if args.json:
        print(json.dumps([{"path": p, "source": s, "chunks": c} for p, s, c in rows]))
        return 0
    for p, s, c in rows:
        print(f"{s}\t{c}\t{p}")
    return 0


def cmd_rm(args) -> int:
    conn = open_db(args.db)
    target = str(Path(args.path).expanduser().resolve())
    if args.dry_run:
        # Count what would be removed without changing anything.
        cur = conn.cursor()
        n = cur.execute(
            "SELECT COUNT(*) FROM chunk_meta WHERE path = ? OR path LIKE ?",
            (target, target.rstrip("/") + "/%"),
        ).fetchone()[0]
        if args.json:
            print(json.dumps({"would_remove_chunks": n, "path": target}))
        else:
            print(f"[dry-run] would remove {n} chunks under {target}")
        return 0
    n = remove_path(conn, target)
    if args.json:
        print(json.dumps({"removed_chunks": n, "path": target}))
    else:
        print(f"removed {n} chunks under {target}")
    return 0


# --- argparse wiring ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eichi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            f"""\
            eichi — local sqlite-vec + sentence-transformers vector search.

            Default DB: {DEFAULT_DB_PATH}
            Embedding model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})

            Subcommands:
              index    Index a file or directory (idempotent, delta-only).
              query    Search the index. Top-K results.
              reindex  Wipe and rebuild for a path or the entire DB.
              stats    Show row count, sources, last-indexed time.
              ls       List indexed files (debug).
              rm       Remove a file or directory from the index.
            """
        ),
    )
    p.add_argument("--version", action="version", version=f"eichi {__version__}")
    p.add_argument("--db", help="override DB path", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser(
        "index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Index a file or directory, or a named connector corpus",
        description=textwrap.dedent(
            """\
            Index documents into eichi's database.

            Two forms:
              eichi index <path>             — walk a filesystem tree
              eichi index --corpus <name>    — run a named connector

            Connectors shipped with eichi:
              claude-jsonl         Claude Code session JSONL transcripts
                                   (default root: ~/.claude/projects)
              claude-watch-queue   claude-watch session-task queue items
                                   (default path: ~/.config/session/queue.json)

            Connector paths can be overridden via eichi.toml's
            [[corpus]] block (set ``path = ...``) or per-connector env
            vars (EICHI_CLAUDE_JSONL_ROOT, EICHI_CLAUDE_QUEUE_PATH).
            """
        ),
    )
    pi.add_argument("path", nargs="?")
    pi.add_argument(
        "--corpus",
        help=(
            "name of a built-in connector to run instead of walking a "
            "filesystem path (e.g. 'claude-jsonl' or "
            "'claude-watch-queue')"
        ),
    )
    pi.add_argument("-v", "--verbose", action="store_true")
    pi.add_argument("-n", "--dry-run", action="store_true")
    pi.add_argument(
        "--force",
        action="store_true",
        help="connector mode only: re-embed even if hash matches",
    )
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=cmd_index)

    pis = sub.add_parser(
        "index-stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Index synthetic documents streamed as JSONL on stdin",
        description=textwrap.dedent(
            """\
            Read JSONL on stdin; each line is one document. Schema:

              {"source": str, "doc_id": str, "text": str,
               "mtime": float (optional), "md": bool (optional)}

            doc_id is the unique key (used as `path` in the schema). Source
            on the line wins; --source is the default for lines that omit it.
            Idempotent: skips docs whose content hash matches what's stored.
            """
        ),
    )
    pis.add_argument("--source", help="default source tag for lines without one")
    pis.add_argument("-v", "--verbose", action="store_true")
    pis.add_argument("-n", "--dry-run", action="store_true")
    pis.add_argument("--force", action="store_true",
                     help="re-embed even if hash matches")
    pis.add_argument("--json", action="store_true")
    pis.set_defaults(func=cmd_index_stream)

    pq = sub.add_parser(
        "query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Search the index",
    )
    pq.add_argument("q")
    pq.add_argument("-k", type=int, default=10, help="top-k (default 10)")
    pq.add_argument("--source", help="filter by source tag")
    pq.add_argument("--json", action="store_true")
    pq.add_argument(
        "--granular",
        action="store_true",
        help=(
            "disable cluster/msg dedupe — return both the cluster hit "
            "and its constituent msg hits when both rank highly"
        ),
    )
    pq.add_argument(
        "--retrieval",
        choices=["hybrid", "vector", "bm25"],
        default="hybrid",
        help=(
            "retrieval strategy: 'hybrid' (default — vec0 KNN union "
            "fts5 BM25 merged via Reciprocal Rank Fusion), 'vector' "
            "(legacy pure-vec, skip BM25), 'bm25' (skip embedding, "
            "exact-term fts5 only — useful for literal-term queries "
            "like 'shonen' or codename matches)"
        ),
    )
    pq.add_argument(
        "--sort",
        choices=["relevance", "added", "year"],
        default="relevance",
        help=(
            "sort order: 'relevance' (default, vec0 distance ascending), "
            "'added' (library_added_at descending — most recently added first), "
            "'year' (release_year descending — newest content first)"
        ),
    )
    pq.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="only return hits with release_year >= N (excludes NULLs)",
    )
    pq.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="only return hits with release_year <= N (excludes NULLs)",
    )
    pq.add_argument(
        "--added-since",
        default=None,
        help=(
            "only return hits where library_added_at is within the given "
            "duration (e.g. 30d, 6m, 1y, 12h). Excludes rows with "
            "unknown library_added_at."
        ),
    )
    pq.add_argument(
        "--caller",
        default=None,
        choices=sorted(ALLOWED_CALLERS),
        help=(
            "label this query's interface for metrics. Defaults to "
            "$EICHI_CALLER (or legacy $VSEARCH_CALLER) then 'cli'. "
            "Allowed: cli|web|agent|mainloop. "
            "The web frontend and the agent harness set this via env so "
            "the metrics dashboard can split traffic by interface."
        ),
    )
    pq.add_argument(
        "--raw-score",
        action="store_true",
        help=(
            "show the raw vec0 L2 distance instead of the projected 0–1 "
            "similarity + band (default rendering: '0.61 [moderate]'). "
            "Useful for regression tests and threshold tuning. The JSON "
            "envelope always carries both `score` (raw) and `similarity`."
        ),
    )
    pq.set_defaults(func=cmd_query)

    pr = sub.add_parser(
        "reindex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Wipe + rebuild for a path or full DB",
    )
    pr.add_argument("path", nargs="?")
    pr.add_argument("-v", "--verbose", action="store_true")
    pr.add_argument("-n", "--dry-run", action="store_true")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_reindex)

    ps = sub.add_parser(
        "stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Show index stats",
    )
    ps.add_argument("--json", action="store_true")
    ps.add_argument(
        "--by-source",
        action="store_true",
        help="break out file/chunk counts + last_indexed by connector source",
    )
    ps.set_defaults(func=cmd_stats)

    pl = sub.add_parser(
        "ls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List indexed files",
    )
    pl.add_argument("source", nargs="?")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_ls)

    pm = sub.add_parser(
        "rm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Remove a file/dir from the index",
    )
    pm.add_argument("path")
    pm.add_argument("-n", "--dry-run", action="store_true")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=cmd_rm)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
