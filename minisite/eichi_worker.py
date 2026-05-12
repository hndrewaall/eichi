"""Persistent eichi worker.

Runs INSIDE the bind-mounted eichi venv (so the heavy ML deps are
already importable). Reads one JSON command per line on stdin, runs
the embed + sqlite-vec query against the open db connection, writes
one JSON object per line back on stdout. Stays alive for the lifetime
of the parent gunicorn worker so the sentence-transformers model only
loads once.

Query log (caller=web)
----------------------
After each successful query the worker appends a JSONL record to
``$EICHI_QUERY_LOG`` (default ``~/.local/share/eichi/query.log``)
mirroring the schema written by the eichi CLI. The record stamps:

  * ``caller="web"`` — splits this query out from CLI / agent / mainloop
    in the metrics dashboard.
  * ``user_uid`` — Firebase uid forwarded from the auth-gate's
    ``X-Auth-Uid`` header (the minisite passes it on the worker request).
  * ``role`` — auth-gate role (forward-compat — currently unused but
    plumbed end-to-end so the dashboard can split admin vs search-user).

Best-effort: a write error never fails the query — the metrics
pipeline is observability, not correctness, and the user-facing
search must keep working if the log volume is wedged.

Wire format (request)
---------------------
``{"id": "<opaque>", "cmd": "query", "q": "<text>", "k": <int>,
   "source": "<tag-or-null>", "user_uid": "<uid|null>",
   "role": "<role|null>"}``

Response (success)
------------------
``{"id": "<opaque>", "ok": true, "results": [{...}, ...]}``

Response (failure)
------------------
``{"id": "<opaque>", "ok": false, "error": "<message>"}``

Each request/response is a single line of JSON. The "id" round-trips so
the parent can interleave concurrent requests without keeping a strict
queue (though in practice gunicorn's threaded worker holds a single
mutex and serializes calls — the id is mostly defensive).

Startup
-------
Reads EICHI_DB from argv[1] (default: eichi's own DEFAULT_DB_PATH).
Opens the DB + loads the embedding model BEFORE writing the first
``{"event": "ready"}`` line. The Flask side blocks on that ready line
the first time it spawns the worker so cold-start latency lands on the
gunicorn boot path, not on the first user query.

Failure mode
------------
On any exception in the per-query path we emit an error response and
keep the loop alive — a single bad query must NOT kill the worker.
The worker only exits on stdin EOF (parent gone) or an explicit
``{"cmd": "shutdown"}``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional


def _query_log_path() -> Path:
    """Mirror eichi.cli._query_log_path so the web frontend writes
    into the same JSONL ring the CLI writes into. Both ``$EICHI_QUERY_LOG``
    overrides land in the same place — the eichi-exporter tails this
    file and feeds the histogram + per-caller / per-user counters.

    ``VSEARCH_QUERY_LOG`` / ``~/.local/share/vsearch/query.log`` are
    honored as backwards-compat fallbacks for one release so an
    in-place upgrade doesn't silently start writing to a new location.
    """
    override = os.environ.get("EICHI_QUERY_LOG") or os.environ.get("VSEARCH_QUERY_LOG")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "eichi" / "query.log"


def _log_query(record: dict) -> None:
    """Best-effort append to the RTT log. NEVER raise — the user-facing
    query path must not fail because metrics logging hit a disk error /
    permission issue / out-of-space."""
    if os.environ.get("EICHI_NO_QUERY_LOG") or os.environ.get("VSEARCH_NO_QUERY_LOG"):
        return
    try:
        path = _query_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _local_parse_duration(spec: str) -> Optional[float]:
    """Fallback duration parser for older eichi checkouts that do
    not expose ``eichi.cli._parse_duration``. Mirrors the CLI's
    suffix vocabulary: ``s`` / ``min`` / ``h`` / ``d`` / ``w`` / ``mo``
    / ``m`` (months) / ``y``. Bare seconds for unsuffixed input.
    """
    s = (spec or "").strip().lower()
    if not s:
        return None
    units: list[tuple[str, float]] = [
        ("min", 60.0),
        ("s", 1.0),
        ("h", 3600.0),
        ("d", 86400.0),
        ("w", 604800.0),
        ("mo", 2592000.0),
        ("m", 2592000.0),
        ("y", 31536000.0),
    ]
    for suffix, mult in units:
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            if not num:
                raise ValueError(f"missing number in duration: {spec!r}")
            try:
                return float(num) * mult
            except ValueError:
                raise ValueError(f"bad duration {spec!r}") from None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"unrecognized duration {spec!r}") from None


def main() -> int:
    db_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Imports happen inside main() so an import-time failure surfaces
    # as a clean error event instead of a stack trace at module load.
    try:
        from eichi.store import (
            ensure_fts_backfill,
            fts_count,
            open_db,
            search,
            search_bm25,
            search_hybrid,
        )
        from eichi import embed  # noqa: F401 — triggers model load below
        # Reuse the CLI's `_parse_duration` so the search-minisite
        # accepts the same `1d` / `7d` / `30d` / `6mo` / `1y` vocabulary
        # as `eichi query --added-since`. Failing soft (function
        # missing) — older eichi checkouts don't expose it; we fall
        # back to a tiny inline parser.
        try:
            from eichi.cli import _parse_duration as _vs_parse_duration
        except ImportError:
            _vs_parse_duration = None
        # Reuse the CLI's similarity/band projection so CLI + web emit
        # IDENTICAL similarity numbers for the same raw distance. If the
        # eichi checkout is older (pre-similarity-projection), we fall
        # back to local copies of the formula + thresholds — the minisite
        # must keep showing similarity even when running against an older
        # eichi venv during a rolling upgrade.
        try:
            from eichi.cli import (
                BAND_CUTOFFS as _VS_BAND_CUTOFFS,
                BAND_LABELS as _VS_BAND_LABELS,
                SIMILARITY_THRESHOLD as _VS_SIM_THRESHOLD,
                relevance_band as _vs_relevance_band,
                similarity_from_distance as _vs_similarity_from_distance,
            )
        except ImportError:  # pragma: no cover — only hit during rollouts
            _VS_SIM_THRESHOLD = 1.4
            _VS_BAND_LABELS = ("strong", "moderate", "weak", "distant")
            _VS_BAND_CUTOFFS = (0.7, 1.05, 1.20)

            def _vs_similarity_from_distance(distance):  # type: ignore[no-redef]
                try:
                    d = float(distance)
                except (TypeError, ValueError):
                    return 0.0
                # BM25-only hybrid hits surface negative scores — not L2
                # distances, can't project. Bail out at 0.0.
                if d < 0.0:
                    return 0.0
                sim = (_VS_SIM_THRESHOLD - d) / _VS_SIM_THRESHOLD
                return max(0.0, min(1.0, sim))

            def _vs_relevance_band(distance):  # type: ignore[no-redef]
                try:
                    d = float(distance)
                except (TypeError, ValueError):
                    return _VS_BAND_LABELS[-1]
                if d < 0.0:
                    return _VS_BAND_LABELS[-1]
                for cutoff, label in zip(_VS_BAND_CUTOFFS, _VS_BAND_LABELS):
                    if d <= cutoff:
                        return label
                return _VS_BAND_LABELS[-1]
    except Exception as exc:  # pragma: no cover — diagnostic startup path
        sys.stdout.write(
            json.dumps(
                {"event": "fatal", "error": f"import: {exc}", "trace": traceback.format_exc()}
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1

    try:
        conn = open_db(db_path) if db_path else open_db()
    except Exception as exc:
        sys.stdout.write(
            json.dumps(
                {"event": "fatal", "error": f"open_db: {exc}", "trace": traceback.format_exc()}
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1

    # Trigger a real embed so the model + tokenizer load happen now,
    # not on first query. Cheap throwaway string.
    try:
        embed.encode_one("warmup")
    except Exception as exc:
        sys.stdout.write(
            json.dumps(
                {"event": "fatal", "error": f"embed warmup: {exc}", "trace": traceback.format_exc()}
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1

    sys.stdout.write(json.dumps({"event": "ready"}) + "\n")
    sys.stdout.flush()

    for raw in sys.stdin:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        req: dict[str, Any]
        try:
            req = json.loads(raw)
        except ValueError:
            sys.stdout.write(json.dumps({"ok": False, "error": "non-JSON request"}) + "\n")
            sys.stdout.flush()
            continue
        if not isinstance(req, dict):
            sys.stdout.write(json.dumps({"ok": False, "error": "request must be a JSON object"}) + "\n")
            sys.stdout.flush()
            continue

        rid = req.get("id", "")
        cmd = req.get("cmd")
        if cmd == "shutdown":
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "event": "shutdown"}) + "\n")
            sys.stdout.flush()
            return 0
        if cmd != "query":
            sys.stdout.write(
                json.dumps({"id": rid, "ok": False, "error": f"unknown cmd: {cmd!r}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        q = req.get("q") or ""
        k = req.get("k") or 20
        source = req.get("source")
        # Per-role source allowlist (search-restricted-access design,
        # Phase D backstop). Search-minisite passes this on no-filter
        # queries; eichi.store.search* applies a row-level filter at
        # the SQL/Python layer so the worker can never return a row
        # whose source falls outside the role's allowlist regardless
        # of upstream bugs.
        allowed_sources_raw = req.get("allowed_sources")
        if isinstance(allowed_sources_raw, list):
            allowed_sources_arg: list[str] | None = [
                str(s) for s in allowed_sources_raw if isinstance(s, str)
            ]
        else:
            allowed_sources_arg = None
        # Caller identity for the query log (search-minisite always
        # uses caller=web — the worker is not reachable from CLI /
        # agent / mainloop callers). user_uid + role come from the
        # auth-gate via the Flask request headers.
        user_uid_raw = req.get("user_uid")
        role_raw = req.get("role")
        user_uid = (
            str(user_uid_raw).strip()
            if isinstance(user_uid_raw, str) and user_uid_raw.strip()
            else None
        )
        role = (
            str(role_raw).strip()
            if isinstance(role_raw, str) and role_raw.strip()
            else None
        )
        # Retrieval mode — default hybrid; falls back to legacy
        # pure-vec when caller pins ``retrieval=vector``. ``bm25``
        # available for diagnostics / literal-term debug. Older
        # callers (no ``retrieval`` key) get hybrid for free.
        retrieval = req.get("retrieval") or "hybrid"
        if retrieval not in ("hybrid", "vector", "bm25"):
            retrieval = "hybrid"
        # Filter facets (q-2026-05-02-5864). All optional; None means no
        # filter applied. The Flask side has already validated the
        # year ranges; we coerce defensively here so a hand-rolled
        # JSON request can't crash the worker.
        year_min_raw = req.get("year_min")
        year_max_raw = req.get("year_max")
        added_since_raw = req.get("added_since")
        try:
            year_min = int(year_min_raw) if year_min_raw is not None else None
        except (TypeError, ValueError):
            year_min = None
        try:
            year_max = int(year_max_raw) if year_max_raw is not None else None
        except (TypeError, ValueError):
            year_max = None
        # Resolve ``added_since`` token → unix epoch cutoff. Reuse the
        # CLI's parser so we accept the same vocabulary; on parse error
        # we drop the filter rather than fail the query (the Flask
        # allowlist already gates the dropdown values).
        added_since_unix = None
        if added_since_raw:
            try:
                if _vs_parse_duration is not None:
                    secs = _vs_parse_duration(str(added_since_raw))
                else:
                    secs = _local_parse_duration(str(added_since_raw))
                if secs is not None and secs > 0:
                    import time as _time
                    added_since_unix = _time.time() - secs
            except ValueError:
                added_since_unix = None
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = 20
        if k < 1:
            k = 1
        if k > 200:
            k = 200

        # Timing — encode_ms / search_ms / elapsed_ms split mirrors
        # the CLI shape so the exporter histogram lines up cleanly.
        encode_ms_val = 0.0
        search_ms_val = 0.0
        try:
            t0 = time.monotonic()
            if retrieval == "bm25":
                if fts_count(conn) == 0:
                    ensure_fts_backfill(conn)
                # bm25 has no embed step — encode_ms stays 0.
                t1 = time.monotonic()
                hits = search_bm25(
                    conn,
                    q,
                    k=k,
                    source=source,
                    year_min=year_min,
                    year_max=year_max,
                    added_since_unix=added_since_unix,
                    allowed_sources=allowed_sources_arg,
                )
                t2 = time.monotonic()
            elif retrieval == "vector":
                qvec = embed.encode_one(q)
                t1 = time.monotonic()
                hits = search(
                    conn,
                    qvec,
                    k=k,
                    source=source,
                    year_min=year_min,
                    year_max=year_max,
                    added_since_unix=added_since_unix,
                    allowed_sources=allowed_sources_arg,
                )
                t2 = time.monotonic()
            else:
                qvec = embed.encode_one(q)
                if fts_count(conn) == 0:
                    ensure_fts_backfill(conn)
                t1 = time.monotonic()
                hits = search_hybrid(
                    conn,
                    qvec,
                    q,
                    k=k,
                    source=source,
                    year_min=year_min,
                    year_max=year_max,
                    added_since_unix=added_since_unix,
                    allowed_sources=allowed_sources_arg,
                )
                t2 = time.monotonic()
            encode_ms_val = (t1 - t0) * 1000.0
            search_ms_val = (t2 - t1) * 1000.0
            # ``hits`` is a list of eichi.store.SearchHit dataclasses.
            # Mirror the CLI's _print_hit JSON shape (score, source,
            # path, chunk_idx, offset, snippet) — clip snippet to 200
            # chars same as the CLI so the wire payload matches the
            # documented surface and the front-end can render either
            # backend identically.
            results = []
            for h in hits:
                snippet = (h.text or "").replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "…"
                # Carry mtime + indexed_at_unix through to the front-end
                # so the result card can render a human-readable date.
                # 0.0 sentinel (no upstream timestamp) means the front-end
                # falls back to indexed_at_unix.
                mtime = float(getattr(h, "mtime", 0.0) or 0.0)
                indexed_at_unix = float(getattr(h, "indexed_at_unix", 0.0) or 0.0)
                # Cluster surface — eichi SearchHit carries kind /
                # cluster_id / mtime_end (added in a recent eichi release). Pass
                # them straight through; cluster_size is derived from the
                # text using the same heuristic as the CLI's _print_hit
                # so the front-end can render a "[signal-chat cluster,
                # N msgs]"-style badge. cluster_size==0 for non-cluster
                # rows is the documented sentinel.
                cluster_kind = getattr(h, "kind", "") or ""
                cluster_id = getattr(h, "cluster_id", "") or ""
                mtime_end = float(getattr(h, "mtime_end", 0.0) or 0.0)
                if cluster_kind == "cluster":
                    raw_text = h.text or ""
                    cluster_size = max(
                        1,
                        sum(1 for line in raw_text.split("\n") if line.startswith("[")),
                    )
                else:
                    cluster_size = 0
                # Date-aware surface (eichi date-aware surface). 0 sentinels
                # for unknown — the front-end must not fabricate a date.
                library_added_at_unix = float(
                    getattr(h, "library_added_at", 0.0) or 0.0
                )
                release_year = int(getattr(h, "release_year", 0) or 0)
                # Hybrid retrieval provenance (eichi 0.2 — BM25+RRF).
                # NULL when the doc did not surface in that pass.
                # Pure-vec / pure-bm25 modes leave both NULL.
                vec_rank = getattr(h, "vec_rank", None)
                bm25_rank = getattr(h, "bm25_rank", None)
                # Project the raw vec0 distance to a 0-1 similarity + band
                # so the front-end + JSON consumers don't have to interpret
                # opaque distance numbers. The transform is monotonic and
                # ranking-preserving — sort order is unchanged.
                similarity_val = float(_vs_similarity_from_distance(h.score))
                band_val = str(_vs_relevance_band(h.score))
                results.append(
                    {
                        "score": float(h.score),
                        "similarity": similarity_val,
                        "relevance_band": band_val,
                        "source": h.source,
                        "path": h.path,
                        "chunk_idx": h.chunk_idx,
                        "offset": h.offset,
                        "snippet": snippet,
                        "mtime": mtime,
                        "indexed_at_unix": indexed_at_unix,
                        "kind": cluster_kind,
                        "cluster_id": cluster_id,
                        "cluster_size": cluster_size,
                        "mtime_end": mtime_end,
                        "library_added_at_unix": library_added_at_unix,
                        "release_year": release_year,
                        "vec_rank": vec_rank,
                        "bm25_rank": bm25_rank,
                    }
                )
            payload = {"id": rid, "ok": True, "results": results}
            # Stamp the query log — caller=web, plus user_uid/role for
            # the per-user dashboard. Best-effort; never raises.
            elapsed_ms_val = encode_ms_val + search_ms_val
            _log_query(
                {
                    "ts": time.time(),
                    "elapsed_ms": round(elapsed_ms_val, 3),
                    "encode_ms": round(encode_ms_val, 3),
                    "search_ms": round(search_ms_val, 3),
                    "k": int(k),
                    "hits": len(results),
                    "source": source,
                    "caller": "web",
                    "user_uid": user_uid,
                    "role": role,
                }
            )
        except Exception as exc:
            payload = {
                "id": rid,
                "ok": False,
                "error": f"query: {exc}",
            }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
