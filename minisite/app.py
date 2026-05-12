"""eichi minisite — Flask web frontend for the eichi search CLI.

Single-file Flask app (app.py) that renders a search UI on top of the
local ``eichi`` CLI (sqlite-vec + sentence-transformers). Auth is
gated UPSTREAM by a reverse proxy (e.g. nginx ``auth_request``) — the
proxy enforces the cookie + uid allowlist before any request reaches
us. This container trusts ``X-Auth-Request-Email`` for display only.

Routes
------
``GET /``
    HTML page with the search box and results pane.

``GET /api/search?q=<query>&k=<int>&source=<filter>&year_min=<int>&year_max=<int>&added_since=<dur>&retrieval=<mode>``
    JSON envelope ``{"results": [...], "elapsed_ms": ..., "query": ..., "k": ...,
    "source": ..., "year_min": ..., "year_max": ..., "added_since": ...,
    "retrieval": ..., "error": ...}``.

    Filter facets (q-2026-05-02-5864):

    * ``year_min`` / ``year_max`` — inclusive ``release_year`` bounds. Empty /
      missing = no bound. Rows with ``release_year=0`` (unknown) are
      excluded when EITHER bound is set.
    * ``added_since`` — duration token (``1d``, ``7d``, ``30d``, ``6mo``,
      ``1y``) parsed by eichi's own ``_parse_duration``. Empty = no
      cutoff. Rows with NULL ``library_added_at`` are excluded when this
      filter is set.
    * ``retrieval`` — ``hybrid`` (default) / ``vector`` / ``bm25``. Pure-vec
      and pure-bm25 are exposed for diagnostics.

``GET /healthz``
    Plain ``ok\n`` for monitoring. Bypassed by the proxy auth gate.

eichi integration — persistent worker
-------------------------------------
The gunicorn worker spawns ``eichi_worker.py`` ONCE under the host's
bind-mounted eichi venv interpreter. The worker loads the
sentence-transformers model + opens the sqlite-vec DB at startup,
then loops on stdin reading JSON queries and writing JSON responses.
Each Flask request hands a JSON request to the worker via stdin and
reads the response off stdout — no per-query model reload, no
per-query subprocess spawn, ~50-200 ms warm latency.

Worker lifecycle
----------------
* Lazily started on first ``/api/search`` hit (so a healthz-only
  caller doesn't pay the 3-5s warmup tax).
* Auto-restarted if the subprocess dies (a single bad query cannot
  kill the worker — its main loop swallows exceptions — but a kernel
  OOM-kill or upstream change still gets recovered).
* Bounded by ``WORKER_QUERY_TIMEOUT`` per request.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.parse
from typing import Any

from flask import Flask, jsonify, render_template, request

# eichi interpreter + worker script. The interpreter lives in the
# bind-mounted host venv (see Dockerfile + the docker-compose snippet
# in the README); the worker script is bundled into the image
# alongside app.py.
#
# Both paths MUST be provided by the operator via env vars — there is
# no sensible default for a venv path inside a container, and a
# silent fallback would mask deployment bugs. EICHI_DB falls back to
# eichi's own default-resolution rules (see eichi.store.default_db_path)
# if unset.
EICHI_PYTHON = os.environ.get("EICHI_PYTHON")
EICHI_DB = os.environ.get("EICHI_DB", "")
WORKER_SCRIPT = os.environ.get("EICHI_WORKER", "/app/eichi_worker.py")

# Backwards-compat aliases. Earlier deployments shipped with
# ``VSEARCH_*`` env vars; tolerate them for one release so an
# in-place upgrade doesn't 500 the gunicorn boot. New deployments
# should set ``EICHI_*``.
if not EICHI_PYTHON:
    EICHI_PYTHON = os.environ.get("VSEARCH_PYTHON", "")
if not EICHI_DB:
    EICHI_DB = os.environ.get("VSEARCH_DB", "")

# Site title rendered in the page header + <title>. Override via env
# for a custom brand; defaults to a neutral "eichi search".
SITE_TITLE = os.environ.get("SEARCH_SITE_TITLE", "eichi search")

# Whitelabel branding hooks. All default to empty strings; the template
# renders cleanly with or without each one set. A deployer overrides
# them via env (typically through an env_file mounted on the container).
#
#   SEARCH_SITE_LOGO_URL    — header logo image URL. Empty = no logo
#                             rendered. May be an absolute URL or a
#                             relative path (e.g. /static/foo.png).
#                             When unset, falls back to the bundled
#                             default at /static/eichi-logo.png if
#                             ``SEARCH_SITE_LOGO_DEFAULT=1`` is set.
#   SEARCH_SITE_BRAND       — short brand string rendered in the
#                             footer. Empty = no brand text.
#   SEARCH_SITE_FAVICON_URL — favicon override. Empty = use the
#                             bundled generic favicon.
SITE_LOGO_URL = os.environ.get("SEARCH_SITE_LOGO_URL", "").strip()
SITE_BRAND = os.environ.get("SEARCH_SITE_BRAND", "").strip()
SITE_FAVICON_URL = os.environ.get("SEARCH_SITE_FAVICON_URL", "").strip()
SITE_LOGO_DEFAULT = os.environ.get("SEARCH_SITE_LOGO_DEFAULT", "").strip() in (
    "1",
    "true",
    "yes",
)

DEFAULT_K = int(os.environ.get("SEARCH_DEFAULT_K", "20"))
MAX_K = int(os.environ.get("SEARCH_MAX_K", "100"))
MAX_QUERY_LEN = int(os.environ.get("SEARCH_MAX_QUERY_LEN", "500"))
# Per-query wall-clock cap. Warm queries are <500 ms; the cap is
# generous to absorb a slow disk read on the index.
WORKER_QUERY_TIMEOUT = float(os.environ.get("SEARCH_QUERY_TIMEOUT", "30"))
# How long we wait for the worker's "ready" event on first spawn.
# Cold start (model load + DB open + embed warmup) takes 5-10 s.
WORKER_BOOT_TIMEOUT = float(os.environ.get("SEARCH_WORKER_BOOT_TIMEOUT", "60"))

# Allowlist of retrieval modes accepted via the `retrieval` query parameter.
# Mirrors the eichi worker / CLI vocabulary. Anything outside this set
# gets clamped to "hybrid" (the safe default).
ALLOWED_RETRIEVAL = {"hybrid", "vector", "bm25"}

# Hard ceiling on year inputs to keep absurd values out of the worker.
# release_year is a 4-digit integer in the index; we clamp anything
# outside [1800, current_year+10].
YEAR_MIN_FLOOR = 1800
YEAR_MAX_CEIL = 2100

# Allowlist of `added_since` duration tokens accepted from the UI. Free
# text passes through to eichi's `_parse_duration` server-side, but
# the dropdown emits one of these values — the allowlist is a
# defense-in-depth check that keeps junk out of the worker's argv.
ALLOWED_ADDED_SINCE = {"", "1d", "7d", "30d", "6mo", "1y"}

# Per-role source allowlist. Each role maps to the set of source tags
# it may query.  Replaces the flat ALLOWED_SOURCES set as part of the
# search-restricted-access rollout — see the design doc s6.
#
# `admin` carries the `*` wildcard sentinel which expands to every entry
# in ALL_SOURCES at request time. New connectors added to ALL_SOURCES
# are admin-only by default — `search-user` gains them only via an
# explicit add here. This is the default-deny invariant: the burden of
# proof is on granting access, never on denying it.
SOURCES_BY_ROLE: dict[str, set[str]] = {
    "admin": {
        "*",
    },
    "search-user": {
        "calibre",
        "embiguity-content",
        "kavita",
        "navidrome-albums",
        "navidrome-artists",
        # "grafana-dashboards" — add when that connector lands and is
        # confirmed safe for non-admin readers.
    },
}

# Every source tag known to the index. Used both as the admin-wildcard
# expansion target AND as input validation when a user-supplied
# ?source= value is checked against the per-role allowlist below.
ALL_SOURCES: set[str] = {
    "calibre",
    "embiguity-content",
    "embiguity-requests",
    "kavita",
    "memory",
    "navidrome-albums",
    "navidrome-artists",
    "obsidian",
    "queue-logs",
    "repo-md",
    "signal-chat",
}


def sources_for_role(role: str) -> set[str]:
    """Return the set of source tags ``role`` is allowed to query.

    Unknown / missing roles → empty set (deny everything). The role
    parameter is whatever ``X-Auth-Role`` carried; the auth-gate defaults
    a missing claim to ``search-user`` upstream, so an empty set here
    means "not even search-user" (e.g. an attacker spoofing a typo'd
    role header through a misconfigured proxy).
    """
    allowed = SOURCES_BY_ROLE.get(role, set())
    if "*" in allowed:
        return set(ALL_SOURCES)
    return set(allowed)


# Backwards-compat alias. Existing tests reference `ALLOWED_SOURCES`
# expecting the union of every source. Keep it pointing at ALL_SOURCES
# so legacy assertions still pass; new code paths read SOURCES_BY_ROLE.
ALLOWED_SOURCES = ALL_SOURCES

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Embedding model — read at boot from the eichi DB's meta table
# ---------------------------------------------------------------------------
#
# The footer reports which embedding model is in use. Hardcoding the
# string drifts the moment eichi upgrades the model (q-2026-05-03-d5c5
# spotted "all-MiniLM-L6-v2" in the SPA after eichi had moved to
# all-mpnet-base-v2). Read it from the sqlite-vec DB's ``meta`` table
# instead — that's eichi's own source of truth, written by
# ``ensure_schema`` when the DB is created.
#
# Failure modes (DB missing, table missing, key missing) silently fall
# back to a generic placeholder so the SPA still renders. The footer
# is cosmetic — never block boot or 500 the page over it.

EMBEDDING_MODEL_FALLBACK = "eichi"


def _read_embedding_model_label() -> str:
    """Return ``"<model>"`` (basename, no leading ``sentence-transformers/``).

    Reads ``meta.embedding_model`` from the eichi DB at ``EICHI_DB``.
    Returns ``EMBEDDING_MODEL_FALLBACK`` on any failure (file missing,
    SQL error, key absent). The DB is opened read-only and closed
    immediately — we don't hold the connection.
    """
    try:
        # Open read-only to avoid stomping a writer (the DB is shared
        # with the host eichi CLI / index-tick driver).
        uri = f"file:{EICHI_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            row = conn.execute(
                "SELECT v FROM meta WHERE k = 'embedding_model'"
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return EMBEDDING_MODEL_FALLBACK
    if not row or not row[0]:
        return EMBEDDING_MODEL_FALLBACK
    label = str(row[0])
    # Drop the org/repo prefix for display ("sentence-transformers/all-mpnet-base-v2"
    # → "all-mpnet-base-v2"). Plays nicer with the narrow footer.
    if "/" in label:
        label = label.rsplit("/", 1)[-1]
    return label


EMBEDDING_MODEL_LABEL = _read_embedding_model_label()


# ---------------------------------------------------------------------------
# Deep-link surface
# ---------------------------------------------------------------------------
#
# Each result row gets an "Open in <App>" link when the doc's source maps
# to a user-facing app whose base URL has been configured via env. The
# mapping by source:
#
#   embiguity-content (show / movie / episode) → Emby
#       Needs the Emby item id. Stamped into the indexed body by the
#       embiguity connector under the ``type:`` line as
#       ``emby_id=<id>``. We look it up from chunk_idx=0 of the matched
#       doc, parsed below.
#   navidrome-albums  → Navidrome (album route)
#   navidrome-artists → Navidrome (artist route)
#       Both ids are already in the doc_id (path) field, no DB lookup.
#   kavita            → Kavita (library + series route)
#       Needs libraryId. Stamped into the indexed body by the kavita
#       connector under the header as ``library_id=<n>``.
#
# Calibre, embiguity-requests, memory, obsidian, queue-logs, repo-md,
# signal-chat, grafana-dashboards intentionally produce no deep link.
#
# Configure the user-facing base URLs via env. UNSET = no deep link
# rendered for that source. Defaults are intentionally empty so a
# default-deploy doesn't emit links pointing at someone else's
# hostnames.
#
#   EMBY_BASE_URL
#   NAVIDROME_BASE_URL
#   KAVITA_BASE_URL

EMBY_BASE_URL = os.environ.get("EMBY_BASE_URL", "")
NAVIDROME_BASE_URL = os.environ.get("NAVIDROME_BASE_URL", "")
KAVITA_BASE_URL = os.environ.get("KAVITA_BASE_URL", "")

# Regex pulled out of the indexed body. Anchored to a line so a stray
# ``emby_id=`` substring inside an overview blob can't collide.
_EMBY_ID_RE = re.compile(r"^emby_id=(\S+)$", re.MULTILINE)
_LIBRARY_ID_RE = re.compile(r"^library_id=(\S+)$", re.MULTILINE)


def _fetch_chunk0_metadata(
    targets: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Batch-fetch chunk_idx=0 text for each (source, path) pair and
    parse out the deep-link IDs (``emby_id``, ``library_id``).

    Returns ``{(source, path): {"emby_id": ..., "library_id": ...}}``;
    missing entries imply no IDs were stamped. Errors degrade silently
    — the deep link just doesn't render. Opens the DB read-only so we
    don't contend with the worker's writer.
    """
    if not targets:
        return {}
    # De-dup so the IN clause stays compact even when k=20 includes
    # multiple chunks of the same path.
    unique = list({t for t in targets if t[0] and t[1]})
    if not unique:
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    try:
        uri = f"file:{EICHI_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            # SQLite IN-clause with positional binds. (source, path) pair
            # match — chunk_idx=0 only (the IDs are stamped near the top
            # of the doc, which always lands in the first chunk for the
            # small embiguity / kavita bodies in scope).
            placeholders = ",".join(["(?, ?)"] * len(unique))
            params: list[str] = []
            for src, path in unique:
                params.append(src)
                params.append(path)
            query = (
                "SELECT source, path, text FROM chunk_meta "
                f"WHERE (source, path) IN ({placeholders}) AND chunk_idx = 0"
            )
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return out
    for src, path, text in rows:
        ids: dict[str, str] = {}
        if not isinstance(text, str):
            continue
        m = _EMBY_ID_RE.search(text)
        if m:
            ids["emby_id"] = m.group(1)
        m = _LIBRARY_ID_RE.search(text)
        if m:
            ids["library_id"] = m.group(1)
        if ids:
            out[(src, path)] = ids
    return out


def _sso_deeplink(base_url: str, sso_path: str, target_path: str) -> str:
    """Wrap an SSO-bridge ``?next=`` payload in a fully-qualified URL.

    The four SSO bridges (emby, navidrome, audiobookshelf, kavita) accept
    a ``?next=<urlencoded payload>`` query param. When set, the bridge
    primes the browser session (localStorage + cookies) and redirects
    to the resolved target instead of the app's home view.

    The shape of ``target_path`` depends on which bridge: path-routed
    apps (Kavita, Navidrome, Audiobookshelf) take an absolute SPA path
    starting with ``/`` (e.g. ``/library/1/series/42``); hashbang-routed
    apps (Emby) take JUST the hashbang fragment payload (e.g.
    ``details?id=42``), and the bridge prepends ``/web/index.html#!/``
    server-side. Wrapping a hashbang URL with literal ``#`` in
    ``target_path`` was the regression q-2026-05-04-3889 — the SPA's
    hashbang shim rewrote ``#/details`` to ``#!/index.html%23/details``,
    leaving the user on a blank home view.

    ``target_path`` MAY contain ``?`` / ``=`` / ``&`` and other URL-
    reserved punctuation — they're percent-encoded so the whole thing
    fits inside a single query value. ``#`` would be percent-encoded
    too if a caller passed it, but per the rule above, callers should
    NOT include ``#`` in the payload — the per-app helpers below
    enforce this.

    The bridge runs ``_safe_next_path`` server-side: any value that
    fails its allowlist is rejected and the bridge falls back to its
    default landing page. So this helper can't be turned into an
    open-redirect — even if the search index is poisoned, the bridge
    enforces the same-origin rule.
    """
    encoded = urllib.parse.quote(target_path, safe="")
    return f"{base_url}{sso_path}?next={encoded}"


def _build_deep_link(source: str, path: str, ids: dict[str, str]) -> str | None:
    """Map (source, path, ids) → user-facing app URL.

    Returns None when the source has no public-facing app (calibre,
    obsidian, signal-chat, etc.), when a required id is missing, OR
    when the operator hasn't configured a base URL for the target app
    (``*_BASE_URL`` env unset).

    The link is ALWAYS routed through the matching SSO bridge
    (``/__sso/<svc>?next=...``) so the browser auto-signs-in via the
    upstream session cookie and lands on the deep target instead of
    dropping into the service's native login page when its session
    cookie has expired.
    """
    if not source or not path:
        return None
    src = source.lower()
    if src == "navidrome-albums":
        if not NAVIDROME_BASE_URL:
            return None
        # Doc id format: ``navidrome:album:<id>``. The id IS the
        # Navidrome subsonic id used by the SPA route.
        prefix = "navidrome:album:"
        if not path.startswith(prefix):
            return None
        album_id = path[len(prefix):]
        if not album_id:
            return None
        return _sso_deeplink(
            NAVIDROME_BASE_URL,
            "/__sso/navidrome",
            f"/app/#/album/{album_id}/show",
        )
    if src == "navidrome-artists":
        if not NAVIDROME_BASE_URL:
            return None
        prefix = "navidrome:artist:"
        if not path.startswith(prefix):
            return None
        artist_id = path[len(prefix):]
        if not artist_id:
            return None
        return _sso_deeplink(
            NAVIDROME_BASE_URL,
            "/__sso/navidrome",
            f"/app/#/artist/{artist_id}/show",
        )
    if src == "embiguity-content":
        if not EMBY_BASE_URL:
            return None
        # Show, movie, OR episode all link to the same Emby SPA
        # hashbang route ``#!/details?id=<emby_id>``. Emby uses
        # hashbang routing, so the bridge's ``?next=`` takes JUST
        # the fragment payload (``details?id=<id>``) — no leading
        # ``#``, no ``/web/index.html`` path prefix. The bridge
        # prepends ``/web/index.html#!/`` server-side. Earlier
        # iterations passed the full path-with-fragment shape
        # (``/web/index.html#/details?id=N``) which the SPA's
        # hashbang shim rewrote to ``#!/index.html%23/details``,
        # leaving the user on a blank home view (regression
        # q-2026-05-04-3889 — Andrew screenshot 2026-05-04 09:03 ET).
        # Without emby_id (e.g. older indexed row, before the
        # connector change landed) we degrade silently — no link
        # rendered.
        emby_id = ids.get("emby_id")
        if not emby_id:
            return None
        return _sso_deeplink(
            EMBY_BASE_URL,
            "/__sso/emby",
            f"details?id={emby_id}",
        )
    if src == "kavita":
        if not KAVITA_BASE_URL:
            return None
        prefix = "kavita:series:"
        if not path.startswith(prefix):
            return None
        series_id = path[len(prefix):]
        library_id = ids.get("library_id")
        if not series_id or not library_id:
            return None
        return _sso_deeplink(
            KAVITA_BASE_URL,
            "/__sso/kavita",
            f"/library/{library_id}/series/{series_id}",
        )
    # Calibre, embiguity-requests, memory, obsidian, queue-logs,
    # repo-md, signal-chat, grafana-dashboards: no public-facing app.
    return None


def _app_label_for(source: str) -> str | None:
    """Human-readable app label used by the front-end "Open in X" button.

    Returns None for sources that have no deep-link target.
    """
    src = (source or "").lower()
    if src == "embiguity-content":
        return "Emby"
    if src in ("navidrome-albums", "navidrome-artists"):
        return "Navidrome"
    if src == "kavita":
        return "Kavita"
    return None


# ---------------------------------------------------------------------------
# Persistent eichi worker
# ---------------------------------------------------------------------------


class _Worker:
    """Lazy-spawned, restartable subprocess wrapper around eichi_worker.py.

    Thread-safe: a single mutex serializes both spawn and request/response
    cycles. Gunicorn's threaded worker model means up to 4 concurrent
    request threads will contend on this mutex; given each query lands
    in 50-200 ms once the model is warm, contention is acceptable. If
    that ever becomes a bottleneck the natural next step is a small
    pool — but a pool means N copies of the model in RAM, so we hold
    off until profiling demands it.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._last_error: str | None = None

    def _spawn_locked(self) -> None:
        """Spawn (or respawn) the worker subprocess. Caller holds the lock."""
        # Reap any prior process before respawning.
        self._kill_locked()

        cmd = [EICHI_PYTHON, WORKER_SCRIPT, EICHI_DB]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except OSError as exc:
            self._last_error = f"failed to spawn worker: {exc.strerror or exc}"
            self._proc = None
            return

        # Block on the "ready" event so the very first query lands
        # against a hot worker. Use a deadline-based read so a wedged
        # worker can't hang the gunicorn boot indefinitely.
        deadline = time.monotonic() + WORKER_BOOT_TIMEOUT
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline() if self._proc.stdout else ""
            if not line:
                # EOF on stdout — worker died during boot. Pull stderr
                # for diagnostics.
                stderr_tail = ""
                if self._proc.stderr:
                    try:
                        stderr_tail = self._proc.stderr.read()[-500:]
                    except OSError:
                        pass
                self._last_error = f"worker died during boot: {stderr_tail.strip()}"
                self._kill_locked()
                return
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "ready":
                self._last_error = None
                return
            if event.get("event") == "fatal":
                self._last_error = f"worker fatal: {event.get('error', '?')}"
                self._kill_locked()
                return
        self._last_error = (
            f"worker did not emit 'ready' within {WORKER_BOOT_TIMEOUT:.0f}s"
        )
        self._kill_locked()

    def _kill_locked(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self._proc = None

    def _alive_locked(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def query(
        self,
        q: str,
        *,
        k: int,
        source: str | None,
        year_min: int | None = None,
        year_max: int | None = None,
        added_since: str | None = None,
        retrieval: str | None = None,
        user_uid: str | None = None,
        role: str | None = None,
        allowed_sources: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Send one query to the worker. Spawns it if not already running.

        Returns ``(results, error)`` matching the previous _run_search
        contract so the route handler is unchanged.

        ``user_uid`` / ``role`` are forwarded from the auth-gate's
        ``X-Auth-Uid`` / ``X-Auth-Role`` headers and are stamped into
        the worker's query.log entry alongside ``caller="web"``. The
        per-user / per-role panels on the eichi Grafana dashboard
        read those labels off the resulting Prom counters.
        """
        with self._lock:
            if not self._alive_locked():
                self._spawn_locked()
                if not self._alive_locked():
                    return [], self._last_error or "worker unavailable"

            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            self._req_id += 1
            rid = str(self._req_id)
            req: dict[str, Any] = {
                "id": rid,
                "cmd": "query",
                "q": q,
                "k": k,
                "source": source,
            }
            if year_min is not None:
                req["year_min"] = year_min
            if year_max is not None:
                req["year_max"] = year_max
            if added_since:
                req["added_since"] = added_since
            if retrieval:
                req["retrieval"] = retrieval
            if user_uid:
                req["user_uid"] = user_uid
            if role:
                req["role"] = role
            if allowed_sources:
                # Pass through as a list (sets aren't JSON-serialisable)
                # — eichi's store.search() applies the SQL-level
                # source-IN filter as a defense-in-depth backstop.
                req["allowed_sources"] = list(allowed_sources)
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                # Pipe broken — kill the worker, the next query will respawn.
                self._kill_locked()
                return [], f"worker pipe write failed: {exc}"

            # Read until we see a response with our id, OR the worker
            # dies, OR we hit the per-query timeout. Use a polling loop
            # with select to honor the timeout without blocking forever.
            deadline = time.monotonic() + WORKER_QUERY_TIMEOUT
            while time.monotonic() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    self._kill_locked()
                    return [], "worker died mid-query"
                try:
                    resp = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(resp, dict):
                    continue
                # Skip stray events (e.g. unsolicited diagnostics).
                if "id" not in resp and resp.get("event"):
                    continue
                if resp.get("id") != rid:
                    # Out-of-order response — the lock guarantees this
                    # shouldn't happen, but tolerate it defensively.
                    continue
                if resp.get("ok"):
                    results = resp.get("results") or []
                    if not isinstance(results, list):
                        results = []
                    return results, None
                err = resp.get("error") or "unknown error"
                return [], f"eichi: {err}"
            # Timeout — the worker may be stuck on a pathological
            # query. Kill it and let the next request respawn.
            self._kill_locked()
            return [], f"eichi query timed out after {WORKER_QUERY_TIMEOUT:.0f}s"


_worker = _Worker()


def _run_search(
    query: str,
    *,
    k: int,
    source: str | None,
    year_min: int | None = None,
    year_max: int | None = None,
    added_since: str | None = None,
    retrieval: str | None = None,
    user_uid: str | None = None,
    role: str | None = None,
    allowed_sources: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Public adapter — keeps the route handler agnostic of the worker
    architecture. Tests monkey-patch THIS function to stub out the
    backend without touching ``_Worker``."""
    if not query:
        return [], None
    return _worker.query(
        query,
        k=k,
        source=source,
        year_min=year_min,
        year_max=year_max,
        added_since=added_since,
        retrieval=retrieval,
        user_uid=user_uid,
        role=role,
        allowed_sources=allowed_sources,
    )


# ---------------------------------------------------------------------------
# Result shaping for the UI
# ---------------------------------------------------------------------------


def _pick_ts(rec: dict[str, Any]) -> tuple[float | None, str]:
    """Pick the most relevant timestamp for a result record.

    Returns ``(unix_ts, kind)`` where ``kind`` is ``"mtime"`` (upstream /
    per-connector relevance time, e.g. signal send-time, file mtime) or
    ``"indexed"`` (when eichi ingested the row). Returns ``(None, "")``
    if neither is populated.
    """
    try:
        mtime = float(rec.get("mtime") or 0.0)
    except (TypeError, ValueError):
        mtime = 0.0
    try:
        indexed = float(rec.get("indexed_at_unix") or 0.0)
    except (TypeError, ValueError):
        indexed = 0.0
    if mtime > 0:
        return mtime, "mtime"
    if indexed > 0:
        return indexed, "indexed"
    return None, ""


def _iso_utc(unix_ts: float | None) -> str | None:
    """Render a unix epoch as ISO8601 UTC (``YYYY-MM-DDTHH:MM:SSZ``).

    The browser ``LocalTime`` helper parses this into the viewer's local
    timezone — conversion stays a frontend concern.
    """
    if unix_ts is None or unix_ts <= 0:
        return None
    import datetime as _dt
    try:
        dt = _dt.datetime.fromtimestamp(float(unix_ts), tz=_dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    # Trim microseconds; drop the +00:00 suffix in favor of the shorter Z.
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _shape(
    rec: dict[str, Any],
    *,
    deep_link_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize a eichi result row for the front-end.

    ``deep_link_ids`` is a per-record dict of stamped IDs (``emby_id``,
    ``library_id``) extracted from chunk_idx=0 of the matched doc. When
    present, the shaped row gets ``app_link`` (URL) + ``app_label``
    (e.g. ``"Emby"``) so the front-end can render an "Open in <App>"
    button. None / missing IDs degrade to no link rendered — the rest
    of the result row is unaffected (q-2026-05-03-db00).
    """
    snippet = rec.get("snippet") or ""
    if len(snippet) > 600:
        snippet = snippet[:600].rstrip() + "…"
    score = rec.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    # Similarity / relevance-band — projected from raw vec0 distance by
    # the worker. Older worker rows (during a rolling upgrade) might not
    # carry these; clamp to safe defaults rather than 500. The frontend
    # checks for a non-null ``relevance_band`` before rendering the chip.
    similarity = rec.get("similarity")
    try:
        similarity_f = float(similarity) if similarity is not None else None
    except (TypeError, ValueError):
        similarity_f = None
    if similarity_f is not None:
        # Defensive clamp — keep the JSON contract honest even if the
        # worker emits a slightly out-of-range float.
        if similarity_f < 0.0:
            similarity_f = 0.0
        elif similarity_f > 1.0:
            similarity_f = 1.0
    relevance_band = rec.get("relevance_band")
    if relevance_band is not None and not isinstance(relevance_band, str):
        relevance_band = str(relevance_band)
    ts_unix, ts_kind = _pick_ts(rec)
    # Cluster surface — pass through ``kind`` / ``cluster_id`` /
    # ``cluster_size`` / ``mtime_end`` so the front-end can render a
    # ``[cluster, N msgs]`` badge and a same-day timestamp range.
    # Defaults are empty/0 for legacy rows / non-cluster connectors.
    kind = rec.get("kind") or ""
    cluster_id = rec.get("cluster_id") or ""
    try:
        cluster_size = int(rec.get("cluster_size") or 0)
    except (TypeError, ValueError):
        cluster_size = 0
    try:
        mtime_end = float(rec.get("mtime_end") or 0.0)
    except (TypeError, ValueError):
        mtime_end = 0.0
    # Date-aware fields. ``library_added_at_unix`` is the unix epoch
    # for "when this content landed in the library" (Emby DateCreated /
    # calibre timestamp / navidrome album.created — distinct from
    # eichi ingest time). ``release_year`` is the canonical release
    # year for the content. 0 sentinel for unknown/unset; the frontend
    # must not render a fake date for legacy rows.
    try:
        library_added_at_unix = float(rec.get("library_added_at_unix") or 0.0)
    except (TypeError, ValueError):
        library_added_at_unix = 0.0
    try:
        release_year = int(rec.get("release_year") or 0)
    except (TypeError, ValueError):
        release_year = 0
    return {
        # ``score`` stays the raw vec0 L2 distance — preserves the
        # historical CLI / JSON contract. ``similarity`` is the 0-1
        # projection (monotonic, ranking-preserving). ``relevance_band``
        # is the named bucket: strong / moderate / weak / distant.
        "score": score_f,
        "similarity": similarity_f,
        "relevance_band": relevance_band,
        "source": rec.get("source") or "",
        "path": rec.get("path") or "",
        "chunk_idx": rec.get("chunk_idx"),
        "offset": rec.get("offset"),
        "snippet": snippet,
        "key": f"{rec.get('source')}:{rec.get('path')}:{rec.get('chunk_idx')}",
        # Timestamp surface — mirrors the eichi CLI shape. The frontend
        # uses ``ts_iso`` (UTC ISO8601) and ``LocalTime`` to render in
        # the viewer's local timezone, with ``ts_kind`` driving the
        # tooltip ("upstream" vs "indexed at"). Raw values surfaced so
        # JSON consumers can re-derive whatever they want.
        "mtime": float(rec.get("mtime") or 0.0),
        "indexed_at_unix": float(rec.get("indexed_at_unix") or 0.0),
        "ts": ts_unix,
        "ts_kind": ts_kind,
        "ts_iso": _iso_utc(ts_unix),
        # Cluster fields. ``mtime_end_iso`` is rendered as UTC ISO8601 so
        # the frontend can feed it through LocalTime.hydrate the same
        # way it does ``ts_iso`` — the in-DOM range "13:41–13:48 ET" is
        # composed in JS from both ISO strings.
        "kind": kind,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "mtime_end": mtime_end,
        "mtime_end_iso": _iso_utc(mtime_end) if mtime_end > 0 else None,
        # Date-aware surface. ``library_added_at_iso`` is rendered as
        # UTC ISO8601 so the frontend can feed it through LocalTime if
        # it ever wants to; for now JSON consumers (the public API users) get both the unix int and the ISO string.
        "library_added_at_unix": library_added_at_unix,
        "library_added_at_iso": (
            _iso_utc(library_added_at_unix) if library_added_at_unix > 0 else None
        ),
        "release_year": release_year,
        # Hybrid retrieval provenance (eichi 0.2 — BM25+RRF).
        # Pass-through only — the UI hasn't grown a chip for these
        # yet, but JSON consumers need them to interpret why a result
        # ranked the way it did. NULL when the doc didn't surface in
        # that pass; integers (1-based rank) when it did.
        "vec_rank": rec.get("vec_rank"),
        "bm25_rank": rec.get("bm25_rank"),
        # Deep-link surface (q-2026-05-03-db00). NULL when the source
        # has no public-facing app, or the IDs the app needs aren't
        # stamped on the indexed body. The front-end renders a button
        # only when both keys are non-null.
        "app_link": _build_deep_link(
            rec.get("source") or "",
            rec.get("path") or "",
            deep_link_ids or {},
        ),
        "app_label": _app_label_for(rec.get("source") or ""),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    role = (request.headers.get("X-Auth-Role") or "").strip() or "search-user"
    role_sources = sources_for_role(role)
    return render_template(
        "index.html",
        site_title=SITE_TITLE,
        site_logo_url=SITE_LOGO_URL,
        site_logo_default=SITE_LOGO_DEFAULT,
        site_brand=SITE_BRAND,
        site_favicon_url=SITE_FAVICON_URL,
        user=request.headers.get("X-Auth-Request-Email", ""),
        default_k=DEFAULT_K,
        max_k=MAX_K,
        allowed_sources=sorted(role_sources),
        allowed_added_since=[
            ("", "any time"),
            ("1d", "1 day"),
            ("7d", "7 days"),
            ("30d", "30 days"),
            ("6mo", "6 months"),
            ("1y", "1 year"),
        ],
        allowed_retrieval=[
            ("hybrid", "hybrid"),
            ("vector", "vector"),
            ("bm25", "bm25"),
        ],
        year_min_floor=YEAR_MIN_FLOOR,
        year_max_ceil=YEAR_MAX_CEIL,
        embedding_model=EMBEDDING_MODEL_LABEL,
    )


def _parse_year(value: str | None, label: str) -> tuple[int | None, str | None]:
    """Return ``(year, error)`` for a year query param.

    Empty / missing → ``(None, None)`` (no filter). Non-integer → 400.
    Out-of-range values (outside ``[YEAR_MIN_FLOOR, YEAR_MAX_CEIL]``)
    → 400 — keeps junk values from reaching the worker.
    """
    if value is None:
        return None, None
    s = value.strip()
    if not s:
        return None, None
    try:
        year = int(s)
    except (TypeError, ValueError):
        return None, f"{label} must be an integer"
    if year < YEAR_MIN_FLOOR or year > YEAR_MAX_CEIL:
        return None, (
            f"{label}={year} out of range "
            f"[{YEAR_MIN_FLOOR}, {YEAR_MAX_CEIL}]"
        )
    return year, None


@app.route("/api/search")
def api_search() -> Any:
    raw_query = (request.args.get("q") or "").strip()
    raw_k = request.args.get("k") or str(DEFAULT_K)
    raw_source = (request.args.get("source") or "").strip() or None
    raw_added_since = (request.args.get("added_since") or "").strip() or None
    raw_retrieval = (request.args.get("retrieval") or "").strip() or None

    # Build a default error envelope so each early-return branch has a
    # consistent shape — keeps the JSON contract stable for the UI.
    def _err(msg: str, status: int = 400) -> Any:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": msg,
                    "results": [],
                    "query": raw_query[:MAX_QUERY_LEN],
                    "k": DEFAULT_K,
                    "source": raw_source,
                    "year_min": None,
                    "year_max": None,
                    "added_since": raw_added_since,
                    "retrieval": raw_retrieval,
                    "elapsed_ms": 0,
                }
            ),
            status,
        )

    if len(raw_query) > MAX_QUERY_LEN:
        return _err(f"query too long (max {MAX_QUERY_LEN} chars)")

    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        return _err("k must be an integer")
    if k < 1:
        k = 1
    if k > MAX_K:
        k = MAX_K

    # Per-role source authorisation (search-restricted-access design).
    # Read X-Auth-Role at request time so a misconfigured upstream that
    # forgets to forward the header lands the most-restrictive role
    # rather than admin-as-default. Auth-gate emits the header on every
    # 200, defaulting missing claims to ``search-user``.
    role = (request.headers.get("X-Auth-Role") or "").strip() or "search-user"
    role_sources = sources_for_role(role)
    if raw_source is not None and raw_source not in role_sources:
        # 403, not 400 — the source is real, the user just isn't
        # allowed to query it. The error body lists ONLY the role's
        # allowed sources to avoid leaking private source names to
        # search-users.
        return _err(
            f"source {raw_source!r} not allowed for this role; allowed: "
            f"{', '.join(sorted(role_sources)) or '(none)'}",
            status=403,
        )

    year_min, ymin_err = _parse_year(request.args.get("year_min"), "year_min")
    if ymin_err:
        return _err(ymin_err)
    year_max, ymax_err = _parse_year(request.args.get("year_max"), "year_max")
    if ymax_err:
        return _err(ymax_err)
    if year_min is not None and year_max is not None and year_min > year_max:
        return _err(f"year_min={year_min} > year_max={year_max}")

    if raw_added_since is not None and raw_added_since not in ALLOWED_ADDED_SINCE:
        return _err(
            f"unknown added_since token; allowed: "
            f"{', '.join(sorted(t for t in ALLOWED_ADDED_SINCE if t))}"
        )

    retrieval: str | None
    if raw_retrieval is None:
        retrieval = None
    elif raw_retrieval in ALLOWED_RETRIEVAL:
        retrieval = raw_retrieval
    else:
        return _err(
            f"unknown retrieval mode; allowed: "
            f"{', '.join(sorted(ALLOWED_RETRIEVAL))}"
        )

    if not raw_query:
        return jsonify(
            {
                "ok": True,
                "results": [],
                "query": "",
                "k": k,
                "source": raw_source,
                "year_min": year_min,
                "year_max": year_max,
                "added_since": raw_added_since,
                "retrieval": retrieval,
                "elapsed_ms": 0,
                "error": None,
            }
        )

    # Identity headers from the auth-gate (forwarded by nginx via
    # ``auth_request_set $auth_uid $upstream_http_x_auth_uid;`` then
    # ``proxy_set_header X-Auth-Uid $auth_uid;``). Empty when missing —
    # the worker treats empty as NULL and the exporter treats NULL as
    # ``_none`` (the user-counter explicitly skips ``_none`` entries).
    # ``role`` was already resolved above (request-time, default
    # search-user) and is reused here so the worker logs the same role
    # the authorisation check used.
    user_uid = (request.headers.get("X-Auth-Uid") or "").strip() or None
    # role_for_log is the raw header value (or None) — keeps the
    # query-log row honest about whether the request came in WITH a
    # role claim or WITHOUT. The authorisation decision uses the
    # default-search-user resolution above; the log records what the
    # gate actually emitted.
    role_for_log = (request.headers.get("X-Auth-Role") or "").strip() or None

    # If the caller did NOT pin a specific source, hand the worker the
    # role's allowed-source set as a hard constraint. Without this, a
    # search-user querying without ?source= would hit every source the
    # eichi index knows about — including the private ones we are
    # trying to wall off. The eichi backstop (Phase D) re-applies
    # the same constraint at the SQL layer.
    allowed_sources_for_worker: list[str] | None
    if raw_source is None:
        allowed_sources_for_worker = sorted(role_sources)
    else:
        # An explicit ?source= was already vetted above against the
        # role's allowlist — no need to re-pass the broader set.
        allowed_sources_for_worker = None

    started = time.monotonic()
    raw_results, err = _run_search(
        raw_query,
        k=k,
        source=raw_source,
        year_min=year_min,
        year_max=year_max,
        added_since=raw_added_since,
        retrieval=retrieval,
        user_uid=user_uid,
        role=role_for_log,
        allowed_sources=allowed_sources_for_worker,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    # Deep-link enrichment (q-2026-05-03-db00). For embiguity-content
    # and kavita rows we need the emby_id / library_id stamped in
    # chunk_idx=0 of the doc. Single batched SELECT keeps the latency
    # bounded (~few ms even for k=20) and only runs when there's at
    # least one deep-link-eligible row in the result set.
    deep_link_targets: list[tuple[str, str]] = []
    for r in raw_results:
        src = (r.get("source") or "").lower()
        if src in ("embiguity-content", "kavita"):
            path = r.get("path") or ""
            if path:
                deep_link_targets.append((src, path))
    deep_link_ids_by_path = (
        _fetch_chunk0_metadata(deep_link_targets) if deep_link_targets else {}
    )

    shaped: list[dict[str, Any]] = []
    for r in raw_results:
        ids = deep_link_ids_by_path.get(
            ((r.get("source") or "").lower(), r.get("path") or "")
        )
        shaped.append(_shape(r, deep_link_ids=ids))

    return jsonify(
        {
            "ok": err is None,
            "results": shaped,
            "query": raw_query,
            "k": k,
            "source": raw_source,
            "year_min": year_min,
            "year_max": year_max,
            "added_since": raw_added_since,
            "retrieval": retrieval,
            "elapsed_ms": elapsed_ms,
            "error": err,
        }
    )


@app.route("/healthz")
def healthz() -> Any:
    return ("ok\n", 200, {"Content-Type": "text/plain"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
