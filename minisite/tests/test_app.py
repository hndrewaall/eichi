"""Unit tests for the search-minisite Flask app.

Run:
    cd <repo-root>/minisite
    uv venv --python 3.12
    uv pip install -p .venv/bin/python flask pytest
    .venv/bin/python -m pytest tests/ -v

The tests stub out the eichi subprocess via monkey-patching
``app._run_search`` so they don't require the host's heavy ML stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as search_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """Flask test client with a stubbed eichi backend."""
    flask_app = search_app.app
    flask_app.config["TESTING"] = True

    # Default stub: return a fixed two-result list. Individual tests
    # override via monkeypatch.setattr(search_app, '_run_search', ...).
    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.42,
                    "source": "signal-chat",
                    "path": "signal-chat:dm:andrew:1",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": f"first hit for {query}",
                },
                {
                    "score": 0.51,
                    "source": "repo-md",
                    "path": "media-tools/README.md",
                    "chunk_idx": 1,
                    "offset": 240,
                    "snippet": f"second hit for {query}",
                },
            ][:k],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    return flask_app.test_client()


# ----------------------------------------------------------------------
# happy paths
# ----------------------------------------------------------------------


def test_healthz_unauth(client):
    """/healthz must work without an auth header (proxy bypasses gate)."""
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.data == b"ok\n"
    assert r.headers["Content-Type"].startswith("text/plain")


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"eichi search" in r.data
    assert b"search-q" in r.data
    assert b"morphdom-2.7.4.min.js" in r.data
    assert b"eichi" in r.data  # bottombar credit


def test_search_happy_path(client):
    r = client.get("/api/search?q=test+query")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["query"] == "test query"
    assert body["k"] == 20
    assert body["source"] is None
    assert body["error"] is None
    assert isinstance(body["elapsed_ms"], int)
    assert len(body["results"]) == 2
    first = body["results"][0]
    # _shape() stamps a stable composite key for morphdom.
    assert first["key"] == "signal-chat:signal-chat:dm:andrew:1:0"
    assert first["snippet"] == "first hit for test query"


def test_empty_query_returns_no_results_no_cli_call(client, monkeypatch):
    """Empty query short-circuits — eichi must NOT be invoked."""
    called = {"n": 0}

    def _boom(query, *, k, source, **_kw):
        called["n"] += 1
        raise AssertionError("_run_search must not be called for empty query")

    monkeypatch.setattr(search_app, "_run_search", _boom)
    r = client.get("/api/search?q=")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["results"] == []
    assert body["query"] == ""
    assert called["n"] == 0


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------


def test_bad_k_param(client):
    r = client.get("/api/search?q=foo&k=notanumber")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "integer" in body["error"]


def test_k_clamped(client):
    """k > MAX_K should be clamped down, not rejected."""
    r = client.get(f"/api/search?q=foo&k={search_app.MAX_K + 50}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["k"] == search_app.MAX_K


def test_bad_source_filter(client):
    # Unknown / typo'd source falls outside any role's allowlist → 403
    # (per the search-restricted-access design, the rejection no longer
    # leaks ALL_SOURCES — only the role's own allowed set is named).
    r = client.get("/api/search?q=foo&source=evilsource")
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False
    assert "not allowed" in body["error"]


def test_query_too_long(client):
    r = client.get("/api/search?q=" + ("x" * (search_app.MAX_QUERY_LEN + 1)))
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "too long" in body["error"]


def test_source_filter_passes_through(client, monkeypatch):
    """A valid source filter must reach _run_search as the `source` kwarg."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured["query"] = query
        captured["k"] = k
        captured["source"] = source
        captured["kw"] = kw
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    # signal-chat is NOT in the default search-user role's allowlist, so
    # we send X-Auth-Role: admin to keep the historical happy-path
    # (previously every source passed through unconditionally).
    r = client.get(
        "/api/search?q=foo&source=signal-chat&k=5",
        headers={"X-Auth-Role": "admin"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert captured["query"] == "foo"
    assert captured["k"] == 5
    assert captured["source"] == "signal-chat"
    # No filter facets passed → all kwargs default to None.  An
    # explicit ?source= was passed, so allowed_sources stays None
    # (the explicit source is its own constraint).
    assert captured["kw"] == {
        "year_min": None,
        "year_max": None,
        "added_since": None,
        "retrieval": None,
        "user_uid": None,
        "role": "admin",
        "allowed_sources": None,
    }


# ----------------------------------------------------------------------
# error surface
# ----------------------------------------------------------------------


def test_search_error_surfaces(client, monkeypatch):
    """When _run_search reports an error string, it must reach the JSON."""

    def _err(query, *, k, source, **_kw):
        return ([], "eichi exit 1: boom")

    monkeypatch.setattr(search_app, "_run_search", _err)
    r = client.get("/api/search?q=anything")
    assert r.status_code == 200  # 200 with ok=False — the UI surfaces the banner
    body = r.get_json()
    assert body["ok"] is False
    assert body["error"] == "eichi exit 1: boom"
    assert body["results"] == []


def test_search_subprocess_failure_surfaces_safely(monkeypatch):
    """Worker spawn failure (binary missing) must surface a clean error.

    Exercises the real _Worker spawn path — bypasses the fixture stub
    by reaching for the module-level _worker singleton directly.
    """
    monkeypatch.setattr(search_app, "EICHI_PYTHON", "/nonexistent/python")
    # Reset the worker so it has to re-spawn under the bad python.
    search_app._worker = search_app._Worker()
    flask_app = search_app.app
    flask_app.config["TESTING"] = True
    c = flask_app.test_client()
    r = c.get("/api/search?q=hello")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert "spawn worker" in (body["error"] or "")


# ----------------------------------------------------------------------
# shape / key stability
# ----------------------------------------------------------------------


def test_result_shape_has_stable_key(client):
    r = client.get("/api/search?q=anything")
    body = r.get_json()
    keys = [it["key"] for it in body["results"]]
    # Distinct keys per row + composed of source:path:chunk_idx.
    assert len(set(keys)) == len(keys)
    for it in body["results"]:
        assert it["key"].startswith(it["source"] + ":")


def test_long_snippet_truncated():
    """_shape() must clip snippets > 600 chars."""
    long_snip = "x" * 800
    rec = {"score": 0.5, "source": "x", "path": "p", "chunk_idx": 0, "offset": 0, "snippet": long_snip}
    out = search_app._shape(rec)
    assert len(out["snippet"]) <= 601  # 600 + "…"
    assert out["snippet"].endswith("…")


# ----------------------------------------------------------------------
# timestamp surface
# ----------------------------------------------------------------------


def test_shape_prefers_mtime_over_indexed():
    """When both timestamps are present, mtime wins; ts_iso is UTC ISO8601."""
    rec = {
        "score": 0.4,
        "source": "signal-chat",
        "path": "signal-chat:dm:andrew:1",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "yo",
        "mtime": 1_700_000_000.0,
        "indexed_at_unix": 1_777_000_000.0,
    }
    out = search_app._shape(rec)
    assert out["ts"] == 1_700_000_000.0
    assert out["ts_kind"] == "mtime"
    # 2023-11-14T22:13:20Z  (1700000000)
    assert out["ts_iso"] == "2023-11-14T22:13:20Z"


def test_shape_falls_back_to_indexed_when_mtime_missing():
    rec = {
        "score": 0.4,
        "source": "memory",
        "path": "/m/foo.md",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "bar",
        "mtime": 0.0,
        "indexed_at_unix": 1_777_000_000.0,
    }
    out = search_app._shape(rec)
    assert out["ts"] == 1_777_000_000.0
    assert out["ts_kind"] == "indexed"
    assert out["ts_iso"] == "2026-04-24T03:06:40Z"


def test_shape_returns_null_ts_when_neither_populated():
    rec = {
        "score": 0.4,
        "source": "x",
        "path": "p",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "z",
    }
    out = search_app._shape(rec)
    assert out["ts"] is None
    assert out["ts_kind"] == ""
    assert out["ts_iso"] is None


def test_api_search_carries_timestamp_fields(client, monkeypatch):
    """End-to-end: the JSON envelope round-trips ts / ts_iso / ts_kind /
    mtime / indexed_at_unix from the worker shape through _shape."""
    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.42,
                    "source": "signal-chat",
                    "path": "signal-chat:dm:andrew:1",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "hi",
                    "mtime": 1_700_000_000.0,
                    "indexed_at_unix": 1_777_000_000.0,
                },
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    r = client.get("/api/search?q=anything")
    assert r.status_code == 200
    body = r.get_json()
    first = body["results"][0]
    assert first["mtime"] == 1_700_000_000.0
    assert first["indexed_at_unix"] == 1_777_000_000.0
    assert first["ts"] == 1_700_000_000.0
    assert first["ts_kind"] == "mtime"
    assert first["ts_iso"] == "2023-11-14T22:13:20Z"


def test_iso_utc_helper_handles_garbage():
    """_iso_utc must defensively return None for None / 0 / negative / NaN."""
    assert search_app._iso_utc(None) is None
    assert search_app._iso_utc(0) is None
    assert search_app._iso_utc(-1) is None
    # Float that's well within range (1700000000 = 2023-11-14T22:13:20Z).
    assert search_app._iso_utc(1_700_000_000.0) == "2023-11-14T22:13:20Z"


# ----------------------------------------------------------------------
# cluster surface (kind / cluster_id / cluster_size / mtime_end)
# ----------------------------------------------------------------------


def test_shape_passes_cluster_fields_through():
    """Cluster-row record from the worker must round-trip kind /
    cluster_id / cluster_size / mtime_end / mtime_end_iso into the API
    response shape."""
    rec = {
        "score": 0.31,
        "source": "signal-chat",
        "path": "signal-chat:dm:andrew:2026-05-02:cluster:42",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "[2026-05-02 13:41 ET] andrew: yo\n[2026-05-02 13:48 ET] andrew: more",
        "mtime": 1_777_000_000.0,       # cluster start
        "mtime_end": 1_777_000_420.0,   # +7 minutes
        "indexed_at_unix": 1_777_500_000.0,
        "kind": "cluster",
        "cluster_id": "cluster:42",
        "cluster_size": 5,
    }
    out = search_app._shape(rec)
    assert out["kind"] == "cluster"
    assert out["cluster_id"] == "cluster:42"
    assert out["cluster_size"] == 5
    assert out["mtime_end"] == 1_777_000_420.0
    assert out["mtime_end_iso"] == search_app._iso_utc(1_777_000_420.0)
    # Non-cluster fields still populated.
    assert out["ts"] == 1_777_000_000.0
    assert out["ts_kind"] == "mtime"


def test_shape_msg_row_has_empty_cluster_fields():
    """kind="msg" rows expose the field but cluster_size==0 / mtime_end==0
    so the front-end's `kind === "cluster"` guard skips badge rendering."""
    rec = {
        "score": 0.4,
        "source": "signal-chat",
        "path": "signal-chat:dm:andrew:2026-05-02:msg:7",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "single message",
        "mtime": 1_777_000_000.0,
        "indexed_at_unix": 0.0,
        "kind": "msg",
        "cluster_id": "cluster:42",
        "cluster_size": 0,
        "mtime_end": 0.0,
    }
    out = search_app._shape(rec)
    assert out["kind"] == "msg"
    assert out["cluster_id"] == "cluster:42"
    assert out["cluster_size"] == 0
    assert out["mtime_end"] == 0.0
    assert out["mtime_end_iso"] is None


def test_shape_legacy_row_has_empty_cluster_fields():
    """Legacy / non-signal connectors that never carried cluster columns
    must still produce a sane envelope (no KeyError, defaults applied)."""
    rec = {
        "score": 0.5,
        "source": "repo-md",
        "path": "media-tools/README.md",
        "chunk_idx": 1,
        "offset": 240,
        "snippet": "legacy",
    }
    out = search_app._shape(rec)
    assert out["kind"] == ""
    assert out["cluster_id"] == ""
    assert out["cluster_size"] == 0
    assert out["mtime_end"] == 0.0
    assert out["mtime_end_iso"] is None


def test_shape_cluster_size_coerces_garbage():
    """Defensive coercion — an exotic upstream value (string / None) must
    not blow up _shape()."""
    rec = {
        "score": 0.4,
        "source": "signal-chat",
        "path": "p",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "z",
        "kind": "cluster",
        "cluster_size": "not-an-int",
        "mtime_end": "garbage",
    }
    out = search_app._shape(rec)
    assert out["cluster_size"] == 0
    assert out["mtime_end"] == 0.0
    assert out["mtime_end_iso"] is None


def test_api_search_carries_cluster_fields(client, monkeypatch):
    """End-to-end: cluster fields land in the JSON envelope under
    /api/search and survive the _shape pass intact."""
    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.31,
                    "source": "signal-chat",
                    "path": "signal-chat:dm:andrew:cluster:42",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "yo … more",
                    "mtime": 1_777_000_000.0,
                    "mtime_end": 1_777_000_420.0,
                    "indexed_at_unix": 0.0,
                    "kind": "cluster",
                    "cluster_id": "cluster:42",
                    "cluster_size": 5,
                },
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    r = client.get("/api/search?q=anything")
    assert r.status_code == 200
    body = r.get_json()
    first = body["results"][0]
    assert first["kind"] == "cluster"
    assert first["cluster_id"] == "cluster:42"
    assert first["cluster_size"] == 5
    assert first["mtime_end"] == 1_777_000_420.0
    assert first["mtime_end_iso"] == search_app._iso_utc(1_777_000_420.0)


def test_api_search_msg_row_has_no_cluster_size(client, monkeypatch):
    """A msg row from the worker must surface cluster_size==0 / mtime_end==0
    so the UI omits the badge and timestamp range."""
    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.4,
                    "source": "signal-chat",
                    "path": "signal-chat:dm:andrew:msg:7",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "hi",
                    "mtime": 1_777_000_000.0,
                    "indexed_at_unix": 0.0,
                    "kind": "msg",
                    "cluster_id": "cluster:42",
                    "cluster_size": 0,
                    "mtime_end": 0.0,
                },
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    r = client.get("/api/search?q=anything")
    body = r.get_json()
    first = body["results"][0]
    assert first["kind"] == "msg"
    assert first["cluster_size"] == 0
    assert first["mtime_end"] == 0.0
    assert first["mtime_end_iso"] is None


# ----------------------------------------------------------------------
# date-aware surface (library_added_at_unix / library_added_at_iso /
# release_year)
# ----------------------------------------------------------------------


def test_shape_carries_date_aware_fields():
    """library_added_at_unix + library_added_at_iso + release_year must
    flow through _shape unchanged. ISO is UTC; 0 sentinel → None."""
    rec = {
        "score": 0.4,
        "source": "calibre",
        "path": "calibre:book:42",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "Some Book (1999)",
        "mtime": 1_777_000_000.0,
        "indexed_at_unix": 1_780_000_000.0,
        "library_added_at_unix": 1_700_000_000.0,
        "release_year": 1999,
    }
    out = search_app._shape(rec)
    assert out["library_added_at_unix"] == 1_700_000_000.0
    assert out["library_added_at_iso"] == "2023-11-14T22:13:20Z"
    assert out["release_year"] == 1999


def test_shape_handles_missing_date_aware_fields():
    """Legacy worker rows that don't carry the new fields must default
    cleanly: library_added_at_unix=0.0, library_added_at_iso=None,
    release_year=0."""
    rec = {
        "score": 0.4,
        "source": "memory",
        "path": "/m/foo.md",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "z",
    }
    out = search_app._shape(rec)
    assert out["library_added_at_unix"] == 0.0
    assert out["library_added_at_iso"] is None
    assert out["release_year"] == 0


# ----------------------------------------------------------------------
# filter facets (q-2026-05-02-5864) — year_min / year_max / added_since /
# retrieval URL params reach the worker stub.
# ----------------------------------------------------------------------


def test_year_filter_passes_through(client, monkeypatch):
    """year_min + year_max must reach _run_search as ints; the JSON
    envelope must echo them back so deep-link UIs can render the active
    filter chip."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured.update(kw)
        captured["query"] = query
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get("/api/search?q=film&year_min=2010&year_max=2020")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["year_min"] == 2010
    assert body["year_max"] == 2020
    assert captured["year_min"] == 2010
    assert captured["year_max"] == 2020
    # Untouched facets must be None when the URL omits them.
    assert captured["added_since"] is None
    assert captured["retrieval"] is None


def test_added_since_passes_through(client, monkeypatch):
    """added_since duration token (e.g. ``30d``) is passed through as
    a string — the worker resolves it to a unix cutoff. The JSON
    envelope echoes the raw token."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured.update(kw)
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get("/api/search?q=meeting&added_since=30d")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["added_since"] == "30d"
    assert captured["added_since"] == "30d"


def test_no_filter_params_regression(client, monkeypatch):
    """Regression: the API still works when called with only ?q= (no
    facet params), and the new echoed-back fields surface as None / no
    crash. Mirrors the original happy-path contract pre-q-5864."""
    seen_kw = {}

    def _spy(query, *, k, source, **kw):
        seen_kw.update(kw)
        return ([{"score": 0.4, "source": "memory", "path": "m/foo.md",
                  "chunk_idx": 0, "offset": 0, "snippet": "hi"}], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get("/api/search?q=hello")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["year_min"] is None
    assert body["year_max"] is None
    assert body["added_since"] is None
    assert body["retrieval"] is None
    # No X-Auth-Role header → role_for_log is None; allowed_sources
    # is the default search-user set (default role when header absent).
    assert seen_kw["year_min"] is None
    assert seen_kw["year_max"] is None
    assert seen_kw["added_since"] is None
    assert seen_kw["retrieval"] is None
    assert seen_kw["user_uid"] is None
    assert seen_kw["role"] is None
    assert seen_kw["allowed_sources"] == sorted(
        search_app.sources_for_role("search-user")
    )
    assert len(body["results"]) == 1


def test_retrieval_mode_passes_through(client, monkeypatch):
    """retrieval=vector / bm25 must reach the worker; unknown modes 400."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured.update(kw)
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get("/api/search?q=foo&retrieval=vector")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["retrieval"] == "vector"
    assert captured["retrieval"] == "vector"

    r = client.get("/api/search?q=foo&retrieval=evilmode")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "retrieval" in body["error"]


def test_year_range_inverted_rejected(client):
    """year_min > year_max is a 400 — keeps the worker from running a
    query that's guaranteed to return zero results."""
    r = client.get("/api/search?q=foo&year_min=2025&year_max=2010")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "year_min" in body["error"] and "year_max" in body["error"]


def test_year_out_of_range_rejected(client):
    """Year outside [YEAR_MIN_FLOOR, YEAR_MAX_CEIL] is a 400."""
    r = client.get(f"/api/search?q=foo&year_min={search_app.YEAR_MIN_FLOOR - 1}")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "out of range" in body["error"]


def test_year_non_integer_rejected(client):
    """Non-integer year is a 400."""
    r = client.get("/api/search?q=foo&year_min=notayear")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "integer" in body["error"]


def test_unknown_added_since_token_rejected(client):
    """An added_since outside the dropdown allowlist is a 400."""
    r = client.get("/api/search?q=foo&added_since=99weeks")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "added_since" in body["error"]


def test_index_renders_filter_inputs(client):
    """The HTML form must include the new filter fields after q-5864."""
    r = client.get("/")
    assert r.status_code == 200
    assert b"search-year-min" in r.data
    assert b"search-year-max" in r.data
    assert b"search-added-since" in r.data
    assert b"search-retrieval" in r.data


# ----------------------------------------------------------------------
# auth-gate identity forwarding (caller=web user_uid + role)
# ----------------------------------------------------------------------


def test_auth_uid_header_forwarded_to_worker(client, monkeypatch):
    """X-Auth-Uid from the auth-gate must reach _run_search as user_uid."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured["kw"] = kw
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get(
        "/api/search?q=foo",
        headers={
            "X-Auth-Uid": "firebase-uid-andrew",
            "X-Auth-Role": "admin",
        },
    )
    assert r.status_code == 200
    assert captured["kw"]["user_uid"] == "firebase-uid-andrew"
    assert captured["kw"]["role"] == "admin"


def test_auth_uid_missing_passes_none(client, monkeypatch):
    """Missing auth headers (e.g. healthz / dev mode) → None, not empty
    string. The worker treats None as 'no identity', which is what the
    exporter wants."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured["kw"] = kw
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get("/api/search?q=foo")
    assert r.status_code == 200
    assert captured["kw"]["user_uid"] is None
    assert captured["kw"]["role"] is None


def test_auth_uid_whitespace_normalized_to_none(client, monkeypatch):
    """A whitespace-only header (header injected but empty) should
    coerce to None — matches the worker's truthiness check."""
    captured = {}

    def _spy(query, *, k, source, **kw):
        captured["kw"] = kw
        return ([], None)

    monkeypatch.setattr(search_app, "_run_search", _spy)
    r = client.get(
        "/api/search?q=foo",
        headers={"X-Auth-Uid": "   ", "X-Auth-Role": ""},
    )
    assert r.status_code == 200
    assert captured["kw"]["user_uid"] is None
    assert captured["kw"]["role"] is None


# ----------------------------------------------------------------------
# similarity / relevance-band surface
#
# The eichi worker projects raw vec0 L2 distance into a 0–1 similarity
# + named band (strong/moderate/weak/distant). The frontend renders the
# similarity number + band chip; raw `score` (distance) is preserved for
# backwards compat. These tests pin both the pass-through path and the
# defensive defaults for legacy / older-worker rows.
# ----------------------------------------------------------------------


def test_shape_passes_similarity_and_band_through():
    """A worker record carrying similarity + relevance_band must round-trip
    through _shape unchanged."""
    rec = {
        "score": 0.85,
        "similarity": 0.39,
        "relevance_band": "moderate",
        "source": "memory",
        "path": "/m/foo.md",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "yo",
    }
    out = search_app._shape(rec)
    assert out["score"] == 0.85
    assert out["similarity"] == pytest.approx(0.39)
    assert out["relevance_band"] == "moderate"


def test_shape_clamps_out_of_range_similarity():
    """Defensive: a worker that ships similarity > 1 / < 0 must be clamped
    rather than passed through — keeps the UI's 0-1 contract honest."""
    rec_high = {
        "score": -0.1,
        "similarity": 1.7,
        "relevance_band": "strong",
        "source": "x", "path": "p", "chunk_idx": 0, "offset": 0, "snippet": "z",
    }
    rec_low = {
        "score": 5.0,
        "similarity": -0.3,
        "relevance_band": "distant",
        "source": "x", "path": "p", "chunk_idx": 0, "offset": 0, "snippet": "z",
    }
    assert search_app._shape(rec_high)["similarity"] == 1.0
    assert search_app._shape(rec_low)["similarity"] == 0.0


def test_shape_legacy_worker_row_has_null_similarity():
    """Older worker rows (rolling upgrade) won't carry the new fields. _shape
    must default similarity → None and relevance_band → None rather than
    crash, so the frontend can fall through to the bare similarity render."""
    rec = {
        "score": 0.85,
        "source": "memory",
        "path": "/m/foo.md",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "yo",
    }
    out = search_app._shape(rec)
    assert out["similarity"] is None
    assert out["relevance_band"] is None
    # Raw score still present — this is the backwards-compat fallback.
    assert out["score"] == 0.85


def test_shape_handles_garbage_similarity():
    """Non-numeric similarity (string/None) coerces to None defensively."""
    rec = {
        "score": 0.5,
        "similarity": "not-a-float",
        "relevance_band": "moderate",
        "source": "x", "path": "p", "chunk_idx": 0, "offset": 0, "snippet": "z",
    }
    out = search_app._shape(rec)
    assert out["similarity"] is None
    assert out["relevance_band"] == "moderate"


def test_api_search_carries_similarity_and_band(client, monkeypatch):
    """End-to-end: similarity + relevance_band reach the JSON envelope under
    /api/search."""
    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.6,
                    "similarity": 0.57,
                    "relevance_band": "strong",
                    "source": "calibre",
                    "path": "calibre:book:1",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "some matching text",
                },
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    r = client.get("/api/search?q=shonen")
    assert r.status_code == 200
    body = r.get_json()
    first = body["results"][0]
    assert first["score"] == 0.6
    assert first["similarity"] == pytest.approx(0.57)
    assert first["relevance_band"] == "strong"


def test_shape_coerces_non_string_band():
    """A worker that accidentally ships an integer band (regression bait)
    must still produce a string in the envelope — the frontend builds
    a CSS class name from this value."""
    rec = {
        "score": 0.5,
        "similarity": 0.6,
        "relevance_band": 42,
        "source": "x", "path": "p", "chunk_idx": 0, "offset": 0, "snippet": "z",
    }
    out = search_app._shape(rec)
    assert out["relevance_band"] == "42"


# ----------------------------------------------------------------------
# embedding model label
# ----------------------------------------------------------------------


def test_read_embedding_model_label_from_db(monkeypatch, tmp_path):
    """The footer's model name comes from the eichi DB's meta table.

    Hard-coding the string drifts when eichi upgrades the embedder
    (q-2026-05-03-d5c5 — SPA was still showing "all-MiniLM-L6-v2"
    after eichi had moved to all-mpnet-base-v2).
    """
    import sqlite3

    db_path = tmp_path / "v.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute(
        "INSERT INTO meta VALUES ('embedding_model', "
        "'sentence-transformers/all-mpnet-base-v2')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(search_app, "EICHI_DB", str(db_path))
    label = search_app._read_embedding_model_label()
    # Org/repo prefix stripped for the footer.
    assert label == "all-mpnet-base-v2"


def test_read_embedding_model_label_falls_back_when_db_missing(monkeypatch, tmp_path):
    """Missing DB → fallback string, never an exception."""
    monkeypatch.setattr(search_app, "EICHI_DB", str(tmp_path / "nope.db"))
    assert search_app._read_embedding_model_label() == search_app.EMBEDDING_MODEL_FALLBACK


def test_read_embedding_model_label_falls_back_when_key_missing(monkeypatch, tmp_path):
    """DB without the meta key → fallback string."""
    import sqlite3
    db_path = tmp_path / "v.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(search_app, "EICHI_DB", str(db_path))
    assert search_app._read_embedding_model_label() == search_app.EMBEDDING_MODEL_FALLBACK


def test_index_template_renders_embedding_model(client, monkeypatch):
    """The footer reads ``embedding_model`` from the render context."""
    monkeypatch.setattr(search_app, "EMBEDDING_MODEL_LABEL", "all-mpnet-base-v2")
    r = client.get("/")
    assert r.status_code == 200
    # The new dynamic label.
    assert b"all-mpnet-base-v2" in r.data
    # The stale string must not leak through.
    assert b"all-MiniLM-L6-v2" not in r.data



# ----------------------------------------------------------------------
# Restricted-access — per-role source authorisation (T1-T14 from the
# search-site-restricted-access design doc s9).
#
# The auth-gate stamps X-Auth-Role on every authenticated request after
# Phase A. search-minisite resolves that to a per-role source allowlist;
# tests below exercise the allowlist + the worker call surface (the
# eichi backstop is exercised separately in eichi's own suite).
# ----------------------------------------------------------------------


PRIVATE_SOURCES = {
    "signal-chat",
    "obsidian",
    "repo-md",
    "queue-logs",
    "memory",
    "embiguity-requests",
}
SEARCH_USER_SOURCES = {
    "calibre",
    "embiguity-content",
    "kavita",
    "navidrome-albums",
    "navidrome-artists",
}


def _spy_run_search(monkeypatch, captured):
    """Install a stub that captures kwargs + returns the requested rows."""

    def _stub(query, *, k, source, **kw):
        captured["query"] = query
        captured["k"] = k
        captured["source"] = source
        captured["kw"] = kw
        # Return a synthetic result whose source matches the explicit
        # filter (when set), or one row from each allowed source —
        # enough material for the privacy assertions to bite.
        if source:
            return (
                [
                    {
                        "score": 0.4,
                        "source": source,
                        "path": f"{source}:row:1",
                        "chunk_idx": 0,
                        "offset": 0,
                        "snippet": "stub",
                    }
                ],
                None,
            )
        # source=None — synthesise one row per allowed source so the
        # caller can verify the filter.
        allowed = kw.get("allowed_sources") or sorted(SEARCH_USER_SOURCES)
        return (
            [
                {
                    "score": 0.4 + (i * 0.01),
                    "source": s,
                    "path": f"{s}:row:1",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "stub",
                }
                for i, s in enumerate(allowed)
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)


# T1 — Unauthenticated request: handled by nginx + auth-gate, not by
# search-minisite. We skip the unit test here (it lives in
# auth-gate/app_test.py — VerifyEndpointTests).


# T2 — search-user explicit private source → 403.
def test_t2_search_user_signal_chat_403(client):
    r = client.get(
        "/api/search?q=foo&source=signal-chat",
        headers={"X-Auth-Role": "search-user"},
    )
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False
    assert "not allowed" in body["error"]


# T3 — search-user explicit obsidian → 403.
def test_t3_search_user_obsidian_403(client):
    r = client.get(
        "/api/search?q=foo&source=obsidian",
        headers={"X-Auth-Role": "search-user"},
    )
    assert r.status_code == 403


# T4 — search-user explicit repo-md → 403.
def test_t4_search_user_repo_md_403(client):
    r = client.get(
        "/api/search?q=foo&source=repo-md",
        headers={"X-Auth-Role": "search-user"},
    )
    assert r.status_code == 403


# T5 — search-user without ?source= → results from media sources only,
# allowed_sources passed through to the worker.
def test_t5_search_user_no_filter_constrains_to_media_sources(
    client, monkeypatch
):
    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get(
        "/api/search?q=foo", headers={"X-Auth-Role": "search-user"}
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    # Worker received the search-user allowlist.
    assert sorted(captured["kw"]["allowed_sources"]) == sorted(
        SEARCH_USER_SOURCES
    )
    # Result rows carry only allowed source tags. (The eichi backstop
    # would enforce this at the SQL layer too — the unit test here pins
    # the contract on the search-minisite side.)
    for row in body["results"]:
        assert row["source"] in SEARCH_USER_SOURCES


# T6 — search-user explicit allowed source.
def test_t6_search_user_calibre_200(client, monkeypatch):
    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get(
        "/api/search?q=foo&source=calibre",
        headers={"X-Auth-Role": "search-user"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert captured["source"] == "calibre"
    # Explicit source overrides the broader allowlist hint.
    assert captured["kw"]["allowed_sources"] is None


# T7 — admin without ?source= → may include any source, allowed_sources
# is the full ALL_SOURCES set.
def test_t7_admin_no_filter_unconstrained(client, monkeypatch):
    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get("/api/search?q=foo", headers={"X-Auth-Role": "admin"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert sorted(captured["kw"]["allowed_sources"]) == sorted(
        search_app.ALL_SOURCES
    )


# T8 — admin explicit private source → 200.
def test_t8_admin_signal_chat_200(client, monkeypatch):
    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get(
        "/api/search?q=foo&source=signal-chat",
        headers={"X-Auth-Role": "admin"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert captured["source"] == "signal-chat"


# T9 — missing X-Auth-Role → treated as search-user (most restrictive).
def test_t9_missing_role_header_treated_as_search_user(client, monkeypatch):
    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get("/api/search?q=foo")
    assert r.status_code == 200
    assert sorted(captured["kw"]["allowed_sources"]) == sorted(
        SEARCH_USER_SOURCES
    )


# T9b — A request with the missing-header default may NOT pin a private
# source (defence-in-depth: same outcome as an explicit search-user role).
def test_t9b_missing_role_header_rejects_private_source(client):
    r = client.get("/api/search?q=foo&source=signal-chat")
    assert r.status_code == 403


# T10 — unknown role string → treated as not-in-SOURCES_BY_ROLE → empty
# allowed set → every explicit ?source= is rejected, source=None hands
# the worker an empty list.
def test_t10_unknown_role_yields_empty_allowlist(client, monkeypatch):
    # Empty allowlist — explicit source rejected.
    r = client.get(
        "/api/search?q=foo&source=calibre",
        headers={"X-Auth-Role": "superuser"},
    )
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False

    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get(
        "/api/search?q=foo", headers={"X-Auth-Role": "superuser"}
    )
    assert r.status_code == 200
    # Empty allowed_sources set — sorted([]) == [].
    assert captured["kw"]["allowed_sources"] == []


# T11 — A new source that lands in ALL_SOURCES but NOT in
# SOURCES_BY_ROLE["search-user"] is admin-only by default.
def test_t11_new_source_default_deny_for_search_user(client, monkeypatch):
    monkeypatch.setattr(
        search_app, "ALL_SOURCES", search_app.ALL_SOURCES | {"new-connector"}
    )
    # SOURCES_BY_ROLE is unchanged → admin still wildcard, search-user still
    # without the new source.
    r = client.get(
        "/api/search?q=foo&source=new-connector",
        headers={"X-Auth-Role": "search-user"},
    )
    assert r.status_code == 403


# T12 — Error message must NOT leak private source names back to a
# search-user. The 403 body lists ONLY the role's own allowed sources.
def test_t12_403_body_does_not_leak_private_sources(client):
    r = client.get(
        "/api/search?q=foo&source=signal-chat",
        headers={"X-Auth-Role": "search-user"},
    )
    assert r.status_code == 403
    body = r.get_json()
    err = body["error"]
    # Allowed sources mentioned (it's the role's own allowlist).
    for src in SEARCH_USER_SOURCES:
        assert src in err
    # Private sources NOT mentioned. (signal-chat IS the rejected
    # input → we tolerate it appearing once as the rejected value, but
    # not the OTHER private sources.)
    for src in PRIVATE_SOURCES - {"signal-chat"}:
        assert src not in err, (
            f"private source {src!r} leaked in error message: {err!r}"
        )


def _extract_source_options(html: str) -> set[str]:
    """Pull the ``<option value="...">`` values out of the source picker
    select (the only one whose id is ``search-source``).

    Avoids false positives from prose elsewhere in the page (the search
    placeholder text mentions ``memory``, etc., which would otherwise
    fail the privacy assertion).
    """
    import re

    m = re.search(
        r'<select[^>]*id="search-source"[^>]*>(.*?)</select>',
        html,
        flags=re.DOTALL,
    )
    if not m:
        return set()
    options_block = m.group(1)
    return {v for v in re.findall(r'<option value="([^"]+)"', options_block) if v}


# T13 — UI source picker only shows allowed sources for the role.
def test_t13_ui_source_picker_filtered_per_role(client):
    r = client.get("/", headers={"X-Auth-Role": "search-user"})
    assert r.status_code == 200
    html = r.data.decode("utf-8", errors="replace")
    options = _extract_source_options(html)
    # Allowed sources surface in the dropdown.
    assert options == SEARCH_USER_SOURCES
    # Private sources MUST NOT appear among the option values.
    for src in PRIVATE_SOURCES:
        assert src not in options, (
            f"private source {src!r} leaked in search-user UI"
        )


def test_t13b_ui_source_picker_admin_sees_everything(client):
    r = client.get("/", headers={"X-Auth-Role": "admin"})
    assert r.status_code == 200
    html = r.data.decode("utf-8", errors="replace")
    options = _extract_source_options(html)
    assert options == set(search_app.ALL_SOURCES)


# T14 — eichi backstop: search-minisite forwards allowed_sources to the
# worker on a no-filter query, so the worker can re-apply the filter at
# the SQL layer. The eichi repo's own suite covers the SQL-level
# enforcement; this test pins the wire-level contract that the kwarg
# arrives at the worker.
def test_t14_eichi_backstop_receives_allowed_sources(client, monkeypatch):
    captured: dict = {}
    _spy_run_search(monkeypatch, captured)
    r = client.get(
        "/api/search?q=foo", headers={"X-Auth-Role": "search-user"}
    )
    assert r.status_code == 200
    # Wire kwarg present, list-typed, equals the role's allowlist.
    assert captured["kw"]["allowed_sources"] is not None
    assert isinstance(captured["kw"]["allowed_sources"], list)
    assert sorted(captured["kw"]["allowed_sources"]) == sorted(
        SEARCH_USER_SOURCES
    )


# ----------------------------------------------------------------------
# sources_for_role() helper (s6a contract).
# ----------------------------------------------------------------------


def test_sources_for_role_admin_wildcard_expands_to_all():
    assert search_app.sources_for_role("admin") == set(search_app.ALL_SOURCES)


def test_sources_for_role_search_user_excludes_private():
    allowed = search_app.sources_for_role("search-user")
    assert allowed == SEARCH_USER_SOURCES
    for src in PRIVATE_SOURCES:
        assert src not in allowed


def test_sources_for_role_unknown_returns_empty_set():
    assert search_app.sources_for_role("supersecret") == set()
    assert search_app.sources_for_role("") == set()

