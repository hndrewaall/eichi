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

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as search_app  # noqa: E402


# Deep-link tests assert against ``search_app.NAVIDROME_BASE_URL`` /
# ``EMBY_BASE_URL`` / ``KAVITA_BASE_URL`` — those default to empty
# strings (no link rendered) in a generic open-source build. Seed
# them with example.com placeholders here so the existing assertions
# still verify the URL-construction logic without leaking real
# hostnames into the source tree.
@pytest.fixture(autouse=True)
def _seed_deep_link_base_urls(monkeypatch):
    monkeypatch.setattr(search_app, "EMBY_BASE_URL", "https://emby.example.com")
    monkeypatch.setattr(
        search_app, "NAVIDROME_BASE_URL", "https://navidrome.example.com"
    )
    monkeypatch.setattr(
        search_app, "KAVITA_BASE_URL", "https://kavita.example.com"
    )


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
        "source": "navidrome-artists",
        "path": "artist:Foo",
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
        "source": "embiguity-content",
        "path": "embiguity:content:show:42",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "One Piece (1999)",
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
                    "source": "embiguity-content",
                    "path": "embiguity:content:show:1",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "shonen anime",
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
# Deep-link surface (q-2026-05-03-db00)
# ----------------------------------------------------------------------
#
# Each result row gets an "Open in <App>" link when the source maps to
# a public-facing user app. Sources covered:
#
#   * embiguity-content (show / movie / episode) → Emby
#   * navidrome-albums                            → Navidrome
#   * navidrome-artists                           → Navidrome
#   * kavita                                      → Kavita
#
# Calibre + memory + signal-chat etc. intentionally produce no link.


def test_app_label_navidrome_albums():
    assert search_app._app_label_for("navidrome-albums") == "Navidrome"


def test_app_label_navidrome_artists():
    assert search_app._app_label_for("navidrome-artists") == "Navidrome"


def test_app_label_kavita():
    assert search_app._app_label_for("kavita") == "Kavita"


def test_app_label_embiguity_content():
    assert search_app._app_label_for("embiguity-content") == "Emby"


def test_app_label_unmapped_source_returns_none():
    """Calibre + signal-chat + memory + obsidian + queue-logs + repo-md +
    grafana-dashboards + embiguity-requests all return None — no public
    deep-link target. Calibre-Web is admin-only per CLAUDE.md."""
    for src in (
        "calibre",
        "signal-chat",
        "memory",
        "obsidian",
        "queue-logs",
        "repo-md",
        "grafana-dashboards",
        "embiguity-requests",
        "",
        "unknown-source",
    ):
        assert search_app._app_label_for(src) is None, src


def test_app_label_case_insensitive():
    assert search_app._app_label_for("NAVIDROME-ALBUMS") == "Navidrome"


def test_build_deep_link_navidrome_album():
    """Album deep links route through ``/__sso/navidrome?next=...`` so
    the browser auto-signs-in via the upstream session cookie before
    landing on the album view (q-2026-05-04-e0c6 fix). The ``#`` in
    the SPA hash route is percent-encoded inside the next= value;
    otherwise the browser would strip it client-side and the bridge
    would never see the deeplink."""
    url = search_app._build_deep_link(
        "navidrome-albums", "navidrome:album:abc123", {}
    )
    assert url == (
        f"{search_app.NAVIDROME_BASE_URL}/__sso/navidrome"
        "?next=%2Fapp%2F%23%2Falbum%2Fabc123%2Fshow"
    )


def test_build_deep_link_navidrome_artist():
    """Artist deep links route through the SSO bridge for the same
    reason as albums (q-2026-05-04-e0c6)."""
    url = search_app._build_deep_link(
        "navidrome-artists", "navidrome:artist:xyz789", {}
    )
    assert url == (
        f"{search_app.NAVIDROME_BASE_URL}/__sso/navidrome"
        "?next=%2Fapp%2F%23%2Fartist%2Fxyz789%2Fshow"
    )


def test_build_deep_link_navidrome_malformed_path_returns_none():
    """Malformed path (no ``navidrome:album:`` prefix) → no link."""
    assert search_app._build_deep_link("navidrome-albums", "garbage", {}) is None


def test_build_deep_link_navidrome_empty_id_returns_none():
    """Empty id after the prefix → no link."""
    assert (
        search_app._build_deep_link("navidrome-albums", "navidrome:album:", {})
        is None
    )


def test_build_deep_link_emby_show():
    """Show deep links route through ``/__sso/emby?next=...`` so a
    click auto-signs-in AND lands on the item detail view (the bare
    ``/web/index.html#!/details?id=N`` URL would drop to native login
    when the MediaBrowser cookie is missing/expired —
    q-2026-05-04-e0c6).

    ``next=`` carries JUST the hashbang fragment payload
    (``details?id=N``); the bridge prepends ``/web/index.html#!/``
    server-side. Earlier iterations passed the full path-with-
    fragment shape (``/web/index.html#/details?id=N``) which the
    SPA's hashbang shim rewrote to ``#!/index.html%23/details``,
    leaving the user on a blank home (regression
    q-2026-05-04-3889).
    """
    url = search_app._build_deep_link(
        "embiguity-content",
        "embiguity:content:show:123",
        {"emby_id": "456"},
    )
    assert url == (
        f"{search_app.EMBY_BASE_URL}/__sso/emby"
        "?next=details%3Fid%3D456"
    )


def test_build_deep_link_emby_movie():
    """Movies route through the SSO bridge for the same reason as
    shows (q-2026-05-04-e0c6).

    Same hashbang-fragment-only ``next=`` shape as shows
    (q-2026-05-04-3889).
    """
    url = search_app._build_deep_link(
        "embiguity-content",
        "embiguity:content:movie:7",
        {"emby_id": "888"},
    )
    assert url == (
        f"{search_app.EMBY_BASE_URL}/__sso/emby"
        "?next=details%3Fid%3D888"
    )


def test_build_deep_link_emby_episode():
    """Episode rows carry the EPISODE's emby_id (not the show's) so the
    deep link lands on the episode-detail view (after SSO bootstrap
    via the bridge — q-2026-05-04-e0c6).

    Same hashbang-fragment-only ``next=`` shape as shows
    (q-2026-05-04-3889).
    """
    url = search_app._build_deep_link(
        "embiguity-content",
        "embiguity:content:episode:42:S1E2",
        {"emby_id": "ep-9999"},
    )
    assert url == (
        f"{search_app.EMBY_BASE_URL}/__sso/emby"
        "?next=details%3Fid%3Dep-9999"
    )


def test_build_deep_link_emby_next_has_no_hash():
    """Regression guard for q-2026-05-04-3889 — the rendered URL must
    NOT contain a literal ``#`` (or its percent-encoded form ``%23``)
    in the ``next=`` value. Earlier shapes that did caused Emby's
    SPA to mis-route deep links to a blank home view.
    """
    url = search_app._build_deep_link(
        "embiguity-content",
        "embiguity:content:show:abc",
        {"emby_id": "618115"},
    )
    assert url is not None
    parsed = urllib.parse.urlsplit(url)
    next_value = urllib.parse.parse_qs(parsed.query)["next"][0]
    assert "#" not in next_value
    # The bridge expects exactly ``details?id=<id>`` after URL-decoding.
    assert next_value == "details?id=618115"


def test_build_deep_link_emby_missing_id_returns_none():
    """No emby_id in the IDs dict (legacy indexed row, before the
    connector stamp landed) → no link rendered. The row still shows
    up; just no clickable button."""
    assert (
        search_app._build_deep_link(
            "embiguity-content", "embiguity:content:show:1", {}
        )
        is None
    )


def test_build_deep_link_kavita():
    """Kavita series deep links route through ``/__sso/kavita?next=...``
    so a click auto-signs-in AND lands on the series page
    (q-2026-05-04-e0c6)."""
    url = search_app._build_deep_link(
        "kavita",
        "kavita:series:1258",
        {"library_id": "1"},
    )
    assert url == (
        f"{search_app.KAVITA_BASE_URL}/__sso/kavita"
        "?next=%2Flibrary%2F1%2Fseries%2F1258"
    )


def test_build_deep_link_kavita_missing_library_id_returns_none():
    """Kavita's series-detail URL requires ``libraryId`` — without it
    the SPA can't resolve the route, so we omit the link entirely."""
    assert (
        search_app._build_deep_link(
            "kavita", "kavita:series:1258", {}
        )
        is None
    )


def test_sso_deeplink_percent_encodes_hash_and_query():
    """Hash + query inside ``target_path`` must be percent-encoded so
    the whole thing fits in a single ``next=`` query value. Without
    encoding, a browser visiting ``/__sso/X?next=/web/index.html#/...?id=N``
    would strip everything past the literal ``#`` BEFORE sending the
    request — the bridge would see ``next=/web/index.html`` and the
    deeplink would silently degrade to the home page."""
    url = search_app._sso_deeplink(
        "https://emby.example.com",
        "/__sso/emby",
        "/web/index.html#/details?id=42",
    )
    # ``#`` -> %23, ``?`` -> %3F, ``=`` -> %3D, ``/`` -> %2F.
    assert url == (
        "https://emby.example.com/__sso/emby"
        "?next=%2Fweb%2Findex.html%23%2Fdetails%3Fid%3D42"
    )


def test_sso_deeplink_round_trips_through_urldecode():
    """The ``next=`` value must round-trip through ``urllib.parse.
    unquote`` to the exact path the bridge will serve, so the bridge's
    same-origin validator (``_safe_next_path``) sees a clean ``/`` -
    prefixed path."""
    target = "/library/1/series/1258"
    url = search_app._sso_deeplink(
        "https://kavita.example.com", "/__sso/kavita", target,
    )
    # Pull the next= value back out.
    parsed = urllib.parse.urlsplit(url)
    next_value = urllib.parse.parse_qs(parsed.query)["next"][0]
    assert next_value == target


def test_build_deep_link_calibre_returns_none():
    """Calibre-Web is admin-only — must NOT surface to users (CLAUDE.md
    rule: the Kavita base URL is the only public-facing book surface)."""
    assert search_app._build_deep_link("calibre", "calibre:1", {}) is None


def test_build_deep_link_unknown_source_returns_none():
    assert search_app._build_deep_link("signal-chat", "x", {}) is None


def test_build_deep_link_empty_inputs_return_none():
    assert search_app._build_deep_link("", "", {}) is None
    assert search_app._build_deep_link("kavita", "", {}) is None


def test_shape_with_deep_link_ids_renders_app_link():
    """When ``deep_link_ids`` carries an emby_id, _shape() emits both
    ``app_link`` (URL) and ``app_label`` (human label) for the
    front-end."""
    rec = {
        "score": 0.4,
        "source": "embiguity-content",
        "path": "embiguity:content:show:42",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "Show: Foo",
    }
    out = search_app._shape(rec, deep_link_ids={"emby_id": "999"})
    assert out["app_link"] == (
        f"{search_app.EMBY_BASE_URL}/__sso/emby"
        "?next=details%3Fid%3D999"
    )
    assert out["app_label"] == "Emby"


def test_shape_without_deep_link_ids_renders_no_app_link():
    """No ``deep_link_ids`` → ``app_link`` is None but ``app_label``
    still surfaces for the source. Front-end checks for non-null
    app_link before rendering the button."""
    rec = {
        "score": 0.4,
        "source": "embiguity-content",
        "path": "embiguity:content:show:42",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "Show: Foo",
    }
    out = search_app._shape(rec)
    assert out["app_link"] is None
    assert out["app_label"] == "Emby"


def test_shape_navidrome_album_no_db_lookup_needed():
    """Navidrome album/artist deep links derive purely from the path —
    no per-row DB lookup required (the IDs are already in the doc_id).
    """
    rec = {
        "score": 0.4,
        "source": "navidrome-albums",
        "path": "navidrome:album:abc123",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "Album: Foo",
    }
    out = search_app._shape(rec)
    assert out["app_link"] == (
        f"{search_app.NAVIDROME_BASE_URL}/__sso/navidrome"
        "?next=%2Fapp%2F%23%2Falbum%2Fabc123%2Fshow"
    )
    assert out["app_label"] == "Navidrome"


def test_shape_calibre_has_no_app_link():
    rec = {
        "score": 0.4,
        "source": "calibre",
        "path": "calibre:7",
        "chunk_idx": 0,
        "offset": 0,
        "snippet": "Book: ...",
    }
    out = search_app._shape(rec)
    assert out["app_link"] is None
    assert out["app_label"] is None


def test_fetch_chunk0_metadata_extracts_emby_and_library_ids(monkeypatch, tmp_path):
    """The DB-side fetcher reads chunk_idx=0 of each (source, path) and
    parses ``emby_id=`` / ``library_id=`` lines out of the body."""
    import sqlite3

    db_path = tmp_path / "v.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunk_meta (rowid INTEGER PRIMARY KEY, source TEXT, "
        "path TEXT, chunk_idx INTEGER, offset INTEGER, text TEXT, "
        "content_hash TEXT, indexed_at REAL)"
    )
    conn.execute(
        "INSERT INTO chunk_meta (source, path, chunk_idx, offset, text, "
        "content_hash, indexed_at) VALUES (?, ?, 0, 0, ?, '', 0)",
        (
            "embiguity-content",
            "embiguity:content:show:1",
            "Show: Foo (2024)\ntype: tv-show\nemby_id=12345\ninstance=emby\ntvdb_id=42",
        ),
    )
    conn.execute(
        "INSERT INTO chunk_meta (source, path, chunk_idx, offset, text, "
        "content_hash, indexed_at) VALUES (?, ?, 0, 0, ?, '', 0)",
        (
            "kavita",
            "kavita:series:99",
            "Series: Bar [Books]\nlibrary_id=1\nType: Books",
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(search_app, "EICHI_DB", str(db_path))
    out = search_app._fetch_chunk0_metadata(
        [
            ("embiguity-content", "embiguity:content:show:1"),
            ("kavita", "kavita:series:99"),
        ]
    )
    assert out[("embiguity-content", "embiguity:content:show:1")] == {
        "emby_id": "12345",
    }
    assert out[("kavita", "kavita:series:99")] == {
        "library_id": "1",
    }


def test_fetch_chunk0_metadata_empty_targets_short_circuits():
    """Empty targets list → empty dict, no DB hit."""
    assert search_app._fetch_chunk0_metadata([]) == {}


def test_fetch_chunk0_metadata_db_missing_returns_empty(monkeypatch, tmp_path):
    """A missing / unreadable DB path → empty dict (no link rendered),
    never an exception. Search must keep working when the deep-link
    side-channel is broken."""
    monkeypatch.setattr(search_app, "EICHI_DB", str(tmp_path / "nope.db"))
    out = search_app._fetch_chunk0_metadata(
        [("embiguity-content", "embiguity:content:show:1")]
    )
    assert out == {}


def test_fetch_chunk0_metadata_skips_missing_rows(monkeypatch, tmp_path):
    """Rows that aren't in chunk_meta → simply absent from the result;
    no exception. Rows whose body lacks the stamps → absent (we only
    populate keys we actually parsed)."""
    import sqlite3

    db_path = tmp_path / "v.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunk_meta (rowid INTEGER PRIMARY KEY, source TEXT, "
        "path TEXT, chunk_idx INTEGER, offset INTEGER, text TEXT, "
        "content_hash TEXT, indexed_at REAL)"
    )
    # A row without any stamps — pre-q-db00 doc.
    conn.execute(
        "INSERT INTO chunk_meta (source, path, chunk_idx, offset, text, "
        "content_hash, indexed_at) VALUES (?, ?, 0, 0, ?, '', 0)",
        (
            "embiguity-content",
            "embiguity:content:show:legacy",
            "Show: Old (2010)\ntype: tv-show\ntvdb_id=1",
        ),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(search_app, "EICHI_DB", str(db_path))
    out = search_app._fetch_chunk0_metadata(
        [
            ("embiguity-content", "embiguity:content:show:legacy"),
            ("embiguity-content", "embiguity:content:show:not-in-db"),
        ]
    )
    assert ("embiguity-content", "embiguity:content:show:legacy") not in out
    assert ("embiguity-content", "embiguity:content:show:not-in-db") not in out


def test_api_search_renders_app_link_for_emby_content(client, monkeypatch, tmp_path):
    """End-to-end: the JSON envelope carries app_link + app_label for
    embiguity-content rows whose chunk_meta has a stamped emby_id."""
    import sqlite3

    db_path = tmp_path / "v.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunk_meta (rowid INTEGER PRIMARY KEY, source TEXT, "
        "path TEXT, chunk_idx INTEGER, offset INTEGER, text TEXT, "
        "content_hash TEXT, indexed_at REAL)"
    )
    conn.execute(
        "INSERT INTO chunk_meta (source, path, chunk_idx, offset, text, "
        "content_hash, indexed_at) VALUES (?, ?, 0, 0, ?, '', 0)",
        (
            "embiguity-content",
            "embiguity:content:show:1",
            "Show: Foo\ntype: tv-show\nemby_id=8675309\n",
        ),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(search_app, "EICHI_DB", str(db_path))

    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.42,
                    "source": "embiguity-content",
                    "path": "embiguity:content:show:1",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "Show: Foo",
                }
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    r = client.get("/api/search?q=foo")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    first = body["results"][0]
    assert first["app_link"] == (
        f"{search_app.EMBY_BASE_URL}/__sso/emby"
        "?next=details%3Fid%3D8675309"
    )
    assert first["app_label"] == "Emby"


def test_api_search_navidrome_album_no_db_lookup(client, monkeypatch):
    """Navidrome album rows derive their deep link purely from the path
    — even when the chunk_meta DB lookup fails (e.g. missing DB), the
    link still renders. Verify by pointing EICHI_DB at a nonexistent
    path; the navidrome row's app_link must still be populated."""
    monkeypatch.setattr(search_app, "EICHI_DB", "/nonexistent/path.db")

    def _stub(query, *, k, source, **_kw):
        return (
            [
                {
                    "score": 0.42,
                    "source": "navidrome-albums",
                    "path": "navidrome:album:abc",
                    "chunk_idx": 0,
                    "offset": 0,
                    "snippet": "Album: Foo",
                }
            ],
            None,
        )

    monkeypatch.setattr(search_app, "_run_search", _stub)
    r = client.get("/api/search?q=foo")
    assert r.status_code == 200
    body = r.get_json()
    first = body["results"][0]
    assert first["app_link"] == (
        f"{search_app.NAVIDROME_BASE_URL}/__sso/navidrome"
        "?next=%2Fapp%2F%23%2Falbum%2Fabc%2Fshow"
    )
    assert first["app_label"] == "Navidrome"


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

