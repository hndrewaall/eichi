"""Tests for cluster/msg dedupe in cmd_query, --granular bypass, and the
index-stream metadata pickup path.

The dedupe logic doesn't need an embedding model (no encode), so we
build SearchHit fixtures directly. The index-stream test stubs the
encoder."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eichi import EMBEDDING_DIM
from eichi.cli import (
    _cluster_size_from_text,
    _dedupe_cluster_overlap,
    _format_hit_timestamp,
    _print_hit,
    build_parser,
)
from eichi.store import SearchHit, open_db, search, stats


def _hit(*, score, kind="", cluster_id="", path="x", mtime=0.0,
         mtime_end=0.0, text="snippet"):
    return SearchHit(
        rowid=1,
        score=score,
        source="signal-chat",
        path=path,
        chunk_idx=0,
        offset=0,
        text=text,
        mtime=mtime,
        indexed_at_unix=0.0,
        kind=kind,
        cluster_id=cluster_id,
        mtime_end=mtime_end,
    )


# --- dedupe tests ---------------------------------------------------------


def test_dedupe_cluster_wins_drops_msg_inside():
    """Cluster ranks higher than msg → msg suppressed."""
    cluster = _hit(score=0.10, kind="cluster", cluster_id="c1",
                   path="signal-chat:cluster:dm:a:1")
    msg = _hit(score=0.20, kind="msg", cluster_id="c1",
               path="signal-chat:dm:a:1")
    out = _dedupe_cluster_overlap([cluster, msg])
    assert out == [cluster]


def test_dedupe_msg_wins_keeps_both():
    """Msg ranks higher AND outside the 5% slack → cluster doesn't shadow."""
    msg = _hit(score=0.10, kind="msg", cluster_id="c1",
               path="signal-chat:dm:a:1")
    cluster = _hit(score=0.50, kind="cluster", cluster_id="c1",
                   path="signal-chat:cluster:dm:a:1")
    out = _dedupe_cluster_overlap([msg, cluster])
    # 0.50 > 0.10 * 1.05 = 0.105, so msg is NOT shadowed by the cluster.
    assert out == [msg, cluster]


def test_dedupe_msg_within_slack_dropped():
    """Cluster ranks within 5% of the msg → msg dropped (cluster represents it)."""
    msg = _hit(score=0.100, kind="msg", cluster_id="c1",
               path="signal-chat:dm:a:1")
    # cluster_score 0.104 <= msg_score * 1.05 = 0.105 → drop msg.
    cluster = _hit(score=0.104, kind="cluster", cluster_id="c1",
                   path="signal-chat:cluster:dm:a:1")
    out = _dedupe_cluster_overlap([msg, cluster])
    assert out == [cluster]


def test_dedupe_msg_from_unrelated_cluster_kept():
    """Msg whose cluster isn't in the result set is unaffected."""
    cluster = _hit(score=0.10, kind="cluster", cluster_id="c1",
                   path="signal-chat:cluster:dm:a:1")
    other_msg = _hit(score=0.20, kind="msg", cluster_id="c2",
                     path="signal-chat:dm:b:99")
    out = _dedupe_cluster_overlap([cluster, other_msg])
    assert out == [cluster, other_msg]


def test_dedupe_legacy_rows_passthrough():
    """Rows with empty kind (legacy file connectors) bypass dedupe entirely."""
    a = _hit(score=0.10, path="/m/a.md")  # kind="" by default
    b = _hit(score=0.20, path="/m/b.md")
    out = _dedupe_cluster_overlap([a, b])
    assert out == [a, b]


def test_dedupe_msg_with_no_cluster_id_kept():
    """Msg with empty cluster_id (e.g. cluster never emitted yet) is kept."""
    msg = _hit(score=0.10, kind="msg", cluster_id="",
               path="signal-chat:dm:a:1")
    cluster = _hit(score=0.05, kind="cluster", cluster_id="c1",
                   path="signal-chat:cluster:dm:b:1")
    out = _dedupe_cluster_overlap([msg, cluster])
    assert out == [msg, cluster]


# --- helper test ---------------------------------------------------------


def test_cluster_size_counts_headers():
    text = (
        "[dm:a] Alice: hello\n"
        "[dm:a] Alice: how are you\n"
        "[dm:a] Alice: still there?"
    )
    assert _cluster_size_from_text(text) == 3


def test_cluster_size_falls_back_to_one_when_no_headers():
    assert _cluster_size_from_text("plain body no header") == 1


# --- print rendering tests -----------------------------------------------


def test_print_hit_text_cluster_prefix():
    h = _hit(
        score=0.4,
        kind="cluster",
        cluster_id="c1",
        path="signal-chat:cluster:dm:a:1",
        text="[dm:a] X: hi\n[dm:a] X: bye",
        mtime=1_700_000_000.0,
        mtime_end=1_700_000_300.0,
    )
    line = _print_hit(h, json_out=False)
    parts = line.split("\t")
    # 5-col layout: path:offset, score, kind_tag, ts, snippet
    assert len(parts) == 5
    assert parts[2] == "[signal-chat cluster, 2 msgs]"
    # Range timestamp: same day, "HH:MM–HH:MM ET" inside the brackets.
    assert "–" in parts[3]


def test_print_hit_text_msg_prefix():
    h = _hit(
        score=0.4,
        kind="msg",
        cluster_id="c1",
        path="signal-chat:dm:a:1",
        text="[dm:a] X: hi",
        mtime=1_700_000_000.0,
    )
    line = _print_hit(h, json_out=False)
    parts = line.split("\t")
    assert parts[2] == "[signal-chat msg]"


def test_print_hit_json_carries_cluster_fields():
    h = _hit(
        score=0.4,
        kind="cluster",
        cluster_id="c1",
        path="signal-chat:cluster:dm:a:1",
        text="[dm:a] X: hi\n[dm:a] X: bye",
        mtime=1_700_000_000.0,
        mtime_end=1_700_000_300.0,
    )
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["kind"] == "cluster"
    assert rec["cluster_id"] == "c1"
    assert rec["cluster_size"] == 2
    assert rec["mtime_end"] == pytest.approx(1_700_000_300.0)


def test_format_cluster_timestamp_range_same_day():
    h = _hit(
        score=0.4,
        kind="cluster",
        cluster_id="c1",
        path="signal-chat:cluster:dm:a:1",
        # 2025-06-01 10:00:00 ET = 1748786400 (approx). Use specific values.
        mtime=1_748_786_400.0,
        # ~7 minutes later, same day.
        mtime_end=1_748_786_820.0,
    )
    out = _format_hit_timestamp(h)
    # Same-day range looks like "[YYYY-MM-DD HH:MM–HH:MM ET]" with optional age.
    assert "–" in out
    # Only ONE date stamp on a same-day range.
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}–\d{2}:\d{2}", out), out


# --- index-stream metadata pickup ----------------------------------------


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
    p = Path(tmp) / "cluster.db"
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


def test_index_stream_picks_up_metadata_kind_and_cluster_id(db_path):
    payload = "\n".join(
        json.dumps(d)
        for d in [
            {
                "source": "signal-chat",
                "doc_id": "signal-chat:dm:a:1",
                "text": "first message",
                "mtime": 1_700_000_000.0,
                "metadata": {
                    "kind": "msg",
                    "cluster_id": "signal-chat:cluster:dm:a:1",
                },
            },
            {
                "source": "signal-chat",
                "doc_id": "signal-chat:cluster:dm:a:1",
                "text": "[dm:a] X: first message\n[dm:a] X: second message",
                "mtime": 1_700_000_000.0,
                "metadata": {
                    "kind": "cluster",
                    "cluster_id": "signal-chat:cluster:dm:a:1",
                    "mtime_end": 1_700_000_300.0,
                },
            },
        ]
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT path, kind, cluster_id, mtime_end FROM files ORDER BY path"
    ).fetchall()
    by_path = {r[0]: r for r in rows}
    assert by_path["signal-chat:dm:a:1"][1] == "msg"
    assert by_path["signal-chat:dm:a:1"][2] == "signal-chat:cluster:dm:a:1"
    assert by_path["signal-chat:dm:a:1"][3] is None
    assert by_path["signal-chat:cluster:dm:a:1"][1] == "cluster"
    assert by_path["signal-chat:cluster:dm:a:1"][2] == "signal-chat:cluster:dm:a:1"
    assert by_path["signal-chat:cluster:dm:a:1"][3] == pytest.approx(1_700_000_300.0)


def test_search_surfaces_kind_and_cluster_id(db_path):
    """End-to-end: index a cluster doc, search returns SearchHit with
    kind/cluster_id/mtime_end populated."""
    payload = json.dumps(
        {
            "source": "signal-chat",
            "doc_id": "signal-chat:cluster:dm:a:1",
            "text": "[dm:a] X: hi",
            "mtime": 1_700_000_000.0,
            "metadata": {
                "kind": "cluster",
                "cluster_id": "signal-chat:cluster:dm:a:1",
                "mtime_end": 1_700_000_120.0,
            },
        }
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    conn = open_db(db_path)
    # Use a stub embedding similar to the doc's stored vector.
    qvec = _stub_encode(["[dm:a] X: hi"])[0]
    hits = search(conn, qvec, k=1)
    assert hits
    h = hits[0]
    assert h.kind == "cluster"
    assert h.cluster_id == "signal-chat:cluster:dm:a:1"
    assert h.mtime_end == pytest.approx(1_700_000_120.0)


# --- cmd_query end-to-end with --granular -------------------------------


def test_cmd_query_granular_disables_dedupe(db_path, capsys):
    """When both a cluster and its constituent msg are indexed and rank
    similarly, default cmd_query suppresses the msg; --granular keeps it."""
    payload = "\n".join(
        json.dumps(d)
        for d in [
            {
                "source": "signal-chat",
                "doc_id": "signal-chat:dm:a:1",
                "text": "alpha bravo charlie",
                "mtime": 1_700_000_000.0,
                "metadata": {
                    "kind": "msg",
                    "cluster_id": "signal-chat:cluster:dm:a:1",
                },
            },
            {
                "source": "signal-chat",
                "doc_id": "signal-chat:cluster:dm:a:1",
                "text": "[dm:a] X: alpha bravo charlie",
                "mtime": 1_700_000_000.0,
                "metadata": {
                    "kind": "cluster",
                    "cluster_id": "signal-chat:cluster:dm:a:1",
                    "mtime_end": 1_700_000_120.0,
                },
            },
        ]
    )
    rc = _run(["index-stream"], payload, db_path)
    assert rc == 0
    # Drain whatever index-stream printed (its summary line) before we
    # capture the query output.
    capsys.readouterr()

    # Build a fake encode_one that returns the same stub embedding as the
    # cluster doc (which makes both msg + cluster perfectly score-tied
    # since they share identical embeddings via the deterministic stub).
    def fake_encode_one(text):
        return _stub_encode([text])[0]

    parser = build_parser()
    # Default — dedupe ON. Expect cluster only (msg is suppressed because
    # cluster ranks within slack).
    args = parser.parse_args([
        "--db", str(db_path), "query", "alpha bravo", "-k", "5", "--json"
    ])
    with patch("eichi.embed.encode_one", side_effect=fake_encode_one), \
         patch("eichi.embed.encode", side_effect=_stub_encode), \
         patch.dict("os.environ", {"VSEARCH_NO_QUERY_LOG": "1"}):
        rc = args.func(args)
    assert rc == 0
    captured = capsys.readouterr().out
    lines = [l for l in captured.splitlines() if l.strip().startswith("{")]
    kinds_default = [json.loads(l)["kind"] for l in lines]

    args = parser.parse_args([
        "--db", str(db_path), "query", "alpha bravo", "-k", "5",
        "--json", "--granular",
    ])
    with patch("eichi.embed.encode_one", side_effect=fake_encode_one), \
         patch("eichi.embed.encode", side_effect=_stub_encode), \
         patch.dict("os.environ", {"VSEARCH_NO_QUERY_LOG": "1"}):
        rc = args.func(args)
    assert rc == 0
    captured = capsys.readouterr().out
    lines_g = [l for l in captured.splitlines() if l.strip().startswith("{")]
    kinds_granular = sorted(json.loads(l)["kind"] for l in lines_g)
    # --granular MUST surface both rows.
    assert "msg" in kinds_granular
    assert "cluster" in kinds_granular
    # Default mode must surface AT MOST what granular surfaces (dedupe is
    # subtractive). If the dedupe kicked in, kinds_default has fewer rows.
    assert len(kinds_default) <= len(kinds_granular)
    # And specifically: msg should NOT appear in default mode (cluster
    # represents it, score-tied within the 5% window).
    assert "msg" not in kinds_default
