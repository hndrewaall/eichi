"""Coverage for the ``--caller`` flag and ``EICHI_CALLER`` env var on
``eichi query``.

Adds a stamp to every query.log entry that splits CLI / web / agent /
mainloop traffic apart in the metrics dashboard. The CLI default is
``cli``; ``$EICHI_CALLER`` (or legacy ``$VSEARCH_CALLER``) overrides when
no flag was passed; the ``--caller`` flag wins outright.

Anything outside the allowlist (cli|web|agent|mainloop) collapses to
``_other`` at log-write time — defense in depth so a typo doesn't
explode label cardinality on the exporter side.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import types
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Allow tests to import the in-repo package without an install.
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from eichi import EMBEDDING_DIM  # noqa: E402
from eichi.cli import (  # noqa: E402
    ALLOWED_CALLERS,
    _resolve_caller,
    build_parser,
)
from eichi.store import add_chunks, open_db  # noqa: E402


def _stub_encode(texts):
    """Deterministic embedding stub — keeps tests offline and fast."""
    out = []
    for t in texts:
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[hash(t) % EMBEDDING_DIM] = 1.0
        out.append(v)
    return np.stack(out)


def _stub_encode_one(text):
    return _stub_encode([text])[0]


def _seed(db_path: Path) -> None:
    """Tiny seeded index so cmd_query has at least one row to chew on."""
    conn = open_db(str(db_path))
    text = "alpha bravo charlie delta"
    embeddings = _stub_encode([text])
    add_chunks(
        conn,
        source="memory",
        path="/seed/doc.md",
        mtime=time.time(),
        file_hash="abc123",
        chunks=[(0, 0, text)],
        embeddings=embeddings,
    )
    conn.close()


@pytest.fixture
def db_and_log(tmp_path, monkeypatch):
    """Per-test DB + query log path. Clears VSEARCH_NO_QUERY_LOG so the
    write actually happens."""
    db_path = tmp_path / "index.db"
    log_path = tmp_path / "query.log"
    _seed(db_path)
    monkeypatch.delenv("VSEARCH_NO_QUERY_LOG", raising=False)
    monkeypatch.setenv("VSEARCH_QUERY_LOG", str(log_path))
    return db_path, log_path


def _run_query(argv, db_path):
    parser = build_parser()
    args = parser.parse_args(["--db", str(db_path)] + argv)
    buf = io.StringIO()
    with patch("eichi.embed.encode_one", side_effect=_stub_encode_one), \
         patch("eichi.embed.encode", side_effect=_stub_encode):
        with redirect_stdout(buf):
            rc = args.func(args)
    return rc, buf.getvalue()


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


# --- _resolve_caller unit tests ---------------------------------------------


def test_resolve_caller_default_is_cli(monkeypatch):
    monkeypatch.delenv("VSEARCH_CALLER", raising=False)
    args = types.SimpleNamespace(caller=None)
    assert _resolve_caller(args) == "cli"


def test_resolve_caller_env_wins_when_no_flag(monkeypatch):
    monkeypatch.setenv("VSEARCH_CALLER", "agent")
    args = types.SimpleNamespace(caller=None)
    assert _resolve_caller(args) == "agent"


def test_resolve_caller_flag_beats_env(monkeypatch):
    monkeypatch.setenv("VSEARCH_CALLER", "agent")
    args = types.SimpleNamespace(caller="web")
    assert _resolve_caller(args) == "web"


def test_resolve_caller_unknown_collapses_to_other(monkeypatch):
    monkeypatch.setenv("VSEARCH_CALLER", "garbage-value")
    args = types.SimpleNamespace(caller=None)
    assert _resolve_caller(args) == "_other"


def test_resolve_caller_allowed_set_matches_doc():
    # Sanity — keep the contract obvious.
    assert ALLOWED_CALLERS == {"cli", "web", "agent", "mainloop"}


# --- end-to-end: query stamps caller into query.log ------------------------


def test_query_log_caller_default_cli(db_and_log, monkeypatch):
    db, log = db_and_log
    monkeypatch.delenv("VSEARCH_CALLER", raising=False)
    rc, _ = _run_query(["query", "alpha", "-k", "1"], db)
    assert rc == 0
    rows = _read_log(log)
    assert rows, "query.log should have at least one entry"
    assert rows[-1]["caller"] == "cli"


def test_query_log_caller_env_override(db_and_log, monkeypatch):
    db, log = db_and_log
    monkeypatch.setenv("VSEARCH_CALLER", "agent")
    rc, _ = _run_query(["query", "alpha", "-k", "1"], db)
    assert rc == 0
    rows = _read_log(log)
    assert rows[-1]["caller"] == "agent"


def test_query_log_caller_flag_wins(db_and_log, monkeypatch):
    db, log = db_and_log
    monkeypatch.setenv("VSEARCH_CALLER", "agent")
    rc, _ = _run_query(
        ["query", "alpha", "-k", "1", "--caller", "mainloop"], db
    )
    assert rc == 0
    rows = _read_log(log)
    assert rows[-1]["caller"] == "mainloop"


def test_query_log_caller_each_allowed_value(db_and_log, monkeypatch):
    db, log = db_and_log
    monkeypatch.delenv("VSEARCH_CALLER", raising=False)
    for caller in ("cli", "web", "agent", "mainloop"):
        rc, _ = _run_query(
            ["query", "alpha", "-k", "1", "--caller", caller], db
        )
        assert rc == 0
    rows = _read_log(log)
    callers = [r["caller"] for r in rows]
    assert callers[-4:] == ["cli", "web", "agent", "mainloop"]
