"""Tests for the claude-watch-queue connector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eichi.connectors import claude_watch_queue


def _write_queue(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


def test_emits_one_doc_per_terminal_item(tmp_path):
    qpath = tmp_path / "queue.json"
    _write_queue(
        qpath,
        [
            {
                "id": "q-1",
                "created_at": "2026-05-01T10:00:00+00:00",
                "completed_at": "2026-05-01T11:00:00+00:00",
                "summary": "rebuild the dashboard",
                "description": "Rebuild the grafana dashboard with new panels",
                "status": "done",
                "scope": ["repo:dashboards"],
                "group_id": "g-A",
                "depends_on": [],
            },
            {
                "id": "q-2",
                "created_at": "2026-05-02T10:00:00+00:00",
                "completed_at": "2026-05-02T11:00:00+00:00",
                "summary": "cancelled work",
                "description": "Did not finish",
                "status": "abandoned",
                "scope": ["repo:thing"],
            },
            {
                "id": "q-3",
                "created_at": "2026-05-03T10:00:00+00:00",
                "summary": "in-flight",
                "description": "Should be skipped",
                "status": "running",
                "scope": ["repo:thing"],
            },
        ],
    )
    docs = list(
        claude_watch_queue.iter_documents(config={"path": str(qpath)})
    )
    assert len(docs) == 2
    ids = {d["metadata"]["queue_id"] for d in docs}
    assert ids == {"q-1", "q-2"}
    # All carry the right source tag.
    assert {d["source"] for d in docs} == {"claude-watch-queue"}
    # Text contains the structured markdown sections.
    text = next(d["text"] for d in docs if d["metadata"]["queue_id"] == "q-1")
    assert "# Queue: q-1" in text
    assert "rebuild the dashboard" in text
    assert "repo:dashboards" in text


def test_include_in_flight(tmp_path):
    """``include_in_flight=True`` indexes running/pending items too."""
    qpath = tmp_path / "queue.json"
    _write_queue(
        qpath,
        [
            {
                "id": "q-running",
                "created_at": "2026-05-03T10:00:00+00:00",
                "summary": "live",
                "description": "in-flight",
                "status": "running",
                "scope": [],
            }
        ],
    )
    docs = list(
        claude_watch_queue.iter_documents(
            config={"path": str(qpath), "include_in_flight": True}
        )
    )
    assert len(docs) == 1
    assert docs[0]["metadata"]["queue_id"] == "q-running"


def test_no_op_when_path_missing(tmp_path):
    docs = list(
        claude_watch_queue.iter_documents(
            config={"path": str(tmp_path / "nope.json")}
        )
    )
    assert docs == []


def test_state_cursor_is_respected(tmp_path):
    qpath = tmp_path / "queue.json"
    _write_queue(
        qpath,
        [
            {
                "id": "q-1",
                "created_at": "2026-05-01T10:00:00+00:00",
                "completed_at": "2026-05-01T11:00:00+00:00",
                "summary": "thing",
                "description": "did it",
                "status": "done",
                "scope": ["repo:x"],
            }
        ],
    )
    state: dict = {}
    docs1 = list(
        claude_watch_queue.iter_documents(
            state=state, config={"path": str(qpath)}
        )
    )
    assert len(docs1) == 1
    docs2 = list(
        claude_watch_queue.iter_documents(
            state=state, config={"path": str(qpath)}
        )
    )
    # Cursor recorded — no re-emit on unchanged content.
    assert docs2 == []


def test_metadata_includes_depends_on_and_scope(tmp_path):
    qpath = tmp_path / "queue.json"
    _write_queue(
        qpath,
        [
            {
                "id": "q-7",
                "created_at": "2026-05-01T10:00:00+00:00",
                "completed_at": "2026-05-01T11:00:00+00:00",
                "summary": "with deps",
                "description": "blocked work",
                "status": "done",
                "scope": ["repo:a", "repo:b"],
                "depends_on": ["q-6", "q-5"],
            }
        ],
    )
    docs = list(
        claude_watch_queue.iter_documents(config={"path": str(qpath)})
    )
    meta = docs[0]["metadata"]
    assert meta["depends_on"] == ["q-6", "q-5"]
    assert meta["scope"] == ["repo:a", "repo:b"]
    assert "depends on: q-6, q-5" in docs[0]["text"]


def test_tolerates_bad_json(tmp_path):
    qpath = tmp_path / "queue.json"
    qpath.write_text("not-json", encoding="utf-8")
    docs = list(
        claude_watch_queue.iter_documents(config={"path": str(qpath)})
    )
    assert docs == []


def test_env_var_overrides_config(tmp_path, monkeypatch):
    qpath = tmp_path / "real.json"
    _write_queue(
        qpath,
        [
            {
                "id": "q-env",
                "created_at": "2026-05-01T10:00:00+00:00",
                "completed_at": "2026-05-01T11:00:00+00:00",
                "summary": "via env",
                "description": "x",
                "status": "done",
                "scope": [],
            }
        ],
    )
    monkeypatch.setenv("EICHI_CLAUDE_QUEUE_PATH", str(qpath))
    docs = list(
        claude_watch_queue.iter_documents(
            config={"path": str(tmp_path / "absent.json")}
        )
    )
    assert len(docs) == 1
    assert docs[0]["metadata"]["queue_id"] == "q-env"
    monkeypatch.delenv("EICHI_CLAUDE_QUEUE_PATH", raising=False)
