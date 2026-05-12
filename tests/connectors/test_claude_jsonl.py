"""Tests for the claude-jsonl connector."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

from eichi.connectors import claude_jsonl


def _write_session(project_dir: Path, events: list) -> Path:
    """Write a synthetic session JSONL.

    Pads the file with a small ``isMeta`` system-init line at the start
    so the size threshold (1024 bytes) is met without manual fudging.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    session_uuid = str(uuid.uuid4())
    path = project_dir / f"{session_uuid}.jsonl"
    # Always lead with a meta-init line — connector filters it out.
    lines = [
        json.dumps(
            {
                "type": "user",
                "isMeta": True,
                "timestamp": "2026-05-11T10:00:00Z",
                "message": {"content": "session-init"},
            }
        )
    ]
    for ev in events:
        lines.append(json.dumps(ev))
    # Pad to satisfy MIN_FILE_BYTES.
    payload = "\n".join(lines) + "\n"
    if len(payload) < claude_jsonl.MIN_FILE_BYTES:
        # Pad with a comment-like assistant turn we control + count.
        pad_len = claude_jsonl.MIN_FILE_BYTES - len(payload) + 16
        pad_text = "x" * pad_len
        payload += (
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-05-11T10:00:00Z",
                    "message": {
                        "content": [{"type": "text", "text": pad_text}]
                    },
                }
            )
            + "\n"
        )
    path.write_text(payload, encoding="utf-8")
    return path


def test_emit_per_turn_and_cluster_and_summary(tmp_path):
    """A synthetic session with 4 turns yields 4 msg docs + ≥1 cluster
    + 1 summary."""
    project = tmp_path / "-home-test"
    events = [
        {
            "type": "user",
            "timestamp": "2026-05-11T10:01:00Z",
            "message": {"content": "fix the deploy script"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-05-11T10:01:30Z",
            "message": {
                "content": [{"type": "text", "text": "looking now"}]
            },
        },
        {
            "type": "user",
            "timestamp": "2026-05-11T10:02:00Z",
            "message": {"content": "thanks"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-05-11T10:02:30Z",
            "message": {
                "content": [
                    {"type": "text", "text": "deploy is now green"},
                    {"type": "tool_use", "name": "Bash"},
                ]
            },
        },
    ]
    _write_session(project, events)

    docs = list(claude_jsonl.iter_documents(config={"root": str(tmp_path)}))
    kinds = [d["metadata"]["kind"] for d in docs]
    # 4 msg docs + at least 1 cluster + 1 summary.
    assert kinds.count("msg") >= 4
    assert "cluster" in kinds
    assert kinds.count("summary") == 1

    # Source tag is correct on every doc.
    assert {d["source"] for d in docs} == {"claude-jsonl"}

    # Per-turn msg docs carry message_role.
    roles = [
        d["metadata"]["message_role"]
        for d in docs
        if d["metadata"]["kind"] == "msg"
    ]
    assert "user" in roles
    assert "assistant" in roles

    # Tool name surfaces in metadata when present.
    tool_docs = [
        d
        for d in docs
        if d["metadata"]["kind"] == "msg"
        and (d["metadata"].get("tools") or [])
    ]
    assert tool_docs, "expected an assistant turn with a tool_use block"
    assert "Bash" in (tool_docs[0]["metadata"]["tools"] or [])


def test_skips_thinking_and_tool_result_and_meta(tmp_path):
    """Events that don't carry user-visible content are filtered."""
    project = tmp_path / "-home-test"
    events = [
        {
            "type": "user",
            "timestamp": "2026-05-11T10:01:00Z",
            "message": {"content": [{"type": "tool_result"}]},
        },
        {
            "type": "assistant",
            "timestamp": "2026-05-11T10:01:30Z",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "private chain"},
                    {"type": "tool_use", "name": "Read"},
                ]
            },
        },
        {
            "type": "system",
            "timestamp": "2026-05-11T10:01:31Z",
            "message": {"content": "ignore me"},
        },
        {
            "type": "user",
            "timestamp": "2026-05-11T10:02:00Z",
            "message": {"content": "real user prompt"},
        },
    ]
    _write_session(project, events)

    docs = list(claude_jsonl.iter_documents(config={"root": str(tmp_path)}))
    msg_docs = [d for d in docs if d["metadata"]["kind"] == "msg"]
    # The tool-result-only user, the system event, and the thinking
    # block all get dropped. The pad-text assistant is preserved.
    # The assistant-with-tool-use-only turn is preserved (tools list
    # populated), the user turn after is preserved.
    texts = [d["text"] for d in msg_docs]
    assert any("real user prompt" in t for t in texts)
    # Thinking content NEVER appears in any doc text.
    assert not any("private chain" in t for t in texts)


def test_no_op_when_root_missing(tmp_path):
    """A missing root directory yields zero docs without error."""
    docs = list(
        claude_jsonl.iter_documents(
            config={"root": str(tmp_path / "does-not-exist")}
        )
    )
    assert docs == []


def test_state_cursor_is_respected(tmp_path):
    """Second run with the same state yields zero new docs."""
    project = tmp_path / "-home-test"
    events = [
        {
            "type": "user",
            "timestamp": "2026-05-11T10:01:00Z",
            "message": {"content": "hello"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-05-11T10:01:30Z",
            "message": {
                "content": [{"type": "text", "text": "hi"}]
            },
        },
    ]
    _write_session(project, events)

    state: dict = {}
    docs1 = list(
        claude_jsonl.iter_documents(
            state=state, config={"root": str(tmp_path)}
        )
    )
    assert docs1, "first run should emit something"

    # Second run — no file changed, cursor is recorded, should be empty.
    docs2 = list(
        claude_jsonl.iter_documents(
            state=state, config={"root": str(tmp_path)}
        )
    )
    assert docs2 == []


def test_env_var_overrides_config(tmp_path, monkeypatch):
    """``$EICHI_CLAUDE_JSONL_ROOT`` beats the config dict."""
    real_root = tmp_path / "real"
    _write_session(
        real_root / "-home-x",
        [
            {
                "type": "user",
                "timestamp": "2026-05-11T10:01:00Z",
                "message": {"content": "yo"},
            }
        ],
    )
    monkeypatch.setenv("EICHI_CLAUDE_JSONL_ROOT", str(real_root))
    docs = list(
        claude_jsonl.iter_documents(
            config={"root": str(tmp_path / "absent")}
        )
    )
    assert docs, "env override should let real_root yield documents"
    monkeypatch.delenv("EICHI_CLAUDE_JSONL_ROOT", raising=False)
