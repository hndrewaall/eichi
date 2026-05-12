"""Claude Code session JSONL connector.

Indexes the JSONL transcripts the Anthropic CLI writes under
``~/.claude/projects/<workspace-encoded>/<session-uuid>.jsonl``. Each
line in those files is one event (``user``, ``assistant``, ``system``,
``attachment``, ``tool_use``, ``tool_result``, ``summary``, ...).

Emits three flavors of document per session:

1. **Per-turn ``kind="msg"`` docs** — one Document per user/assistant
   turn. ``thinking`` / ``tool_use`` / ``tool_result`` blocks and
   ``isMeta=true`` system-init lines are dropped (high noise, low
   semantic value). Body capped at 1500 chars.

2. **``kind="cluster"`` conversation docs** — rolling window of
   consecutive user/assistant turns, bounded by 10-min gap / 8-turn
   count / 1500-char body budget. Anchors "what was the topic" queries.

3. **``kind="summary"`` session doc** — first user prompt + first ~3 KiB
   of last assistant turn. Anchors "what did we do on YYYY-MM-DD"
   queries.

doc_id format::

    transcripts:msg:<session-uuid>:<turn-idx>
    transcripts:cluster:conv:<session-uuid>:<seed-turn-idx>
    transcripts:summary:<session-uuid>

Cursor: per-file mtime in seconds. State::

    {"files": {"<session-uuid>": <mtime_float>, ...}}

Configurable: ``config["root"]`` overrides the default
``~/.claude/projects`` root. No-op (yields nothing) if the root is missing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


DEFAULT_ROOT = Path.home() / ".claude" / "projects"

# Cluster tunables — mirror the scoping report recommendation.
CLUSTER_GAP_SEC = 10 * 60
CLUSTER_MAX_TURNS = 8
CLUSTER_MAX_CHARS = 1500

PER_TURN_BODY_CAP = 1500
SESSION_SUMMARY_CAP = 3000

# Skip files smaller than this — empty / aborted sessions.
MIN_FILE_BYTES = 1024


def _parse_iso(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    s = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _flatten_assistant_content(content: Any) -> Tuple[str, List[str]]:
    """Walk an assistant message.content list.

    Returns ``(prose, tool_names)`` — text blocks concatenated, plus the
    ordered list of tool names invoked. Drops ``thinking`` / ``tool_use``
    content body (the names are kept separately as metadata).
    """
    prose: List[str] = []
    tools: List[str] = []
    if isinstance(content, str):
        return content.strip(), []
    if not isinstance(content, list):
        return "", []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text", "")
            if isinstance(txt, str) and txt.strip():
                prose.append(txt)
        elif btype == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                tools.append(name)
        # thinking / other block types: drop
    return "\n".join(prose).strip(), tools


def _flatten_user_content(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        s = content.strip()
        if s in (
            "<local-command-stdout></local-command-stdout>",
            "<local-command-stderr></local-command-stderr>",
        ):
            return ""
        return s
    # list content = tool_result, drop
    return ""


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 12].rstrip() + "...[trunc]"


def _strip_harness_envelope(text: str) -> str:
    """Strip / drop harness-internal envelopes (slash commands,
    system-reminders, task-notifications, self-clear-resume boilerplate).

    Returns "" for envelopes that should be dropped entirely; the
    original text for plain user content.
    """
    if "<command-name>" in text:
        return ""
    if text.startswith("<task-notification>"):
        return ""
    if text.startswith("<system-reminder>") and "</system-reminder>" in text:
        return ""
    if text.startswith("[SELF-CLEAR-RESUME]"):
        return ""
    if text.startswith("[SYSTEM NOTIFICATION"):
        return ""
    return text


def _read_turns(path: Path) -> List[Dict[str, Any]]:
    """Walk one JSONL into a list of indexable turns.

    Each turn dict has::

        {"kind": "user"|"assistant", "text": str, "tools": [str, ...],
         "ts": datetime|None, "idx": int}

    Tool-only assistant turns (no prose, only tool_use blocks) are
    INCLUDED — they still convey "claude did X here" which is useful
    cluster context. Their ``text`` is empty; the cluster builder
    renders them as ``[tool: <name>, ...]``.
    """
    turns: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("type")
                if t not in ("user", "assistant"):
                    continue
                if obj.get("isSidechain"):
                    # Subagent transcripts get their own indexing path.
                    continue
                if t == "user" and obj.get("isMeta"):
                    # First system-init metadata line.
                    continue
                ts = _parse_iso(obj.get("timestamp"))
                msg = obj.get("message", {})
                if t == "user":
                    text = _flatten_user_content(msg)
                    if not text:
                        continue
                    text = _strip_harness_envelope(text)
                    if not text.strip():
                        continue
                    turns.append(
                        {
                            "kind": "user",
                            "text": _truncate(text, PER_TURN_BODY_CAP),
                            "tools": [],
                            "ts": ts,
                            "idx": len(turns),
                        }
                    )
                elif t == "assistant":
                    content = (
                        msg.get("content") if isinstance(msg, dict) else None
                    )
                    prose, tools = _flatten_assistant_content(content)
                    if not prose and not tools:
                        continue
                    turns.append(
                        {
                            "kind": "assistant",
                            "text": _truncate(prose, PER_TURN_BODY_CAP),
                            "tools": tools,
                            "ts": ts,
                            "idx": len(turns),
                        }
                    )
    except OSError:
        return []
    return turns


def _format_ts(ts: Optional[datetime]) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if ts else "?"


def _msg_text(turn: Dict[str, Any], session_uuid: str) -> str:
    header = (
        f"[{session_uuid[:8]} {_format_ts(turn.get('ts'))}] {turn['kind']}:"
    )
    body = turn["text"]
    tools = turn.get("tools") or []
    if turn["kind"] == "assistant" and tools:
        body = (f"{body}\n[tool: {', '.join(tools)}]"
                if body else f"[tool: {', '.join(tools)}]")
    return f"{header} {body}"


def _form_conversation_clusters(
    turns: List[Dict[str, Any]], session_uuid: str
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Bucket turns into rolling-window conversation clusters."""
    clusters: List[Tuple[str, List[Dict[str, Any]]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0

    def body_len(t: Dict[str, Any]) -> int:
        return len(t.get("text", "")) + sum(
            len(x) for x in t.get("tools") or []
        )

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        seed = current[0]["idx"]
        cid = f"transcripts:cluster:conv:{session_uuid}:{seed}"
        clusters.append((cid, list(current)))
        current = []
        current_chars = 0

    for turn in turns:
        bl = body_len(turn)
        if not current:
            current = [turn]
            current_chars = bl
            continue
        prev = current[-1]
        gap_sec = 0.0
        if turn.get("ts") and prev.get("ts"):
            gap_sec = (turn["ts"] - prev["ts"]).total_seconds()
        projected_turns = len(current) + 1
        projected_chars = current_chars + bl
        if (
            (gap_sec and gap_sec > CLUSTER_GAP_SEC)
            or projected_turns > CLUSTER_MAX_TURNS
            or projected_chars > CLUSTER_MAX_CHARS
        ):
            flush()
            current = [turn]
            current_chars = bl
        else:
            current.append(turn)
            current_chars = projected_chars
    flush()
    return clusters


def _cluster_text(
    members: List[Dict[str, Any]], session_uuid: str
) -> str:
    return "\n".join(_msg_text(m, session_uuid) for m in members)


def _session_summary_text(
    turns: List[Dict[str, Any]], session_uuid: str, path: Path
) -> str:
    if not turns:
        return ""
    first_user = next((t for t in turns if t["kind"] == "user"), None)
    last_assist = next(
        (t for t in reversed(turns)
         if t["kind"] == "assistant" and t["text"]),
        None,
    )
    first_ts = next((t["ts"] for t in turns if t.get("ts")), None)
    last_ts = next((t["ts"] for t in reversed(turns) if t.get("ts")), None)
    date_str = first_ts.strftime("%Y-%m-%d") if first_ts else "?"
    span = (
        f"{_format_ts(first_ts)} .. {_format_ts(last_ts)} "
        f"({len(turns)} turns)"
    )
    lines = [
        f"# Claude session: {date_str} {session_uuid[:8]}",
        "## Path",
        str(path),
        "## First prompt",
        _truncate(first_user["text"], 1000) if first_user else "-",
        "## Last assistant",
        _truncate(last_assist["text"], 1500) if last_assist else "-",
        "## Span",
        span,
    ]
    return _truncate("\n".join(lines), SESSION_SUMMARY_CAP)


def _iter_session_files(root: Path) -> Iterator[Tuple[Path, str, float, int]]:
    """Yield ``(path, session_uuid, mtime, size)`` for every JSONL under
    ``root``. ``session_uuid`` is the file stem.
    """
    if not root.is_dir():
        return
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        try:
            entries = list(project_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if entry.suffix != ".jsonl":
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            if st.st_size < MIN_FILE_BYTES:
                continue
            yield entry, entry.stem, st.st_mtime, st.st_size


def _emit_for_session(
    path: Path, session_uuid: str, mtime: float
) -> Iterator[Dict[str, Any]]:
    """Emit per-turn + cluster + summary docs for one session JSONL."""
    turns = _read_turns(path)
    if not turns:
        return

    clusters = _form_conversation_clusters(turns, session_uuid)
    cluster_by_idx: Dict[int, str] = {}
    for cid, members in clusters:
        for m in members:
            cluster_by_idx[m["idx"]] = cid

    project_slug = path.parent.name

    # Per-turn msg docs.
    for turn in turns:
        ts = turn.get("ts")
        cid = cluster_by_idx.get(turn["idx"], "")
        yield {
            "source": "claude-jsonl",
            "doc_id": (
                f"transcripts:msg:{session_uuid}:{turn['idx']}"
            ),
            "text": _msg_text(turn, session_uuid),
            "mtime": ts.timestamp() if ts else mtime,
            "metadata": {
                "kind": "msg",
                "session_id": session_uuid,
                "project_slug": project_slug,
                "message_role": turn["kind"],
                "turn_idx": turn["idx"],
                "cluster_id": cid,
                "timestamp": ts.isoformat() if ts else None,
                "tools": turn.get("tools") or None,
            },
        }

    # Conversation clusters.
    for cid, members in clusters:
        seed_ts = members[0].get("ts")
        end_ts = members[-1].get("ts")
        yield {
            "source": "claude-jsonl",
            "doc_id": cid,
            "text": _cluster_text(members, session_uuid),
            "mtime": seed_ts.timestamp() if seed_ts else mtime,
            "metadata": {
                "kind": "cluster",
                "session_id": session_uuid,
                "project_slug": project_slug,
                "turn_count": len(members),
                "cluster_id": cid,
                "ts_start": seed_ts.isoformat() if seed_ts else None,
                "ts_end": end_ts.isoformat() if end_ts else None,
                "mtime_end": end_ts.timestamp() if end_ts else None,
            },
        }

    # Session summary.
    summary_text = _session_summary_text(turns, session_uuid, path)
    if summary_text:
        first_ts = next((t["ts"] for t in turns if t.get("ts")), None)
        last_ts = next(
            (t["ts"] for t in reversed(turns) if t.get("ts")), None
        )
        yield {
            "source": "claude-jsonl",
            "doc_id": f"transcripts:summary:{session_uuid}",
            "text": summary_text,
            "mtime": first_ts.timestamp() if first_ts else mtime,
            "metadata": {
                "kind": "summary",
                "session_id": session_uuid,
                "project_slug": project_slug,
                "turn_count": len(turns),
                "ts_start": first_ts.isoformat() if first_ts else None,
                "ts_end": last_ts.isoformat() if last_ts else None,
            },
        }


def iter_documents(
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield JSONL-shaped doc dicts for every changed session JSONL.

    ``config["root"]`` overrides the default
    ``~/.claude/projects`` root. ``$EICHI_CLAUDE_JSONL_ROOT`` env var also
    honored (env wins over config).

    ``state["files"]`` is the per-session mtime cursor — touched in place
    so the caller can persist it.
    """
    cfg = config or {}
    if state is None:
        state = {}
    files_cursor: Dict[str, float] = state.setdefault("files", {})

    root_override = (
        os.environ.get("EICHI_CLAUDE_JSONL_ROOT")
        or cfg.get("root")
    )
    root = (
        Path(os.path.expanduser(str(root_override)))
        if root_override
        else DEFAULT_ROOT
    )

    if not root.is_dir():
        return

    for path, session_uuid, mtime, _size in _iter_session_files(root):
        prev = float(files_cursor.get(session_uuid, 0.0))
        if mtime <= prev:
            continue
        yield from _emit_for_session(path, session_uuid, mtime)
        files_cursor[session_uuid] = mtime
