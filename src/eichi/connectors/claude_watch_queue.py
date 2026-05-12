"""claude-watch session-task queue connector.

Indexes the queue items in ``~/.config/session/queue.json`` — the JSON file
that ``claude-watch`` / ``session-task queue`` reads and writes. One
Document is emitted per queue item.

Despite the connector name ("queue"), the underlying store is a single
JSON document (fcntl-locked on every RMW), not SQLite. Schema for one
item::

    {
      "id": "q-2026-04-16-a3df",
      "created_at": "2026-04-16T15:28:43+00:00",
      "completed_at": "2026-04-16T15:39:02+00:00",
      "created_by": "andrew|cron|migrate|...",
      "description": "free-text prompt to the agent",
      "summary": "~10 word headline (optional)",
      "scope": ["repo:foo", "agent-proto:bar", ...],
      "priority": 5,
      "status": "pending|ready|running|done|abandoned|...",
      "group_id": "g-...",
      "group_head": false,
      "context": {"legacy_body": "..."},   // optional free-form body
      "depends_on": [...],                  // optional
      "log_archive_path": "q-...jsonl"      // set on done/abandon
    }

doc_id format::

    queue:item:<id>

By default only items with terminal status (``done``, ``abandoned``,
``wedged``) are indexed — in-flight descriptions get rewritten as the
agent runs, and indexing those would just churn the embeddings.
``config["include_in_flight"] = true`` overrides this.

Configurable: ``config["path"]`` (or ``$EICHI_CLAUDE_QUEUE_PATH``) points
the connector at a non-default queue JSON file. No-op when the file is
missing.

Cursor: per-item completed_at (or created_at for in-flight items) so a
re-run only re-emits items that have actually changed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


DEFAULT_QUEUE_PATH = Path.home() / ".config" / "session" / "queue.json"

# Terminal statuses — these are stable and worth indexing. In-flight
# statuses (pending / ready / running / blocked) are skipped by default.
TERMINAL_STATUSES = {"done", "abandoned", "wedged"}


def _parse_iso_to_epoch(value: Any) -> float:
    """Parse an ISO8601 timestamp into epoch seconds. 0.0 on any error.

    Accepts trailing ``Z`` (UTC), explicit offsets, and naive timestamps
    (assumed UTC).
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    s = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return 0.0


def _resolve_path(config: Optional[Dict[str, Any]]) -> Path:
    cfg = config or {}
    override = (
        os.environ.get("EICHI_CLAUDE_QUEUE_PATH")
        or cfg.get("path")
    )
    if override:
        return Path(os.path.expanduser(str(override)))
    return DEFAULT_QUEUE_PATH


def _load_items(path: Path) -> List[Dict[str, Any]]:
    """Read ``queue.json`` and return the items list. Tolerates a missing
    file (empty list) and the legacy bare-list shape.
    """
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items
        return []
    if isinstance(data, list):
        return data
    return []


def _render_item_text(item: Dict[str, Any]) -> str:
    """Build the indexed text body for one queue item.

    Layout (markdown headers so the embedder picks up structure)::

        # Queue: <id>
        ## Summary
        <summary or '-'>
        ## Status / scope / completed
        <status> | <scope-joined> | <completed_at>
        ## Description
        <description>
        ## Body
        <context.legacy_body>
        ## Deps
        depends on: <id1>, <id2>     (omitted when empty)
    """
    qid = str(item.get("id") or "?")
    summary = str(item.get("summary") or "").strip()
    description = str(item.get("description") or "").strip()
    status = str(item.get("status") or "").strip() or "?"
    completed = str(item.get("completed_at") or "").strip() or "-"
    scope_raw = item.get("scope")
    if isinstance(scope_raw, list):
        scope = ", ".join(str(s) for s in scope_raw if s) or "-"
    elif isinstance(scope_raw, str):
        scope = scope_raw or "-"
    else:
        scope = "-"
    body = ""
    context_raw = item.get("context")
    if isinstance(context_raw, dict):
        legacy = context_raw.get("legacy_body")
        if isinstance(legacy, str) and legacy.strip():
            body = legacy.strip()
    deps_raw = item.get("depends_on")
    deps_list: List[str] = []
    if isinstance(deps_raw, list):
        deps_list = [str(d) for d in deps_raw if isinstance(d, str) and d]

    lines: List[str] = [
        f"# Queue: {qid}",
        "## Summary",
        summary or "-",
        "## Status / scope / completed",
        f"{status} | {scope} | {completed}",
    ]
    if description and description != summary:
        lines.append("## Description")
        lines.append(description)
    if body:
        lines.append("## Body")
        lines.append(body)
    if deps_list:
        lines.append("## Deps")
        lines.append("depends on: " + ", ".join(deps_list))
    return "\n".join(lines)


def _build_doc(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qid = item.get("id")
    if not isinstance(qid, str) or not qid:
        return None
    text = _render_item_text(item)
    created_at = _parse_iso_to_epoch(item.get("created_at"))
    completed_at = _parse_iso_to_epoch(item.get("completed_at"))
    mtime = completed_at or created_at
    library_added_at = created_at or None

    scope_raw = item.get("scope")
    scope_list: List[str] = []
    if isinstance(scope_raw, list):
        scope_list = [str(s) for s in scope_raw if isinstance(s, str) and s]

    deps_raw = item.get("depends_on")
    deps_list: List[str] = []
    if isinstance(deps_raw, list):
        deps_list = [
            str(d) for d in deps_raw if isinstance(d, str) and d
        ]

    return {
        "source": "claude-watch-queue",
        "doc_id": f"queue:item:{qid}",
        "text": text,
        "mtime": mtime,
        "metadata": {
            "queue_id": qid,
            "scope": scope_list or None,
            "group_id": item.get("group_id") or None,
            "status": item.get("status") or None,
            "depends_on": deps_list or None,
            "created_at": item.get("created_at") or None,
            "completed_at": item.get("completed_at") or None,
            "priority": item.get("priority"),
            "library_added_at": library_added_at,
        },
    }


def iter_documents(
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield JSONL-shaped doc dicts for each queue item.

    Filtering rules:
      - by default only terminal-status items (``done``, ``abandoned``,
        ``wedged``) are emitted. Set ``config["include_in_flight"]=True``
        (or ``$EICHI_CLAUDE_QUEUE_INCLUDE_IN_FLIGHT=1``) to lift this.
      - per-item cursor: items already at the cursor's recorded
        ``mtime`` (or newer) are skipped.

    State shape::

        {"items": {"<queue-id>": <mtime_seconds_float>, ...}}
    """
    cfg = config or {}
    if state is None:
        state = {}
    cursor: Dict[str, float] = state.setdefault("items", {})

    include_in_flight = bool(
        cfg.get("include_in_flight")
        or os.environ.get("EICHI_CLAUDE_QUEUE_INCLUDE_IN_FLIGHT")
    )

    path = _resolve_path(config)
    items = _load_items(path)

    for item in items:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if not include_in_flight and status not in TERMINAL_STATUSES:
            continue
        doc = _build_doc(item)
        if doc is None:
            continue
        qid = doc["metadata"]["queue_id"]
        prev = float(cursor.get(qid, 0.0))
        if doc["mtime"] and doc["mtime"] <= prev:
            continue
        yield doc
        cursor[qid] = doc["mtime"] or prev
