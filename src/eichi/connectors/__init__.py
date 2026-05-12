"""eichi connectors.

A connector is a small Python module that knows how to enumerate documents
from some external data source and emit them as JSONL-shaped dicts compatible
with ``eichi index-stream``. Each dict has the keys ``source``, ``doc_id``,
``text``, ``mtime`` (optional), and ``metadata`` (optional).

The standard contract is a function::

    def iter_documents(state: dict | None = None,
                       config: dict | None = None) -> Iterator[dict]: ...

``state`` is an opaque per-connector cursor that the driver loads/saves so
incremental runs only re-emit changed content. ``config`` is per-connector
configuration loaded from ``eichi.toml`` (path overrides, etc.).

Connectors shipped with eichi:

- :mod:`eichi.connectors.claude_jsonl` — Anthropic CLI session JSONL files
  (``~/.claude/projects/*/*.jsonl``). One doc per user/assistant turn plus
  rolling conversation clusters per session.
- :mod:`eichi.connectors.claude_watch_queue` — claude-watch ``queue.json``
  task queue. One doc per finished queue item.

Both connectors no-op gracefully when their underlying paths are absent,
so it's safe to enable them in ``eichi.toml`` even on a fresh machine.
"""

from __future__ import annotations

from typing import Callable, Dict

from . import claude_jsonl, claude_watch_queue

# Public registry — name → module. The CLI walks this so callers do
# ``eichi index --corpus <name>`` and we dispatch to the right module.
REGISTRY: Dict[str, Callable] = {
    "claude-jsonl": claude_jsonl.iter_documents,
    "claude-watch-queue": claude_watch_queue.iter_documents,
}

__all__ = ["REGISTRY", "claude_jsonl", "claude_watch_queue"]
