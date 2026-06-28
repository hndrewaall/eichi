# Agents

Guidance for AI agents and LLM-driven tools integrating eichi as a
search backend.

## Search-first behavior

When wired into an agent's tool loop, eichi should be the **default
first lookup** for open-ended recall ("where is X", "what did we
decide about Y", "find the conversation where Z"). The decision tree:

1. **Concept-level question** (fuzzy, semantic) -> `eichi query "<q>"`
2. **Exact-string question** (function name, error message, config key)
   -> `grep -r` or code search
3. **Structured data** (metrics, timestamps, statuses) -> domain-specific
   tool (Prometheus, DB query, etc.)

If a concept-level query returns no results or weak scores (all
`[distant]`), THEN fall back to grep — not before.

## Invocation patterns

### CLI (host-side or in a container with the venv available)

```bash
# Basic semantic search (hybrid vec+BM25 by default):
eichi query "alerting tier design decisions" -k 5

# Filter by recency:
eichi query "docker networking" --added-since 7d

# Sort by index time (most recently added first):
eichi query "PR feedback" --sort added -k 10

# Specific source tag:
eichi query "meeting notes" --source obsidian -k 5

# Check corpus health:
eichi stats        # doc count, last indexed, DB size
eichi ls           # list indexed files with chunk counts
```

### Web API (from inside a container without direct CLI access)

```bash
curl -s "http://localhost:8001/api/search?q=alerting+tiers&k=5" | jq .
```

Query params: `q` (required), `k` (top-K, default 20), `source`
(filter), `added_since` (duration: `1d`, `7d`, `30d`), `retrieval`
(`hybrid`|`vector`|`bm25`).

## Interpreting results

Each result line (CLI) or object (API) includes:

- **Score** with a human-readable label: `[strong]` > `[moderate]` >
  `[weak]` > `[distant]`. Treat `[distant]` as noise unless the query
  is highly specialized.
- **Source tag** (`[file]`, `[obsidian]`, `[transcripts]`, etc.) — helps
  disambiguate provenance.
- **Timestamp** — when the document was last modified (mtime) or added
  to the index.

## Re-indexing guidance

- `eichi index <path>` is **idempotent and delta-only** — safe to run
  frequently. Only re-embeds files whose content hash changed.
- Check staleness via `eichi stats` — if `last indexed at` is older
  than recent corpus activity, re-index.
- `eichi index-stream` accepts JSONL on stdin for programmatic ingestion
  (connectors, CI artifacts, conversation logs).

## Wiring into an agent framework

To add "search-first" behavior to an agent:

1. Add eichi to the agent's tool set (CLI wrapper or HTTP client).
2. In the agent's system prompt, instruct: "Before grepping or asking
   the user for context, query eichi for semantic matches."
3. For container-based agents that can't run the CLI directly, point at
   the minisite API (`/api/search`).

See the [claude-watch container baked-CLAUDE.md](https://github.com/hndrewaall/claude-watch/blob/main/container/baked-CLAUDE.md)
for a worked example of how this guidance is baked into a production
agent loop.
