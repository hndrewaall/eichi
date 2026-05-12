<p align="center">
  <img src="minisite/static/eichi-logo.png" alt="eichi" width="180">
</p>

# eichi

Local-first semantic search CLI built on [sqlite-vec](https://github.com/asg017/sqlite-vec)
and [sentence-transformers](https://www.sbert.net/). Point it at a directory
of notes, chat logs, agent transcripts, or code; query in natural language; get
top-K hits in roughly 30 ms.

The name (Japanese 叡智, *eichi*) means "wisdom".

## Features

- **Hybrid retrieval**: vec0 KNN union FTS5 BM25, fused via Reciprocal Rank
  Fusion (default). Pure-vector and pure-BM25 modes also exposed for
  diagnostics.
- **Offline at runtime**: the embedding model loads from the local Hugging Face
  cache only. A one-shot bootstrap download runs on first use; afterwards
  `HF_HUB_OFFLINE=1` is enforced.
- **Local data**: everything lives in one SQLite file under
  `~/.local/share/eichi/index.db`. No network calls except the bootstrap.
- **Streaming connector input**: pipe JSONL into `eichi index-stream` to index
  synthetic documents (chat messages, queue items, anything you can render as
  one JSON-per-line).
- **Date / metadata filters**: `--year-min`, `--year-max`, `--added-since 30d`,
  `--sort relevance|added|year`.
- **Per-call attribution**: `--caller cli|web|agent|mainloop` (or
  `$EICHI_CALLER`) tags every query in a local JSONL log, useful for
  partitioning RTT metrics.

## Quick start

```sh
git clone <repo-url> eichi
cd eichi
uv venv --python 3.11
uv pip install -e .

# index some notes (first run downloads the embedding model, ~400 MB)
eichi index ~/Documents/notes

# search
eichi query 'how did I decide to ship the new logging pipeline?' -k 5
```

If you don't use [uv](https://github.com/astral-sh/uv), `python -m venv .venv
&& . .venv/bin/activate && pip install -e .` works too.

## Subcommands

```
eichi index <path>       Index a file or directory (idempotent, delta-only)
eichi index-stream       Index JSONL docs from stdin (connector input)
eichi query <q> [-k N]   Search; print top-K results
eichi reindex [<path>]   Wipe + rebuild for path or full DB
eichi stats              Row count, sources, last-indexed time
eichi ls [<source>]      List indexed files
eichi rm <path>          Remove a file or directory from the index
```

All subcommands support `--json` for machine output. `index | reindex | rm`
support `-n` (dry run) and `-v` (verbose).

## Configuration

eichi works with zero configuration — `eichi index <path>` is enough for ad-hoc
use. For a canonical list of corpora to re-index on a schedule, drop a
`eichi.toml` at one of:

- `$EICHI_CONFIG` (any path you point at)
- `$XDG_CONFIG_HOME/eichi/eichi.toml`
- `~/.config/eichi/eichi.toml`

See [`eichi.toml.example`](./eichi.toml.example) for the format.

Other environment variables:

| Var | Meaning |
|-----|---------|
| `EICHI_DB` | Override the SQLite database path. |
| `EICHI_QUERY_LOG` | Override the local query-log path. |
| `EICHI_NO_QUERY_LOG=1` | Disable query logging entirely. |
| `EICHI_CALLER` | Tag the caller (default `cli`). One of `cli|web|agent|mainloop`. |
| `XDG_DATA_HOME` | Roots the default DB / query-log directory. |

## Web minisite (`minisite/`)

A small Flask app under [`minisite/`](./minisite/) wraps `eichi query`
for use behind a reverse proxy. It is fully whitelabel-able through
environment variables — the public build ships generic defaults.

| Var | Default | Meaning |
|-----|---------|---------|
| `SEARCH_SITE_TITLE` | `eichi search` | Page `<title>` + header text. |
| `SEARCH_SITE_LOGO_URL` | *(empty)* | Header logo `<img>` src. Absolute URL or `/static/…` path. Empty = no logo. |
| `SEARCH_SITE_LOGO_DEFAULT` | *(empty)* | Set to `1` to render the bundled `static/eichi-logo.png` when `SEARCH_SITE_LOGO_URL` is empty. |
| `SEARCH_SITE_BRAND` | *(empty)* | Optional brand string appended to the footer. |
| `SEARCH_SITE_FAVICON_URL` | *(empty)* | Favicon override. Empty = use the bundled generic favicon. |
| `SEARCH_DEFAULT_K` | `20` | Default top-K. |
| `SEARCH_MAX_K` | `100` | Max top-K accepted via query string. |
| `SEARCH_QUERY_TIMEOUT` | `30` | Per-query wall-clock cap (seconds). |
| `EMBY_BASE_URL`, `NAVIDROME_BASE_URL`, `KAVITA_BASE_URL` | *(empty)* | Optional deep-link bases for media-source result rows. Empty = no deep link rendered. |

**Default logo**: `minisite/static/eichi-logo.png` — an OpenAI
image-gen-generated abstract glyph (white on dark), fair-use safe (no
third-party brand IP). Opt in via `SEARCH_SITE_LOGO_DEFAULT=1`, or
ignore and ship your own via `SEARCH_SITE_LOGO_URL`.

## Supported corpus types

**Today** (built into the core):

- Filesystem trees of plain text / Markdown / source-code files
  (heuristic source tags: `transcripts` for `.claude/projects/`,
  `obsidian` for `/Notes/`, `memory` for `/memory/`, `file` otherwise).
- Streaming JSONL documents via `eichi index-stream` — anything you can
  render as one `{"source": ..., "doc_id": ..., "text": ..., "mtime": ...}`
  per line.

**Coming**:

- A reference set of example connectors (agent session JSONL
  transcripts, a generic queue/task log connector) shipping under
  `examples/connectors/` in a follow-up release.

## Performance

- ~30 ms median query latency on a corpus of ~150 markdown files
  (CPU-only, mpnet 768-dim embeddings, hybrid retrieval).
- ~5-10 docs/sec indexing throughput on the same machine; bulk
  re-indexing of a 100k-row corpus runs in single-digit minutes.

Numbers vary heavily with corpus size, document length, CPU, and
embedding-model choice; the values above are a rough order-of-magnitude.

## Stack

- Python 3.11+.
- `sentence-transformers/all-mpnet-base-v2` (768-dim, CPU). Override via
  the `EMBEDDING_MODEL` constant in `src/eichi/__init__.py`; changing it
  triggers an automatic DB wipe + rebuild on next open.
- `sqlite-vec` extension on top of stock SQLite. FTS5 also stock.

## Schema

```
chunks       vec0 virtual table, embedding float[EMBEDDING_DIM]
chunks_fts   fts5 virtual table (unicode61 tokenizer) for BM25
chunk_meta   rowid, source, path, chunk_idx, offset, text, content_hash, indexed_at
files        path PK, source, mtime, file_hash, indexed_at
meta         k/v (embedding_model, schema_version)
```

## Telemetry / privacy

eichi never phones home. The only network call is a single bootstrap
download of the embedding model on first use (after which
`HF_HUB_OFFLINE=1` is enforced). A local query-log at
`~/.local/share/eichi/query.log` records per-query latency for an optional
metrics exporter; the file never leaves the machine and can be disabled
with `EICHI_NO_QUERY_LOG=1`.

## Tests

```sh
.venv/bin/python -m pytest tests/ -v
```

## Acknowledgments

Built on:

- [sqlite-vec](https://github.com/asg017/sqlite-vec) — Alex Garcia's
  vector-search extension for SQLite.
- [sentence-transformers](https://www.sbert.net/) — Reimers & Gurevych.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers).

## License

MIT — see [LICENSE](./LICENSE).
