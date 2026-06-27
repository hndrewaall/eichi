# eichi

Local-first semantic search CLI. Python package in `src/eichi/`. Web UI
in `minisite/`. Storage backend: SQLite + the [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
extension. Embedding model: `sentence-transformers/all-mpnet-base-v2`
(768-dim, CPU).

## Install

```bash
uv sync
```

`uv sync` creates the venv and installs eichi (editable) plus its deps
resolved from the committed `uv.lock`. As a fallback without
[uv](https://github.com/astral-sh/uv), `python -m venv .venv &&
. .venv/bin/activate && pip install -e .` works too — the project is a
pure-Python wheel with `sqlite-vec` and `sentence-transformers` as
runtime deps.

## Dev loop

```bash
make install   # uv sync — venv + editable install from uv.lock (one-time bootstrap)
make test      # pytest tests/
make lint      # ruff check src tests
make all       # lint + test
```

Or directly: `.venv/bin/python -m pytest tests/ -v` and
`.venv/bin/ruff check src tests`.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff +
pytest on Python 3.11 and 3.12 against every PR + push to main. Don't
push without `make all` green locally.

## Web UI

The Flask app in [`minisite/`](minisite/) wraps `eichi query` for use
behind a reverse proxy. Iterate locally with:

```bash
make serve-minisite          # docker compose build + up against minisite/Dockerfile
```

See [`minisite/README.md`](minisite/README.md) for env vars + branding
knobs. The minisite ships generic defaults; brand identity goes through
env vars, not source edits.

## Key files

- `src/eichi/cli.py` — argparse entrypoint (`eichi` console script).
- `src/eichi/store.py` — SQLite schema + sqlite-vec load + chunk-table
  CRUD. The on-disk DB lives at `~/.local/share/eichi/index.db` (or
  `$EICHI_DB` if set).
- `src/eichi/embed.py` — sentence-transformers wrapper with a one-shot
  bootstrap download then `HF_HUB_OFFLINE=1` enforcement.
- `src/eichi/chunk.py` — document → chunk splitter.
- `src/eichi/connectors/` — per-corpus adapters (e.g. claude-jsonl,
  claude-watch-queue). Add new sources here.
- `minisite/app.py` — Flask app (`/`, `/api/search`).
- `minisite/eichi_worker.py` — persistent worker process that imports
  the eichi package once and answers queries over a pipe.

## Conventions

- **Don't hard-fail at runtime on a missing embedding model.** Bootstrap
  downloads once on first use (model is ~400 MB), then enforces
  `HF_HUB_OFFLINE=1`. If you add a new model, keep the bootstrap-then-
  offline pattern.
- **No network calls in tests.** All tests run against tempdir SQLite +
  a tiny stub embedder where needed.
- **Schema is stable.** Migrations land in `store.py`; bump
  `SCHEMA_VERSION` and write a one-shot upgrade. Don't break the
  `chunks` / `chunks_fts` / `chunk_meta` / `files` / `meta` shape
  without a migration.

## Don't touch

- `EMBEDDING_MODEL` in `src/eichi/__init__.py` without thinking about
  re-indexing — changing it triggers an automatic DB wipe + rebuild on
  next open.
- The packaged `sqlite-vec` shared object — it loads from the
  pip-installed wheel via `sqlite_vec.loadable_path()`, no system
  package needed.
