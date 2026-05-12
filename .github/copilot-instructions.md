# Copilot instructions

See [`CLAUDE.md`](../CLAUDE.md) at the repo root for agent-facing
guidance: install, dev loop, key files, conventions.

Quick reference:

- Install: `uv venv --python 3.11 && uv pip install -e .` (or `make install`).
- Tests: `pytest tests/` (or `make test`).
- Lint: `ruff check src tests` (or `make lint`).
- CI runs ruff + pytest on Python 3.11 and 3.12.
- Minisite under `minisite/`; iterate via `make serve-minisite`.
