# eichi minisite

Flask web frontend for the [eichi](../README.md) semantic-search CLI.

Single-file Flask app ([`app.py`](app.py)) that delegates queries to a
persistent worker ([`eichi_worker.py`](eichi_worker.py)) which imports
the host's eichi package + opens the sqlite-vec DB once at startup.

Designed to sit **behind** a reverse proxy (oauth2-proxy, nginx
`auth_request`, etc.). The app itself does NOT enforce access control —
it trusts `X-Auth-Request-Email`, `X-Auth-Uid`, and `X-Auth-Role`
headers for display + per-role source authorisation. The upstream gate
is responsible for setting them honestly. Don't bind to a public IP
without a gate.

## Container is web-only — no CLI inside

This container ships gunicorn + the Flask app + the Python deps needed
to load the embedding model for query-time lookups. It does **not**
include the `eichi` CLI wrapper. Commands like

```bash
docker compose exec eichi-search eichi --version   # WON'T WORK
docker compose exec eichi-search eichi index ...   # WON'T WORK
```

will fail — there is no `eichi` binary on `$PATH` inside the container.
This is intentional: the container is a read-only web surface; the
host-side CLI owns indexing + admin. For any CLI operation (version
probe, indexing, stats, removals) run the wrapper at the repo root on
the host:

```bash
cd /path/to/eichi
./bin/eichi --version
./bin/eichi index ~/Documents/notes
./bin/eichi stats
```

Both surfaces share one SQLite index DB (the container bind-mounts it
read-write so the worker can open it, but only the host CLI writes to
it).

## Run

Easiest from the repo root via the Makefile:

```bash
make serve-minisite
```

That builds `eichi-minisite:dev` and runs it on `http://localhost:8001/`
bind-mounting the host's eichi venv + index DB + Hugging Face cache.

Standalone `docker run`:

```bash
docker build -t eichi-minisite:dev -f minisite/Dockerfile .
docker run --rm -it -p 8001:8000 \
  -v "$HOME/repos/eichi:/home/hndrewaall/repos/eichi:ro" \
  -v "$HOME/.local/share/eichi:/home/hndrewaall/.local/share/eichi:rw" \
  -v "$HOME/.cache/huggingface:/home/hndrewaall/.cache/huggingface:ro" \
  -e EICHI_DB=/home/hndrewaall/.local/share/eichi/index.db \
  -e EICHI_PYTHON=/home/hndrewaall/repos/eichi/.venv/bin/python \
  eichi-minisite:dev
```

The minisite is also wired into the [`claude-watch`](https://github.com/hndrewaall/claude-watch)
fresh-laptop compose stack under `examples/compose/`. From the
claude-watch repo: `make compose-up` brings both this minisite (port
8001) and the queue-minisite (port 8000) up together.

## Pre-flight

The container needs a populated index DB to return results. Bootstrap
one on the host first:

```bash
cd /path/to/eichi
uv sync
eichi index ~/Documents/notes        # any corpus
```

The host venv is bind-mounted into the container at the matching
absolute path so the venv's editable .pth (which uses an absolute path)
keeps resolving. The Makefile target hardcodes
`~/repos/eichi`; adjust the bind mount if your clone lives elsewhere.

## Branding

The minisite ships generic defaults; brand identity lives in env vars
so the public image is whitelabel-able without forking:

| Var | Default | Meaning |
|-----|---------|---------|
| `SEARCH_SITE_TITLE` | `eichi search` | Page `<title>` + header text. |
| `SEARCH_SITE_LOGO_URL` | *(empty)* | Header logo `<img>` src. Empty = no logo unless `SEARCH_SITE_LOGO_DEFAULT=1`. |
| `SEARCH_SITE_LOGO_DEFAULT` | *(empty)* | Set to `1` to render the bundled `static/eichi-logo.png`. |
| `SEARCH_SITE_BRAND` | *(empty)* | Footer brand string. Empty = no footer. |
| `SEARCH_SITE_FAVICON_URL` | *(empty)* | Favicon override. Empty falls back to the bundled favicons. |

## Environment

| Var | Default | Meaning |
|-----|---------|---------|
| `EICHI_DB` | `/home/hndrewaall/.local/share/eichi/index.db` | Path to the sqlite-vec DB inside the container. |
| `EICHI_PYTHON` | `/home/hndrewaall/repos/eichi/.venv/bin/python` | Bind-mounted host venv interpreter that imports the eichi package. |
| `SEARCH_DEFAULT_K` | `20` | Default top-K. |
| `SEARCH_MAX_K` | `100` | Max top-K accepted via query string. |
| `SEARCH_QUERY_TIMEOUT` | `30` | Per-query wall-clock cap (seconds). |
| `EICHI_SOURCES_CONFIG` | *(empty)* | Path to the local source-map TOML (see [Source map](#source-map)). |

See [`.env.example`](.env.example) for a template.

## Source map

The minisite is service-agnostic — every per-deployment thing about
result display (badge label, outbound URL template, badge colour) and
per-role source authorisation (which connector-stamped tags each role
may query) lives in a local TOML config, not in the code. Source tags
stamped by connectors are treated as opaque tokens; the local config
maps them to display rendering.

Configuration is OPTIONAL. With no config file the minisite boots
with an empty `ALL_SOURCES` set — the admin role's wildcard expands to
nothing, and every result row renders neutrally (label = literal
source id, no outbound link). Drop a config file once you wire up a
connector.

File resolution: `$EICHI_SOURCES_CONFIG` env override → else
`$XDG_CONFIG_HOME/eichi/sources.toml` → else
`~/.config/eichi/sources.toml`. See
[`examples/sources.example.toml`](../examples/sources.example.toml) at
the repo root for the schema, including the `{doc_id}` /
`{doc_id_suffix}` URL-template substitution tokens.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v   # from the repo root
```

Pytest collects `minisite/tests/` alongside the package tests. The
minisite tests use Flask's test client + a stub worker — no Docker, no
real index.
