# eichi minisite — Flask web frontend for the eichi semantic search CLI.
#
# Single-file Flask app (app.py) that shells out to a persistent worker
# (eichi_worker.py) which imports the eichi package + opens the
# sqlite-vec DB once at startup. The container installs the full eichi
# dep set (sentence-transformers, sqlite-vec, numpy) so it works on
# any host architecture (Linux, macOS ARM, etc.) without relying on
# the host's venv binaries being Linux-native.
#
# Build context
# -------------
# The Dockerfile expects to be built with the repo root as context so
# the `bin/eichi` wrapper script and `src/` are reachable. Typical
# invocation from a sibling docker-compose:
#
#   build:
#     context: ../eichi
#     dockerfile: minisite/Dockerfile
#
# Auth
# ----
# This image performs NO authentication. Deploy it behind a reverse
# proxy that gates the front door (nginx auth_request, oauth2-proxy,
# etc.). The app trusts `X-Auth-Request-Email`, `X-Auth-Uid`, and
# `X-Auth-Role` headers for display + per-role source authorisation;
# the upstream gate is responsible for setting them honestly.

FROM python:3.11-slim

WORKDIR /app

# Install flask, gunicorn, AND the eichi package with all its ML deps.
# This makes the container self-contained — no reliance on host venv
# binaries being Linux-native (critical for macOS ARM hosts where the
# host venv contains Mach-O binaries that can't exec in Linux).
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    gunicorn==23.0.0 \
    /app/

# Backwards-compat: bridge /usr/bin/python3.11 -> /usr/local/bin so
# legacy EICHI_PYTHON paths that point at a venv symlink still resolve.
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11

# Give the runtime UID a real /etc/passwd entry. Some Python deps in
# the eichi venv (huggingface_hub / filelock) call pwd.getpwuid()
# during model load and crash on a UID with no entry. Default to
# 1000:1000 — override via docker-compose `user:` if your host
# operator runs under a different UID.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} eichi \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/sh eichi

# Copy the eichi `bin/eichi` wrapper into the image PATH so an operator
# `docker exec <container> eichi …` works for debugging. The wrapper
# itself expects the host venv to be bind-mounted at the documented
# path (see compose example); this is a convenience, not a dependency.
COPY bin/eichi /usr/local/bin/eichi

COPY minisite/app.py /app/app.py
COPY minisite/eichi_worker.py /app/eichi_worker.py
COPY minisite/templates /app/templates
COPY minisite/static /app/static

# Run as the eichi UID so the bind-mounted index DB / venv / HF cache
# are readable without surprise permission gymnastics.
USER eichi

EXPOSE 8000

# 2 workers x 4 threads. No in-process state to share, fine to scale.
# 30s default gunicorn timeout is fine — SEARCH_QUERY_TIMEOUT is 30s
# already and any cold-start beyond that is a real failure.
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
