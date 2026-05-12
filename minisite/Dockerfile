# eichi minisite — Flask web frontend for the eichi semantic search CLI.
#
# Single-file Flask app (app.py) that shells out to a persistent worker
# (eichi_worker.py) which imports the host's eichi package + opens the
# sqlite-vec DB once at startup. The container stays slim: only flask
# + gunicorn are installed here. The heavy ML stack (sentence-transformers,
# sqlite-vec, numpy) is supplied by the *host* eichi venv via a
# bind-mount, so a single shared model + cache serves both the CLI and
# the web UI without doubling RAM or disk.
#
# Build context
# -------------
# The Dockerfile expects to be built with the repo root as context so
# the `bin/eichi` wrapper script is reachable for COPY. Typical
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

# flask + gunicorn — that's the entire container-side dep set. The
# eichi query runs under the bind-mounted host venv's interpreter, so
# the container does not need ML deps installed.
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    gunicorn==23.0.0

# The bind-mounted host venv's python interpreter is typically a
# symlink to /usr/bin/python3.11 (Debian / Ubuntu default location).
# Slim images put python at /usr/local/bin/python3.11. Bridge the two
# so the host venv's interpreter symlink resolves on exec without
# having to rewrite it.
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11

# Give the runtime UID a real /etc/passwd entry. Some Python deps in
# the eichi venv (huggingface_hub / filelock) call pwd.getpwuid()
# during model load and crash on a UID with no entry. Default to
# 1000:1000 — override via docker-compose `user:` if your host
# operator runs under a different UID.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} eichi \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --no-create-home --shell /bin/sh eichi

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
