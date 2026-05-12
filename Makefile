.PHONY: install test lint all serve-minisite minisite-build minisite-down clean

# Bootstrap a Python 3.11 venv with eichi installed in editable mode.
# Idempotent — re-running just re-resolves deps in the existing venv.
install:
	uv venv --python 3.11
	uv pip install -e .

# Run the pytest suite.
test:
	.venv/bin/python -m pytest tests/ -v

# Run ruff against the package + tests. CI is non-blocking on lint, but
# fix what you can locally.
lint:
	.venv/bin/ruff check src tests

# Combined lint + test, matching the CI gate.
all: lint test

# Build the minisite image. Build context is the repo root because the
# Dockerfile copies bin/eichi + minisite/. Re-run after editing
# minisite/app.py or minisite/eichi_worker.py.
minisite-build:
	docker build -t eichi-minisite:dev -f minisite/Dockerfile .

# Build + run the minisite under docker compose if a compose file
# exists; otherwise fall back to a bare docker run. The bare path
# expects a host venv at .venv/ and an index DB at
# ~/.local/share/eichi/.
serve-minisite: minisite-build
	@if [ -f minisite/docker-compose.yml ]; then \
		docker compose -f minisite/docker-compose.yml up; \
	else \
		docker run --rm -it -p 8001:8000 \
			-v "$$HOME/repos/eichi:/home/hndrewaall/repos/eichi:ro" \
			-v "$$HOME/.local/share/eichi:/home/hndrewaall/.local/share/eichi:rw" \
			-v "$$HOME/.cache/huggingface:/home/hndrewaall/.cache/huggingface:ro" \
			-e EICHI_DB=/home/hndrewaall/.local/share/eichi/index.db \
			-e EICHI_PYTHON=/home/hndrewaall/repos/eichi/.venv/bin/python \
			eichi-minisite:dev; \
	fi

# Tear down minisite (compose mode only).
minisite-down:
	@if [ -f minisite/docker-compose.yml ]; then \
		docker compose -f minisite/docker-compose.yml down; \
	else \
		echo "No minisite/docker-compose.yml — nothing to do (bare docker run uses --rm)."; \
	fi

# Remove the local venv. The index DB at ~/.local/share/eichi/ is
# intentionally left alone.
clean:
	rm -rf .venv eichi.egg-info build dist
