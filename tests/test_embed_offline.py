"""Tests for the offline-by-default embed loader.

These tests do NOT load the actual sentence-transformers model. They verify:

1. ``_is_cached()`` correctly detects a populated snapshot dir layout.
2. ``_load()`` sets ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` /
   ``HF_HUB_DISABLE_PROGRESS_BARS`` before importing sentence-transformers.
3. ``_load()`` passes ``local_files_only=True`` to SentenceTransformer.

Any runtime HF traffic from eichi is treated as a bug — the one allowed
exception is a single bootstrap download when the cache is empty.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from eichi import EMBEDDING_MODEL, embed


def test_is_cached_false_when_no_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert embed._is_cached() is False


def test_is_cached_true_with_populated_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    safe = "models--" + EMBEDDING_MODEL.replace("/", "--")
    snap = tmp_path / "hub" / safe / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert embed._is_cached() is True


def test_is_cached_false_when_snapshot_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    safe = "models--" + EMBEDDING_MODEL.replace("/", "--")
    snap = tmp_path / "hub" / safe / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    # No config.json — counts as not cached.
    assert embed._is_cached() is False


def test_load_sets_offline_env_and_local_files_only(monkeypatch):
    """_load() must set HF offline env vars and pass local_files_only=True.

    We stub out the actual SentenceTransformer constructor to avoid loading
    the model, and stub _is_cached() to return True so the bootstrap path
    is skipped.
    """
    # Reset module cache so _load() does the work.
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(embed, "_is_cached", lambda: True)

    # Clear env vars so we can verify they get set.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_HUB_DISABLE_PROGRESS_BARS"):
        monkeypatch.delenv(var, raising=False)

    captured = {}

    class _FakeST:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            captured["env_at_init"] = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_HUB_DISABLE_PROGRESS_BARS":
                    os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
            }

    fake_module = MagicMock()
    fake_module.SentenceTransformer = _FakeST

    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        embed._load()

    assert captured["name"] == EMBEDDING_MODEL
    assert captured["kwargs"].get("local_files_only") is True
    assert captured["env_at_init"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env_at_init"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["env_at_init"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_load_runs_bootstrap_when_uncached(monkeypatch):
    """If cache is missing, _load() must run the bootstrap (online) once
    and not pre-set the offline env vars before that bootstrap call."""
    monkeypatch.setattr(embed, "_model", None)

    cached_state = {"value": False}

    def _is_cached_stub():
        return cached_state["value"]

    monkeypatch.setattr(embed, "_is_cached", _is_cached_stub)

    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)

    bootstrap_calls = []
    final_calls = []

    class _FakeST:
        def __init__(self, name, **kwargs):
            if not kwargs.get("local_files_only"):
                # Bootstrap path: must NOT have offline env set.
                bootstrap_calls.append({
                    "name": name,
                    "kwargs": kwargs,
                    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                })
                # Mark cache populated for the second call.
                cached_state["value"] = True
            else:
                final_calls.append({
                    "name": name,
                    "kwargs": kwargs,
                    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                })

    fake_module = MagicMock()
    fake_module.SentenceTransformer = _FakeST

    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        embed._load()

    assert len(bootstrap_calls) == 1
    assert bootstrap_calls[0]["HF_HUB_OFFLINE"] in (None, "")
    assert len(final_calls) == 1
    assert final_calls[0]["kwargs"].get("local_files_only") is True
    assert final_calls[0]["HF_HUB_OFFLINE"] == "1"


def test_encode_empty_returns_zero_array():
    """Empty input must short-circuit before touching the model."""
    import numpy as np

    from eichi import EMBEDDING_DIM

    arr = embed.encode([])
    assert arr.shape == (0, EMBEDDING_DIM)
    assert arr.dtype == np.float32
