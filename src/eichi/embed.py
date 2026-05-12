"""Lazy-loaded sentence-transformers embedding wrapper.

The model is loaded once on first encode() call and cached in the module.
HF cache lives at the user's default ``~/.cache/huggingface/``.

Network policy: at runtime we never hit HuggingFace. The model loads from
the local HF cache only. If the cache is missing we run a one-shot bootstrap
download with a clear stderr message and then load the cached files.

Mechanics:

* Before importing ``sentence_transformers`` / ``transformers`` /
  ``huggingface_hub`` we set ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``.
  These env vars are read at import time by those libraries and switch off
  the cache-validity HEAD requests + the "unauthenticated requests" warning.
* ``SentenceTransformer(..., local_files_only=True)`` is a belt-and-suspenders
  guard at the call site.
* ``_ensure_cached()`` checks for a populated snapshot dir and only enables
  network for that one bootstrap call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from . import EMBEDDING_DIM, EMBEDDING_MODEL

_model = None


def _hf_cache_dir() -> Path:
    base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(base) / "hub"


def _model_cache_dir() -> Path:
    # HF rewrites "org/name" to "models--org--name".
    safe = "models--" + EMBEDDING_MODEL.replace("/", "--")
    return _hf_cache_dir() / safe


def _is_cached() -> bool:
    """True iff a populated snapshot for EMBEDDING_MODEL exists locally."""
    snap_root = _model_cache_dir() / "snapshots"
    if not snap_root.is_dir():
        return False
    for snap in snap_root.iterdir():
        if not snap.is_dir():
            continue
        # A usable snapshot has at minimum a config.json (resolves through
        # symlinks into blobs/). If config.json is present and readable we
        # treat the cache as good — sentence-transformers will surface any
        # deeper missing-file error itself.
        cfg = snap / "config.json"
        if cfg.exists():
            return True
    return False


def _ensure_cached() -> None:
    """First-run bootstrap: download the model exactly once, online.

    All subsequent loads are offline; runtime HF traffic is treated as a bug.
    This is the one allowed exception, gated on the cache being empty.
    """
    if _is_cached():
        return
    print(
        f"eichi: first-run bootstrap — downloading {EMBEDDING_MODEL} "
        f"to {_hf_cache_dir()} (one-time, ~80 MB)",
        file=sys.stderr,
    )
    # Network is needed here. Do NOT set HF_HUB_OFFLINE for this call.
    from sentence_transformers import SentenceTransformer  # noqa: F401

    SentenceTransformer(EMBEDDING_MODEL)
    if not _is_cached():
        raise RuntimeError(
            f"eichi: failed to populate HF cache for {EMBEDDING_MODEL} at "
            f"{_model_cache_dir()}"
        )
    print("eichi: bootstrap done; future runs will be offline.", file=sys.stderr)


def _load():
    global _model
    if _model is None:
        _ensure_cached()
        # Force offline mode for the actual load. These env vars are read
        # at import time by the HF libraries, so set them BEFORE the import.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Also silence the "unauthenticated requests" warning belt-and-suspenders.
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        # Silence the local "Loading weights" tqdm bar that transformers
        # 5.x prints while iterating safetensors keys (purely local I/O).
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _model


def encode(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Encode a sequence of strings to a (N, EMBEDDING_DIM) float32 array.

    Empty input returns an empty (0, EMBEDDING_DIM) array.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _load()
    arr = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return arr.astype(np.float32, copy=False)


def encode_one(text: str) -> np.ndarray:
    """Convenience: encode a single string. Returns shape (EMBEDDING_DIM,)."""
    return encode([text])[0]


def is_loaded() -> bool:
    return _model is not None
