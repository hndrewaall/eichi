"""Tests for the similarity / relevance-band projection added on top of the
raw vec0 L2 distance.

The transform is monotonic and ranking-preserving — these tests pin the
formula, the band thresholds, and the CLI / JSON surface so a future tweak
of THRESHOLD or BAND_CUTOFFS doesn't silently break downstream web
frontends or JSON consumers.
"""

from __future__ import annotations

import json

import pytest

from eichi.cli import (
    BAND_CUTOFFS,
    BAND_LABELS,
    SIMILARITY_THRESHOLD,
    _print_hit,
    relevance_band,
    similarity_from_distance,
)
from eichi.store import SearchHit


def _hit(score: float, source: str = "memory") -> SearchHit:
    return SearchHit(
        rowid=1,
        score=score,
        source=source,
        path=f"/x/{source}.md",
        chunk_idx=0,
        offset=0,
        text="example chunk text",
    )


# ---------------------------------------------------------------------------
# similarity_from_distance — transform 0–1 range, clamps at extremes
# ---------------------------------------------------------------------------


def test_similarity_zero_distance_is_one():
    """A perfect-match (distance=0) projects to similarity 1.0."""
    assert similarity_from_distance(0.0) == pytest.approx(1.0)


def test_similarity_at_threshold_is_zero():
    """distance=THRESHOLD lands exactly at similarity 0.0 (formula edge)."""
    assert similarity_from_distance(SIMILARITY_THRESHOLD) == pytest.approx(0.0)


def test_similarity_clamps_above_threshold():
    """distance > THRESHOLD must clamp at 0.0 — never negative similarity."""
    assert similarity_from_distance(2.0) == 0.0
    assert similarity_from_distance(10.0) == 0.0


def test_similarity_negative_distance_is_zero():
    """A negative distance can't come from vec0 (always >= 0) — it shows up
    when hybrid retrieval surfaces a BM25-only hit (fts5's bm25() returns
    negative ranks). We can't project that into an L2 similarity, so we
    bail out at 0.0 rather than mis-paint it as a perfect match."""
    assert similarity_from_distance(-0.5) == 0.0
    assert similarity_from_distance(-5.381) == 0.0


def test_band_negative_distance_is_distant():
    """Mirror of the similarity rule — a BM25-only hit can't be banded
    sensibly, default to ``distant`` rather than ``strong``."""
    assert relevance_band(-0.5) == "distant"
    assert relevance_band(-5.381) == "distant"


def test_similarity_midpoint_within_range():
    """Spot-check the linear interpolation midway."""
    half = SIMILARITY_THRESHOLD / 2.0
    sim = similarity_from_distance(half)
    assert sim == pytest.approx(0.5)


def test_similarity_monotonic_decreasing():
    """Closer distances ALWAYS produce higher similarity — ranking preserved."""
    distances = [0.0, 0.3, 0.6, 0.9, 1.2, 1.4, 1.8]
    sims = [similarity_from_distance(d) for d in distances]
    for a, b in zip(sims, sims[1:]):
        assert a >= b, f"non-monotonic: {sims}"


def test_similarity_handles_garbage():
    """None / NaN / non-numeric must not raise."""
    assert similarity_from_distance(None) == 0.0
    assert similarity_from_distance("not a number") == 0.0


# ---------------------------------------------------------------------------
# relevance_band — threshold buckets
# ---------------------------------------------------------------------------


def test_band_strong_at_low_distance():
    """distance ≤ 0.7 → strong (close match band)."""
    assert relevance_band(0.0) == "strong"
    assert relevance_band(0.5) == "strong"
    assert relevance_band(0.7) == "strong"


def test_band_moderate_in_middle():
    """0.7 < distance ≤ 1.05 → moderate."""
    assert relevance_band(0.71) == "moderate"
    assert relevance_band(0.9) == "moderate"
    assert relevance_band(1.05) == "moderate"


def test_band_weak_in_upper_middle():
    """1.05 < distance ≤ 1.20 → weak."""
    assert relevance_band(1.06) == "weak"
    assert relevance_band(1.15) == "weak"
    assert relevance_band(1.20) == "weak"


def test_band_distant_above_top_cutoff():
    """distance > 1.20 → distant (the catch-all)."""
    assert relevance_band(1.21) == "distant"
    assert relevance_band(1.5) == "distant"
    assert relevance_band(2.0) == "distant"


def test_band_cutoffs_align_with_labels():
    """Sanity check on the constants — one fewer cutoff than labels (the
    last label is the catch-all). Pins the data shape so a refactor that
    breaks the zip() walk fails loudly."""
    assert len(BAND_LABELS) == len(BAND_CUTOFFS) + 1
    assert BAND_LABELS[-1] == "distant"  # catch-all is always last


def test_band_handles_garbage():
    """Bad input → distant (defensive default — never raise)."""
    assert relevance_band(None) == "distant"
    assert relevance_band("nope") == "distant"


# ---------------------------------------------------------------------------
# _print_hit — JSON output carries similarity + relevance_band
# ---------------------------------------------------------------------------


def test_print_hit_json_carries_similarity_and_band():
    """JSON envelope must include both new fields alongside raw score."""
    h = _hit(score=0.85)
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["score"] == pytest.approx(0.85)  # raw distance preserved
    assert "similarity" in rec
    assert rec["similarity"] == pytest.approx(
        similarity_from_distance(0.85), abs=1e-6
    )
    assert rec["relevance_band"] == "moderate"


def test_print_hit_json_strong_band():
    h = _hit(score=0.4)
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["relevance_band"] == "strong"
    assert rec["similarity"] > 0.7  # close match, high similarity


def test_print_hit_json_distant_band_clamps_similarity():
    h = _hit(score=1.8)
    rec = json.loads(_print_hit(h, json_out=True))
    assert rec["relevance_band"] == "distant"
    assert rec["similarity"] == 0.0


# ---------------------------------------------------------------------------
# _print_hit — text rendering: similarity + band by default, raw with flag
# ---------------------------------------------------------------------------


def test_print_hit_text_default_renders_similarity_and_band():
    """Default text output: '0.NN [band]' replaces raw distance."""
    h = _hit(score=0.85)
    line = _print_hit(h, json_out=False)
    parts = line.split("\t")
    score_field = parts[1]
    # Format: "<2-decimal-similarity> [<band>]"
    assert "[" in score_field and "]" in score_field
    assert "moderate" in score_field
    # The 2-decimal projection of similarity_from_distance(0.85).
    expected_sim = f"{similarity_from_distance(0.85):.2f}"
    assert score_field.startswith(expected_sim)


def test_print_hit_text_raw_score_flag_restores_raw_distance():
    """--raw-score reverts to the legacy 4-decimal raw distance render."""
    h = _hit(score=0.85)
    line = _print_hit(h, json_out=False, raw_score=True)
    parts = line.split("\t")
    score_field = parts[1]
    assert score_field == "0.8500"
    # Band annotation must NOT leak into raw-score mode (regression guard).
    assert "[" not in score_field
    assert "moderate" not in score_field


def test_print_hit_text_distant_band_renders_zero_similarity():
    """A no-relationship hit (distance > THRESHOLD) renders '0.00 [distant]'."""
    h = _hit(score=2.0)
    line = _print_hit(h, json_out=False)
    parts = line.split("\t")
    assert parts[1] == "0.00 [distant]"
