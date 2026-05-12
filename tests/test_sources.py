"""Tests for ``eichi.sources`` — local source-map config loader.

The minisite reads a TOML file mapping each connector-stamped source
tag to display rendering + per-role allowlist. These tests exercise:

* config loading from a tempfile (happy path + schema errors),
* rendering with known and unknown source ids,
* per-role allowlist resolution (including the ``"*"`` wildcard),
* URL template substitution (``{doc_id}`` / ``{doc_id_suffix}``).

None of the tests touch the operator's real ``~/.config/eichi/`` —
every call passes an explicit path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eichi import sources as sm


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "sources.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_missing_config_file_is_silent(tmp_path):
    """A nonexistent config path returns an empty SourceMap, no exception."""
    sm_obj = sm.load(tmp_path / "does-not-exist.toml")
    assert isinstance(sm_obj, sm.SourceMap)
    assert sm_obj.all_sources == frozenset()
    assert sm_obj.sources_for_role("admin") == frozenset()
    assert sm_obj.sources_for_role("search-user") == frozenset()


def test_empty_file_is_silent(tmp_path):
    """An empty TOML file is structurally valid; load returns an empty map."""
    p = _write_toml(tmp_path, "")
    sm_obj = sm.load(p)
    assert sm_obj.all_sources == frozenset()


def test_malformed_toml_raises(tmp_path):
    """A non-TOML file must raise SourcesConfigError, not return silently."""
    p = _write_toml(tmp_path, "[sources.broken\nlabel = nope")
    with pytest.raises(sm.SourcesConfigError) as exc:
        sm.load(p)
    assert "invalid TOML" in str(exc.value)


def test_source_missing_label_raises(tmp_path):
    """[sources.<id>] without `label` is a schema error."""
    p = _write_toml(
        tmp_path,
        '[sources.no-label]\nbadge_color = "#cb4b16"\n',
    )
    with pytest.raises(sm.SourcesConfigError) as exc:
        sm.load(p)
    assert "label" in str(exc.value)


def test_source_label_must_be_string(tmp_path):
    p = _write_toml(tmp_path, "[sources.bad]\nlabel = 42\n")
    with pytest.raises(sm.SourcesConfigError):
        sm.load(p)


def test_role_must_be_list(tmp_path):
    p = _write_toml(
        tmp_path,
        '[sources.a]\nlabel = "A"\n[roles]\nadmin = "everything"\n',
    )
    with pytest.raises(sm.SourcesConfigError) as exc:
        sm.load(p)
    assert "must be a list" in str(exc.value)


def test_role_entries_must_be_strings(tmp_path):
    p = _write_toml(
        tmp_path,
        '[sources.a]\nlabel = "A"\n[roles]\nadmin = ["a", 7]\n',
    )
    with pytest.raises(sm.SourcesConfigError):
        sm.load(p)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _make_map(
    *,
    sources: dict[str, dict] | None = None,
    roles: dict[str, list[str]] | None = None,
) -> sm.SourceMap:
    """Construct a SourceMap directly from raw dicts (test-only helper)."""
    entries = {
        sid: sm._SourceEntry(
            label=raw["label"],
            url_template=raw.get("url"),
            badge_color=raw.get("badge_color"),
            badge_class=raw.get("badge_class"),
        )
        for sid, raw in (sources or {}).items()
    }
    role_map = {
        r: frozenset(srcs) for r, srcs in (roles or {}).items()
    }
    return sm.SourceMap(sources=entries, roles=role_map)


def test_unknown_source_renders_neutral():
    """An unmapped source_id → render() returns None (caller falls back)."""
    smap = _make_map(sources={"a": {"label": "Alpha"}})
    assert smap.render("unknown-source", "unknown-source:item:1") is None


def test_known_source_uses_config_label_and_color():
    """A configured source_id → render() returns the configured fields."""
    smap = _make_map(
        sources={
            "alpha": {
                "label": "Alpha Library",
                "url": "http://example.com/a/{doc_id}",
                "badge_color": "#cb4b16",
                "badge_class": "alpha-badge",
            }
        }
    )
    out = smap.render("alpha", "alpha:item:42")
    assert out is not None
    assert out.source_id == "alpha"
    assert out.label == "Alpha Library"
    assert out.badge_color == "#cb4b16"
    assert out.badge_class == "alpha-badge"


def test_render_with_no_url_template():
    """A source without a `url` template renders with href = None."""
    smap = _make_map(sources={"alpha": {"label": "Alpha"}})
    out = smap.render("alpha", "alpha:item:42")
    assert out is not None
    assert out.href is None


def test_url_template_substitutes_doc_id():
    smap = _make_map(
        sources={
            "alpha": {
                "label": "Alpha",
                "url": "http://example.com/raw/{doc_id}",
            }
        }
    )
    out = smap.render("alpha", "alpha:thing:42")
    assert out is not None
    assert out.href == "http://example.com/raw/alpha:thing:42"


def test_url_template_substitutes_doc_id_suffix():
    """``{doc_id_suffix}`` strips the leading ``<source_id>:`` prefix."""
    smap = _make_map(
        sources={
            "alpha": {
                "label": "Alpha",
                "url": "http://example.com/item/{doc_id_suffix}",
            }
        }
    )
    out = smap.render("alpha", "alpha:thing:42")
    assert out is not None
    assert out.href == "http://example.com/item/thing:42"


def test_url_template_doc_id_suffix_without_prefix_is_full_id():
    """When doc_id has no ``<source_id>:`` prefix, ``{doc_id_suffix}`` is
    the full id (defensive: don't slice off something arbitrary)."""
    smap = _make_map(
        sources={
            "alpha": {
                "label": "Alpha",
                "url": "http://example.com/item/{doc_id_suffix}",
            }
        }
    )
    out = smap.render("alpha", "bare-doc-id")
    assert out is not None
    assert out.href == "http://example.com/item/bare-doc-id"


# ---------------------------------------------------------------------------
# Per-role allowlist
# ---------------------------------------------------------------------------


def test_role_wildcard_expands_to_all_sources():
    smap = _make_map(
        sources={"a": {"label": "A"}, "b": {"label": "B"}, "c": {"label": "C"}},
        roles={"admin": ["*"]},
    )
    assert smap.sources_for_role("admin") == frozenset({"a", "b", "c"})


def test_role_filters_unknown_sources():
    """A role referencing an unknown source id is silently dropped — the
    role is only granted access to sources actually declared under
    [sources]."""
    smap = _make_map(
        sources={"a": {"label": "A"}},
        roles={"user": ["a", "ghost"]},
    )
    assert smap.sources_for_role("user") == frozenset({"a"})


def test_unknown_role_is_empty():
    smap = _make_map(
        sources={"a": {"label": "A"}},
        roles={"admin": ["*"]},
    )
    assert smap.sources_for_role("unknown") == frozenset()


# ---------------------------------------------------------------------------
# End-to-end load → render
# ---------------------------------------------------------------------------


def test_load_full_config_roundtrip(tmp_path):
    """A complete TOML config round-trips through load() correctly."""
    p = _write_toml(
        tmp_path,
        """
[roles]
admin = ["*"]
"search-user" = ["alpha", "beta"]

[sources.alpha]
label = "Alpha Library"
url = "http://example.com/a/{doc_id_suffix}"
badge_color = "#cb4b16"

[sources.beta]
label = "Beta Library"
badge_color = "#859900"

[sources.gamma]
label = "Gamma Library"
""",
    )
    smap = sm.load(p)
    assert smap.all_sources == frozenset({"alpha", "beta", "gamma"})
    assert smap.sources_for_role("admin") == frozenset(
        {"alpha", "beta", "gamma"}
    )
    assert smap.sources_for_role("search-user") == frozenset({"alpha", "beta"})
    alpha = smap.render("alpha", "alpha:show:42")
    assert alpha is not None
    assert alpha.label == "Alpha Library"
    assert alpha.href == "http://example.com/a/show:42"
    assert alpha.badge_color == "#cb4b16"
    beta = smap.render("beta", "beta:album:7")
    assert beta is not None
    assert beta.label == "Beta Library"
    assert beta.href is None  # no url template


def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    """``$EICHI_SOURCES_CONFIG`` points the loader at an explicit file."""
    p = _write_toml(tmp_path, '[sources.a]\nlabel = "A"\n')
    monkeypatch.setenv("EICHI_SOURCES_CONFIG", str(p))
    sm.reset_cache()
    out = sm.get_default()
    assert out.all_sources == frozenset({"a"})
    sm.reset_cache()


def test_get_default_with_missing_env_and_file_is_empty(tmp_path, monkeypatch):
    """No env var + a missing default path → empty SourceMap, silent."""
    monkeypatch.delenv("EICHI_SOURCES_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    sm.reset_cache()
    out = sm.get_default()
    assert isinstance(out, sm.SourceMap)
    assert out.all_sources == frozenset()
    sm.reset_cache()
