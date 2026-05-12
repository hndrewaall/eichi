"""Source-map configuration for the eichi minisite.

The minisite stamps a ``source`` tag onto every result. Source tags are
opaque wire-protocol vocabulary emitted by whichever connectors the
operator runs against the index — eichi itself treats them as bytes.
This module loads a LOCAL config that maps each known source tag to:

* a human-readable label (for the badge / tooltip),
* an optional URL template (for an outbound link from the path),
* an optional badge / border colour.

The config also declares the per-role source allowlist used by the
restricted-access feature: every role names the source ids it may
query, with the literal ``"*"`` token expanding to every key declared
under ``[sources]``.

Config resolution (first non-empty wins):

  1. ``$EICHI_SOURCES_CONFIG`` env var (absolute path to a TOML file).
  2. ``$XDG_CONFIG_HOME/eichi/sources.toml`` if XDG_CONFIG_HOME is set.
  3. ``~/.config/eichi/sources.toml``.

The file is OPTIONAL. When it is missing the minisite still boots; the
admin role gets an empty allowlist (no sources known) and unknown
source tags render neutrally (label = literal source id, no outbound
link, neutral colour). Schema errors (malformed TOML, wrong types) are
LOUD — we raise ``SourcesConfigError`` so a misconfigured deploy fails
fast instead of silently dropping access rules.

URL templating
--------------

The ``url`` field may contain two substitution tokens:

* ``{doc_id}`` — the full doc id string as stamped by the connector
  (the ``path`` column on the result row).
* ``{doc_id_suffix}`` — same, but with the leading ``"<source>:"``
  prefix stripped when present. Useful when a connector encodes a
  service-specific id as e.g. ``mysource:item:123`` and you want the
  external URL to receive just ``item:123``.

Example
-------

::

    # ~/.config/eichi/sources.toml

    [roles]
    admin = ["*"]
    "search-user" = ["video-content", "music-album"]

    [sources.video-content]
    label = "Video Library"
    url = "http://example.com/video/{doc_id_suffix}"
    badge_color = "#cb4b16"

    [sources.music-album]
    label = "Music Library"
    url = "http://example.com/album/{doc_id_suffix}"
    badge_color = "#859900"

See ``examples/sources.example.toml`` in the repo root.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]


class SourcesConfigError(ValueError):
    """Raised when the sources config file is malformed.

    Missing file is NOT an error — load() returns an empty SourceMap in
    that case. Only structural / type errors raise.
    """


@dataclass(frozen=True)
class RenderedSource:
    """Resolved display surface for one search result.

    Returned by ``SourceMap.render()`` when the result's source id is
    known to the config. Callers use ``label`` for the badge text +
    tooltip, ``href`` (if set) for an outbound link from the path, and
    ``badge_color`` / ``badge_class`` for visual differentiation.
    """

    source_id: str
    label: str
    href: Optional[str]
    badge_color: Optional[str]
    badge_class: Optional[str]


@dataclass
class _SourceEntry:
    label: str
    url_template: Optional[str]
    badge_color: Optional[str]
    badge_class: Optional[str]


@dataclass
class SourceMap:
    """In-memory representation of the resolved sources config.

    The minisite reads this at module-load time. Tests construct
    instances directly to exercise the rendering / authorisation
    surface without touching the filesystem.
    """

    sources: Mapping[str, _SourceEntry] = field(default_factory=dict)
    roles: Mapping[str, frozenset[str]] = field(default_factory=dict)

    # --- introspection ---------------------------------------------------

    @property
    def all_sources(self) -> frozenset[str]:
        """Every source id known to the config (the keys of ``[sources]``).

        ``"*"`` in a role's allowlist expands to this set.
        """
        return frozenset(self.sources.keys())

    def sources_for_role(self, role: str) -> frozenset[str]:
        """Return the set of source ids ``role`` may query.

        Unknown role → empty set (deny everything). The literal ``"*"``
        wildcard in a role's allowlist expands to ``all_sources``.
        """
        allowed = self.roles.get(role)
        if allowed is None:
            return frozenset()
        if "*" in allowed:
            return self.all_sources
        # Filter to known sources so a stale role config doesn't leak a
        # tag the index doesn't know about.
        return frozenset(s for s in allowed if s in self.sources)

    # --- rendering -------------------------------------------------------

    def render(self, source_id: str, doc_id: str) -> Optional[RenderedSource]:
        """Resolve a (source_id, doc_id) pair to display fields.

        Returns ``None`` when ``source_id`` is unknown to the config —
        callers fall back to a neutral rendering (label = source_id
        literal, no href, neutral colour).
        """
        entry = self.sources.get(source_id)
        if entry is None:
            return None
        href: Optional[str] = None
        if entry.url_template:
            href = _expand_template(entry.url_template, source_id, doc_id)
        return RenderedSource(
            source_id=source_id,
            label=entry.label,
            href=href,
            badge_color=entry.badge_color,
            badge_class=entry.badge_class,
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def config_path() -> Path:
    """Resolve the sources config file path (may not exist on disk)."""
    override = os.environ.get("EICHI_SOURCES_CONFIG")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "eichi" / "sources.toml"


def load(path: Optional[Path] = None) -> SourceMap:
    """Load the sources config file. Returns an empty SourceMap if absent.

    Raises ``SourcesConfigError`` on a malformed file (invalid TOML,
    wrong types, unknown role wildcard form, missing required ``label``
    on a source entry).
    """
    p = path or config_path()
    if not p.exists():
        return SourceMap()
    try:
        with open(p, "rb") as fh:
            data = _toml.load(fh)
    except _toml.TOMLDecodeError as exc:
        raise SourcesConfigError(f"invalid TOML in {p}: {exc}") from exc
    except OSError as exc:
        raise SourcesConfigError(f"failed to read {p}: {exc}") from exc

    raw_sources = data.get("sources", {})
    if not isinstance(raw_sources, dict):
        raise SourcesConfigError(
            f"{p}: [sources] must be a table (got {type(raw_sources).__name__})"
        )
    sources: dict[str, _SourceEntry] = {}
    for sid, raw in raw_sources.items():
        if not isinstance(raw, dict):
            raise SourcesConfigError(
                f"{p}: [sources.{sid}] must be a table "
                f"(got {type(raw).__name__})"
            )
        label = raw.get("label")
        if label is None or not isinstance(label, str) or not label.strip():
            raise SourcesConfigError(
                f"{p}: [sources.{sid}] missing required string field `label`"
            )
        url_template = raw.get("url")
        if url_template is not None and not isinstance(url_template, str):
            raise SourcesConfigError(
                f"{p}: [sources.{sid}].url must be a string"
            )
        badge_color = raw.get("badge_color")
        if badge_color is not None and not isinstance(badge_color, str):
            raise SourcesConfigError(
                f"{p}: [sources.{sid}].badge_color must be a string"
            )
        badge_class = raw.get("badge_class")
        if badge_class is not None and not isinstance(badge_class, str):
            raise SourcesConfigError(
                f"{p}: [sources.{sid}].badge_class must be a string"
            )
        sources[str(sid)] = _SourceEntry(
            label=label.strip(),
            url_template=url_template or None,
            badge_color=badge_color or None,
            badge_class=badge_class or None,
        )

    raw_roles = data.get("roles", {})
    if not isinstance(raw_roles, dict):
        raise SourcesConfigError(
            f"{p}: [roles] must be a table (got {type(raw_roles).__name__})"
        )
    roles: dict[str, frozenset[str]] = {}
    for role, raw in raw_roles.items():
        if not isinstance(raw, list):
            raise SourcesConfigError(
                f"{p}: roles.{role} must be a list of source ids "
                f"(got {type(raw).__name__})"
            )
        cleaned: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise SourcesConfigError(
                    f"{p}: roles.{role} entries must be strings "
                    f"(got {type(item).__name__})"
                )
            cleaned.append(item)
        roles[str(role)] = frozenset(cleaned)

    return SourceMap(sources=sources, roles=roles)


@lru_cache(maxsize=1)
def _cached_load() -> SourceMap:
    """Process-lifetime cache for the default-path load.

    The minisite calls this once per request via :func:`get_default`;
    the lru_cache means we hit the filesystem exactly once per process.
    Tests that need to swap the config use :func:`reset_cache` or call
    :func:`load` directly with an explicit path.
    """
    return load()


def get_default() -> SourceMap:
    """Return the process-cached SourceMap loaded from the default path."""
    return _cached_load()


def reset_cache() -> None:
    """Invalidate the cached default SourceMap. Test-only."""
    _cached_load.cache_clear()


# ---------------------------------------------------------------------------
# URL template expansion
# ---------------------------------------------------------------------------


def _expand_template(template: str, source_id: str, doc_id: str) -> str:
    """Substitute ``{doc_id}`` and ``{doc_id_suffix}`` in ``template``.

    Unknown placeholders are left as-is rather than raising — keeps the
    URL emission robust against a future template variable that the
    runtime doesn't yet know about. Both substitution values are
    inserted verbatim; callers are expected to URL-quote the output
    themselves if it will be embedded in surrounding HTML / JSON.
    """
    prefix = f"{source_id}:"
    if doc_id.startswith(prefix):
        suffix = doc_id[len(prefix):]
    else:
        suffix = doc_id
    out = template
    out = out.replace("{doc_id_suffix}", suffix)
    out = out.replace("{doc_id}", doc_id)
    return out
