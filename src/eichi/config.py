"""Optional user configuration for eichi.

The config file is OPTIONAL — eichi works with zero configuration. The
config file is useful if you want to declare a set of corpora that you
re-index on a schedule, and have a single `eichi index` invocation walk
all of them.

File location (first match wins):
  1. ``$EICHI_CONFIG`` env var (absolute path to TOML file).
  2. ``$XDG_CONFIG_HOME/eichi/eichi.toml`` if XDG_CONFIG_HOME is set.
  3. ``~/.config/eichi/eichi.toml``.

Format (TOML):

    # eichi.toml — declare named corpora to index.
    [[corpus]]
    name = "notes"
    path = "~/Documents/notes"
    extensions = ["md", "txt"]

    [[corpus]]
    name = "code-readmes"
    path = "~/repos"
    extensions = ["md"]

See ``eichi.toml.example`` in the repo root.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]


@dataclass
class Corpus:
    name: str
    path: Path
    extensions: List[str] = field(default_factory=list)


@dataclass
class Config:
    corpora: List[Corpus] = field(default_factory=list)


def config_path() -> Path:
    """Resolve the config file path (may not exist on disk)."""
    override = os.environ.get("EICHI_CONFIG")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "eichi" / "eichi.toml"


def load(path: Optional[Path] = None) -> Config:
    """Load the config file. Returns an empty :class:`Config` if absent."""
    p = path or config_path()
    if not p.exists():
        return Config()
    with open(p, "rb") as fh:
        data = _toml.load(fh)
    corpora: List[Corpus] = []
    for raw in data.get("corpus", []) or []:
        name = raw.get("name")
        cpath = raw.get("path")
        if not name or not cpath:
            continue
        corpora.append(
            Corpus(
                name=str(name),
                path=Path(os.path.expanduser(str(cpath))),
                extensions=[str(x).lstrip(".") for x in raw.get("extensions", [])],
            )
        )
    return Config(corpora=corpora)
