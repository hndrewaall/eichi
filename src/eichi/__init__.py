"""eichi — local sqlite-vec + sentence-transformers vector search CLI.

Modules:
    chunk: split text into overlapping windows.
    embed: lazy-loaded sentence-transformers model wrapper.
    store: sqlite-vec backed storage (open, add, search, stats, remove).
    cli:   argparse subcommand dispatch.
"""

__version__ = "0.4.0"
# Bigger embedder for better semantic synonymy. all-MiniLM-L6-v2 (384d) is the
# bootstrap-quality default; mpnet (768d) is the standard quality-upgrade path
# with only ~3-5x indexing-time cost. Changing this constant triggers an
# automatic DB wipe + rebuild on next open_db() — see store._maybe_migrate_embedder.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
SCHEMA_VERSION = "3"  # bumped when embedder upgraded from MiniLM-L6 (384d) to mpnet (768d)
