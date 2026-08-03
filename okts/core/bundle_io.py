"""Load / save an OKT bundle to a directory of markdown files.

Layout on disk::

    bundle/
      index.md            # optional: category hierarchy (YAML frontmatter)
      github.create_issue.md
      github.update_issue.md
      ...

Each ``*.md`` (other than ``index.md``) is one OKT concept. ``index.md`` carries
the category hierarchy under a ``hierarchy:`` key in its frontmatter, e.g.::

    ---
    hierarchy:
      github/issues: [github.create_issue, github.update_issue, github.list_issues]
      github/repos:  [github.get_repo]
    ---
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from okts.core.model import Bundle
from okts.core.serialize import (
    concept_from_markdown,
    concept_to_markdown,
    split_frontmatter,
)

INDEX_FILENAME = "index.md"

# Characters that are unsafe in a filename: path separators, parent-dir markers,
# drive/scheme colons, and anything non-portable. Concept ids are dot-namespaced
# (``github.create_issue``) so this normally only strips hostile/edge ids.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _concept_filename(concept_id: str) -> str:
    """Map a concept id to a SAFE ``<name>.md`` filename.

    A concept id can originate from an untrusted source (an MCP server's tool
    name, an OpenAPI ``operationId``, an agent card), so it must never be used
    as a path verbatim — an id like ``../../etc/x`` or ``/abs/path`` would let a
    hostile source write outside the bundle directory. We slug the id to a flat,
    separator-free stem. The authoritative id is preserved inside the file's
    ``id:`` frontmatter, so ``load_bundle`` round-trips correctly regardless of
    the on-disk filename.
    """
    stem = _UNSAFE_FILENAME_CHARS.sub("_", concept_id).strip(". ")
    if not stem or set(stem) <= {"."}:
        stem = "tool"
    return f"{stem}.md"


def load_bundle(directory: str | os.PathLike) -> Bundle:
    """Load every ``*.md`` in ``directory`` into a :class:`Bundle`."""
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"not a bundle directory: {directory}")

    bundle = Bundle()
    for md in sorted(path.glob("*.md")):
        if md.name == INDEX_FILENAME:
            fm, _ = split_frontmatter(md.read_text(encoding="utf-8"))
            hierarchy = fm.get("hierarchy") or {}
            if isinstance(hierarchy, dict):
                bundle.hierarchy = {k: list(v or []) for k, v in hierarchy.items()}
            continue
        concept = concept_from_markdown(md.read_text(encoding="utf-8"))
        bundle.add(concept)
    return bundle


def save_bundle(bundle: Bundle, directory: str | os.PathLike) -> None:
    """Write each concept to a safe ``<id>.md`` file and the hierarchy to
    ``index.md``. Filenames are slugged from the concept id (see
    :func:`_concept_filename`) so an untrusted id can never escape ``directory``.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    base = path.resolve()

    used: dict[str, str] = {}
    for concept in bundle:
        filename = _concept_filename(concept.id)
        # Guard against post-slug collisions clobbering a distinct concept. Use a
        # STABLE digest (not the salted built-in hash) so output is reproducible.
        if filename in used and used[filename] != concept.id:
            import hashlib

            digest = hashlib.sha1(concept.id.encode("utf-8")).hexdigest()[:8]
            filename = f"{filename[:-3]}_{digest}.md"
        used[filename] = concept.id

        target = (path / filename).resolve()
        if base != target.parent:  # defense in depth: never write outside base
            raise ValueError(f"refusing to write concept {concept.id!r} outside bundle dir")
        target.write_text(concept_to_markdown(concept), encoding="utf-8")

    if bundle.hierarchy:
        import yaml

        fm = yaml.safe_dump(
            {"hierarchy": bundle.hierarchy}, sort_keys=False, allow_unicode=True
        ).strip()
        (path / INDEX_FILENAME).write_text(f"---\n{fm}\n---\n", encoding="utf-8")
