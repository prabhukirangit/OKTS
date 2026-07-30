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
from pathlib import Path

from okts.core.model import Bundle
from okts.core.serialize import (
    concept_from_markdown,
    concept_to_markdown,
    split_frontmatter,
)

INDEX_FILENAME = "index.md"


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
    """Write each concept to ``<id>.md`` and the hierarchy to ``index.md``."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    for concept in bundle:
        (path / f"{concept.id}.md").write_text(
            concept_to_markdown(concept), encoding="utf-8"
        )

    if bundle.hierarchy:
        import yaml

        fm = yaml.safe_dump(
            {"hierarchy": bundle.hierarchy}, sort_keys=False, allow_unicode=True
        ).strip()
        (path / INDEX_FILENAME).write_text(f"---\n{fm}\n---\n", encoding="utf-8")
