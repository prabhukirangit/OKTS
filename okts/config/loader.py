"""Loader for ``tools.config.yaml``.

Example::

    sources:
      - interface: mcp
        servers: [github-mcp, slack-mcp, linear-mcp]
      - interface: http
        openapi: ./specs/stripe.yaml
      - interface: function
        module: ./my_local_tools.py
    retrieval: { mode: hybrid, graph_expand: true }

The loader is intentionally permissive: it validates the shape it knows and
stashes everything else on ``Source.options`` so new adapter knobs don't require
loader changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Source:
    """One entry under ``sources:``."""

    interface: str
    options: dict[str, Any] = field(default_factory=dict)

    # convenience accessors for the common keys
    @property
    def servers(self) -> list[str]:
        return list(self.options.get("servers") or [])

    @property
    def openapi(self) -> str | None:
        return self.options.get("openapi")

    @property
    def module(self) -> str | None:
        return self.options.get("module")


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"          # "bm25" | "dense" | "hybrid"
    graph_expand: bool = True
    hierarchy_prefilter: bool = True
    k: int = 5


@dataclass
class Config:
    sources: list[Source] = field(default_factory=list)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # Where the built OKT bundle is written/read.
    bundle_dir: str = "./okt-bundle"
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    """Parse a ``tools.config.yaml`` file into a :class:`Config`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")

    sources: list[Source] = []
    for entry in data.get("sources") or []:
        if not isinstance(entry, dict) or "interface" not in entry:
            raise ValueError(f"each source needs an 'interface': {entry!r}")
        interface = entry["interface"]
        options = {k: v for k, v in entry.items() if k != "interface"}
        sources.append(Source(interface=interface, options=options))

    r = data.get("retrieval") or {}
    retrieval = RetrievalConfig(
        mode=r.get("mode", "hybrid"),
        graph_expand=bool(r.get("graph_expand", True)),
        hierarchy_prefilter=bool(r.get("hierarchy_prefilter", True)),
        k=int(r.get("k", 5)),
    )

    return Config(
        sources=sources,
        retrieval=retrieval,
        bundle_dir=data.get("bundle_dir", "./okt-bundle"),
        raw=data,
    )
