"""OKTS — Open Knowledge Tool Search.

Give your agent 300 tools; it only ever sees 3.

Public surface (stable, forever): ``search_tools``, ``load_tool``, ``call_tool``.
This top-level package re-exports the core data model so downstream layers
(adapters, index, serve, eval) import from one place.

Logging
-------
Every stage logs under the ``okts`` logger hierarchy (``okts.adapters.mcp``,
``okts.index.retriever``, ``okts.serve.service``, ...) using the standard
library ``logging`` module. Following library convention, OKTS attaches a
``NullHandler`` and configures nothing else, so it is **silent by default** and
never touches the root logger. To see what each of the four stages is doing —
what a source adapted to, why a query ranked the way it did, which dispatcher
handled a call — turn it on:

    import okts
    okts.enable_debug_logging()          # DEBUG for everything, to stderr
    okts.enable_debug_logging("okts.index")   # just the retrieval stage

or wire it into your own ``logging.basicConfig`` — the ``okts.*`` records flow
into whatever handlers you configure. Levels used: DEBUG = per-item tracing
(each concept, each hit + score), INFO = stage milestones (bundle built,
served), WARNING = graceful degradations (skipped/malformed input, missing
credential, dispatch fallback).
"""

import logging as _logging

from okts.core.model import (
    Bundle,
    Cost,
    Interface,
    OKTConcept,
    SideEffects,
)
from okts.core.serialize import concept_from_markdown, concept_to_markdown
from okts.core.validator import ConformanceError, validate_concept, validate_bundle

# Library best practice: emit records but configure no handlers, so importing
# OKTS never spams stderr or hijacks the root logger. Users opt in (below).
_logging.getLogger("okts").addHandler(_logging.NullHandler())


def enable_debug_logging(
    name: str = "okts",
    level: int = _logging.DEBUG,
    stream: "object | None" = None,
) -> _logging.Logger:
    """Convenience: route an ``okts`` logger to a stderr ``StreamHandler``.

    A one-liner for debugging without setting up ``logging`` yourself. ``name``
    scopes it (e.g. ``"okts.index"`` for just retrieval, ``"okts.serve"`` for
    just serving/dispatch); ``level`` and ``stream`` are the usual handler
    knobs. Idempotent — calling it again won't stack duplicate handlers.

    Returns the configured :class:`logging.Logger`.
    """
    logger = _logging.getLogger(name)
    logger.setLevel(level)
    tag = "_okts_debug_handler"
    if not any(getattr(h, tag, False) for h in logger.handlers):
        handler = _logging.StreamHandler(stream)  # type: ignore[arg-type]
        handler.setFormatter(_logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        setattr(handler, tag, True)
        logger.addHandler(handler)
    return logger


__all__ = [
    "Bundle",
    "Cost",
    "Interface",
    "OKTConcept",
    "SideEffects",
    "concept_from_markdown",
    "concept_to_markdown",
    "ConformanceError",
    "validate_concept",
    "validate_bundle",
    "enable_debug_logging",
]

__version__ = "0.1.0"
