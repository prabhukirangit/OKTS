"""OKTS — Open Knowledge Tool Search.

Give your agent 300 tools; it only ever sees 3.

Public surface (stable, forever): ``search_tools``, ``load_tool``, ``call_tool``.
This top-level package re-exports the core data model so downstream layers
(adapters, index, serve, eval) import from one place.
"""

from okts.core.model import (
    Bundle,
    Cost,
    Interface,
    OKTConcept,
    SideEffects,
)
from okts.core.serialize import concept_from_markdown, concept_to_markdown
from okts.core.validator import ConformanceError, validate_concept, validate_bundle

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
]

__version__ = "0.1.0"
