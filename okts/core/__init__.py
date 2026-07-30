"""Core: the OKT concept/bundle data model, (de)serialization, validator, protocols.

This subpackage is the FROZEN CONTRACT. Adapters, index, serve, and eval all
import from here and must not redefine these types. Change with care.
"""

from okts.core.model import (
    Bundle,
    Cost,
    Interface,
    OKTConcept,
    SideEffects,
    REQUIRED_MINIMUM,
)
from okts.core.serialize import concept_from_markdown, concept_to_markdown
from okts.core.validator import ConformanceError, validate_concept, validate_bundle
from okts.core.protocols import Adapter, Dispatcher, Enricher, Retriever, SearchHit

__all__ = [
    "Bundle",
    "Cost",
    "Interface",
    "OKTConcept",
    "SideEffects",
    "REQUIRED_MINIMUM",
    "concept_from_markdown",
    "concept_to_markdown",
    "ConformanceError",
    "validate_concept",
    "validate_bundle",
    "Adapter",
    "Dispatcher",
    "Enricher",
    "Retriever",
    "SearchHit",
]
