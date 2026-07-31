"""Logging behavior across the four stages.

Two guarantees are covered:

1. **Silent by default** — importing/using OKTS must never configure the root
   logger or spam stderr. The package attaches only a ``NullHandler`` (library
   best practice); nothing is emitted unless the user opts in.
2. **Verbose when asked** — turning logging on (via ``caplog`` here, or
   ``okts.enable_debug_logging`` in real use) surfaces records at the right
   levels from every stage: adapters (layer 1), enrich (1½), retrieval (3),
   and serving/dispatch (4).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import okts
from okts.build import build_bundle_from_config, build_service
from okts.config.loader import config_from_dict
from okts.core.model import Bundle, Interface, OKTConcept
from okts.serve.dispatch import DispatcherRegistry, MockDispatcher
from okts.serve.service import OKTSService
from okts.serve.mcp_server import NaiveFallbackRetriever

# A tiny all-source config so one build exercises every adapter's logging.
_CONFIG = {
    "sources": [
        {
            "interface": "mcp",
            "servers": {
                "github-mcp": {
                    "tools": [
                        {
                            "name": "create_issue",
                            "description": "Open a new issue in a GitHub repository.",
                            "inputSchema": {
                                "type": "object",
                                "required": ["repo"],
                                "properties": {"repo": {"type": "string"}},
                            },
                        }
                    ]
                }
            },
        },
        {
            "interface": "function",
            "schemas": [
                {
                    "name": "math.add",
                    "description": "Add two numbers.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    ]
}


def test_package_logger_is_silent_by_default():
    """The ``okts`` logger carries a NullHandler and no other configuration,
    so nothing is emitted and the root logger is left untouched."""
    logger = logging.getLogger("okts")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    # We never set a level or propagate=False on the package logger.
    assert logger.level == logging.NOTSET
    assert logger.propagate is True


def test_enable_debug_logging_is_idempotent():
    name = "okts.test_idempotent"
    okts.enable_debug_logging(name)
    okts.enable_debug_logging(name)
    logger = logging.getLogger(name)
    tagged = [h for h in logger.handlers if getattr(h, "_okts_debug_handler", False)]
    assert len(tagged) == 1
    assert logger.level == logging.DEBUG
    # cleanup so we don't leak a stderr handler into other tests
    for h in tagged:
        logger.removeHandler(h)


def test_build_pipeline_emits_stage_logs(caplog):
    """Adapters + enrich + build all log at INFO when logging is enabled."""
    with caplog.at_level(logging.INFO, logger="okts"):
        config = config_from_dict(_CONFIG)
        build_bundle_from_config(config)

    messages = [r.getMessage() for r in caplog.records]
    text = "\n".join(messages)
    # adapters (layer 1)
    assert "mcp adapter" in text
    assert "function adapter" in text
    # enrich (layer 1½)
    assert "enriched" in text
    # build milestones
    assert "assembling bundle" in text
    assert "passed OKF conformance" in text
    # every record is under the okts.* hierarchy
    assert all(r.name.startswith("okts") for r in caplog.records)


def test_retrieval_emits_debug_trace(caplog):
    """The retriever logs the query, ranked hits, and (when present) graph
    expansion at DEBUG — the highest-value debugging surface."""
    config = config_from_dict(_CONFIG)
    bundle = build_bundle_from_config(config)
    service = build_service(bundle=bundle)

    with caplog.at_level(logging.DEBUG, logger="okts.index"):
        service.search_tools("create a new issue", k=3)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "search q=" in text
    assert "ranked" in text


def test_serve_and_dispatch_emit_logs(caplog):
    concept = OKTConcept(
        id="demo.ping",
        title="Ping",
        description="Ping the demo backend.",
        input_schema={"type": "object", "properties": {}},
        interface=Interface.FUNCTION,
        target="demo",
    )
    bundle = Bundle()
    bundle.add(concept)
    registry = DispatcherRegistry(dispatchers={Interface.FUNCTION: MockDispatcher()})
    service = OKTSService(bundle, NaiveFallbackRetriever(), registry)

    with caplog.at_level(logging.DEBUG, logger="okts.serve"):
        service.call_tool("demo.ping", {})

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "phase 3 call_tool" in text
    assert "registry routing" in text


def test_missing_dispatcher_backend_warns(caplog):
    """A graceful degradation (no backend wired) logs at WARNING, not silence."""
    concept = OKTConcept(
        id="demo.ping",
        title="Ping",
        description="Ping the demo backend.",
        input_schema={"type": "object", "properties": {}},
        interface=Interface.SEARCH,
        target="demo",
    )
    bundle = Bundle()
    bundle.add(concept)
    # registry has nothing registered for `search`
    service = OKTSService(bundle, NaiveFallbackRetriever(), DispatcherRegistry())

    with caplog.at_level(logging.WARNING, logger="okts.serve"):
        with pytest.raises(Exception):
            service.call_tool("demo.ping", {})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("no dispatcher" in r.getMessage() for r in warnings)
