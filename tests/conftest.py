"""Shared pytest fixtures for all layers."""

from pathlib import Path

import pytest

from okts.core.bundle_io import load_bundle

FIXTURE_BUNDLE = Path(__file__).parent / "fixtures" / "bundle"


# ---------------------------------------------------------------------------
# `live` marker: opt-in tests that hit a real network/subprocess path (e.g. a
# live stdio MCP server). Skipped by default so CI stays network-/key-free;
# run locally with `pytest --run-live`.
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked `live` (real subprocess/network paths).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live test — pass --run-live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def bundle_dir() -> Path:
    return FIXTURE_BUNDLE


@pytest.fixture
def bundle():
    """The loaded fixture bundle (11 concepts across github/slack/stripe)."""
    return load_bundle(FIXTURE_BUNDLE)
