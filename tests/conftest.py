"""Shared pytest fixtures for all layers."""

from pathlib import Path

import pytest

from okts.core.bundle_io import load_bundle

FIXTURE_BUNDLE = Path(__file__).parent / "fixtures" / "bundle"


@pytest.fixture
def bundle_dir() -> Path:
    return FIXTURE_BUNDLE


@pytest.fixture
def bundle():
    """The loaded fixture bundle (11 concepts across github/slack/stripe)."""
    return load_bundle(FIXTURE_BUNDLE)
