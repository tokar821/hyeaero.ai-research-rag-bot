"""Phase 53 — alias expansion engine tests."""

import pytest

from services.catalog.alias_expansion_engine import (
    expand_shorthand_token,
    resolve_comparison_models_from_query,
)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("CJ4", "Citation CJ4"),
        ("cj4", "Citation CJ4"),
        ("Phenom", "Phenom 300"),
        ("phenom 300", "Phenom 300"),
        ("Falcon", "Falcon 2000"),
        ("falcon 2000", "Falcon 2000"),
        ("Longitude", "Citation Longitude"),
        ("Challenger", "Challenger 350"),
        ("Latitude", "Citation Latitude"),
    ],
)
def test_expand_shorthand(token, expected):
    canon, conf = expand_shorthand_token(token)
    assert canon == expected
    assert conf >= 80


def test_cj4_vs_phenom_comparison():
    models = resolve_comparison_models_from_query("CJ4 vs Phenom 300")
    assert len(models) >= 2
    assert "Citation CJ4" in models
    assert any("Phenom" in m for m in models)


def test_longitude_vs_falcon2000():
    models = resolve_comparison_models_from_query("Longitude vs Falcon 2000")
    assert len(models) >= 2
    assert "Citation Longitude" in models
    assert "Falcon 2000" in models
