"""Tests for Aircraft Query Builder (isolated query only)."""

from services.aircraft_query_builder import build_aircraft_image_search_seed


def test_builder_rebuilds_from_scratch_no_history_noise():
    q = "Challenger 300 cabin interior nice best under 12M previous"
    out = build_aircraft_image_search_seed(isolated_query=q, resolved_entity="Challenger 300")
    assert "previous" not in out.lower()
    assert "under" not in out.lower()
    assert "12m" not in out.lower()
    assert "challenger" in out.lower()
    assert "cabin" in out.lower() or "interior" in out.lower()


def test_builder_good_example_matches_shape():
    out = build_aircraft_image_search_seed(
        isolated_query="interior", resolved_entity="Challenger 300"
    )
    assert out.startswith("Challenger 300")
    assert "high" in out.lower()
    assert "resolution" in out.lower()
    assert len(out.split()) <= 6


def test_builder_deictic_only_uses_entity_and_default_facet():
    out = build_aircraft_image_search_seed(isolated_query="show me", resolved_entity="N807JS")
    assert out.startswith("N807JS")
    assert "aircraft" in out.lower() or "photo" in out.lower()


def test_builder_no_entity_returns_cleaned_query_or_empty():
    # New concept, no aircraft: builder should not invent one.
    out = build_aircraft_image_search_seed(isolated_query="best cabin under $12M", resolved_entity=None)
    assert "challenger" not in out.lower()


def test_builder_tail_photo_browse_does_not_force_cabin():
    out = build_aircraft_image_search_seed(
        isolated_query="N604WM aircraft photos",
        resolved_entity="N604WM",
    )
    low = out.lower()
    assert "n604wm" in low
    assert "cabin" not in low
    assert "photo" in low or "aircraft" in low


def test_builder_scrubs_unrelated_past_aircraft_mentions():
    # Simulate pasted context with another aircraft; builder must remove it completely.
    out = build_aircraft_image_search_seed(
        isolated_query="show interior like G650 but use Challenger 300 cabin",
        resolved_entity="Challenger 300",
    )
    low = out.lower()
    assert "challenger" in low
    assert "300" in low
    assert "g650" not in low

