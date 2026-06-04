"""Cabin visual intent must drop jetphotos exterior shots without interior cues."""

from services.broker_execution.gallery_visual_intent import (
    filter_gallery_by_visual_intent,
    row_matches_visual_facet,
)


def test_exterior_jetphotos_not_cabin():
    row = {
        "url": "https://cdn.jetphotos.com/400/6/797261_1708513782.jpg",
        "title": "N807JS Cessna Citation Excel",
        "source": "jetphotos",
    }
    assert not row_matches_visual_facet(row, "cabin")


def test_cabin_interior_row_matches():
    row = {
        "url": "https://example.com/x.jpg",
        "title": "Citation Excel cabin interior seating",
    }
    assert row_matches_visual_facet(row, "cabin")


def test_filter_drops_exterior_keeps_cabin():
    images = [
        {"url": "https://cdn.jetphotos.com/a.jpg", "title": "N807JS ramp taxi"},
        {"url": "https://example.com/b.jpg", "title": "bizjet cabin interior galley"},
    ]
    out = filter_gallery_by_visual_intent(images, facet="cabin", max_out=5)
    assert len(out) == 1
    assert "cabin" in (out[0].get("title") or "").lower()
