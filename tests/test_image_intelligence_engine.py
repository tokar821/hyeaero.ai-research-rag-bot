"""Aviation Image Intelligence Engine — structured output + validation."""

from __future__ import annotations

from unittest.mock import patch

from services.image_intelligence_engine import (
    classify_visual_category,
    resolve_aircraft_identity,
    run_aircraft_image_intelligence,
    run_image_intelligence,
)


def test_classify_visual_category_cockpit():
    assert classify_visual_category("n123ab cockpit view flight deck") == "cockpit"


def test_resolve_without_db_not_authoritative():
    ac, auth, reason = resolve_aircraft_identity(tail="N628TS", db=None)
    assert ac == "" and auth is False


def test_run_image_intelligence_invalid_query_empty():
    out = run_image_intelligence("Falcon 9000 cockpit photos", db=None)
    assert out["images"] == []


@patch("services.image_intelligence_engine._optional_db", return_value=None)
@patch("services.image_intelligence_engine.fetch_ranked_searchapi_aircraft_images")
@patch("services.image_intelligence_engine.searchapi_aircraft_images_enabled", return_value=True)
def test_run_image_intelligence_filters_stock_and_generic(_en, fetch_mock, _db):
    fetch_mock.return_value = (
        [
            {
                "url": "https://cdn.jetphotos.com/photo/1.jpg",
                "description": "N807JS cabin layout",
                "source": "JetPhotos",
                "page_url": "https://www.jetphotos.com/photo/1",
            },
            {
                "url": "https://www.shutterstock.com/x.jpg",
                "description": "luxury jet interior stock",
                "source": "web",
                "page_url": "",
            },
            {
                "url": "https://random-cdn.example.com/a.jpg",
                "description": "business jet",
                "source": "unknown",
                "page_url": "",
            },
        ],
        {},
    )
    out = run_image_intelligence("N807JS cabin", db=None)
    assert len(out["images"]) == 1
    assert out["images"][0]["url"].startswith("https://cdn.jetphotos.com")
    assert out["images"][0]["match_type"] == "tail_match"
    assert out["tail_number"] == "N807JS"


@patch("services.image_intelligence_engine._optional_db", return_value=None)
@patch("services.image_intelligence_engine.fetch_ranked_searchapi_aircraft_images")
@patch("services.image_intelligence_engine.searchapi_aircraft_images_enabled", return_value=True)
def test_run_aircraft_image_intelligence_schema(_en, fetch_mock, _db):
    fetch_mock.return_value = (
        [
            {
                "url": "https://cdn.jetphotos.com/photo/1.jpg",
                "description": "N807JS cabin layout club seating",
                "source": "JetPhotos",
                "page_url": "https://www.jetphotos.com/photo/1",
            },
        ],
        {},
    )
    out = run_aircraft_image_intelligence("N807JS cabin photos", db=None)
    assert set(out.keys()) == {"aircraft", "image_type", "images", "insight"}
    assert out["image_type"] == "cabin"
    assert len(out["images"]) == 1
    img = out["images"][0]
    assert set(img.keys()) == {"url", "confidence", "source", "tags"}
    assert img["url"].startswith("https://cdn.jetphotos.com")
    assert 0.0 <= img["confidence"] <= 1.0
    assert "cabin" in img["tags"]
    assert isinstance(out["insight"], str) and len(out["insight"]) > 20
