"""HyeAero image rank & filter engine (deterministic, CLIP-free)."""

from services.aviation_image_rank_filter_engine import (
    rank_and_filter_aviation_images,
    searchapi_image_rank_filter_engine_enabled,
)


def test_rank_filter_rejects_house_hotel():
    qi = {"aircraft": "Challenger 350", "section": "cabin", "type": "interior"}
    images = [
        {
            "url": "https://cdn.example/cabin.jpg",
            "title": "Challenger 350 cabin interior",
            "source": "web",
            "alt": "",
        },
        {
            "url": "https://hotel.example/lobby.jpg",
            "title": "Grand hotel suite interior wood cabin style",
            "source": "x",
            "alt": "",
        },
    ]
    out = rank_and_filter_aviation_images(query_intent=qi, images=images, min_selected=1, max_selected=5)
    assert len(out["selected_images"]) >= 1
    assert all("hotel" not in (s.get("url") or "") for s in out["selected_images"])


def test_rank_filter_rejects_exterior_when_cabin_requested():
    qi = {"aircraft": "Falcon 900", "section": "cabin", "type": "interior"}
    images = [
        {
            "url": "https://jp.example/f900-ramp.jpg",
            "title": "Dassault Falcon 900 ramp planespotting takeoff",
            "source": "jp",
            "alt": "",
        },
        {
            "url": "https://cdn.example/f900-cabin.jpg",
            "title": "Falcon 900 cabin interior galley",
            "source": "web",
            "alt": "",
        },
    ]
    out = rank_and_filter_aviation_images(query_intent=qi, images=images, min_selected=1, max_selected=5)
    urls = [s["url"] for s in out["selected_images"]]
    assert any("cabin" in u for u in urls)
    assert not any("ramp" in u for u in urls)


def test_rank_filter_returns_empty_when_only_one_passes_precision():
    qi = {"aircraft": "Challenger 350", "section": "cabin", "type": "interior"}
    images = [
        {
            "url": "https://a.example/c.jpg",
            "title": "Challenger 350 cabin interior layout",
            "source": "s",
            "alt": "",
        },
    ]
    out = rank_and_filter_aviation_images(query_intent=qi, images=images, min_selected=2, max_selected=5)
    assert out["selected_images"] == []


def test_searchapi_rank_filter_env_default_off(monkeypatch):
    monkeypatch.delenv("SEARCHAPI_IMAGE_RANK_FILTER_ENGINE", raising=False)
    assert searchapi_image_rank_filter_engine_enabled() is False


def test_searchapi_rank_filter_env_on(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_RANK_FILTER_ENGINE", "1")
    assert searchapi_image_rank_filter_engine_enabled() is True
