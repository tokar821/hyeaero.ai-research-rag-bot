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


def test_rank_filter_rejects_interior_design_furniture():
    qi = {"aircraft": "Challenger 350", "section": "cabin", "type": "interior"}
    images = [
        {
            "url": "https://cdn.example/cabin.jpg",
            "title": "Challenger 350 cabin interior layout",
            "source": "web",
            "alt": "",
        },
        {
            "url": "https://houzz.example/x.jpg",
            "title": "Modern interior design furniture living room",
            "source": "x",
            "alt": "",
        },
    ]
    out = rank_and_filter_aviation_images(query_intent=qi, images=images, min_selected=1, max_selected=5)
    assert out["valid_images"] == out["selected_images"]
    assert len(out["valid_images"]) >= 1
    assert all("interior design" not in (v.get("reason") or "").lower() for v in out["valid_images"])
    assert all("houzz" not in (v.get("url") or "") for v in out["valid_images"])


def test_rank_filter_scores_are_0_to_100_ints():
    qi = {"aircraft": "Challenger 350", "section": "cabin", "type": "interior"}
    images = [
        {
            "url": "https://jetphotos.example/1.jpg",
            "title": "Challenger 350 cabin interior jetphotos",
            "source": "jetphotos",
            "alt": "",
        },
    ]
    out = rank_and_filter_aviation_images(query_intent=qi, images=images, min_selected=1, max_selected=5)
    assert out["valid_images"]
    assert isinstance(out["valid_images"][0]["score"], int)
    assert 0 <= out["valid_images"][0]["score"] <= 100


def test_rank_filter_rejects_target_racing_sim_under_cockpit_intent():
    qi = {"aircraft": "Global 7500", "section": "cockpit", "type": "cockpit"}
    images = [
        {
            "url": "https://target.scene7.com/is/image/Target/guest-sim.jpg",
            "title": "Racing sim cockpit Thrustmaster Logitech",
            "source": "Target",
            "page_url": "https://www.target.com/p/dardoo-racing-sim-cockpit/-/A-1",
            "alt": "",
        },
        {
            "url": "https://cdn.jetphotos.com/g7500-deck.jpg",
            "title": "Global 7500 flight deck cockpit",
            "source": "JetPhotos",
            "page_url": "https://www.jetphotos.com/photo/1",
            "alt": "",
        },
    ]
    out = rank_and_filter_aviation_images(query_intent=qi, images=images, min_selected=1, max_selected=5)
    assert out["selected_images"]
    assert all("target.com" not in (s.get("url") or "").lower() for s in out["selected_images"])
