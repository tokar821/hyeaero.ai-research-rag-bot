from services.aviation_image_relevance_filter import (
    evaluate_aviation_image_relevance,
    filter_aviation_images_by_relevance,
)


def test_accept_oem_cabin_interior():
    r = evaluate_aviation_image_relevance(
        {
            "title": "Gulfstream G650 cabin interior seating layout",
            "url": "https://cdn.example.com/g650.jpg",
        }
    )
    assert r["accepted"] is True
    assert r["confidence"] >= 0.68
    assert "oem" in r["reason"] or "cues" in r["reason"] or "registration" in r["reason"]


def test_reject_hotel():
    r = evaluate_aviation_image_relevance(
        {
            "title": "Luxury hotel suite interior pool view",
            "url": "https://hotel.example.com/img.jpg",
        }
    )
    assert r["accepted"] is False
    assert "hotel" in r["reason"] or "pool" in r["reason"] or "house" in r["reason"]


def test_reject_reddit():
    r = evaluate_aviation_image_relevance(
        {"title": "funny meme", "url": "https://i.redd.it/abc123.jpg"},
    )
    assert r["accepted"] is False


def test_reject_uncertain_generic():
    r = evaluate_aviation_image_relevance(
        {"title": "nice room", "url": "https://images.example.com/a.jpg"},
    )
    assert r["accepted"] is False
    assert r["confidence"] < 0.68


def test_tail_plus_jetphotos_accepts():
    r = evaluate_aviation_image_relevance(
        {
            "title": "N650GA at LAX",
            "url": "https://www.jetphotos.net/photo/12345",
            "description": "departure",
        }
    )
    assert r["accepted"] is True


def test_filter_list():
    rows = [
        {"url": "https://x.com/1.jpg", "title": "Gulfstream G550 aircraft cabin interior"},
        {"url": "https://x.com/2.jpg", "title": "Zillow home listing interior"},
    ]
    kept = filter_aviation_images_by_relevance(rows)
    assert len(kept) == 1
    assert "relevance_filter" in kept[0]
