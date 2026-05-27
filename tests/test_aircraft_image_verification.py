"""Strict aircraft image verification pipeline."""

from __future__ import annotations

from services.aircraft_image_verification import (
    VERIFIED_FAILURE_MESSAGE,
    classify_source_tier,
    verify_aircraft_image_rows,
    verify_gallery_images,
)
from services.aircraft_image_verification.rejection import (
    ImageVerificationContext,
    evaluate_rejection,
)
from services.aircraft_image_verification.source_ranking import SourceTier


def test_stock_imagery_rejected():
    row = {
        "url": "https://image.shutterstock.com/shutterstock/photos/123.jpg",
        "title": "Luxury private jet cabin interior",
        "source": "Shutterstock",
    }
    reason = evaluate_rejection(
        row,
        ImageVerificationContext(model="Gulfstream G650", section="cabin"),
    )
    assert reason == "stock_imagery_unverified"


def test_generic_residential_cabin_rejected():
    row = {
        "url": "https://example.com/cabin.jpg",
        "title": "Log cabin rental smoky mountains vacation",
        "source": "web",
    }
    reason = evaluate_rejection(
        row,
        ImageVerificationContext(model="Challenger 350", section="cabin"),
    )
    assert reason == "generic_or_residential_interior"


def test_wrong_variant_rejected():
    row = {
        "url": "https://jetphotos.com/photo.jpg",
        "title": "Falcon 8X on ramp",
        "source": "JetPhotos",
        "_source_page": "https://jetphotos.com/photo/123",
    }
    reason = evaluate_rejection(
        row,
        ImageVerificationContext(model="Falcon 2000", section="exterior"),
    )
    assert reason == "wrong_aircraft_variant"


def test_exact_model_passes_verification():
    rows = [
        {
            "url": "https://www.gulfstream.com/en/aircraft/g650",
            "title": "Gulfstream G650 exterior ramp",
            "source": "Gulfstream",
            "_source_page": "https://www.gulfstream.com/en/aircraft/g650",
        }
    ]
    result = verify_aircraft_image_rows(
        rows,
        model="Gulfstream G650",
        section="exterior",
        min_confidence=0.5,
    )
    assert not result.empty
    assert result.images[0]["_verification_match_type"] == "model_exact"


def test_low_confidence_returns_empty_message():
    rows = [
        {
            "url": "https://random-blog.example/img.jpg",
            "title": "aircraft",
            "source": "blog",
        }
    ]
    result = verify_aircraft_image_rows(
        rows,
        model="Gulfstream G650",
        section="exterior",
        min_confidence=0.7,
    )
    assert result.empty
    assert result.message == VERIFIED_FAILURE_MESSAGE


def test_tail_required_for_tail_mode():
    result = verify_aircraft_image_rows(
        [{"url": "https://jetphotos.com/x.jpg", "title": "N650GX Gulfstream G650", "source": "JetPhotos"}],
        tail="N650GX",
        section="exterior",
        min_confidence=0.55,
    )
    assert not result.empty or result.message == VERIFIED_FAILURE_MESSAGE


def test_manufacturer_outranks_stock_in_source_tier():
    tier_m, _ = classify_source_tier(
        url="https://www.gulfstream.com/assets/g650.jpg",
        page_url="https://www.gulfstream.com/en/aircraft/g650",
        title="Gulfstream G650",
    )
    tier_s, _ = classify_source_tier(
        url="https://shutterstock.com/image.jpg",
        title="jet interior",
    )
    assert tier_m == SourceTier.MANUFACTURER
    assert tier_s == SourceTier.STOCK_UNVERIFIED


def test_gallery_verification_strips_unverified():
    gallery = [
        {
            "url": "https://www.gulfstream.com/img.jpg",
            "description": "Gulfstream G650 cabin interior galley",
            "source": "searchapi",
            "page_url": "https://www.gulfstream.com/en/aircraft/g650",
        },
        {
            "url": "https://shutterstock.com/generic.jpg",
            "description": "luxury cabin stock photo",
            "source": "searchapi",
        },
    ]
    out, meta = verify_gallery_images(
        gallery,
        model="Gulfstream G650",
        section="cabin",
        min_confidence=0.5,
    )
    assert len(out) <= 1
    assert meta.get("aircraft_image_verification")


def test_missing_identity_preflight():
    result = verify_aircraft_image_rows(
        [{"url": "https://example.com/a.jpg", "title": "private jet"}],
        min_confidence=0.7,
    )
    assert result.empty
    assert result.message == VERIFIED_FAILURE_MESSAGE
