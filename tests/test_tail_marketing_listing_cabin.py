"""Tail cabin images from broker marketing pages (Virtual Hangar, etc.)."""

from unittest.mock import patch

from services.aircraft_image_verification.rejection import (
    ImageVerificationContext,
    evaluate_rejection,
)
from services.broker_execution.gallery_visual_intent import (
    filter_gallery_by_visual_intent,
    row_matches_visual_facet,
)
from services.tail_marketing_listing_images import (
    append_tail_marketing_cabin_queries,
    canonical_virtualhangar_tail_url,
    discover_tail_marketing_listing_urls,
    row_is_tail_listing_cabin_candidate,
)


def test_virtualhangar_tail_url():
    assert canonical_virtualhangar_tail_url("n807js") == "https://virtualhangar.com/aircraft/n807js"


def test_discover_includes_virtualhangar():
    urls = discover_tail_marketing_listing_urls("N807JS", [])
    assert any("virtualhangar.com/aircraft/n807js" in u for u in urls)


def test_append_marketing_cabin_queries():
    qs = append_tail_marketing_cabin_queries(["N807JS cabin"], "N807JS")
    assert any("virtualhangar" in q.lower() for q in qs)


def test_listing_cdn_row_matches_cabin_facet():
    row = {
        "url": "https://assets-global.website-files.com/5ebd/636a91330e1bd22dcba85aed_2.jpg",
        "title": "",
        "page_url": "https://virtualhangar.com/aircraft/n807js",
    }
    assert row_is_tail_listing_cabin_candidate(row, "N807JS")
    assert row_matches_visual_facet(row, "cabin")


def test_verification_accepts_listing_page_cabin_without_cdn_keywords():
    row = {
        "url": "https://assets-global.website-files.com/5ebd/636a91330e1bd22dcba85aed_2.jpg",
        "description": "N807JS Citation Excel cabin interior",
        "page_url": "https://virtualhangar.com/aircraft/n807js",
        "source": "listing_og",
    }
    ctx = ImageVerificationContext(tail="N807JS", section="cabin")
    assert evaluate_rejection(row, ctx) is None


def test_filter_keeps_virtualhangar_cabin():
    images = [
        {
            "url": "https://cdn.jetphotos.com/a.jpg",
            "title": "N807JS ramp",
            "page_url": "https://jetphotos.com/1",
        },
        {
            "url": "https://assets-global.website-files.com/x.jpg",
            "title": "N807JS interior",
            "page_url": "https://virtualhangar.com/aircraft/n807js",
        },
    ]
    out = filter_gallery_by_visual_intent(images, facet="cabin", max_out=5)
    assert len(out) == 1
    assert "virtualhangar" in (out[0].get("page_url") or "")


def test_enrich_gallery_fetches_virtualhangar_cabin_from_html():
    from services.tail_marketing_listing_images import enrich_gallery_from_tail_marketing_listings

    cabin_url = (
        "https://assets-global.website-files.com/5ebd6216f527da304f85bcca/"
        "636a91330e1bd22dcba85aed_2.jpg"
    )
    with patch(
        "services.tail_marketing_listing_images.fetch_marketing_listing_page_images",
        return_value=[cabin_url],
    ):
        out = enrich_gallery_from_tail_marketing_listings(
            tail="N807JS",
            phly_rows=[],
            max_out=3,
            facet="cabin",
        )
    assert len(out) >= 1
    assert "website-files.com" in (out[0].get("url") or "")
    assert "virtualhangar" in (out[0].get("page_url") or "")


def test_build_consultant_strict_tail_cabin_returns_images(monkeypatch):
    """Regression: undefined ``_pi_for_fetch`` must not swallow the SearchAPI gallery path."""
    from dotenv import load_dotenv

    load_dotenv()
    from services.consultant_aircraft_images import build_consultant_aircraft_images

    phly = [
        {
            "registration_number": "N807JS",
            "manufacturer": "Cessna",
            "model": "Citation Excel",
        }
    ]
    meta: dict = {}
    with patch(
        "services.tail_marketing_listing_images.fetch_marketing_listing_page_images",
        return_value=[
            "https://assets-global.website-files.com/5ebd6216f527da304f85bcca/"
            "636a91330e1bd22dcba85aed_2.jpg"
        ],
    ):
        imgs = build_consultant_aircraft_images(
            {},
            phly,
            required_tail="N807JS",
            strict_tail_page_match=True,
            user_query="show me n807js cabin",
            gallery_meta_out=meta,
        )
    assert len(imgs) >= 1
    assert meta.get("consultant_tail_listing_cabin_enriched") or any(
        "website-files" in str(r.get("url") or "") for r in imgs
    )


def test_listing_row_rejects_vh_logo_assets():
    row = {
        "url": "https://virtualhangar.com/wp-content/uploads/2024/09/UberJets_VirtualHangar_Logos_White.png",
        "page_url": "https://virtualhangar.com/aircraft/n807js",
    }
    assert not row_is_tail_listing_cabin_candidate(row, "N807JS")


def test_junk_logos_rejected_from_vh_html():
    from services.tail_marketing_listing_images import fetch_marketing_listing_page_images

    html = """
    <img src="https://assets-global.website-files.com/5ebd/636a91330e1bd22dcba85aed_2.jpg">
    <img src="https://virtualhangar.com/wp-content/uploads/2024/09/UberJets_VirtualHangar_Logos_White.png">
    <img src="https://virtualhangar.io/wp-content/uploads/2023/10/lear45.png">
    """
    with patch(
        "services.consultant_aircraft_images.fetch_listing_page_html",
        return_value=html,
    ):
        imgs = fetch_marketing_listing_page_images(
            "https://virtualhangar.com/aircraft/n807js",
            tail="N807JS",
            want_cabin=True,
            max_images=6,
        )
    assert len(imgs) == 1
    assert "website-files.com" in imgs[0]


def test_exterior_og_not_used_when_html_has_cabin():
    from services.tail_marketing_listing_images import fetch_marketing_listing_page_images

    html = (
        '<img src="https://assets-global.website-files.com/x/cabin_2.jpg">'
        '<meta property="og:image" content="https://s3.example.com/exteriortailimage/N807JS-exterior.png">'
    )
    with patch(
        "services.consultant_aircraft_images.fetch_listing_page_html",
        return_value=html,
    ):
        imgs = fetch_marketing_listing_page_images(
            "https://virtualhangar.com/aircraft/n807js",
            tail="N807JS",
            want_cabin=True,
        )
    assert imgs
    assert "website-files" in imgs[0]
    assert "exterior" not in imgs[0].lower()
