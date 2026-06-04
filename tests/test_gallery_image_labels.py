"""Gallery UI label annotation."""

from services.broker_execution.gallery_image_labels import (
    annotate_consultant_gallery_images,
    resolve_gallery_row_label,
)


def test_listing_cabin_gets_exact_tail_label():
    row = {
        "url": "https://assets-global.website-files.com/x/cabin.jpg",
        "source": "listing_scrape",
        "description": "N807JS cabin interior",
        "page_url": "https://virtualhangar.com/aircraft/n807js",
        "image_provenance": "tail_marketing_listing",
    }
    labels = resolve_gallery_row_label(row, tail="N807JS")
    assert labels["gallery_label"] == "Listing cabin (exact tail)"
    assert labels["visual_facet"] == "cabin"


def test_representative_model_cabin_label():
    row = {
        "url": "https://cdn.example.com/cabin.jpg",
        "source": "searchapi",
        "image_provenance": "representative_model_cabin",
        "description": "Citation Excel cabin",
    }
    labels = resolve_gallery_row_label(
        row,
        tail="N807JS",
        gallery_meta={"consultant_cabin_image_tier": "representative_model"},
    )
    assert "Representative" in labels["gallery_label"]
    assert labels["image_provenance"] == "representative_model_cabin"


def test_annotate_preserves_urls():
    images = [
        {
            "url": "https://assets-global.website-files.com/a.jpg",
            "source": "listing_scrape",
            "image_provenance": "tail_marketing_listing",
        }
    ]
    out = annotate_consultant_gallery_images(
        images,
        tail="N807JS",
        gallery_meta={"consultant_tail_listing_cabin_enriched": True},
        user_query="show n807js cabin",
    )
    assert out[0]["gallery_label"]
    assert out[0]["url"].startswith("https://")
