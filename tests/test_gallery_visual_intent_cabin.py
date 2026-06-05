"""Cabin facet discipline — exterior URLs rejected; interior preferred over LOPA."""

from services.broker_execution.gallery_visual_intent import (
    filter_gallery_by_visual_intent,
    row_matches_visual_facet,
)


def test_angled_exterior_rejected_for_cabin():
    row = {
        "url": "https://www.globalaircharters.com/wp-content/uploads/Gulfstream-GV-N5616-angled-side-view.png",
        "description": "Gulfstream GV N5616",
        "page_url": "https://www.globalaircharters.com/fleet/gulfstream-gv-n5616/",
    }
    assert not row_matches_visual_facet(row, "cabin")


def test_aviapages_interior_preferred_over_lopa():
    interior = {
        "url": "https://md.aviapages.com/media/thmb/2022/09/09/q90/g640x384/crcenter/upscale/GV-sn-616-Spec_7.jpg",
        "description": "N5616 Gulfstream V interior",
        "page_url": "https://aviapages.com/aircraft/n5616/",
    }
    lopa = {
        "url": "https://md.aviapages.com/media/2025/05/08/2303-GAC-LOPA-GV-N5616.webp",
        "description": "N5616 LOPA",
        "page_url": "https://aviapages.com/aircraft/n5616/",
    }
    out = filter_gallery_by_visual_intent([lopa, interior], facet="cabin", max_out=2)
    assert len(out) == 2
    assert "Spec_7" in (out[0].get("url") or "")
