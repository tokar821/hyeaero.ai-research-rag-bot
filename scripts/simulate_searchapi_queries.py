"""
Simulate five user gallery queries with mocked SearchAPI Bing image hits.

Run:  python scripts/simulate_searchapi_queries.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import patch

# Ensure SearchAPI path is "on" for this simulation only.
os.environ["SEARCHAPI_API_KEY"] = "simulation-key"

from rag.aviation_tail import find_visual_gallery_tail_candidates
from services.searchapi_aircraft_images import (
    compose_manufacturer_model_phrase,
    fetch_ranked_searchapi_aircraft_images,
    resolve_queries_for_consultant_gallery,
)


def _mock_search_aircraft_images(query: str, **kwargs: Any) -> List[Dict[str, str]]:
    """Deterministic fake Bing-image rows keyed by substrings of ``q``."""
    q = (query or "").lower()

    def row(
        url: str,
        title: str,
        source: str,
        page: str = "",
    ) -> Dict[str, str]:
        return {
            "url": url,
            "title": title,
            "source": source,
            "_source_page": page,
        }

    if "n807js" in q:
        return [
            row(
                "https://cdn.jetphotos.com/photo/807js-jetphotos.jpg",
                "N807JS — Citation Excel on ramp",
                "JetPhotos",
                "https://www.jetphotos.com/photo/807js",
            ),
            row(
                "https://images.planespotters.net/n807js-1.jpg",
                "N807JS parked",
                "Planespotters",
                "https://www.planespotters.net/photo/n807js",
            ),
            row(
                "https://www.avbuyer.com/media/wrong-n807jt.jpg",
                "N807JT similar serial (wrong tail in title)",
                "AvBuyer",
                "https://www.avbuyer.com/listing/x",
            ),
            row(
                "https://globalair.com/gallery/n807js.jpg",
                "N807JS exterior",
                "GlobalAir",
                "https://www.globalair.com/foo",
            ),
        ]

    if "n628ts" in q:
        return [
            row(
                "https://cdn.jetphotos.com/photo/n628ts-g650.jpg",
                "N628TS Gulfstream G650 departure",
                "JetPhotos",
                "https://www.jetphotos.com/photo/n628ts",
            ),
            row(
                "https://controller.com/cdn/n628ts-hero.jpg",
                "N628TS — listing hero",
                "Controller",
                "https://www.controller.com/listing/for-sale/999/",
            ),
        ]

    if "falcon" in q and "2000" in q:
        return [
            row(
                "https://cdn.jetphotos.com/falcon-2000-cabin.jpg",
                "Dassault Falcon 2000 cabin interior",
                "JetPhotos",
                "https://www.jetphotos.com/photo/f2k-cabin",
            ),
            row(
                "https://www.avbuyer.com/falcon900-cabin.jpg",
                "Falcon 900 cabin (wrong family — confusable)",
                "AvBuyer",
                "https://www.avbuyer.com/articles/f900",
            ),
            row(
                "https://www.avbuyer.com/f2k-dealer.jpg",
                "Dassault Falcon 2000 interior — AvBuyer",
                "AvBuyer",
                "https://www.avbuyer.com/listing/f2k-dealer",
            ),
            row(
                "https://images.planespotters.net/f2k-ext.jpg",
                "Falcon 2000 exterior line-up",
                "Planespotters",
                "https://www.planespotters.net/photo/f2k",
            ),
            row(
                "https://controller.com/cdn/f2k-walk.jpg",
                "Falcon 2000 walkaround still",
                "Controller",
                "https://www.controller.com/listing/f2k",
            ),
        ]

    if "challenger" in q:
        return [
            row(
                "https://cdn.jetphotos.com/cl350-ext.jpg",
                "Bombardier Challenger 350 on approach",
                "JetPhotos",
                "https://www.jetphotos.com/photo/cl350",
            ),
            row(
                "https://www.aircraftexchange.com/img/cl350.jpg",
                "Challenger 350 — dealer photo",
                "AircraftExchange",
                "https://www.aircraftexchange.com/foo",
            ),
            row(
                "https://random-cdn.example/challenger605.jpg",
                "Challenger 605 (different model)",
                "ExampleHost",
                "https://example.com/news/ch605",
            ),
        ]

    if "n000zzz" in q:
        # Simulate Bing returning nothing useful for a non-standard mark string.
        return []

    return []


def run_case(label: str, user_query: str) -> None:
    # Match consultant_retrieval: gallery tails include loose ``N``+digit marks (e.g. N000ZZZ).
    tails = find_visual_gallery_tail_candidates(user_query, None)
    required_tail = tails[0] if tails else None
    strict_tail = bool(tails)
    inferred_mm: str | None = None
    strict_model = False
    if not strict_tail:
        # Mirror consultant_retrieval: inferred marketing type when no Phly row.
        try:
            from rag.consultant_query_expand import _detect_manufacturers, _detect_models

            blob = (user_query or "").strip()
            mans = _detect_manufacturers(blob.lower())
            mdls = _detect_models(blob)
            inferred_mm = (
                compose_manufacturer_model_phrase(
                    mans[0] if mans else "",
                    mdls[0] if mdls else "",
                ).strip()
                or (mdls[0] if mdls else None)
            )
            strict_model = bool(inferred_mm and ("cabin" in blob.lower() or "exterior" in blob.lower()))
        except Exception:
            inferred_mm = None

    queries, canon_tail, mm_for_score = resolve_queries_for_consultant_gallery(
        user_query=user_query,
        phly_rows=[],
        required_tail=required_tail,
        strict_tail_mode=strict_tail,
        required_marketing_type=inferred_mm if strict_model else None,
        strict_model_mode=strict_model,
    )

    marketing = None
    if strict_tail:
        marketing = None
    elif strict_model and inferred_mm:
        marketing = inferred_mm
    else:
        marketing = mm_for_score

    with patch("services.searchapi_aircraft_images.search_aircraft_images", _mock_search_aircraft_images):
        out, _gallery_meta = fetch_ranked_searchapi_aircraft_images(
            queries=queries,
            canonical_tail=canon_tail if strict_tail else None,
            strict_tail_mode=strict_tail,
            marketing_type_for_model_match=marketing,
            max_out=5,
            user_query=user_query,
        )

    print("=" * 72)
    print(f"QUERY {label}: {user_query!r}")
    print(f"  Strict tail mode: {strict_tail}  required_tail={required_tail!r}")
    print(f"  Strict model mode: {strict_model}  marketing_type={marketing!r}")
    print(f"  Fan-out queries ({len(queries)}):")
    for i, qq in enumerate(queries, 1):
        print(f"    {i}. {qq}")
    print(f"  Final gallery ({len(out)} images):")
    if not out:
        print("    (empty)")
        if strict_tail:
            print(
                "    -> With strict tail filtering, no image had the exact tail in title/URL/page "
                "with zero conflicting registrations (or mock returned no rows)."
            )
        else:
            print("    -> Mock returned no hits for this resolved query string.")
    for i, im in enumerate(out, 1):
        print(f"    {i}. url: {im.get('url')}")
        print(f"       description: {im.get('description')!r}")
        print(f"       page_url: {im.get('page_url')!r}")
        print(f"       source: {im.get('source')!r}")
    print()


def main() -> None:
    run_case("1", "show me N807JS")
    run_case("2", "show me Falcon 2000 cabin")
    run_case("3", "show me N000ZZZ")
    run_case("4", "Challenger 350 exterior")
    run_case("5", "N628TS aircraft")


if __name__ == "__main__":
    main()
