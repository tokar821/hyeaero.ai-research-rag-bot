"""SearchAPI Bing Images helpers — query fan-out, strict tail filter, ranking."""

from __future__ import annotations

from unittest.mock import patch

from services.searchapi_aircraft_images import (
    MIN_TAIL_MATCH_SCORE,
    apply_intent_boost,
    build_aircraft_image_search_queries,
    compose_manufacturer_model_phrase,
    compute_tail_match_score,
    detect_query_image_intent,
    fetch_ranked_searchapi_aircraft_images,
    normalize_aircraft_name,
    resolve_queries_for_consultant_gallery,
    strip_domains,
    strict_tail_search_hit_confirmed,
)


def test_build_queries_tail_branch():
    qs = build_aircraft_image_search_queries(canonical_tail="N123AB", manufacturer=None, model=None)
    assert qs[0] == "N123AB aircraft"
    assert "jetphotos.com" in qs[2]
    assert "planespotters.net" in qs[3]


def test_build_queries_model_branch():
    qs = build_aircraft_image_search_queries(
        canonical_tail=None, manufacturer="Gulfstream", model="G650"
    )
    assert any("exterior" in q for q in qs)
    assert any("cabin" in q for q in qs)
    assert any("private jet" in q for q in qs)
    assert any("cockpit" in q for q in qs)
    assert any("interior" in q for q in qs)
    assert any("walkaround" in q for q in qs)


def test_compose_manufacturer_model_dedupes_falcon():
    assert compose_manufacturer_model_phrase("Dassault Falcon", "Falcon 2000") == "Dassault Falcon 2000"


def test_compose_returns_model_only_when_manufacturer_is_substring_of_model():
    assert compose_manufacturer_model_phrase("Challenger", "Challenger 350") == "Challenger 350"


def test_strict_tail_accepts_planespotters_net_url():
    ok = strict_tail_search_hit_confirmed(
        "N807JS",
        {
            "title": "N807JS parked",
            "url": "https://images.planespotters.net/n807js-1.jpg",
            "source": "Planespotters",
            "_source_page": "https://www.planespotters.net/photo/n807js",
        },
    )
    assert ok is True


def test_strict_tail_rejects_other_mark_in_title():
    ok = strict_tail_search_hit_confirmed(
        "N807JS",
        {
            "title": "N807JS at JetPhotos",
            "url": "https://cdn.jetphotos.com/photo/1.jpg",
            "source": "JetPhotos",
            "_source_page": "https://www.jetphotos.com/photo/123",
        },
    )
    assert ok is True

    bad = strict_tail_search_hit_confirmed(
        "N807JS",
        {
            "title": "N807JT at JetPhotos",
            "url": "https://cdn.jetphotos.com/photo/1.jpg",
            "source": "JetPhotos",
            "_source_page": "https://www.jetphotos.com/photo/123",
        },
    )
    assert bad is False


def test_fetch_ranked_merges_queries(monkeypatch):
    calls: list[str] = []

    def fake_search(q: str, **kwargs):
        calls.append(q)
        if "jetphotos" in q:
            return [
                {
                    "url": "https://cdn.jetphotos.com/a.jpg",
                    "title": "N999ZZ ramp",
                    "source": "jp",
                    "_source_page": "https://jetphotos.com/x",
                }
            ]
        return [
            {
                "url": "https://cdn.jetphotos.com/b.jpg",
                "title": "N111AA ramp",
                "source": "jp",
                "_source_page": "https://jetphotos.com/y",
            }
        ]

    with patch("services.searchapi_aircraft_images.search_aircraft_images", side_effect=fake_search):
        out, _ = fetch_ranked_searchapi_aircraft_images(
            queries=["N111AA aircraft", "N111AA site:jetphotos.com"],
            canonical_tail="N111AA",
            strict_tail_mode=True,
            marketing_type_for_model_match=None,
            max_out=5,
        )
    assert len(out) == 1
    assert "jetphotos" in out[0]["url"]
    assert calls  # two fan-out queries


def test_avbuyer_dropped_when_three_high_tier_strict_tail():
    """With 3+ strong non-AvBuyer hosts, AvBuyer rows are removed entirely."""

    def fake_search(_q: str, **kwargs):
        return [
            {
                "url": "https://cdn.jetphotos.com/n111aa-1.jpg",
                "title": "N111AA at JetPhotos",
                "source": "jp",
                "_source_page": "https://jetphotos.com/1",
            },
            {
                "url": "https://images.planespotters.net/n111aa.jpg",
                "title": "N111AA parked",
                "source": "ps",
                "_source_page": "https://planespotters.net/2",
            },
            {
                "url": "https://controller.com/cdn/n111aa.jpg",
                "title": "N111AA listing",
                "source": "ctl",
                "_source_page": "https://controller.com/listing/1",
            },
            {
                "url": "https://www.avbuyer.com/n111aa.jpg",
                "title": "N111AA on AvBuyer",
                "source": "ab",
                "_source_page": "https://avbuyer.com/x",
            },
        ]

    with patch("services.searchapi_aircraft_images.search_aircraft_images", side_effect=fake_search):
        out, _ = fetch_ranked_searchapi_aircraft_images(
            queries=["N111AA aircraft"],
            canonical_tail="N111AA",
            strict_tail_mode=True,
            marketing_type_for_model_match=None,
            max_out=10,
        )
    assert len(out) == 3
    assert all("avbuyer" not in (r.get("url") or "").lower() for r in out)


def test_falcon_900_rejected_when_marketing_is_falcon_2000():
    def fake_search(_q: str, **kwargs):
        return [
            {
                "url": "https://cdn.example.com/f900.jpg",
                "title": "Dassault Falcon 900 on approach",
                "source": "host",
                "_source_page": "https://example.com/f900",
            }
        ]

    with patch("services.searchapi_aircraft_images.search_aircraft_images", side_effect=fake_search):
        out, _ = fetch_ranked_searchapi_aircraft_images(
            queries=["Falcon 2000 cabin"],
            canonical_tail=None,
            strict_tail_mode=False,
            marketing_type_for_model_match="Falcon 2000",
            max_out=5,
            user_query="Falcon 2000 cabin",
        )
    assert out == []


def test_detect_challenger_350_exterior():
    from rag.consultant_query_expand import _detect_models

    assert "Challenger 350" in _detect_models("Challenger 350 exterior")


def test_visual_gallery_tail_includes_loose_us_n():
    from rag.aviation_tail import find_visual_gallery_tail_candidates, find_strict_tail_candidates_in_text

    assert find_strict_tail_candidates_in_text("show me N000ZZZ") == []
    assert find_visual_gallery_tail_candidates("show me N000ZZZ", None) == ["N000ZZZ"]


def test_normalize_cl350_alias():
    assert normalize_aircraft_name("Bombardier CL350 exterior") == "Challenger 350"


def test_resolve_queries_phly_row_normalizes_model_token():
    qs, _, mm = resolve_queries_for_consultant_gallery(
        user_query="show photos",
        phly_rows=[{"manufacturer": "Bombardier", "model": "CL350"}],
        required_tail=None,
        strict_tail_mode=False,
        required_marketing_type=None,
        strict_model_mode=False,
    )
    assert mm and "Challenger" in mm
    assert any("Challenger 350" in q for q in qs)


def test_tail_query_ranks_controller_above_jetphotos():
    def fake_search(_q: str, **kwargs):
        return [
            {
                "url": "https://cdn.jetphotos.com/n807js.jpg",
                "title": "N807JS ramp",
                "source": "jp",
                "_source_page": "https://jetphotos.com/1",
            },
            {
                "url": "https://www.controller.com/cdn/n807js.jpg",
                "title": "N807JS listing photo",
                "source": "ctl",
                "_source_page": "https://controller.com/listing/1",
            },
        ]

    with patch("services.searchapi_aircraft_images.search_aircraft_images", side_effect=fake_search):
        out, _ = fetch_ranked_searchapi_aircraft_images(
            queries=["N807JS aircraft"],
            canonical_tail="N807JS",
            strict_tail_mode=True,
            marketing_type_for_model_match=None,
            max_out=5,
            user_query="N807JS for sale",
        )
    assert out
    assert "controller.com" in (out[0].get("url") or "").lower()


def test_compute_tail_match_score_high_for_exact_title_low_for_other_tail():
    good = {"title": "N807JS parked", "url": "https://x.com/a.jpg", "_source_page": ""}
    assert compute_tail_match_score(good, "N807JS") >= 120
    bad = {"title": "N807JT JetPhotos", "url": "https://x.com/a.jpg", "_source_page": ""}
    assert compute_tail_match_score(bad, "N807JS") < 80


def test_compute_tail_match_score_cdn_only_title_empty_survives():
    row = {
        "title": "   ",
        "url": "https://cdn.jetphotos.com/photo/807js/full.jpg",
        "_source_page": "https://www.jetphotos.com/photo/123",
    }
    assert compute_tail_match_score(row, "N807JS") >= MIN_TAIL_MATCH_SCORE


def test_strip_domains_removes_host_tokens():
    s = "N807JS SPOTTED AT PLANESPOTTERS.NET IN EUROPE"
    out = strip_domains(s)
    assert "NET" not in out.split()  # standalone NET token from TLD
    assert "N807JS" in out


def test_detect_query_intent_prefers_cockpit_over_exterior():
    assert detect_query_image_intent("Global 7500 cockpit photos") == "cockpit"
    assert detect_query_image_intent("Global 7500 exterior walkaround") == "exterior"
    assert detect_query_image_intent("Falcon 2000 cabin layout") == "cabin"


def test_apply_intent_boost_cabin_penalizes_exterior_in_metadata():
    row = {
        "title": "Dassault Falcon 2000 exterior ramp",
        "url": "https://cdn.example.com/a.jpg",
        "_source_page": "",
    }
    base = 100.0
    boosted = apply_intent_boost(base, row, "cabin")
    assert boosted < base + 40.0


def test_gallery_meta_documents_empty_tail_search():
    meta: dict = {}

    def fake_empty(_q: str, **kwargs):
        return []

    with patch("services.searchapi_aircraft_images.search_aircraft_images", side_effect=fake_empty):
        out, _ = fetch_ranked_searchapi_aircraft_images(
            queries=["N000ZZZ aircraft"],
            canonical_tail="N000ZZZ",
            strict_tail_mode=True,
            marketing_type_for_model_match=None,
            max_out=5,
            gallery_meta=meta,
        )
    assert out == []
    assert meta.get("consultant_gallery_empty") is True
    assert isinstance(meta.get("consultant_gallery_suggestions"), list)
    assert len(meta.get("consultant_gallery_suggestions") or []) >= 1


def test_max_per_domain_limits_repetition():
    rows = []
    for i in range(5):
        rows.append(
            {
                "url": f"https://cdn.jetphotos.com/p{i}.jpg",
                "title": "N111AA ramp",
                "source": "jp",
                "_source_page": f"https://jetphotos.com/{i}",
            }
        )

    def fake_search(_q: str, **kwargs):
        return list(rows)

    with patch("services.searchapi_aircraft_images.search_aircraft_images", side_effect=fake_search):
        out, _ = fetch_ranked_searchapi_aircraft_images(
            queries=["N111AA aircraft"],
            canonical_tail="N111AA",
            strict_tail_mode=True,
            marketing_type_for_model_match=None,
            max_out=10,
        )
    assert len(out) == 2
    assert all("jetphotos" in (r.get("url") or "").lower() for r in out)


def test_prioritize_listings_orders_controller_before_avbuyer():
    from rag.consultant_market_lookup import prioritize_and_deduplicate_listing_rows

    rows = [
        {
            "source_platform": "avbuyer",
            "source_listing_id": "1",
            "listing_url": "https://avbuyer.com/a",
            "ask_price": 10.0,
            "updated_at": "2024-01-02",
        },
        {
            "source_platform": "controller",
            "source_listing_id": "2",
            "listing_url": "https://controller.com/b",
            "ask_price": 9.0,
            "updated_at": "2024-01-01",
        },
    ]
    out = prioritize_and_deduplicate_listing_rows(rows)
    assert out[0]["source_platform"] == "controller"
