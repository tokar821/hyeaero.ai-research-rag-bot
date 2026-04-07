"""Hybrid consultant image intent (keywords + LLM + thread follow-ups)."""

from unittest.mock import patch

from rag.conversation_guard import ConversationMessageType, evaluate_conversation_guard
from rag.consultant_image_intent import (
    broad_keyword_suggests_image_request,
    resolve_hybrid_image_gallery_intent,
    thread_has_aircraft_context,
    visual_followup_suggests_image_request,
)
from rag.consultant_market_lookup import (
    build_aircraft_photo_focus_tavily_query,
    wants_consultant_aircraft_images_in_answer,
)
from services.consultant_aircraft_images import (
    build_consultant_aircraft_images,
    filter_tavily_images_for_phly,
)
from services.consultant_report_image_proxy import consultant_report_image_url_allowed


def test_broad_keyword_image_phrases():
    assert broad_keyword_suggests_image_request("Any photos of the interior?") is True
    assert broad_keyword_suggests_image_request("show me a Falcon 2000") is True
    assert broad_keyword_suggests_image_request("can you show me the G650?") is True
    assert broad_keyword_suggests_image_request("What does it look like?") is True
    assert broad_keyword_suggests_image_request("Can I see it?") is False
    assert broad_keyword_suggests_image_request("Compare range and price") is False


def test_broad_keyword_google_images_and_typo_imags():
    assert broad_keyword_suggests_image_request('search pls on google "N508JA aircraft imags"') is True
    assert broad_keyword_suggests_image_request("can you google images for N508JA") is True
    assert broad_keyword_suggests_image_request("What is Google Flights?") is False


def test_thread_has_aircraft_context_models():
    hist = [{"role": "user", "content": "Tell me about the Falcon 2000"}]
    assert thread_has_aircraft_context("can I see it?", hist) is True


def test_visual_followup_requires_aircraft_thread():
    hist = [{"role": "user", "content": "Tell me about the Falcon 2000"}]
    assert visual_followup_suggests_image_request("can I see it?", hist) is True
    assert visual_followup_suggests_image_request("let me see it", hist) is True
    assert visual_followup_suggests_image_request("can I see it?", None) is False
    assert visual_followup_suggests_image_request("What is the capital of France?", hist) is False


def test_visual_followup_wanna_see_try_to_see_means_photos_not_maps():
    hist = [{"role": "user", "content": "What about N807JS?"}]
    assert visual_followup_suggests_image_request("I wanna see that", hist) is True
    assert visual_followup_suggests_image_request("wanna see it", hist) is True
    assert visual_followup_suggests_image_request("trying to see it", hist) is True
    assert visual_followup_suggests_image_request("try to see that", hist) is True
    assert visual_followup_suggests_image_request("I'd like to see that", hist) is True


def test_strict_triggers_wanna_see_aircraft_and_try_tail():
    assert wants_consultant_aircraft_images_in_answer("I wanna see the aircraft") is True
    assert wants_consultant_aircraft_images_in_answer("trying to see the jet") is True
    assert wants_consultant_aircraft_images_in_answer("try to see N508JS") is True


def test_broad_keyword_wanna_see_the_plane():
    assert broad_keyword_suggests_image_request("I wanna see the plane") is True
    assert broad_keyword_suggests_image_request("trying to see the aircraft") is True


def test_tail_in_thread_do_you_know_registration():
    """Regression: prior user line 'do you know n807js' must anchor 'can I see that?'."""
    hist = [{"role": "user", "content": "do you know n807js?"}]
    assert thread_has_aircraft_context("can I see that?", hist) is True
    assert visual_followup_suggests_image_request("can I see that?", hist) is True


def test_conversation_guard_forces_aviation_for_visual_followup():
    hist = [{"role": "user", "content": "do you know n807js?"}]
    r = evaluate_conversation_guard("can I see that?", hist, openai_api_key="", chat_model="")
    assert r.message_type == ConversationMessageType.AVIATION_QUERY
    assert r.reply is None


def test_tavily_image_filter_drops_different_tail_embedded_in_url():
    """Article/CDN paths often name another registration (e.g. N503EA) than the aircraft in Phly."""
    phly = [
        {
            "registration_number": "N508JA",
            "manufacturer": "Eclipse",
            "model": "EA500",
        }
    ]
    wrong = [
        {
            "url": "https://static0.simpleflyingimages.com/wordpress/wp-content/uploads/2022/08/Eclipse-N503EA-060825-23-16.jpg",
            "description": "Eclipse very light jet on ramp",
        }
    ]
    kept = filter_tavily_images_for_phly(
        wrong, phly, max_out=6, trust_tail_biased_search=True
    )
    assert kept == []


def test_tavily_image_filter_keeps_matching_tail_in_url():
    phly = [
        {
            "registration_number": "N508JA",
            "manufacturer": "Eclipse",
            "model": "EA500",
        }
    ]
    row = [
        {
            "url": "https://cdn.example.com/aircraft/N508JA-thumb.jpg",
            "description": "Eclipse EA500",
        }
    ]
    kept = filter_tavily_images_for_phly(
        row, phly, max_out=6, trust_tail_biased_search=True
    )
    assert len(kept) == 1


def test_pdf_proxy_allows_simple_flying_cdn():
    u = "https://static0.simpleflyingimages.com/wordpress/wp-content/uploads/2022/08/Eclipse-N503EA.jpg"
    assert consultant_report_image_url_allowed(u) is True


def test_pdf_proxy_allows_bjtonline_and_cirrus():
    assert (
        consultant_report_image_url_allowed(
            "https://www.bjtonline.com/sites/bjtonline.com/files/aircraft/eclipse500.jpg"
        )
        is True
    )
    assert (
        consultant_report_image_url_allowed(
            "https://cirrusaircraft.com/wp-content/uploads/2026/01/2026-cirrus-aircraft-sr-series-g7-interior-1-1024x576.jpeg"
        )
        is True
    )


def test_pdf_proxy_allows_aircharterservice_and_navalnews():
    assert (
        consultant_report_image_url_allowed(
            "https://images.aircharterservice.com/global/aircraft-guide/private-charter/gulfstream-g650.jpg"
        )
        is True
    )
    assert (
        consultant_report_image_url_allowed(
            "https://www.navalnews.com/wp-content/uploads/2025/01/French-Navy-Falcon-2000LXS-Albatros-AVSIMAR-aircraft-1024x682.jpg"
        )
        is True
    )


def test_strict_tail_mode_requires_tail_on_source_page_option_b():
    tav = {
        "results": [
            {
                "url": "https://example.com/spotting/n807js-cessna-citation-excel",
                "title": "N807JS Cessna Citation Excel",
                "content": "Spotted N807JS arriving yesterday.",
                "images": [
                    {"url": "https://cdn.example.com/img/a.jpg", "description": "N807JS on approach"},
                    {"url": "https://cdn.example.com/img/b.jpg", "description": "Cabin shot"},
                ],
            },
            {
                "url": "https://example.com/spotting/n807jt-cessna",
                "title": "N807JT Cessna",
                "content": "Different tail.",
                "images": [{"url": "https://cdn.example.com/img/c.jpg", "description": "N807JT"}],
            },
        ]
    }
    # Listing rows should be ignored in strict-tail mode (no substitution).
    listing_rows = [
        {
            "source_platform": "controller",
            "source_listing_id": "x",
            "listing_url": "https://controller.example/listing/1",
            "gallery_image_urls": ["https://controller.example/cdn/1.jpg"],
        }
    ]
    out = build_consultant_aircraft_images(
        tav,
        [],
        listing_rows=listing_rows,
        required_tail="N807JS",
        strict_tail_page_match=True,
        max_gallery_images=5,
    )
    assert out
    assert all("cdn.example.com" in (r.get("url") or "") for r in out)
    assert all(r.get("page_url") == "https://example.com/spotting/n807js-cessna-citation-excel" for r in out)


def test_strict_model_mode_filters_and_ranks_by_host_priority():
    tav = {
        "results": [
            {
                "url": "https://example.com/story/falcon-2000",
                "title": "Dassault Falcon 2000 — cabin and exterior",
                "content": "Falcon 2000 photos",
                "images": [
                    # Opaque CDN path; would fail URL/desc-only matching without result-linked page check.
                    {"url": "https://cdn.jetphotos.com/full/1/1.jpg", "description": "Taxi"},
                    {"url": "https://images.planespotters.net/photo/1.jpg", "description": "Parked"},
                ],
            }
        ],
        "images": [
            {"url": "https://example.com/img/other.jpg", "description": "Falcon 900 on ramp"},
        ],
    }
    out = build_consultant_aircraft_images(
        tav,
        [],
        required_marketing_type="Dassault Falcon 2000",
        strict_model_title_alt_match=True,
        max_gallery_images=5,
    )
    assert len(out) == 2
    assert "jetphotos" in (out[0].get("url") or "").lower()


def test_tavily_image_filter_requires_make_model_when_phly_full_row():
    """Drop generic bizjet hits that lack Citation/Excel cues when Phly type is known."""
    phly = [
        {
            "registration_number": "N807JS",
            "manufacturer": "Cessna",
            "model": "Citation Excel",
        }
    ]
    wrong = [
        {
            "url": "https://cdn.jetphotos.com/photo/global.jpg",
            "description": "Bombardier Global 6000 on approach",
        }
    ]
    kept = filter_tavily_images_for_phly(
        wrong, phly, max_out=6, trust_tail_biased_search=True
    )
    assert kept == []


def test_resolve_hybrid_keywords_strict_before_llm():
    show, src, llm = resolve_hybrid_image_gallery_intent(
        "Show me photos of N12345",
        None,
        api_key="",
        model="gpt-4o-mini",
    )
    assert show is True
    assert src == "keywords_strict"
    assert llm is None


def test_resolve_hybrid_llm_aircraft_image_request():
    with patch.dict("os.environ", {"CONSULTANT_IMAGE_INTENT_LLM": "1"}):
        with patch(
            "rag.consultant_image_intent.classify_consultant_hybrid_intent_llm",
            return_value="aircraft_image_request",
        ):
            show, src, llm = resolve_hybrid_image_gallery_intent(
                "I'd love a visual of that jet we discussed",
                [{"role": "user", "content": "Falcon 900"}],
                api_key="sk-test",
                model="gpt-4o-mini",
            )
    assert show is True
    assert src == "llm_aircraft_image_request"
    assert llm == "aircraft_image_request"


def test_resolve_hybrid_llm_disabled():
    with patch.dict("os.environ", {"CONSULTANT_IMAGE_INTENT_LLM": "0"}):
        show, src, llm = resolve_hybrid_image_gallery_intent(
            "I'd love a visual of that jet",
            None,
            api_key="sk-test",
            model="gpt-4o-mini",
        )
    assert show is False
    assert src == "llm_disabled"


def test_photo_focus_prefers_phly_make_model():
    rows = [
        {
            "registration_number": "N807JS",
            "serial_number": "560-5354",
            "manufacturer": "Cessna",
            "model": "Citation Excel",
        }
    ]
    q = build_aircraft_photo_focus_tavily_query("show me N807JS", rows, None)
    assert q
    assert "Citation Excel" in q
    assert "N807JS" not in q.replace("n807js", "")
    low = q.lower()
    assert "aircraft exterior" in low
    assert "private jet" in low


def test_suspicious_falcon_9000_note():
    from rag.consultant_suspicious_model import consultant_suspicious_aircraft_model_note

    note = consultant_suspicious_aircraft_model_note("Show me a Falcon 9000")
    assert note
    assert "9000" in note
    assert "Falcon 900" in note or "900" in note


def test_resolve_hybrid_do_you_have_photos():
    show, src, llm = resolve_hybrid_image_gallery_intent(
        "Do you have any photos of that one?",
        [{"role": "user", "content": "N807JS Citation Excel"}],
        api_key="",
        model="gpt-4o-mini",
    )
    assert show is True
    assert src == "keywords_strict"
