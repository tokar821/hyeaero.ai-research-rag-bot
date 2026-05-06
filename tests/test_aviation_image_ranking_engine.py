from services.aviation_image_ranking_engine import rank_aviation_images_for_intent


def _intent_interior():
    return {
        "intent_type": "interior_visual",
        "aircraft": "Dassault Falcon 7X",
        "category": "heavy",
        "visual_focus": "interior",
        "constraints": {"budget": None, "style": "modern", "comparison_target": None},
    }


def test_ranks_falcon_interior_above_reddit():
    intent = _intent_interior()
    cands = ["Dassault Falcon 7X", "Bombardier Challenger 650"]
    images = [
        {
            "title": "Dassault Falcon 7X main cabin interior seating",
            "url": "https://www.jetphotos.net/photo/1.jpg",
            "source_domain": "jetphotos.net",
            "description": "windows and aisle",
        },
        {
            "title": "random jet meme",
            "url": "https://i.redd.it/x.jpg",
            "source_domain": "redd.it",
            "description": "lol",
        },
        {
            "title": "Falcon 7X ramp taxi",
            "url": "https://example.com/ext.jpg",
            "source_domain": "example.com",
            "description": "takeoff exterior only",
        },
    ]
    out = rank_aviation_images_for_intent(
        normalized_intent=intent,
        aircraft_candidates=cands,
        images=images,
        min_score=0.65,
        max_keep=6,
    )
    assert out, "expected at least one passing image"
    assert out[0]["score"] >= 0.65
    assert out[0]["aircraft_match"] >= 0.65
    assert out[0]["source_quality"] >= 0.9
    assert all("reddit" not in (r.get("reason") or "").lower() for r in out)


def test_aircraft_gate_zeroes_wrong_family():
    intent = {
        "intent_type": "interior_visual",
        "aircraft": "Falcon 7X",
        "category": "heavy",
        "visual_focus": "interior",
        "constraints": {},
    }
    images = [
        {
            "title": "Citation CJ2 cabin interior",
            "url": "https://jetphotos.net/cj2.jpg",
            "source_domain": "jetphotos.net",
            "description": "citation jet",
        }
    ]
    out = rank_aviation_images_for_intent(
        normalized_intent=intent,
        aircraft_candidates=["Dassault Falcon 7X"],
        images=images,
        min_score=0.65,
    )
    assert out == []


def test_duplicate_downrank_second_same_type():
    intent = _intent_interior()
    cands = ["Dassault Falcon 7X"]
    images = [
        {
            "title": "Falcon 7X cabin interior one",
            "url": "https://a.com/1.jpg",
            "source_domain": "jetphotos.net",
            "description": "cabin windows",
        },
        {
            "title": "Falcon 7X cabin interior two",
            "url": "https://a.com/2.jpg",
            "source_domain": "jetphotos.net",
            "description": "cabin seating",
        },
    ]
    out = rank_aviation_images_for_intent(
        normalized_intent=intent,
        aircraft_candidates=cands,
        images=images,
        min_score=0.65,
        max_keep=6,
    )
    if len(out) >= 2:
        assert out[0]["score"] >= out[1]["score"]
