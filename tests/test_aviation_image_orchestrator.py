from services.aviation_image_orchestrator import orchestrate_aviation_image_pipeline


def _falcon_pool():
    """Six distinct rows so ranking can pass >=3 over the 0.65 gate."""
    out = []
    for i in range(6):
        out.append(
            {
                "url": f"https://jetphotos.net/photo/{8000 + i}.jpg",
                "title": f"Dassault Falcon 7X aircraft cabin interior seating windows aisle {i}",
                "description": f"Dassault Falcon 7X aircraft cabin interior seating windows aisle {i}",
                "source_domain": "jetphotos.net",
            }
        )
    return out


def test_pipeline_happy_path():
    pool = _falcon_pool()
    norm = {"intent_type": "interior_visual", "visual_focus": "interior", "aircraft": None, "constraints": {}}
    out = orchestrate_aviation_image_pipeline(
        normalized_intent=norm,
        aircraft_candidates=["Dassault Falcon 7X"],
        answer_text="The Dassault Falcon 7X offers a wide cabin with good windows.",
        raw_images=pool,
        filtered_images=pool,
        max_retries=2,
    )
    assert out["pipeline_decisions"]["ranking_applied"] is True
    assert out["pipeline_decisions"]["alignment_applied"] is True
    assert len(out["final_images"]) >= 3
    assert all(float(im.get("score", 0) or 0) >= 0.65 for im in out["final_images"])


def test_pipeline_empty_returns_failsafe():
    out = orchestrate_aviation_image_pipeline(
        normalized_intent={"intent_type": "interior_visual", "visual_focus": "interior", "constraints": {}},
        aircraft_candidates=["Global 7500"],
        answer_text="Nothing here.",
        raw_images=[],
        filtered_images=[],
        max_retries=2,
    )
    assert out["final_images"] == []
    assert out["pipeline_decisions"]["fallback_used"] is True
