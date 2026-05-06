"""
Contract tests for ``consultant_response_controller.generate_consultant_response``.

Patches isolate OpenAI / SearchAPI; the real rank → align → orchestrate stack still runs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services import consultant_response_controller as crc


def _challenger_cabin_pool(n: int = 6):
    """Rows tuned to pass relevance, ranking (≥0.65), alignment, and orchestrator validation."""
    out = []
    for i in range(n):
        out.append(
            {
                "url": f"https://www.jetphotos.net/photo/91000{i}/challenger-cabin-{i}.jpg",
                "title": (
                    f"Bombardier Challenger 650 main cabin interior VIP club seating "
                    f"galley oval windows aisle {i}"
                ),
                "description": (
                    "Bombardier Challenger 650 aircraft cabin interior business jet "
                    "narrow cabin jetphotos"
                ),
                "source_domain": "jetphotos.net",
                "source": "JetPhotos",
            }
        )
    return out


def _falcon_900_pool(n: int = 6):
    out = []
    for i in range(n):
        out.append(
            {
                "url": f"https://www.jetphotos.net/photo/92000{i}/falcon900-cabin-{i}.jpg",
                "title": (
                    f"Dassault Falcon 900EX main cabin interior seating windows "
                    f"galley VIP {i}"
                ),
                "description": (
                    "Dassault Falcon 900 aircraft cabin interior business jet cabin windows"
                ),
                "source_domain": "jetphotos.net",
                "source": "JetPhotos",
            }
        )
    return out


@pytest.fixture
def patch_intent_and_search(monkeypatch):
    """Default: no-op overrides; tests set side_effect / return_value per case."""

    def _default_fetch(user_query, normalized_intent, matching, queries):
        return _challenger_cabin_pool(6)

    monkeypatch.setattr(crc, "default_searchapi_fetch", _default_fetch)


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_show_me_interior_mid_conversation_returns_images(
    mock_norm, mock_match, mock_genq, patch_intent_and_search,
):
    history = [
        {"role": "user", "content": "We are comparing Gulfstream G650 against Globals."},
        {"role": "assistant", "content": "Here is a spec summary without photos."},
    ]
    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "Gulfstream G650",
        "visual_focus": "interior",
        "category": "heavy jet",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": [
            "Bombardier Challenger 650",
            "Dassault Falcon 7X",
        ],
        "hard_fail": False,
        "reasoning": "test",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["challenger 650 interior cabin"]}

    out = crc.generate_consultant_response("show me interior", history)

    assert out["meta"]["visual_trigger"] is True
    assert len(out["images"]) >= 3
    assert all("url" in im for im in out["images"])
    assert "cannot find images" not in (out["answer"] or "").lower()
    cand = " ".join(out["meta"]["aircraft_candidates"]).lower()
    assert "challenger" in cand or "falcon" in cand


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_g650_cheaper_peer_class_in_answer_and_meta(
    mock_norm, mock_match, mock_genq,
):
    mock_norm.return_value = {
        "intent_type": "comparison",
        "aircraft": "Gulfstream G650",
        "visual_focus": "interior",
        "category": "ultra long range",
        "constraints": {"comparison_target": "lower cost"},
    }
    mock_match.return_value = {
        "aircraft_candidates": [
            "Bombardier Challenger 650",
            "Dassault Falcon 7X",
            "Bombardier Global 5000",
        ],
        "hard_fail": False,
        "reasoning": "ULR anchor",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["g650 peer interior"]}

    out = crc.generate_consultant_response(
        "something like G650 but cheaper",
        [],
    )

    assert out["meta"]["visual_trigger"] is False
    assert out["images"] == []
    ans = (out["answer"] or "").lower()
    peers = " ".join(out["meta"]["aircraft_candidates"]).lower()
    assert "challenger" in peers or "falcon" in peers or "global" in peers
    assert any(x in ans for x in ("challenger", "falcon", "global"))


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_falcon_9000_interior_corrects_model_and_images(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda uq, ni, m, q: _falcon_900_pool(6))

    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "Falcon 9000",
        "visual_focus": "interior",
        "category": "large cabin",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": [],
        "hard_fail": False,
        "reasoning": "no anchor",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["falcon 900 cabin interior"]}

    out = crc.generate_consultant_response("Falcon 9000 interior", [])

    assert out["meta"]["visual_trigger"] is True
    assert "9000" in (out["answer"] or "") or "falcon 900" in (out["answer"] or "").lower()
    assert len(out["images"]) >= 3
    joined = " ".join(str(im.get("title", "")) for im in out["images"]).lower()
    assert "falcon" in joined and ("900" in joined or "900ex" in joined)


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_irrelevant_history_does_not_block_visual_output(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda uq, ni, m, q: _challenger_cabin_pool(6))

    noise = [{"role": "user", "content": f"unrelated note {i}"} for i in range(15)]
    mock_norm.return_value = {
        "intent_type": "cabin_search",
        "aircraft": "Bombardier Challenger 650",
        "visual_focus": "cabin",
        "category": "large cabin",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": ["Bombardier Challenger 650"],
        "hard_fail": False,
        "reasoning": "test",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["challenger cabin"]}

    out = crc.generate_consultant_response(
        "show me the cabin interior",
        noise,
    )

    assert out["meta"]["visual_trigger"] is True
    assert len(out["images"]) >= 3


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_empty_image_search_returns_empty_not_junk(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda *a, **k: [])

    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "Global 7500",
        "visual_focus": "interior",
        "category": "ultra long range",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": ["Bombardier Global 6000"],
        "hard_fail": False,
        "reasoning": "test",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["global 7500 interior"]}

    out = crc.generate_consultant_response("photo of the interior", [])

    assert out["meta"]["visual_trigger"] is True
    assert out["images"] == []


def test_visual_trigger_keyword_detection():
    norm = {"intent_type": "comparison", "visual_focus": "exterior", "aircraft": None}
    assert crc._visual_trigger("show me interior", [], norm) is True


def test_visual_trigger_from_prior_history_blob():
    norm = {"intent_type": "comparison", "visual_focus": "exterior", "aircraft": None}
    hist = [{"role": "user", "content": "earlier we looked at cabin photos"}]
    assert crc._visual_trigger("what about fuel burn?", hist, norm) is True


def test_scan_last_aircraft_from_assistant_turn():
    hist = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "The Gulfstream G650 is a strong ULR option."},
    ]
    assert crc.scan_last_aircraft_mention(hist, "show me interior") == "Gulfstream G650"


def test_apply_recovery_fills_empty_aircraft_from_thread():
    hist = [{"role": "assistant", "content": "Compared to the Challenger 650 cabin width…"}]
    norm = {"intent_type": "interior_visual", "aircraft": "", "visual_focus": "interior", "constraints": {}}
    out = crc.apply_aircraft_context_recovery(norm, hist, "photo of the cabin please")
    assert "Challenger 650" in (out.get("aircraft") or "")


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_g650_intent_answer_must_not_name_g500_when_not_in_candidates(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda uq, ni, m, q: _challenger_cabin_pool(6))
    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "Gulfstream G650",
        "visual_focus": "interior",
        "category": "ultra long range",
        "constraints": {},
    }
    peers = [
        "Bombardier Challenger 650",
        "Dassault Falcon 7X",
        "Bombardier Global 6000",
    ]
    mock_match.return_value = {"aircraft_candidates": peers, "hard_fail": False, "reasoning": "x", "hard_fail_reason": None}
    mock_genq.return_value = {"queries": ["g650 interior peers"]}
    out = crc.generate_consultant_response("show me the interior", [])
    ans = (out["answer"] or "").lower()
    assert "g500" not in ans
    assert "challenger" in ans or "falcon" in ans or "global" in ans
    meta_c = " ".join(out["meta"]["aircraft_candidates"]).lower()
    assert "g500" not in meta_c


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_visual_reply_bans_common_clarification_templates(
    mock_norm, mock_match, mock_genq, patch_intent_and_search,
):
    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "Bombardier Challenger 650",
        "visual_focus": "interior",
        "category": "large cabin",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": ["Bombardier Challenger 650"],
        "hard_fail": False,
        "reasoning": "test",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["cabin"]}
    out = crc.generate_consultant_response("show interior", [])
    banned = ("could you clarify which aircraft", "which aircraft did you mean", "which jet did you mean")
    ans = (out["answer"] or "").lower()
    assert all(b not in ans for b in banned)


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_random_math_turn_then_cabin_still_visual(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda uq, ni, m, q: _challenger_cabin_pool(6))
    hist = [{"role": "user", "content": "what is sqrt(941)"}, {"role": "assistant", "content": "≈30.67"}]
    mock_norm.return_value = {
        "intent_type": "cabin_search",
        "aircraft": "Bombardier Challenger 650",
        "visual_focus": "cabin",
        "category": "heavy",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": ["Bombardier Challenger 650"],
        "hard_fail": False,
        "reasoning": "test",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["challenger cabin"]}
    out = crc.generate_consultant_response("show me cabin", hist)
    assert out["meta"]["visual_trigger"] is True
    assert len(out["images"]) >= 3


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_visual_with_no_thread_aircraft_falls_back_without_clarifying(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda uq, ni, m, q: _challenger_cabin_pool(6))
    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "",
        "visual_focus": "interior",
        "category": "",
        "constraints": {},
    }

    mock_match.return_value = {
        "aircraft_candidates": [],
        "hard_fail": False,
        "reasoning": "noop",
        "hard_fail_reason": None,
    }
    mock_genq.return_value = {"queries": ["midsize cabin interior"]}
    out = crc.generate_consultant_response("show Gulfstream G650 cabin interior gallery", [])
    assert out["meta"]["visual_trigger"] is True
    assert "could you clarify which aircraft" not in (out["answer"] or "").lower()
    assert out["meta"]["aircraft_candidates"]


@patch("rag.intent.generate_aviation_image_queries")
@patch("rag.intent.run_aircraft_matching_engine")
@patch("rag.intent.normalize_aviation_intent")
def test_visual_matching_hard_fail_still_attaches_candidates_and_runs_orchestrator(
    mock_norm, mock_match, mock_genq, monkeypatch,
):
    """Classifier hard_fail must not turn off images when the turn is visual."""
    monkeypatch.setattr(crc, "default_searchapi_fetch", lambda uq, ni, m, q: _challenger_cabin_pool(6))
    mock_norm.return_value = {
        "intent_type": "interior_visual",
        "aircraft": "Gulfstream G650",
        "visual_focus": "interior",
        "category": "ultra long range",
        "constraints": {},
    }
    mock_match.return_value = {
        "aircraft_candidates": [],
        "hard_fail": True,
        "reasoning": "policy",
        "hard_fail_reason": "x",
    }
    mock_genq.return_value = {"queries": ["cabin interior"]}
    out = crc.generate_consultant_response("interior gallery", [])
    assert out["meta"]["visual_trigger"] is True
    assert out["meta"]["aircraft_candidates"]
    assert len(out["images"]) >= 3
