"""Phase 22 — Recommendation Confidence & Evidence Engine tests."""

from __future__ import annotations

from services.confidence import recommendation_confidence_engine as rcee
from services.confidence.recommendation_confidence_engine import (
    EvidenceItem,
    attach_recommendation_confidence_if_enabled,
    build_recommendation_confidence,
    confidence_band,
    evaluate_data_completeness,
    evaluate_recommendation_confidence_hooks,
    recommendation_confidence_enabled,
)
from services.consultant.mission_state import MissionState


def _mission(
    *,
    pax: int | None = 8,
    budget_m: float | None = 12.0,
    routes: list[str] | None = None,
) -> MissionState:
    return MissionState(
        passenger_count=pax,
        routes=routes if routes is not None else ["LA to Miami"],
        budget_usd=budget_m * 1_000_000 if budget_m else None,
        nonstop_requirement=True,
    )


def test_complete_mission_high_confidence():
    mission = _mission(pax=8, budget_m=12.0, routes=["LA to Miami"])
    score, warnings, missing = rcee._mission_confidence_score(mission)
    assert score >= 70
    assert "budget" not in missing
    assert "route" not in missing

    query = "8 passengers LA to Miami under $12M"
    response = {
        "data_used": {
            "consultant_recommendations": [{"model": "Citation Latitude"}],
            "mission_state": {
                "passenger_count": 8,
                "routes": ["LA to Miami"],
                "budget_usd": 12_000_000,
            },
        }
    }
    bundle = build_recommendation_confidence(query, response)
    top = bundle["aircraft_confidence"][0]
    assert top["mission_confidence"] >= 70
    assert top["overall_confidence"] >= 40


def test_missing_budget_moderate_confidence():
    mission = _mission(pax=8, budget_m=None)
    score, _, missing = rcee._mission_confidence_score(mission)
    assert 40 <= score < 70
    assert "budget" in missing

    bundle = build_recommendation_confidence(
        "8 passengers LA to Miami",
        {"data_used": {"consultant_recommendations": [{"model": "Citation Latitude"}]}},
    )
    row = bundle["aircraft_confidence"][0]
    assert row["confidence_band"] in ("LOW", "MODERATE", "HIGH")
    assert "budget" in row["missing_inputs"]


def test_missing_route_low_confidence():
    mission = _mission(pax=8, budget_m=12.0, routes=[])
    score, warnings, missing = rcee._mission_confidence_score(mission)
    assert score < 40
    assert "route" in missing
    assert any("route" in w.lower() for w in warnings)


def test_buy_decision_with_comps():
    score, _, missing = rcee._buy_decision_confidence(
        {
            "status": "OK",
            "expected_market_band_usd": {"low": 8_000_000, "mid": 10_000_000, "high": 12_000_000},
        },
        {"tail": "N123AB", "ask_usd": 10_500_000, "verdict": "Fair Deal"},
    )
    assert score >= 70
    assert "market_comps" not in missing

    bundle = build_recommendation_confidence(
        "Is this a good deal?",
        {
            "data_used": {
                "aircraft_authority_market": {
                    "status": "OK",
                    "canonical_name": "Gulfstream G650",
                    "expected_market_band_usd": {"low": 8e6, "mid": 10e6, "high": 12e6},
                },
                "deal_killer": {"tail": "N123AB", "ask_usd": 10_500_000, "verdict": "Fair Deal"},
            }
        },
    )
    row = bundle["aircraft_confidence"][0]
    assert row["market_confidence"] >= 70


def test_buy_decision_without_comps():
    score, warnings, missing = rcee._buy_decision_confidence({}, {})
    assert score < 40
    assert "market_comps" in missing

    bundle = build_recommendation_confidence(
        "Good deal on a G650?",
        {"data_used": {"deal_killer": {"verdict": "Fair Deal", "hypothetical": True}}},
    )
    row = bundle["aircraft_confidence"][0]
    assert row["market_confidence"] < 70
    assert any("market" in w.lower() for w in row["warnings"])


def test_alias_ambiguity_penalty():
    assert rcee._detect_alias_ambiguity("Longitude") is True

    penalty, warnings, _ = rcee._compute_penalties(
        aircraft="Longitude",
        mission=_mission(),
        data_used={},
        is_buy=False,
    )
    assert penalty >= 10
    assert any("alias" in w.lower() for w in warnings)


def test_evidence_aggregation():
    mission = _mission()
    data_used = {
        "consultant_recommendations": [{"model": "Gulfstream G650"}],
        "authoritative_comparison_dataset": {
            "status": "OK",
            "aircraft": [
                {"canonical_name": "Gulfstream G650", "range_nm": 6000},
                {"canonical_name": "Falcon 8X", "range_nm": 5600},
            ],
        },
        "deal_killer": {"verdict": "Fair Deal"},
    }
    evidence, evidence_score, authority_conf, _ = rcee._collect_evidence(
        aircraft="Gulfstream G650",
        mission=mission,
        data_used=data_used,
        is_buy=True,
    )
    assert evidence_score >= 40
    assert authority_conf > 0
    names = {e.source_name for e in evidence}
    assert "Aircraft Authority" in names
    assert "Deal Killer" in names
    assert "Comparison Dataset" in names

    item = EvidenceItem("test", "Test Source", 0.9, 5.0)
    d = item.to_dict()
    assert d["source_name"] == "Test Source"
    assert d["contribution"] == 5.0


def test_confidence_band_mapping():
    assert confidence_band(95) == "VERY_HIGH"
    assert confidence_band(80) == "HIGH"
    assert confidence_band(55) == "MODERATE"
    assert confidence_band(25) == "LOW"


def test_data_completeness_percentage():
    mission = _mission()
    pct = evaluate_data_completeness(
        aircraft="Gulfstream G650",
        mission=mission,
        data_used={
            "aircraft_authority_market": {"status": "OK"},
            "deal_killer": {"tail": "N999", "ask_usd": 10_000_000},
        },
    )
    assert 50 <= pct <= 100


def test_evaluator_confidence_hooks():
    inflated = {
        "data_used": {
            "recommendation_confidence": {
                "aircraft_confidence": [
                    {
                        "overall_confidence": 95,
                        "evidence_score": 40,
                        "data_completeness_score": 30,
                        "confidence_band": "VERY_HIGH",
                        "missing_inputs": ["budget"],
                    }
                ]
            }
        }
    }
    failures = evaluate_recommendation_confidence_hooks(inflated)
    assert "confidence_inflation" in failures
    assert "unsupported_high_confidence" in failures
    assert "confidence_consistency" in failures


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_RECOMMENDATION_CONFIDENCE", raising=False)
    payload = {"answer": "x", "data_used": {}}
    assert not recommendation_confidence_enabled()
    out = attach_recommendation_confidence_if_enabled("test", payload)
    assert "recommendation_confidence" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_RECOMMENDATION_CONFIDENCE", "1")
    out2 = attach_recommendation_confidence_if_enabled("test", payload)
    assert "recommendation_confidence" in (out2.get("data_used") or {})
