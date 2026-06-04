"""Phase 36.7 — cross-pipeline consistency E2E-style checks."""

from __future__ import annotations

import re

import pytest

from services.consistency.pipeline_agreement_checker import AgreementFlag
from services.consistency.unified_broker_state import UnifiedBrokerState
from services.consultant.answer_recovery import recover_valuation_answer
from services.routing.authority_dispatch import respond_buy_decision

pytestmark = pytest.mark.deterministic


def _audit_cross_pipeline(query: str, answer: str, data_used: dict) -> list[str]:
    failures: list[str] = []
    ubs = data_used.get("unified_broker_state")
    if not isinstance(ubs, dict) or not ubs.get("canonical_model"):
        failures.append("CROSS_MODEL_IDENTITY_UNIQUENESS:missing_unified_state")

    canon = (ubs or {}).get("canonical_model", "")
    bdd = data_used.get("buy_decision_dispatch") or {}
    if bdd.get("model") and canon and str(bdd["model"]).lower() != str(canon).lower():
        failures.append("CROSS_MODEL_IDENTITY_UNIQUENESS:dispatch_model_drift")

    pa = data_used.get("pipeline_agreement") or {}
    if pa.get("flags"):
        for flag in pa["flags"]:
            if flag in (
                AgreementFlag.MODEL_MISMATCH.value,
                AgreementFlag.BAND_MISMATCH.value,
                AgreementFlag.LIQUIDITY_MISMATCH.value,
                AgreementFlag.VERDICT_INCONSISTENCY.value,
            ):
                failures.append(f"PIPELINE_AGREEMENT_STABILITY:{flag}")

    mi = data_used.get("market_intelligence") or {}
    band = mi.get("band") or {}
    ubs_band = (ubs or {}).get("band") or {}
    if band and ubs_band:
        for key in ("low", "mid", "high"):
            if key in band and key in ubs_band and band[key] != ubs_band[key]:
                failures.append("MARKET_BAND_UNIQUENESS_ACROSS_PIPELINES:band_drift")
                break

    if "good deal" in query.lower() or "fair price" in query.lower():
        dq = data_used.get("deal_quality") or {}
        dk = data_used.get("deal_killer") or {}
        dv = str(dq.get("display_verdict") or "").upper().replace(" ", "")
        kv = str(dk.get("verdict") or "").upper().replace(" ", "")
        if dv and kv and dv in ("GOODDEAL", "FAIRDEAL", "OVERPRICED") and kv in (
            "GOODDEAL",
            "FAIRDEAL",
            "OVERPRICED",
        ):
            if dv != kv:
                failures.append("DEAL_VERDICT_CONSISTENCY:verdict_mismatch")

    if failures and "recovery" in query.lower():
        failures.append("RECOVERY_ALIGNMENT_WITH_DISPATCH:recovery_failed")

    return failures


@pytest.mark.parametrize(
    "query",
    [
        "Is a 2015 Citation Latitude for $5M a good deal?",
        "What is a 2019 Citation Latitude worth?",
    ],
)
def test_cross_pipeline_buy_and_valuation(query: str) -> None:
    du: dict = {}
    if "worth" in query.lower():
        body = recover_valuation_answer(query, data_used=du)
    else:
        body = respond_buy_decision(query, db=None, data_used=du)
    assert body.strip()
    failures = _audit_cross_pipeline(query, body, du)
    assert not failures, failures


def test_buy_decision_unified_state_fields() -> None:
    du: dict = {}
    respond_buy_decision("Is a 2015 Citation Latitude for $5M a good deal?", db=None, data_used=du)
    ubs = du["unified_broker_state"]
    assert ubs["canonical_model"] == "Citation Latitude"
    assert "band" in ubs
    assert "liquidity" in ubs
    assert re.search(r"GOOD DEAL|FAIR DEAL|OVERPRICED|HIGH RISK", du["deal_killer"]["verdict"], re.I)


def test_valuation_recovery_alignment() -> None:
    du: dict = {}
    body = recover_valuation_answer("What is a 2019 Falcon 8X worth?", data_used=du)
    assert "Falcon 8X" in body
    failures = _audit_cross_pipeline("valuation recovery", body, du)
    assert not any(f.startswith("RECOVERY_ALIGNMENT") for f in failures), failures
