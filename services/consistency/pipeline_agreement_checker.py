"""Deterministic cross-pipeline agreement validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.consistency.unified_broker_state import UnifiedBrokerState


class AgreementFlag(str, Enum):
    MODEL_MISMATCH = "MODEL_MISMATCH"
    BAND_MISMATCH = "BAND_MISMATCH"
    LIQUIDITY_MISMATCH = "LIQUIDITY_MISMATCH"
    VERDICT_INCONSISTENCY = "VERDICT_INCONSISTENCY"


@dataclass
class PipelineAgreementReport:
    flags: List[AgreementFlag] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    aligned: bool = True

    def add(self, flag: AgreementFlag, detail: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
        self.details.append(detail)
        self.aligned = False


def _norm_model(m: str) -> str:
    return (m or "").strip().lower()


def _band_triple(band: Any) -> Optional[tuple]:
    if band is None:
        return None
    if isinstance(band, dict):
        low, mid, high = band.get("low"), band.get("mid"), band.get("high")
    else:
        low, mid, high = getattr(band, "low", None), getattr(band, "mid", None), getattr(band, "high", None)
    if low is None or high is None:
        return None
    return (float(low), float(mid) if mid is not None else None, float(high))


def _bands_diverge(a: tuple, b: tuple, *, tolerance: float = 0.08) -> bool:
    low_a, mid_a, high_a = a
    low_b, mid_b, high_b = b
    ref = mid_a or (low_a + high_a) / 2.0
    ref_b = mid_b or (low_b + high_b) / 2.0
    if ref <= 0 or ref_b <= 0:
        return False
    return abs(ref - ref_b) / ref > tolerance or abs(low_a - low_b) / ref > tolerance


def check_pipeline_agreement(
    *,
    data_used: Optional[Dict[str, Any]] = None,
    state: Optional["UnifiedBrokerState"] = None,
) -> PipelineAgreementReport:
    """Validate model, band, liquidity, and verdict alignment across layers."""
    report = PipelineAgreementReport()
    du = data_used if isinstance(data_used, dict) else {}

    models: List[str] = []
    if state is not None:
        models.append(state.identity.canonical_model)
    for key in ("buy_decision_dispatch",):
        block = du.get(key)
        if isinstance(block, dict) and block.get("model"):
            models.append(str(block["model"]))
    mi = du.get("market_intelligence")
    if isinstance(mi, dict):
        snap = mi.get("snapshot")
        if isinstance(snap, dict) and snap.get("model"):
            models.append(str(snap["model"]))
    rec = du.get("aircraft_authority_record")
    if isinstance(rec, dict) and rec.get("canonical_name"):
        models.append(str(rec["canonical_name"]))
    verified = du.get("verified_recovery_models")
    if isinstance(verified, list):
        models.extend(str(m) for m in verified[:2])

    canon = state.identity.canonical_model if state else None
    if canon:
        from services.consistency.cross_model_identity import _authority_canonical

        canon_norm = _norm_model(canon)
        for m in models:
            resolved = _authority_canonical(m) or m
            if _norm_model(resolved) != canon_norm and m.strip():
                report.add(
                    AgreementFlag.MODEL_MISMATCH,
                    f"layer model {m!r} != canonical {canon!r}",
                )
                break

    mi_band = None
    auth_band = None
    if state is not None and state.market_band is not None:
        mi_band = _band_triple(state.market_band)
    elif isinstance(mi, dict):
        mi_band = _band_triple(mi.get("band"))
    auth = du.get("aircraft_authority_market")
    if isinstance(auth, dict):
        auth_band = _band_triple(auth.get("expected_market_band_usd"))
    if mi_band and auth_band and _bands_diverge(mi_band, auth_band):
        report.add(
            AgreementFlag.BAND_MISMATCH,
            "market_intelligence band differs from authority_market band",
        )

    liq_mi: Optional[str] = None
    liq_md: Optional[str] = None
    if state is not None and state.liquidity is not None:
        liq_mi = state.liquidity.band.value
    elif isinstance(mi, dict):
        liq_mi = str(mi.get("liquidity", {}).get("band") or "")
    dk = du.get("deal_killer")
    md = state.market_data if state else {}
    if isinstance(md, dict):
        liq_md = str(md.get("liquidity") or "").upper()
    if liq_mi and liq_md and liq_mi.upper() != liq_md.upper():
        report.add(
            AgreementFlag.LIQUIDITY_MISMATCH,
            f"liquidity tier {liq_md} != market_intel {liq_mi}",
        )

    dk_verdict = ""
    dq_verdict = ""
    if isinstance(dk, dict):
        dk_verdict = str(dk.get("verdict") or "").upper().replace("_", " ")
    dq = du.get("deal_quality")
    if isinstance(dq, dict):
        dq_verdict = str(dq.get("display_verdict") or dq.get("verdict") or "").upper().replace("_", " ")
    if state is not None and state.deal_quality is not None:
        dq_verdict = state.deal_quality.display_verdict.upper()
    if dk_verdict and dq_verdict:
        dk_norm = dk_verdict.replace(" ", "")
        dq_norm = dq_verdict.replace(" ", "")
        price_verdicts = {"GOODDEAL", "FAIRDEAL", "OVERPRICED"}
        if dk_norm in price_verdicts and dq_norm in price_verdicts and dk_norm != dq_norm:
            report.add(
                AgreementFlag.VERDICT_INCONSISTENCY,
                f"deal_killer {dk_verdict} != deal_quality {dq_verdict}",
            )

    return report
