"""Unified broker state — single source of truth for response formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consistency.cross_model_identity import CanonicalAircraftIdentity
from services.consistency.pipeline_agreement_checker import PipelineAgreementReport
from services.market_intelligence.deal_quality_engine import DealQualityResult
from services.market_intelligence.liquidity_scoring import LiquidityScore
from services.market_intelligence.listing_analytics import MarketSnapshot
from services.market_intelligence.market_band_builder import MarketBand
from services.market_intelligence.market_intelligence_engine import MarketIntelligenceBundle


@dataclass
class UnifiedBrokerState:
    """Merged deterministic state for buy, valuation, and comparison formatting."""

    identity: CanonicalAircraftIdentity
    market_snapshot: Optional[MarketSnapshot] = None
    liquidity: Optional[LiquidityScore] = None
    market_band: Optional[MarketBand] = None
    deal_quality: Optional[DealQualityResult] = None
    market_bundle: Optional[MarketIntelligenceBundle] = None
    authority_market: Optional[Dict[str, Any]] = None
    authority_record: Optional[Dict[str, Any]] = None
    dispatch_kind: str = ""
    year: Optional[int] = None
    ask_usd: Optional[float] = None
    verdict_payload: Optional[Dict[str, Any]] = None
    market_data: Dict[str, Any] = field(default_factory=dict)
    agreement_report: Optional[PipelineAgreementReport] = None
    recovery_augmented: bool = False
    comparison_identities: Optional[tuple[CanonicalAircraftIdentity, CanonicalAircraftIdentity]] = None
    temporal: Any = None  # Optional[TemporalMarketExtension] — additive Phase 37
    adversarial: Any = None  # Optional[QueryConflictReport dict] — additive Phase 38, not for pricing

    @property
    def canonical_model(self) -> str:
        return self.identity.canonical_model

    def to_data_used_dict(self) -> Dict[str, Any]:
        """Serializable snapshot for ``data_used['unified_broker_state']``."""
        out: Dict[str, Any] = {
            "canonical_model": self.canonical_model,
            "confidence_score": self.identity.confidence_score,
            "aliases_used": list(self.identity.aliases_used),
            "source_layers": list(self.identity.source_layers),
            "dispatch_kind": self.dispatch_kind,
            "year": self.year,
            "ask_usd": self.ask_usd,
        }
        if self.market_band is not None:
            out["band"] = {
                "low": self.market_band.low,
                "mid": self.market_band.mid,
                "high": self.market_band.high,
                "confidence": self.market_band.confidence.value,
            }
        if self.liquidity is not None:
            out["liquidity"] = {
                "score": self.liquidity.score,
                "band": self.liquidity.band.value,
            }
        if self.deal_quality is not None:
            out["deal_quality"] = {
                "verdict": self.deal_quality.verdict.value,
                "display_verdict": self.deal_quality.display_verdict,
                "reason": self.deal_quality.reason,
            }
        if self.agreement_report is not None:
            out["agreement_aligned"] = self.agreement_report.aligned
            out["agreement_flags"] = [f.value for f in self.agreement_report.flags]
        if self.temporal is not None:
            from services.temporal_market.temporal_market_intelligence import temporal_to_data_used_dict

            out["temporal"] = temporal_to_data_used_dict(self.temporal)
        if isinstance(self.adversarial, dict) and self.adversarial:
            out["adversarial"] = self.adversarial
        return out

    @classmethod
    def from_data_used(cls, data_used: Dict[str, Any]) -> Optional["UnifiedBrokerState"]:
        """Rehydrate minimal state when full object was stored as dict."""
        raw = data_used.get("unified_broker_state")
        if not isinstance(raw, dict) or not raw.get("canonical_model"):
            return None
        identity = CanonicalAircraftIdentity(
            canonical_model=str(raw["canonical_model"]),
            aliases_used=tuple(raw.get("aliases_used") or []),
            source_layers=tuple(raw.get("source_layers") or []),
            confidence_score=int(raw.get("confidence_score") or 0),
            resolved_from_query_tokens=(),
        )
        return cls(
            identity=identity,
            dispatch_kind=str(raw.get("dispatch_kind") or ""),
            year=raw.get("year"),
            ask_usd=raw.get("ask_usd"),
        )
