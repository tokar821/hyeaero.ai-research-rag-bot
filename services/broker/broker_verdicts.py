"""
Broker-grade verdict vocabulary — operational tradeoffs, not chatbot absolutes.
"""

from __future__ import annotations

from enum import Enum


class BrokerVerdict(str, Enum):
    PRIMARY_RECOMMENDATION = "PRIMARY RECOMMENDATION"
    VIABLE_WITH_COMPROMISES = "VIABLE WITH COMPROMISES"
    MISSION_RISKY = "MISSION-RISKY"
    NOT_OPERATIONALLY_CREDIBLE = "NOT OPERATIONALLY CREDIBLE"


# Legacy pipeline labels → broker verdicts
_LEGACY_MAP = {
    "BEST FIT": BrokerVerdict.PRIMARY_RECOMMENDATION,
    "GOOD FIT": BrokerVerdict.PRIMARY_RECOMMENDATION,
    "CONDITIONAL FIT": BrokerVerdict.VIABLE_WITH_COMPROMISES,
    "NOT A FIT": BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE,
    "NOT A FIT.": BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE,
}


def normalize_broker_verdict(label: str) -> BrokerVerdict:
    raw = (label or "").strip().upper()
    for verdict in BrokerVerdict:
        if raw == verdict.value:
            return verdict
    return map_legacy_verdict(raw)


def map_legacy_verdict(legacy: str) -> BrokerVerdict:
    key = (legacy or "").strip().upper()
    if key in _LEGACY_MAP:
        return _LEGACY_MAP[key]
    if "NOT" in key and "FIT" in key:
        return BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE
    if "CONDITIONAL" in key or "COMPROMISE" in key:
        return BrokerVerdict.VIABLE_WITH_COMPROMISES
    if "RISK" in key:
        return BrokerVerdict.MISSION_RISKY
    if "PRIMARY" in key or "BEST" in key or "GOOD" in key:
        return BrokerVerdict.PRIMARY_RECOMMENDATION
    return BrokerVerdict.VIABLE_WITH_COMPROMISES


def verdict_from_operational_signals(
    *,
    hard_feasible: bool,
    margin_nm: float,
    penalty_total: float,
    corridor_approved: bool = True,
) -> BrokerVerdict:
    """Map elimination + margin facts to broker verdict — not score alone."""
    if not hard_feasible or not corridor_approved:
        return BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE
    if margin_nm < 0:
        return BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE
    # Mission-risky triggers on either severe penalties or a very tight margin.
    if penalty_total >= 0.28 or margin_nm < 150:
        return BrokerVerdict.MISSION_RISKY
    if margin_nm < 300 or penalty_total >= 0.16:
        return BrokerVerdict.VIABLE_WITH_COMPROMISES
    return BrokerVerdict.PRIMARY_RECOMMENDATION


def verdict_to_storage_string(verdict: BrokerVerdict) -> str:
    return verdict.value
