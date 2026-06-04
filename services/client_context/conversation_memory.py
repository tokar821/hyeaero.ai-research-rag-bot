"""Rolling conversation memory — aircraft, budget, manufacturers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.client_context.client_profile import ClientProfile

_BUDGET_RE = re.compile(
    r"(?is)(?:about|around|have|budget|for)\s+\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil|k)?\b|"
    r"\b(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\s+budget\b",
)
_MANUFACTURER_RE = {
    "Gulfstream": re.compile(r"\b(?:gulfstreams?|g\s*650s?|g\s*700s?|g\s*550s?)\b", re.I),
    "Dassault": re.compile(r"\b(?:dassault|falcon)\b", re.I),
    "Bombardier": re.compile(r"\b(?:bombardier|challenger|global|learjet)\b", re.I),
    "Cessna": re.compile(r"\b(?:citation|cessna|latitude|longitude)\b", re.I),
    "Embraer": re.compile(r"\b(?:embraer|phenom|praetor|legacy)\b", re.I),
}
_COMPARE_PAIR_RE = re.compile(r"\b(?:vs\.?|versus|compare)\b", re.I)
_MODEL_TOKEN_RE: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bg650s?\b", re.I), "Gulfstream G650"),
    (re.compile(r"\bg700s?\b", re.I), "Gulfstream G700"),
    (re.compile(r"\blongitude\b", re.I), "Citation Longitude"),
    (re.compile(r"\blatitude\b", re.I), "Citation Latitude"),
    (re.compile(r"\bpraetor\s*600\b", re.I), "Praetor 600"),
    (re.compile(r"\bpraetor\b", re.I), "Praetor 600"),
    (re.compile(r"\bphenom\b", re.I), "Praetor 600"),
    (re.compile(r"\bchallenger\s*350\b", re.I), "Challenger 350"),
]


@dataclass
class ConversationMemory:
    aircraft_mentions: Dict[str, int] = field(default_factory=dict)
    manufacturer_mentions: Dict[str, int] = field(default_factory=dict)
    budget_mentions_musd: List[float] = field(default_factory=list)
    last_comparison_pair: List[str] = field(default_factory=list)
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft_mentions": dict(self.aircraft_mentions),
            "manufacturer_mentions": dict(self.manufacturer_mentions),
            "budget_mentions_musd": list(self.budget_mentions_musd),
            "last_comparison_pair": list(self.last_comparison_pair),
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> ConversationMemory:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            aircraft_mentions=dict(raw.get("aircraft_mentions") or {}),
            manufacturer_mentions=dict(raw.get("manufacturer_mentions") or {}),
            budget_mentions_musd=[float(x) for x in (raw.get("budget_mentions_musd") or []) if x is not None],
            last_comparison_pair=_str_list(raw.get("last_comparison_pair")),
            turn_count=int(raw.get("turn_count") or 0),
        )


def _str_list(val: Any) -> List[str]:
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()][:4]


def _to_musd(amount: str, unit: str) -> Optional[float]:
    try:
        val = float(amount)
    except ValueError:
        return None
    u = (unit or "m").lower()
    if u == "k":
        return val / 1000.0
    if val < 1000:
        return val
    return val / 1_000_000.0 if val >= 10_000 else val


def _canonical_memory_model(model: str) -> str:
    try:
        from services.aircraft.aircraft_authority_service import resolve_aircraft_alias
        from services.comparison.aircraft_registry_lock import lock_comparison_aircraft

        alias = resolve_aircraft_alias(model) or model
        lock = lock_comparison_aircraft([alias])
        if lock.canonical:
            return lock.canonical[0]
        return alias.strip()
    except Exception:
        return (model or "").strip()


def _detect_models(text: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return list(detect_models_from_text(text or "") or [])
    except Exception:
        return []


def update_memory_from_turn(
    memory: ConversationMemory,
    query: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> ConversationMemory:
    """Incorporate current turn (and recent user history) into rolling memory."""
    memory.turn_count += 1
    blob_parts = [query or ""]
    for h in (history or [])[-8:]:
        if isinstance(h, dict) and str(h.get("role") or "").lower() == "user":
            blob_parts.append(str(h.get("content") or ""))
    blob = " ".join(blob_parts)

    for pat, seed in _MODEL_TOKEN_RE:
        if pat.search(query or ""):
            key = _canonical_memory_model(seed)
            if key:
                memory.aircraft_mentions[key] = memory.aircraft_mentions.get(key, 0) + 2

    for model in _detect_models(query):
        key = _canonical_memory_model(model)
        if key:
            memory.aircraft_mentions[key] = memory.aircraft_mentions.get(key, 0) + 2

    for model in _detect_models(blob):
        key = _canonical_memory_model(model)
        if key:
            memory.aircraft_mentions[key] = memory.aircraft_mentions.get(key, 0) + 1

    for mfr, pat in _MANUFACTURER_RE.items():
        if pat.search(query or ""):
            memory.manufacturer_mentions[mfr] = memory.manufacturer_mentions.get(mfr, 0) + 2
        elif pat.search(blob):
            memory.manufacturer_mentions[mfr] = memory.manufacturer_mentions.get(mfr, 0) + 1

    for m in _BUDGET_RE.finditer(query or ""):
        amt = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4) or "m"
        if amt:
            musd = _to_musd(amt, unit or "m")
            if musd is not None:
                memory.budget_mentions_musd.append(musd)

    if _COMPARE_PAIR_RE.search(query or ""):
        models: List[str] = []
        for pat, seed in _MODEL_TOKEN_RE:
            if pat.search(query or ""):
                canon = _canonical_memory_model(seed)
                if canon and canon not in models:
                    models.append(canon)
        for m in _detect_models(query):
            canon = _canonical_memory_model(m)
            if canon and canon not in models:
                models.append(canon)
        if len(models) >= 2:
            memory.last_comparison_pair = models[:2]

    return memory


def memory_to_profile(memory: ConversationMemory, profile: ClientProfile) -> ClientProfile:
    """Merge rolling memory into client profile."""
    if memory.budget_mentions_musd:
        profile.preferred_budget_musd = memory.budget_mentions_musd[-1]

    ranked_aircraft = sorted(
        memory.aircraft_mentions.items(),
        key=lambda x: (-x[1], x[0]),
    )
    profile.preferred_aircraft = [m for m, _ in ranked_aircraft[:8]]

    ranked_mfr = sorted(
        memory.manufacturer_mentions.items(),
        key=lambda x: (-x[1], x[0]),
    )
    profile.preferred_manufacturers = [m for m, _ in ranked_mfr[:5]]

    if memory.last_comparison_pair:
        profile.inferred_preferences["last_comparison_pair"] = list(memory.last_comparison_pair)

    profile.inferred_preferences["aircraft_mention_counts"] = dict(memory.aircraft_mentions)
    return profile


__all__ = [
    "ConversationMemory",
    "memory_to_profile",
    "update_memory_from_turn",
]
