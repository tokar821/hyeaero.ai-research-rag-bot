"""
Subjective luxury / escalation language → structured retrieval bias.

Used to (1) enrich the semantic rerank anchor and (2) apply small deterministic score
nudges on RAG rows so large-cabin / modern / hotel-vibe evidence rises without replacing
the cross-encoder when enabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class LuxuryEscalationProfile:
    active: bool = False
    """Flavor flags (multiple may be true)."""
    escalate_general: bool = False  # nicer, bigger, step up, wow
    hotel_vibe: bool = False
    private_airline: bool = False
    modern_interior: bool = False
    """Human-readable reasons for logging / data_used."""
    matched_phrases: List[str] = field(default_factory=list)
    """Appended to rerank anchor (BGE query)."""
    rerank_anchor_suffix: str = ""
    """Substring hits in chunk text → score += weight * boost_weight."""
    boost_terms: List[str] = field(default_factory=list)
    """Substring hits → score -= penalty_weight."""
    penalty_terms: List[str] = field(default_factory=list)

    def asdict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "escalate_general": self.escalate_general,
            "hotel_vibe": self.hotel_vibe,
            "private_airline": self.private_airline,
            "modern_interior": self.modern_interior,
            "matched_phrases": list(self.matched_phrases),
            "rerank_anchor_suffix": self.rerank_anchor_suffix,
            "boost_terms": list(self.boost_terms),
            "penalty_terms": list(self.penalty_terms),
        }


_GENERAL_RE = re.compile(
    r"\b("
    r"something\s+nicer|something\s+better|nicer\b|bigger(\s+cabin)?|"
    r"step\s+up|one\s+tier\s+up|more\s+premium|wow\s+factor|visual\s+wow|"
    r"even\s+more\s+luxury|top\s+tier\s+cabin"
    r")\b",
    re.I,
)
_HOTEL_RE = re.compile(
    r"\b("
    r"like\s+a\s+hotel|hotel\s+feel|hotel\s+vibe|luxury\s+hotel|"
    r"spa\s+bath|cream\s+interior|warm\s+lighting|suite\s+feel|"
    r"bedroom\s+layout|divan|berth(?:ing)?"
    r")\b",
    re.I,
)
_AIRLINE_RE = re.compile(
    r"\b("
    r"private\s+airline|airline\s+feel|airline\s+cabin|first\s+class\s+cabin|"
    r"widebody|wide\s+body|flight\s+attendant\s+layout|large\s+galley|lounge\s+zone"
    r")\b",
    re.I,
)
_MODERN_RE = re.compile(
    r"\b("
    r"more\s+modern|contemporary|newer\s+interior|cutting\s+edge\s+cabin|"
    r"latest\s+cabin|state\s+of\s+the\s+art\s+interior"
    r")\b",
    re.I,
)


def _thread_blob(history: Optional[Sequence[Dict[str, str]]], max_chars: int = 1600) -> str:
    parts: List[str] = []
    for h in (history or [])[-12:]:
        if not isinstance(h, dict):
            continue
        if str(h.get("role") or "").strip().lower() != "user":
            continue
        c = str(h.get("content") or "").strip()
        if c:
            parts.append(c)
    blob = " ".join(parts).strip()
    if len(blob) > max_chars:
        blob = blob[-max_chars:]
    return blob


def interpret_luxury_escalation(
    query: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> LuxuryEscalationProfile:
    q = (query or "").strip()
    blob = f"{q} {_thread_blob(history)}".strip()
    low = blob.lower()
    prof = LuxuryEscalationProfile()
    phrases: List[str] = []

    if _GENERAL_RE.search(low):
        prof.escalate_general = True
        phrases.append("general_escalation")
    if _HOTEL_RE.search(low):
        prof.hotel_vibe = True
        phrases.append("hotel_vibe")
    if _AIRLINE_RE.search(low):
        prof.private_airline = True
        phrases.append("private_airline")
    if _MODERN_RE.search(low):
        prof.modern_interior = True
        phrases.append("modern_interior")

    if not phrases:
        return prof

    prof.active = True
    prof.matched_phrases = phrases

    anchor_bits: List[str] = []
    boosts: List[str] = []

    if prof.escalate_general:
        anchor_bits.append(
            "large cabin ultra long range wide cabin premium interior high finish stand-up cabin"
        )
        boosts += [
            "global 7500",
            "global 8000",
            "g700",
            "g650",
            "g600",
            "g500",
            "falcon 8x",
            "falcon 7x",
            "challenger 650",
            "challenger 350",
            "ultra long range",
            "large cabin",
        ]
    if prof.hotel_vibe:
        anchor_bits.append(
            "warm lighting cream interior wide cabin divan bedroom layout ultra long range flagship cabin"
        )
        boosts += [
            "divan",
            "berth",
            "bedroom",
            "suite",
            "global 7500",
            "g700",
            "falcon 8x",
            "falcon 7x",
            "wide cabin",
            "cabin width",
        ]
    if prof.private_airline:
        anchor_bits.append(
            "wide cabin lounge zone large galley flight attendant layout airline style interior"
        )
        boosts += [
            "galley",
            "lounge",
            "divan",
            "global 7500",
            "g700",
            "787",
            "wide body",
        ]
    if prof.modern_interior:
        anchor_bits.append(
            "Gulfstream G500 G600 Falcon 8X Global 7500 Praetor 600 modern cabin newer interior"
        )
        boosts += [
            "g500",
            "g600",
            "falcon 8x",
            "global 7500",
            "praetor 600",
            "challenger 350",
            "symmetry flight deck",
            "pulse cabin",
        ]

    # Dedupe boosts preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for t in boosts:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        deduped.append(t)
    prof.boost_terms = deduped[:40]

    prof.penalty_terms = [
        "hawker",
        "beechjet",
        "learjet 25",
        "learjet 35",
        "lear 25",
        "lear 35",
        "learjet 31",
        "citation ii",
        "citation iii",
        "citation v",
        "citation sii",
        "early citation",
    ]

    prof.rerank_anchor_suffix = " ".join(anchor_bits).strip()[:520]
    return prof


def luxury_rerank_anchor(query: str, profile: LuxuryEscalationProfile) -> str:
    q = (query or "").strip()
    if not profile.active or not profile.rerank_anchor_suffix:
        return q
    return f"{q}\n\n[retrieval bias: {profile.rerank_anchor_suffix}]".strip()


def apply_luxury_escalation_score_adjustments(
    results: List[Dict[str, Any]],
    profile: LuxuryEscalationProfile,
    *,
    boost_weight: float = 0.038,
    penalty_weight: float = 0.065,
    max_total_boost: float = 0.11,
) -> List[Dict[str, Any]]:
    """Nudge ``score`` / ``rerank_score`` within same tier ordering (see structured-first sort)."""
    if not results or not profile.active:
        return results

    def _text(r: Dict[str, Any]) -> str:
        return (r.get("full_context") or r.get("chunk_text") or "").lower()

    out: List[Dict[str, Any]] = []
    for r in results:
        row = dict(r)
        t = _text(row)
        delta = 0.0
        for term in profile.boost_terms:
            if term.lower() in t:
                delta += boost_weight
        for term in profile.penalty_terms:
            if term.lower() in t:
                delta -= penalty_weight
        delta = min(max_total_boost, delta)

        base_keys = ("rerank_score", "score", "pinecone_score")
        base = None
        for k in base_keys:
            v = row.get(k)
            if v is not None:
                try:
                    base = float(v)
                    break
                except (TypeError, ValueError):
                    continue
        if base is None:
            base = 0.0
        new_sc = max(-1.0, min(2.0, base + delta))
        row["score"] = new_sc
        if row.get("rerank_score") is not None:
            row["rerank_score"] = new_sc
        out.append(row)
    return out
