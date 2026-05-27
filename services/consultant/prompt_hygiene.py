"""
Prompt hygiene — contamination scoring and bleed detection before containment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from services.consultant.template_suppression import fallback_contamination_score


@dataclass
class HygieneReport:
    contamination_score: float
    bleed_detected: bool
    repeated_phrases: List[str]
    stale_fragment_count: int
    actions_taken: List[str]

    def to_dict(self) -> Dict:
        return {
            "contamination_score": self.contamination_score,
            "bleed_detected": self.bleed_detected,
            "repeated_phrases": list(self.repeated_phrases),
            "stale_fragment_count": self.stale_fragment_count,
            "actions_taken": list(self.actions_taken),
        }


_STALE_RETRIEVAL_FRAGMENTS: Tuple[re.Pattern, ...] = (
    re.compile(r"\[AUTHORITATIVE\b", re.I),
    re.compile(r"\bFOR USER REPLY\b", re.I),
    re.compile(r"\bphlydata_aircraft\b", re.I),
    re.compile(r"\bpinecone\b", re.I),
    re.compile(r"\bvector\s+search\b", re.I),
)

_REPEATED_PHRASE_SEEDS: Tuple[str, ...] = (
    "assuming 6-8 passengers",
    "assuming 6–8 passengers",
    "here are a few realistic fits",
    "bottom line:",
    "consultant insight:",
    "start with challenger 350",
    "typical business-use constraints",
    "based on typical aircraft performance data",
    "credible alternate if priorities shift",
    "worth keeping on the list",
    "how you're actually using the airplane",
    "frame the mission before anchoring",
    "unless priorities shift",
)


def _count_phrase_repeats(text: str, phrase: str) -> int:
    tl = (text or "").lower()
    p = phrase.lower().replace("–", "-")
    return tl.count(p)


def detect_repeated_phrases(text: str, *, min_count: int = 2) -> List[str]:
    found: List[str] = []
    for seed in _REPEATED_PHRASE_SEEDS:
        if _count_phrase_repeats(text, seed) >= min_count:
            found.append(seed)
    return found


def score_prompt_contamination(text: str) -> float:
    """Aggregate contamination: fallback templates + retrieval bleed + phrase repeats."""
    base = fallback_contamination_score(text)
    bleed = 0.0
    for pat in _STALE_RETRIEVAL_FRAGMENTS:
        if pat.search(text or ""):
            bleed += 0.2
    bleed = min(1.0, bleed)
    repeats = detect_repeated_phrases(text)
    rep_score = min(0.5, len(repeats) * 0.15)
    return min(1.0, base * 0.55 + bleed * 0.25 + rep_score)


def apply_prompt_hygiene(
    text: str,
    *,
    prior_answer: str = "",
    history: Optional[List[Dict[str, str]]] = None,
    turn_seed: str = "",
) -> Tuple[str, HygieneReport]:
    """
    Scrub repeated phrases and stale retrieval carryover; return cleaned text + report.
    """
    s = (text or "").strip()
    actions: List[str] = []
    try:
        from rag.pinpoint_answer import strip_advisory_boilerplate

        cleaned = strip_advisory_boilerplate(s)
        if cleaned != s:
            s = cleaned
            actions.append("stripped_stock_advisory_fallback")
    except Exception:
        pass
    stale = 0
    for pat in _STALE_RETRIEVAL_FRAGMENTS:
        if pat.search(s):
            s = pat.sub("", s).strip()
            stale += 1
            actions.append("stripped_retrieval_fragment")

    for phrase in detect_repeated_phrases(s):
        # Keep first occurrence; drop subsequent blocks containing phrase
        parts = re.split(r"\n\s*\n", s)
        kept = []
        seen_phrase = False
        for part in parts:
            if phrase.lower().replace("–", "-") in part.lower().replace("–", "-"):
                if seen_phrase:
                    actions.append(f"suppressed_repeat:{phrase[:24]}")
                    continue
                seen_phrase = True
            kept.append(part)
        s = "\n\n".join(kept).strip()

    if prior_answer and _count_phrase_repeats(s, "assuming 6") >= 1:
        if _count_phrase_repeats(prior_answer, "assuming 6") >= 1:
            s = re.sub(
                r"\n?\s*Assuming\s+6[–-]8\s+passengers[\s\S]*?(?=\n\s*(?:For |On |I would |\Z))",
                "",
                s,
                flags=re.I,
            ).strip()
            actions.append("dropped_cross_turn_assuming_block")

    try:
        from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

        s, phrase_report = apply_phrase_repetition_guard(
            s,
            history=history,
            prior_answer=prior_answer,
            turn_seed=turn_seed,
        )
        if phrase_report.phrases_varied:
            actions.append("phrase_repetition_variation")
        if phrase_report.phrases_stripped:
            actions.append("phrase_repetition_strip")
        if phrase_report.actions_taken:
            actions.extend(phrase_report.actions_taken)
    except Exception:
        pass

    report = HygieneReport(
        contamination_score=score_prompt_contamination(s),
        bleed_detected=stale > 0,
        repeated_phrases=detect_repeated_phrases(s),
        stale_fragment_count=stale,
        actions_taken=actions,
    )
    return s, report
