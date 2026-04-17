"""
Heuristic evaluator for consultant responses.

Design goals:
- Deterministic, fast, no external calls.
- Adversarial: detect hallucinations, avoidance, shallow lists, wrong-tail mentions.
- Produces 0–5 scores for 7 dimensions plus critical-failure flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


DIMENSIONS: Tuple[str, ...] = (
    "correctness",
    "reasoning_depth",
    "usefulness",
    "specificity",
    "honesty_no_hallucination",
    "structure_clarity",
    "tone_professionalism",
)


_WEAK_PATTERNS = (
    r"\bcheck\s+(jetphotos|google|bing|online|the\s+web)\b",
    r"\bjust\s+search\b",
    r"\bi\s+can(?:not|'t)\s+(show|provide)\s+(images|photos|pictures)\b",
    r"\bvisit\s+jetphotos\b",
    r"\bnot\s+able\s+to\s+access\s+the\s+internet\b",
)

_LIST_WITHOUT_WHY = (
    r"\b(top|best)\s+\d+\b",
    r"\bhere\s+are\s+\d+\b",
)

_WHY_TOKENS = re.compile(r"\b(because|why|so that|which means|tradeoff|downside|upside|fit|fits|works for)\b", re.I)
_ASSUMPTION_TOKENS = re.compile(r"\b(assuming|if you|if we assume|given that|based on)\b", re.I)

_TAIL_RE = re.compile(r"\bN[0-9]{1,5}[A-Z]{0,2}\b", re.I)

# Common fake / suspicious marketing strings (extend over time).
_FAKE_MODEL_PATTERNS = (
    r"\bfalcon\s*9000\b",
    r"\bg\s*[-.]?\s*6500\b",
    r"\bgulfstream\s+g750\b",
    r"\bglobal\s*10000\b",
)


@dataclass(frozen=True)
class EvalResult:
    score_total: int
    breakdown: Dict[str, int]
    issues: List[str]
    severity: str  # low|medium|critical
    critical_failures: List[str]


def _clamp_0_5(x: int) -> int:
    return 0 if x < 0 else (5 if x > 5 else x)


def _extract_tail_from_query(query: str) -> Optional[str]:
    m = _TAIL_RE.search(query or "")
    if not m:
        return None
    return m.group(0).upper()


def _all_tails(text: str) -> List[str]:
    return [m.group(0).upper() for m in _TAIL_RE.finditer(text or "")]


def _contains_any(patterns: Tuple[str, ...], text: str) -> bool:
    t = text or ""
    return any(re.search(p, t, re.I) for p in patterns)


def _detect_mode_hints(query: str) -> Dict[str, bool]:
    ql = (query or "").lower()
    return {
        "visual": any(k in ql for k in ("show me", "see", "photos", "pictures", "images", "cockpit", "cabin", "interior", "exterior", "walkaround")),
        "comparison": (" vs " in ql) or ("versus" in ql),
        "advisory": any(k in ql for k in ("what should i buy", "recommend", "best jet for", "should i own", "own or charter", "works for", "mission")),
        "strategic": any(k in ql for k in ("own vs", "fractional", "charter", "roi", "hours/year", "cost of ownership", "fixed cost", "variable cost")),
        "tail": bool(_extract_tail_from_query(query)),
    }


def evaluate_response(query: str, response: str) -> Dict[str, Any]:
    """
    Main entry point. Returns a dict payload:
    {
      score_total: X/35,
      breakdown: {...},
      issues: [...],
      severity: "low"|"medium"|"critical",
      critical_failures: [...]
    }
    """
    q = query or ""
    r = response or ""
    rl = r.lower()

    issues: List[str] = []
    critical: List[str] = []

    hints = _detect_mode_hints(q)
    q_tail = _extract_tail_from_query(q)
    resp_tails = _all_tails(r)

    # --- Critical failure detection ---
    if q_tail:
        if resp_tails and any(t != q_tail for t in resp_tails):
            critical.append(f"tail_mismatch: query={q_tail} response_mentions={sorted(set(resp_tails))}")
        if _contains_any(_WEAK_PATTERNS, r):
            issues.append("weak_visual_or_tail_deferral")
    if _contains_any(_FAKE_MODEL_PATTERNS, q) and not re.search(r"\b(no\s+such|does\s+not\s+exist|isn'?t\s+real|not\s+a\s+production)\b", rl):
        critical.append("hallucinated_fake_model_treated_as_real")

    # Visual correctness heuristic: for "show me N000ZZZ" type traps, refusing with a precise message is OK.
    if q_tail and q_tail.upper() == "N000ZZZ":
        if "no verified" not in rl and "no confirmed" not in rl and "can't verify" not in rl:
            issues.append("n000zzz_should_acknowledge_no_verified_images")

    # --- Dimension scoring (0–5) ---
    breakdown: Dict[str, int] = {k: 3 for k in DIMENSIONS}  # start neutral

    # Honesty / hallucination: penalize false certainty signals.
    if any(x in rl for x in ("definitely", "confirmed", "verified")) and not any(x in rl for x in ("if", "assuming", "based on", "from the brief")):
        breakdown["honesty_no_hallucination"] -= 1
        issues.append("overconfident_language_without_grounding")
    if "i don't know" in rl or "no verified data" in rl or "can't verify" in rl:
        breakdown["honesty_no_hallucination"] += 1

    # Correctness: heuristic only. Tail mismatch is critical.
    if critical:
        breakdown["correctness"] = 0
    else:
        # obvious fake model requests should be rejected
        if _contains_any(_FAKE_MODEL_PATTERNS, q):
            if re.search(r"\b(no\s+such|does\s+not\s+exist|isn'?t\s+real|not\s+a\s+production)\b", rl):
                breakdown["correctness"] += 2
            else:
                breakdown["correctness"] -= 2

    # Reasoning depth: look for "why" language and tradeoffs.
    if _WHY_TOKENS.search(r):
        breakdown["reasoning_depth"] += 1
    if "tradeoff" in rl or "downside" in rl or "upside" in rl:
        breakdown["reasoning_depth"] += 1
    if hints["comparison"] and "when to choose" in rl:
        breakdown["reasoning_depth"] += 1

    # Usefulness: must give a stance/verdict for comparisons/advisory, not just facts.
    if hints["comparison"] and not re.search(r"\b(verdict|i(?:'d)?\s+pick|better\s+for|wins\s+if)\b", rl):
        breakdown["usefulness"] -= 1
        issues.append("comparison_missing_verdict")
    if hints["advisory"] and not re.search(r"\b(i(?:'d)?\s+look at|shortlist|recommend)\b", rl):
        breakdown["usefulness"] -= 1
        issues.append("advisory_missing_recommendations")
    if hints["advisory"] and "bottom line" not in rl:
        breakdown["structure_clarity"] -= 1
        issues.append("missing_bottom_line")

    # Specificity: penalize generic filler.
    if len(r.strip()) < 80:
        breakdown["specificity"] -= 1
    if re.search(r"\b(it depends|hard to say|varies widely)\b", rl) and not _ASSUMPTION_TOKENS.search(r):
        breakdown["specificity"] -= 1
        issues.append("generic_hedging_without_assumptions")

    # Structure clarity: look for bullets / headings / consultant insight where appropriate.
    has_bullets = ("\n-" in r) or ("\n•" in r)
    if hints["comparison"] and not has_bullets:
        breakdown["structure_clarity"] -= 1
        issues.append("comparison_missing_bullets_structure")
    if hints["advisory"] and "consultant insight" not in rl:
        breakdown["structure_clarity"] -= 1
        issues.append("missing_consultant_insight")

    # Tone: avoid chatbot disclaimers; prefer broker voice.
    if any(x in rl for x in ("as an ai", "i am an ai", "i can't access")):
        breakdown["tone_professionalism"] -= 2
        issues.append("chatbot_disclaimer_tone")
    if "happy to help" in rl and len(r) < 120:
        breakdown["tone_professionalism"] -= 1

    # Weak answer patterns.
    if _contains_any(_WEAK_PATTERNS, r):
        breakdown["usefulness"] -= 2
        breakdown["tone_professionalism"] -= 1
        issues.append("avoidance_or_external_deferral")

    # List-without-why: recommendations must include rationale tokens.
    if hints["advisory"] and _contains_any(_LIST_WITHOUT_WHY, r) and not _WHY_TOKENS.search(r):
        breakdown["reasoning_depth"] -= 2
        breakdown["usefulness"] -= 1
        issues.append("list_without_reasoning")

    # Clamp all to 0–5
    for k in list(breakdown.keys()):
        breakdown[k] = _clamp_0_5(int(breakdown[k]))

    score_total = int(sum(breakdown.values()))

    # Severity
    severity = "low"
    if critical:
        severity = "critical"
    elif score_total <= 18 or len([i for i in issues if "missing" in i or "avoidance" in i]) >= 2:
        severity = "medium"

    out = EvalResult(
        score_total=score_total,
        breakdown=breakdown,
        issues=sorted(set(issues)),
        severity=severity,
        critical_failures=critical,
    )
    return {
        "score_total": f"{out.score_total}/35",
        "breakdown": out.breakdown,
        "issues": out.issues,
        "severity": out.severity,
        "critical_failures": out.critical_failures,
    }

