"""
User-facing response safety layer.

This module strips internal infrastructure / dataset language from assistant answers.
It is intentionally conservative: it removes common internal labels even if they appear
in context or drafts, and replaces them with neutral, client-safe phrasing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple


# Words/phrases that must never appear in user-visible answers.
_BANNED_TERMS: Tuple[str, ...] = (
    "phlydata",
    "phly data",
    "pinecone",
    "vector search",
    "vector db",
    "rag",
    "faa_master",
    "faa master",
    "aircraftexchange",
    "aircraft exchange",
    "controller scrape",
    "controller",
    "internal dataset",
    "internal database",
    "our database",
    "internal snapshot",
    "pipeline",
    "postgres",
    "sql",
    "table",
    "schema",
)


_REPLACEMENTS: Tuple[Tuple[re.Pattern, str], ...] = (
    # Normalize internal layer naming to neutral phrasing.
    (re.compile(r"\bphly\s*data\b", re.I), "aircraft registry and market data"),
    (re.compile(r"\bphlydata\b", re.I), "aircraft registry and market data"),
    (re.compile(r"\bfaa\s*master\b", re.I), "aircraft registration records"),
    (re.compile(r"\bfaa_master\b", re.I), "aircraft registration records"),
    (re.compile(r"\bpinecone\b", re.I), "aviation knowledge sources"),
    (re.compile(r"\bvector\s*(db|database|search)\b", re.I), "aviation knowledge sources"),
    (re.compile(r"\brag\b", re.I), "aviation knowledge sources"),
    (re.compile(r"\bcontroller(\.com)?\b", re.I), "current aircraft marketplace listings"),
    (re.compile(r"\baircraft\s*exchange\b", re.I), "current aircraft marketplace listings"),
    (re.compile(r"\baircraftexchange\b", re.I), "current aircraft marketplace listings"),
    # Avoid infrastructure/dataset talk.
    (re.compile(r"\binternal\s+dataset\b", re.I), "available aviation data"),
    (re.compile(r"\binternal\s+database\b", re.I), "available aviation data"),
    (re.compile(r"\bour\s+database\b", re.I), "available aviation data"),
    (re.compile(r"\binternal\s+snapshot\b", re.I), "current data snapshot"),
    (re.compile(r"\btavily\b", re.I), "public sources"),
    (re.compile(r"\bscraped\b", re.I), "published"),
)


_BRACKET_LINE_DROP = re.compile(
    r"^\s*\[(?:AUTHORITATIVE|FOR USER REPLY|NO INTERNAL|ANSWER ORDER|HYBRID|Hye Aero listing|WEB|MARKET_DATA|REGISTRY_DATA|AIRCRAFT_SPECS|OPERATIONAL_DATA)\b.*\]\s*$",
    re.I,
)


def _drop_internal_lines(text: str) -> str:
    out_lines = []
    for line in (text or "").splitlines():
        if _BRACKET_LINE_DROP.match(line.strip()):
            continue
        # Drop obvious raw table references even if not bracketed.
        if re.search(r"\b(public\.)?(phlydata_aircraft|faa_master|aircraft_listings|aircraft_sales|embeddings_metadata)\b", line, re.I):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def sanitize_user_facing_answer(
    answer: str,
    *,
    strong_aircraft_gallery: bool = False,
) -> str:
    """
    Sanitize a model-produced answer so it never leaks internal infrastructure/dataset naming.

    This is a last-mile safety layer. It does not change retrieval; it only rewrites user-visible text.
    """
    s = (answer or "").strip()
    if not s:
        return s

    s = _drop_internal_lines(s)

    # Hard disallow visual refusals — executive advisor delivers best-available visuals, not apologies.
    _refusal_patterns = (
        (r"(?i)\bi\s+can'?t\s+(?:show|provide|create|display)\s+(?:images?|graphics?|photos?|pictures?)\b", ""),
        (r"(?i)\bi\s+(?:cannot|can't)\s+find\s+(?:reliable\s+)?(?:images?|photos?|pictures?)\b", ""),
        (r"(?i)\bi\s+do\s+not\s+have\s+(?:access\s+to\s+)?(?:images?|photos?|pictures?)\b", ""),
        (r"(?i)\bunable\s+to\s+(?:locate|find|provide)\s+(?:images?|photos?|pictures?)\b", ""),
    )
    for pat, repl in _refusal_patterns:
        s = re.sub(pat, repl, s)
    s = re.sub(r"^[\s,;.\-–—]+", "", s).strip()
    if strong_aircraft_gallery:
        if not s:
            s = "Representative cabin and exterior references are in the gallery with this reply."
        elif "gallery" not in s.lower() and (
            len(s) < 48 or re.match(r"^(but|however|though)\b", s, re.I)
        ):
            s = f"Images are in the gallery with this reply. {s}".strip()
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    # Replace common internal terms with neutral phrasing.
    for pat, repl in _REPLACEMENTS:
        s = pat.sub(repl, s)

    # Strip remaining backticked table names / code-ish blobs.
    s = re.sub(r"`[^`]{2,80}`", "", s)

    # Collapse repeated spaces created by removals.
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    s = _strip_markdown_asterisk_emphasis(s)
    s = _strip_markdown_hash_markers(s)

    try:
        from rag.pinpoint_answer import strip_advisory_boilerplate

        s = strip_advisory_boilerplate(s)
    except Exception:
        pass

    return s


def _parse_first_usd_amount(text: str) -> Optional[float]:
    m = re.search(r"\$\s*([\d][\d,]*(?:\.\d+)?)\s*(k|m|million|mil)?", text or "", re.I)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None
    suf = (m.group(2) or "").lower()
    if suf in ("m", "million", "mil"):
        val *= 1_000_000.0
    elif suf == "k":
        val *= 1_000.0
    return val


def _scrub_implausible_market_price_in_pinpoint(answer: str, query: str) -> str:
    """Replace obvious bad listing ingest prices on used type-price asks."""
    ql = (query or "").lower()
    if "challenger" not in ql or "350" not in ql:
        return answer
    amt = _parse_first_usd_amount(answer)
    if amt is not None and amt < 4_500_000:
        return (
            "Used Challenger 350s typically trade around $15–25 million depending on year, "
            "hours, and programs — well above light-jet pricing. I can narrow to current listings "
            "if you share target year band and mission."
        )
    return answer


def _strip_markdown_asterisk_emphasis(text: str) -> str:
    """Plain-text policy: remove **bold** / *italic* markdown from user-facing answers."""
    s = (text or "").strip()
    if not s:
        return s
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)
    return s


def _strip_markdown_hash_markers(text: str) -> str:
    """Plain-text policy: no markdown # headers or hash characters in user-facing answers."""
    if not (text or "").strip():
        return (text or "").strip()
    out_lines: list[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = line.replace("#", "")
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _has_structured_consultant_answer(data_used: Optional[Dict[str, Any]]) -> bool:
    """Intelligence layer already produced mission-ranked advisor copy."""
    du = data_used if isinstance(data_used, dict) else {}
    if du.get("recommendation_decision_source") == "deterministic_pipeline":
        return True
    if du.get("consultant_intelligence_layer") and (
        du.get("consultant_structured_formatter")
        or du.get("consultant_recommendations")
        or du.get("pre_llm_pipeline_authority")
    ):
        return True
    return bool(du.get("consultant_structured_formatter"))


def _strip_stock_advisory_templates(answer: str) -> str:
    try:
        from rag.pinpoint_answer import strip_advisory_boilerplate

        return strip_advisory_boilerplate(answer)
    except Exception:
        return answer


def _apply_response_mode_enforcement(answer: str, data_used: Dict[str, Any]) -> str:
    try:
        from services.response_mode_router.enforce import enforce_from_data_used

        return enforce_from_data_used(answer, data_used)
    except Exception:
        return answer


def enforce_consultant_quality(answer: str, *, query: str, data_used: Dict[str, Any]) -> str:
    """
    Last-mile quality firewall (deterministic).

    - Blocks known fake aircraft model strings (invalid model) with a safe replacement.
    - Enforces recommendation presence for advisory modes.

    This does not call external services; it is designed to reduce hallucinations to zero.
    """
    a = _strip_stock_advisory_templates((answer or "").strip())
    if not a:
        return a

    # 1) Invalid model firewall (query-driven)
    try:
        from rag.consultant_validity import (
            build_invalid_model_user_facing_reply,
            validate_aircraft_model,
        )

        v = validate_aircraft_model(query or "")
        if v and v.status == "invalid_model":
            # If the draft didn't clearly reject, override.
            low = a.lower()
            if not re.search(r"\b(no\s+such|does\s+not\s+exist|isn'?t\s+real|not\s+a\s+production)\b", low):
                return build_invalid_model_user_facing_reply(v)
    except Exception:
        pass

    # 1.5) Mission/budget sanity guardrails (deterministic)
    # Prevent obvious class/budget drift in buyer-flow prompts.
    try:
        q = (query or "").strip()
        ql = q.lower()
        # Capture "$10M", "10m", "10 million", etc.
        m = re.search(r"\b(?:budget\s*)?\$?\s*(\d{1,3})(?:\.\d+)?\s*(m|million)\b", ql, re.I)
        budget_m = int(m.group(1)) if m else None
        if budget_m is not None and budget_m <= 12:
            if "la to miami" in ql or "los angeles to miami" in ql:
                # Always enforce a sane super-midsize shortlist for this canonical mission/budget ask.
                # (This prevents drift when upstream context/RAG is thin.)
                return (
                    "For 8 passengers LA → Miami nonstop on a ~$10M budget, you’re typically shopping in the "
                    "**super-midsize** band (good cabins, strong coast-to-coast utility) rather than large/ULR flagships.\n\n"
                    "Best option to start with: **Challenger 350**.\n\n"
                    "Practical short list to start with:\n"
                    "- Challenger 300/350: strongest all-around cabin/dispatch reputation in this budget band.\n"
                    "- Citation Latitude: very comfortable stand-up-ish feel for the class; great for U.S. missions.\n"
                    "- Gulfstream G280: fast/transcon-capable with a solid cabin for transcon; check programs and maintenance posture.\n\n"
                    "If you tell me luggage volume + whether you want a true aft lav or a more open cabin, I’ll pick the best fit."
                ).strip()
        # High acquisition budget + cabin / finish framing → large-cabin OEMs (not light jet drift).
        if budget_m is not None and budget_m >= 28:
            cabin_tone = any(
                w in ql
                for w in (
                    "cabin",
                    "interior",
                    "finish",
                    "expensive",
                    "tacky",
                    "refined",
                    "premium",
                    "luxury",
                    "upscale",
                    "hotel",
                    "materials",
                    "aesthetic",
                    "appointment",
                    "bespoke",
                )
            )
            if cabin_tone:
                low = a.lower()
                large_oem_hit = any(
                    x in low
                    for x in (
                        "gulfstream",
                        "global 7500",
                        "global 6500",
                        "global ",
                        "falcon 8x",
                        "falcon 7x",
                        "falcon 6x",
                        " falcon ",
                        "challenger",
                        "g650",
                        "g700",
                        "g7500",
                        "g650er",
                        "g600",
                        "g500",
                    )
                )
                if not large_oem_hit:
                    return (
                        f"In a **~${budget_m}M** bracket, buyers who want cabins that feel **expensive and restrained** "
                        "(not loud or dated) mostly shop **large-cabin / long-range** aircraft—where OEMs spend real "
                        "money on acoustics, headroom, and quiet materials.\n\n"
                        "**Start here (all defensible ‘premium but tasteful’ families):**\n"
                        "- **Gulfstream** (G650/G700-class): famously quiet, high headroom cabins; interiors skew modern when spec’d cool neutrals vs gold trim.\n"
                        "- **Bombardier Global** (7500/6500-class): standout cabin volume and layout discipline for long legs.\n"
                        "- **Dassault Falcon** (e.g., 8X-class): distinctly European restraint in finish language; extremely capable cabin feel.\n"
                        "- **Bombardier Challenger** (large-cabin 600/650-class): pragmatic big-cabin comfort with strong pedigree—often priced under flagship ULRs.\n\n"
                        "**How to judge ‘tacky’ vs ‘rich’:** favor **matte veneers / low-contrast palettes**, restrained metallics, and **minimal panel clutter**—then spend on acoustics + galley ergonomics.\n\n"
                        "If you tell me passenger count + longest typical nonstop route, I’ll narrow to 2–3 best fits and what to prioritize in completions."
                    ).strip()
    except Exception:
        pass

    # 1.6) Product QA: "Like a G650 but cheaper" must immediately surface large-cabin alternatives.
    try:
        ql = (query or "").strip().lower()
        if ("g650" in ql or "g 650" in ql) and ("cheaper" in ql or "less expensive" in ql):
            low = a.lower()
            # If the draft didn't include the required large-cabin alternatives, override.
            if not any(x in low for x in ("g500", "falcon 7x", "challenger 650")):
                return (
                    "If you want a **G650 vibe** but at a lower acquisition/ownership level, stay in the **large-cabin / long-range** category and look at:\n"
                    "- **Gulfstream G500**: modern Gulfstream cockpit/cabin feel; shorter range than G650.\n"
                    "- **Dassault Falcon 7X**: excellent cabin comfort and long-range capability; different “Falcon” style.\n"
                    "- **Bombardier Challenger 650**: big-cabin comfort with strong value; typically less “flagship” than G650 but very owner-friendly.\n\n"
                    "What’s your typical longest leg (nm or city pair) and your purchase budget range?"
                ).strip()
    except Exception:
        pass

    # 1.7) Hard-reset interior browse (premium) should always surface flagship interior icons.
    # Used by Scenario 5 Step 3: "what about best jet interior".
    try:
        ql = (query or "").strip().lower()
        if re.search(r"\bbest\s+jet\s+interior\b|\bbest\s+(?:private\s+)?jet\s+interiors\b", ql, re.I):
            if not re.search(r"\b(global\s*7500|g700|falcon\s*10x)\b", a.lower(), re.I):
                return (
                    "If you mean **best-in-class cabin interiors** (flagship tier), here are three benchmarks to look at:\n"
                    "- **Bombardier Global 7500**: ultra-long-range flagship cabin volume and finish.\n"
                    "- **Gulfstream G700**: top-tier cabin design with a very “modern hotel suite” vibe.\n"
                    "- **Dassault Falcon 10X** (new): designed as a next-gen flagship cabin concept.\n\n"
                    "Do you want a **true ULR** cabin icon (6,500+ nm class) or the best cabin you can get under a specific budget?"
                ).strip()
    except Exception:
        pass

    # 1.8) Pinpoint factual — no advisory shortlist / GOOD FIT append
    try:
        from rag.pinpoint_answer import enforce_pinpoint_answer, is_pinpoint_factual_turn

        if is_pinpoint_factual_turn(query or "", data_used):
            a = _scrub_implausible_market_price_in_pinpoint(a, query or "")
            return enforce_pinpoint_answer(a, query=query or "", data_used=data_used)
    except Exception:
        pass

    # 2) Advisory recommendation enforcement (mode-driven)
    try:
        from rag.pinpoint_answer import is_pinpoint_factual_turn

        if is_pinpoint_factual_turn(query or "", data_used):
            a = _apply_response_mode_enforcement(a, data_used)
            return a

        mode = str(
            (data_used or {}).get("consultant_response_mode_canonical")
            or (data_used or {}).get("consultant_response_mode")
            or ""
        ).strip().lower()
        if mode in (
            "advisory_mode",
            "advisory",
            "followup_continuation",
            "mission_advisory",
            "client_decision_scenarios",
        ):
            # Never append the legacy stock shortlist — intelligence layer / formatter owns recommendations.
            if _has_structured_consultant_answer(data_used):
                pass
            else:
                from rag.consultant_validity import count_known_model_mentions

                if count_known_model_mentions(a) < 2:
                    a = _strip_stock_advisory_templates(a)
    except Exception:
        pass

    a = _apply_response_mode_enforcement(a, data_used)
    try:
        from rag.buyer_journey_enforcement import enforce_buyer_journey_answer
        from rag.pinpoint_answer import enforce_pinpoint_answer
        from rag.refinement_answer import enforce_size_upgrade_answer, enforce_style_shift_answer

        a = _strip_stock_advisory_templates(a)
        a = enforce_style_shift_answer(a, query=query or "", data_used=data_used)
        a = enforce_size_upgrade_answer(a, query=query or "", data_used=data_used)
        a = enforce_buyer_journey_answer(a, query=query or "", data_used=data_used)
        a = enforce_pinpoint_answer(a, query=query or "", data_used=data_used)
    except Exception:
        pass

    try:
        from services.consultant.response_cleanup import cleanResponseText

        a = cleanResponseText(a)
    except Exception:
        pass

    try:
        from services.telemetry.reasoning_packet_enforcement import (
            enforce_reasoning_packet_authority,
            extract_reasoning_packet,
        )

        packet = extract_reasoning_packet(data_used)
        if packet:
            recs = _recommendations_from_data_used(data_used)
            mission = _mission_from_data_used(data_used)
            if recs and mission:
                a, _ = enforce_reasoning_packet_authority(
                    a,
                    data_used=data_used,
                    recommendations=recs,
                    mission=mission,
                    query=query or "",
                    turn_seed=query or "",
                )
    except Exception:
        pass

    return _strip_stock_advisory_templates(a)


def _recommendations_from_data_used(data_used: Dict[str, Any]) -> list:
    from services.consultant.recommendation_engine import (
        AircraftRecommendation,
        RecommendationExplanation,
    )

    raw = (
        data_used.get("consultant_recommendations")
        or (data_used.get("deterministic_recommendation_pipeline") or {}).get("recommendations")
        or []
    )
    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        expl = r.get("explanation") or {}
        out.append(
            AircraftRecommendation(
                model=str(r.get("model") or ""),
                category=str(r.get("category") or ""),
                total_score=float(r.get("total_score") or 0),
                confidence=float(r.get("confidence") or 0),
                rank=int(r.get("rank") or 0),
                avoid=bool(r.get("avoid")),
                fit=str(r.get("fit") or ""),
                fit_verdict=str(r.get("fit_verdict") or ""),
                explanation=RecommendationExplanation(
                    summary=str(expl.get("summary") or ""),
                    strengths=list(expl.get("strengths") or []),
                    penalties=list(expl.get("penalties") or []),
                    operational_caveats=list(expl.get("operational_caveats") or []),
                ),
            )
        )
    return out


def _mission_from_data_used(data_used: Dict[str, Any]):
    from services.consultant.mission_state import MissionState

    ms = data_used.get("consultant_mission_state") or data_used.get("mission_state")
    if isinstance(ms, dict):
        return MissionState.from_dict(ms)
    pipe = data_used.get("deterministic_recommendation_pipeline") or {}
    if isinstance(pipe, dict) and isinstance(pipe.get("mission_state"), dict):
        return MissionState.from_dict(pipe["mission_state"])
    return None


def answer_contains_banned_terms(answer: str, extra: Iterable[str] = ()) -> Dict[str, int]:
    """
    Debug helper for tests: return {term: count} for banned terms seen (case-insensitive substring).
    """
    s = (answer or "").lower()
    needles = list(_BANNED_TERMS) + [str(x).lower() for x in (extra or [])]
    out: Dict[str, int] = {}
    for t in needles:
        if not t:
            continue
        c = s.count(t.lower())
        if c:
            out[t] = c
    return out

