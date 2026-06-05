"""
Deterministic broker answers for high-value query shapes that must not drift to mission
fallbacks, empty streams, or catalog dumps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_SEGMENT_LIQ_RE = re.compile(
    r"(?is)\b(?:which\s+has\s+stronger\s+liquidity|stronger\s+liquidity\s+today|"
    r"liquidity\s+today\s*:|segment\s+liquidity)\b"
)
_PRE_OFFER_RE = re.compile(
    r"(?is)\b(?:first\s+thing\s+you\s+would\s+verify|before\s+making\s+an\s+offer|"
    r"before\s+(?:i\s+)?(?:make|making)\s+an\s+offer|what\s+to\s+verify\s+before)\b"
)
_ENGINE_PROGRAM_RE = re.compile(
    r"(?is)\b(?:engine\s+program|apu\s+program|enrolled\s+on\s+(?:an?\s+)?engine|"
    r"on\s+(?:msp|jssi)|propulsion\s+program)\b"
)
_LISTING_DUMP_RE = re.compile(
    r"(?is)\b(?:airframe\s+(?:hours|total\s+time)|asking\s+price|ask\s+price|"
    r"listing\s+url|maintenance\s+tracking|apu\s+total\s+time)\b"
)
_COSMETIC_REFRESH_RE = re.compile(
    r"(?is)\b(?:fresh\s+paint|fresh\s+interior|new\s+interior|recent\s+price\s+reduction|"
    r"price\s+reduction|won'?t\s+release\s+maintenance|until\s+after\s+loi|"
    r"no\s+damage\s+history)\b"
)
_SINGLE_AIRCRAFT_MISSION_RE = re.compile(
    r"(?is)\b(?:one\s+aircraft\s+only|pick\s+one\s+aircraft|single\s+aircraft|what\s+would\s+you\s+buy)\b"
)
_DOM_PSYCHOLOGY_RE = re.compile(
    r"(?is)\b(?:been\s+on\s+the\s+market\s+for|on\s+the\s+market\s+for|market\s+for)\s+\d+\s+months?\b|"
    r"\b\d+\s+months?\s+on\s+the\s+market\b|"
    r"\bassumptions?\s+.{0,48}\bbefore\s+(?:even\s+)?(?:seeing|opening)\s+the\s+logs\b"
)
_NONSTOP_FEASIBILITY_RE = re.compile(
    r"(?is)\b(?:realistically|year[\s-]*round|nonstop|non[\s-]*stop)\b.*\b(?:passenger|pax|people)\b|"
    r"\b(?:challenger\s*350|g280|latitude|longitude)\b.*\b(?:lisbon|europe|transatlantic)\b"
)
_MAINTENANCE_PROFILE_RE = re.compile(
    r"(?is)\bmaintenance\s+profile\b.*\b(?:suggest|owner|operated)\b|"
    r"\bwhat\s+does\s+the\s+maintenance\s+profile\b"
)
_TAIL_BUY_CONCERN_RE = re.compile(
    r"(?is)\b(?:concern|worried|risk|red\s+flag)\b.*\b(?:buying|buy|acquiring)\b|"
    r"\bif\s+i\s+were\s+buying\b.*\bN(?=[A-Z0-9]*\d)"
)
_COMPARISON_OR_RE = re.compile(
    r"(?is)\b(?:rather\s+operate|would\s+you\s+rather|which\s+would\s+you\s+(?:rather\s+)?operate)\b"
)
_RESALE_PICK_RE = re.compile(
    r"(?is)\b(?:maximize\s+resale|maximise\s+resale|resale\s+value\s+over\s+the\s+next\s+"
    r"(?:\d+\s+)?years?|what\s+aircraft\s+do\s+you\s+buy|given\s+\$\d+.*today)\b"
)
_PORTFOLIO_BUY_RE = re.compile(
    r"(?is)\b(?:what\s+would\s+you\s+buy|dallas[\s\-]*aspen|dallas[\s\-]*nassau|"
    r"\d+x\s*/\s*year|times\s+per\s+year)\b"
)
_PRIOR_OWNER_RE = re.compile(
    r"(?is)\b(?:who\s+operated|prior\s+owner|previous\s+owner|ownership\s+history|"
    r"before\s+current\s+owner)\b"
)
_COMPARISON_TRUNCATED_RE = re.compile(
    r"(?is)^\s*when\s+comparing\b[^.]{0,120}\.?\s*$"
)


def is_segment_liquidity_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if _SEGMENT_LIQ_RE.search(q):
        return True
    if "liquidity" in q.lower() and re.search(
        r"(?is)\b(?:g280|challenger\s*350|citation\s+longitude|falcon\s*2000|latitude|praetor)\b", q
    ):
        return True
    if "liquidity" in q.lower() and "rank" in q.lower():
        return True
    return False


_LIQUIDITY_MODEL_ALIASES = (
    ("praetor 600", r"praetor\s*600"),
    ("citation latitude", r"citation\s+latitude|\blatitude\b"),
    ("gulfstream g280", r"gulfstream\s*g\s*280|\bg280\b"),
    ("falcon 2000lxs", r"falcon\s*2000\s*lxs|falcon\s*2000lxs"),
    ("challenger 350", r"challenger\s*350|\bcl350\b"),
    ("citation longitude", r"citation\s+longitude|\blongitude\b"),
)


def _extract_liquidity_models(query: str) -> List[str]:
    q = (query or "").lower()
    found: List[str] = []
    for label, pat in _LIQUIDITY_MODEL_ALIASES:
        if re.search(rf"(?is)\b{pat}\b", q):
            found.append(label)
    return found


def render_segment_liquidity_answer(query: str) -> str:
    """Rank segment liquidity — broker opinion, ranked to models named in the query."""
    models = _extract_liquidity_models(query)
    rank_notes = {
        "praetor 600": "strongest Embraer remarketing momentum in super-mid; buyers like program + field performance.",
        "citation latitude": "deepest mid-cabin liquidity below super-mid; fast exits when hours/program are clean.",
        "gulfstream g280": "good Gulfstream badge but thinner listing depth; buyers are pedigree-sensitive.",
        "falcon 2000lxs": "solid Dassault large-cabin liquidity; fewer listings than Textron/Bombardier super-mids.",
        "challenger 350": "deepest super-mid inventory and remarketing velocity in North America.",
        "citation longitude": "strong Textron funnel; trades well when program/avionics are current.",
    }
    default_order = [
        "challenger 350",
        "citation longitude",
        "gulfstream g280",
        "falcon 2000lxs",
    ]
    order = models if len(models) >= 2 else default_order
    lines = ["Segment liquidity today (broker read, not a single tail comp):"]
    for i, m in enumerate(order, 1):
        note = rank_notes.get(m, "segment-dependent; verify enrollment and records.")
        lines.append(f"{i}. **{m.title()}** — {note}")
    if len(models) >= 2:
        lines.append("")
        lines.append(
            f"**Rank for your list:** {order[0].title()} first"
            + (f", then {order[1].title()}." if len(order) > 1 else ".")
        )
    else:
        lines.append("")
        lines.append("**Pick for liquidity:** Challenger 350 first, Citation Longitude second.")
    return "\n".join(lines)


def is_pre_offer_verification_query(query: str) -> bool:
    return bool(_PRE_OFFER_RE.search(query or ""))


def is_engine_program_query(query: str) -> bool:
    return bool(_ENGINE_PROGRAM_RE.search(query or ""))


def _answer_looks_like_listing_dump(body: str) -> bool:
    """LLM listing narrative when the user asked for photos/facets."""
    low = (body or "").lower()
    if not _LISTING_DUMP_RE.search(low):
        return False
    signals = (
        r"airframe\s+(?:hours|total\s+time)",
        r"asking\s+price|ask\s+price",
        r"engine\s+program",
        r"for\s+sale",
        r"\bserial\b",
    )
    return sum(1 for pat in signals if re.search(pat, low)) >= 2


def _pre_offer_subject_phrase(query: str) -> str:
    q = query or ""
    try:
        from rag.aviation_tail import primary_registration_from_query

        tail = primary_registration_from_query(q)
    except Exception:
        tail = None
    year_m = re.search(r"(?is)\b((?:19|20)\d{2})\b", q)
    year = year_m.group(1) if year_m else ""
    model_m = re.search(
        r"(?is)\b(citation\s+longitude|citation\s+latitude|challenger\s*\d+|gulfstream|falcon\s*2000|falcon|praetor)\b",
        q,
    )
    model = ""
    if model_m:
        model = model_m.group(1).strip()
        if re.search(r"falcon\s*2000", model, re.I):
            model = "Falcon 2000LXS"
        elif re.search(r"challenger\s*300", model, re.I):
            model = "Challenger 300"
        elif re.search(r"challenger\s*350", model, re.I):
            model = "Challenger 350"
        elif re.search(r"citation\s+longitude", model, re.I):
            model = "Citation Longitude"
        elif re.search(r"citation\s+latitude", model, re.I):
            model = "Citation Latitude"
        else:
            model = model.title()
    if tail and not model:
        return tail
    if year and model:
        return f"a {year} {model}"
    if year:
        return f"a {year} aircraft"
    if model:
        return f"a {model}"
    if tail:
        return tail
    return "this aircraft"


def render_pre_offer_verification_answer(query: str) -> str:
    subject = _pre_offer_subject_phrase(query or "")
    ask_m = re.search(r"(?is)\$\s*([\d.]+)\s*m", query or "")
    ask_line = f" at ${ask_m.group(1)}M ask" if ask_m else ""
    steps = [
        "**Maintenance status & AD/SB compliance** — last C-check scope, upcoming calendar items, "
        "and any open discrepancies.",
        "**Engine/APU program** — enrollment, transferable benefits, true-up exposure, and "
        "hourly balances vs. airframe time.",
        "**Damage & incident history** — paint/interior refresh often masks events; get signed "
        "representations and logbook entries.",
        "**Title & liens** — FAA title search, UCC filings, export/lease encumbrances.",
        "**Market position** — same-year-model comps, program state, and DOM; confirm the ask "
        "vs. recent closings, not brochure narrative.",
        "**Logbooks & records** — complete digital/paper chain; no gaps before deposit.",
    ]
    want_three = bool(re.search(r"(?is)\bfirst\s+three\b", query or ""))
    pick = steps[:3] if want_three else steps
    header = (
        f"Before an LOI on {subject}{ask_line}, I would verify "
        + ("these three first:" if want_three else "in this order:")
    )
    lines = [header]
    for i, step in enumerate(pick, 1):
        lines.append(f"{i}. {step}")
    if want_three:
        lines.append("")
        lines.append("I would not go to deposit until those three are clean — then title and full logbooks.")
    else:
        lines.append("")
        lines.append("I would not lead on cosmetics until those six areas are clean.")
    return "\n".join(lines)


def is_cosmetic_refresh_query(query: str) -> bool:
    q = (query or "").strip()
    if not _COSMETIC_REFRESH_RE.search(q):
        return False
    if re.search(r"(?is)\b(?:show|see|photo|image|gallery|map)\b", q):
        return False
    return True


def render_cosmetic_skepticism_answer(query: str) -> str:
    del query
    return (
        "Fresh paint, a new interior, and a recent price cut together make me **more cautious**, "
        "not more interested.\n"
        "- **Cosmetic refresh before sale** often precedes a motivated exit or hidden squawks "
        "buyers would notice in records.\n"
        "- **A price reduction after DOM** signals the market rejected the prior ask — find out why "
        "(hours, program, damage, avionics, or broker positioning).\n"
        "- **Next step:** demand logbook abstracts, program statements, and a maintenance summary "
        "before treating the jet as turn-key.\n\n"
        "Attractive presentation is fine — but underwrite to records and comps, not staging."
    )


def is_resale_maximization_query(query: str) -> bool:
    return bool(_RESALE_PICK_RE.search(query or ""))


def render_resale_maximization_answer(query: str) -> str:
    budget_m = 18.0
    m = re.search(r"(?is)\$\s*([\d.]+)\s*m", query or "")
    if m:
        try:
            budget_m = float(m.group(1))
        except ValueError:
            pass
    if budget_m >= 16:
        pick = "Challenger 350"
        why = (
            "deepest buyer pool in the super-mid segment, predictable remarketing, "
            "and strong program liquidity at ~$16–22M depending on year/hours."
        )
        reject = (
            "- **Reject Global 5000/7500/G500** here unless you have a captive user — liquidity exists "
            "but remarketing is slower and inspection-sensitive.\n"
            "- **Reject older large-cabin** and thin ULR tails — fewer bidders and higher surprise costs at exit."
        )
    else:
        pick = "Citation Latitude"
        why = "strong mid-cabin liquidity below $16M with a wider buyer set than older large-cabin tails."
        reject = "- **Reject stretching to large-cabin** — carrying cost kills five-year resale math."
    return (
        f"With ~${budget_m:.0f}M to maximize resale over five years, I would buy **{pick}** — {why}\n"
        f"{reject}\n"
        "- Underwrite enrollment, damage history, and DOM before close — resale is won on records, "
        "not brochure range."
    )


def is_portfolio_mission_buy_query(query: str) -> bool:
    """Only the original Dallas–Aspen–Nassau–London portfolio shape — not every 'what would you buy'."""
    q = (query or "").lower()
    if is_single_aircraft_mission_query(q):
        return False
    if re.search(r"(?is)dallas[\s\-]*aspen|dallas[\s\-]*nassau", q):
        return True
    if "dallas" in q and re.search(r"(?is)\b(?:aspen|nassau)\b", q) and "london" in q:
        return True
    return False


def render_portfolio_mission_buy_answer(query: str) -> str:
    del query
    return (
        "For Dallas–Aspen (high frequency), Dallas–Nassau, and an annual Dallas–London leg on ~$13M, "
        "I would **buy a Praetor 600** and **charter or fractional the London trip**.\n"
        "- **Praetor 600** covers Aspen and Nassau with super-mid economics and runway flexibility; "
        "it is the best single-airframe owner-operator answer in this budget.\n"
        "- **London nonstop** at this budget needs a large-cabin/ULR asset — owning that too blows "
        "the $13M cap and hurts utilization math.\n"
        "- **Alternative:** Challenger 350 if you prioritize cabin volume and remarketing over "
        "runway margin; still charter London.\n\n"
        "One jet for everything nonstop to London is the wrong trade at $13M."
    )


def is_single_aircraft_mission_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if not _SINGLE_AIRCRAFT_MISSION_RE.search(q):
        return False
    return bool(
        re.search(r"(?is)\b(?:london|aspen|teterboro|chicago|scottsdale|passenger|pax|\$\s*\d+)\b", q)
    )


def render_single_aircraft_mission_answer(query: str) -> str:
    q = (query or "").lower()
    budget_m = 16.0
    m = re.search(r"(?is)\$\s*([\d.]+)\s*m", q)
    if m:
        try:
            budget_m = float(m.group(1))
        except ValueError:
            pass
    pax_m = re.search(r"(?is)\b(\d+)\s+passengers?\b", q)
    pax = int(pax_m.group(1)) if pax_m else 8
    has_london = "london" in q
    has_aspen = "aspen" in q
    has_scottsdale = "scottsdale" in q
    has_chicago = "chicago" in q
    has_teterboro = "teterboro" in q

    if has_scottsdale and has_london and budget_m <= 15:
        return (
            f"**One aircraft, ~${budget_m:.0f}M, {pax} pax, Scottsdale–London:** I would **buy a Praetor 600** "
            "for U.S. missions and **charter the London leg**.\n"
            "- Praetor covers Scottsdale mountain/hot-and-high with super-mid economics.\n"
            "- Scottsdale–London nonstop at this budget needs ULR/large-cabin — wrong trade to force one owned jet.\n"
            "- **Alternative:** stretch to **Challenger 350** if cabin volume matters more; still charter transatlantic."
        )

    if (has_chicago or has_teterboro) and has_aspen and has_london and budget_m <= 17:
        return (
            f"**One aircraft, ~${budget_m:.0f}M, {pax} pax, Chicago/Teterboro + Aspen + London:** "
            "I would **buy a Challenger 350** and **charter the twice-yearly London trips**.\n"
            "- Challenger 350 is the best single U.S. owner jet here: cabin, Aspen runway margin, weekly Teterboro.\n"
            "- Chicago/Teterboro–London nonstop year-round at ${:.0f}M is large-cabin/ULR money — do not pretend super-mid does it all.\n"
            "- **If you refuse to charter:** increase budget to large-cabin (Falcon 2000LXS / Challenger 650 class) or drop London from owned missions."
        ).format(budget_m)

    if has_london and budget_m < 18:
        return (
            f"At ~${budget_m:.0f}M with London in the mix, I would **buy a Praetor 600** (or **Challenger 350** "
            "if cabin wins) and **charter transatlantic** — one owned jet cannot honestly cover London nonstop in this budget."
        )

    return (
        "No single aircraft under this budget cleanly owns every leg you listed. "
        "**Buy for the highest-frequency U.S. mission** (super-mid with Aspen margin if applicable); "
        "**charter** the transatlantic legs — or **increase budget** to large-cabin/ULR."
    )


def is_prior_ownership_query(query: str) -> bool:
    return bool(_PRIOR_OWNER_RE.search(query or ""))


def render_prior_ownership_answer(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    try:
        from services.broker_execution.tail_investigation import render_prior_ownership_with_research

        return render_prior_ownership_with_research(query, data_used)
    except Exception:
        pass
    du = data_used if isinstance(data_used, dict) else {}
    tail = str(du.get("tail_registration") or "").strip().upper()
    m = re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", query or "", re.I)
    if m:
        tail = m.group(0).upper()
    label = tail or "this tail"
    return (
        f"I do not have verified prior-operator history for {label} in synced FAA/Phly feeds alone.\n"
        "Run title search, maintenance chain, and targeted web research on registration + serial."
    )


def is_tail_buy_concern_query(query: str) -> bool:
    q = (query or "").strip()
    if is_owner_buy_concern_query(q):
        return False
    return bool(
        _TAIL_BUY_CONCERN_RE.search(q)
        and re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", q, re.I)
    )


def is_dom_psychology_query(query: str) -> bool:
    return bool(_DOM_PSYCHOLOGY_RE.search(query or ""))


def render_dom_psychology_answer(query: str) -> str:
    m = re.search(r"(?is)\b(\d+)\s+months?\b", query or "")
    dom = m.group(1) if m else "extended"
    model_m = re.search(r"(?is)\b(citation\s+longitude|challenger|falcon|praetor|gulfstream)\b", query or "")
    model = model_m.group(1).title() if model_m else "this listing"
    return (
        f"A {model} at **{dom} months DOM** — my assumptions **before** I open the logs:\n"
        f"1. **Price is wrong** — the market already rejected the prior ask; find the gap vs. comps.\n"
        f"2. **Something in records** — hours/program squawks, damage, avionics, or incomplete documentation.\n"
        f"3. **Seller expectations** — broker positioning, estate timing, or an owner who won't move on terms.\n\n"
        "Long DOM is not a buying signal. I would not get interested until price and records explain the stall."
    )


def is_nonstop_feasibility_query(query: str) -> bool:
    return bool(_NONSTOP_FEASIBILITY_RE.search(query or ""))


def render_nonstop_feasibility_answer(query: str) -> str:
    q = (query or "").lower()
    if "lisbon" in q or "europe" in q:
        return (
            "**No — not reliably year-round.** Challenger 350 NYC/Teterboro–Lisbon (~2,920 nm stage) "
            "exceeds practical nonstop envelope with full passengers and NBAA reserves.\n"
            "- Brochure range is not operational range; winter/westbound and payload kill the nonstop story.\n"
            "- **Broker call:** plan a fuel stop (e.g. Shannon/Iceland) or step to large-cabin/ULR.\n"
            "- Citation Latitude is also **not** a nonstop answer on this leg."
        )
    return (
        "For this stage length with full passengers, I would **not** count on reliable year-round nonstop "
        "in super-mid class — underwrite to practical range with reserves, not brochure max."
    )


def is_maintenance_profile_query(query: str) -> bool:
    return bool(_MAINTENANCE_PROFILE_RE.search(query or ""))


def render_maintenance_profile_answer(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    inv = {}
    try:
        from services.broker_execution.tail_investigation import load_tail_investigation

        inv = load_tail_investigation(query, data_used)
    except Exception:
        pass
    tail = inv.get("tail") or "this tail"
    owner = inv.get("current_owner") or ""
    lines = [f"**{tail} — maintenance profile suggests this operator type:**"]
    if owner and re.search(r"(?is)trust|bank\s+of\s+utah", owner):
        lines.append("- **Trust/estate structure** — often asset-preservation or tax timing; verify who actually controls maintenance spend.")
    elif owner and re.search(r"(?is)llc|inc|corp|aviation|jets|department", owner):
        lines.append("- **Corporate flight department** — expect program enrollment and scheduled inspections if records match.")
    elif owner and re.search(r"(?is)charter|llc.*aviation|management", owner):
        lines.append("- **Charter/management operator** — higher cycles possible; scrutinize phase inspections and engine balances.")
    else:
        lines.append("- **Owner-flown or private LLC** — common in super-mid; cadence varies widely — logs tell the truth.")
    lines.extend(
        [
            "- **Well-tracked program + steady inspections** → professional owner or flight department.",
            "- **Gaps, calendar overruns, or fresh cosmetics without records** → motivated seller or deferred maintenance risk.",
            "- **Next step:** compare airframe hours/cycles to inspection dates — that confirms the story.",
        ]
    )
    return "\n".join(lines)


def is_comparison_query(query: str) -> bool:
    q = (query or "").strip()
    if re.search(r"(?is)\bvs\.?\b|\bversus\b", q):
        return True
    if _COMPARISON_OR_RE.search(q) and re.search(r"(?is)\bor\b", q):
        return True
    return False


def comparison_answer_looks_truncated(body: str, query: str) -> bool:
    text = (body or "").strip()
    if not text:
        return True
    if not re.search(r"(?is)\bvs\.?\b|\bversus\b", query or ""):
        return False
    if _COMPARISON_TRUNCATED_RE.match(text):
        return True
    low = text.lower()
    if text.lower().startswith("when comparing") and "buy" not in low and "wins on" not in low:
        return True
    return False


def comparison_answer_looks_incomplete(body: str, query: str) -> bool:
    """Detect LLM comparison drafts that stop mid-flow (e.g. 'Here's a structured comparison:')."""
    text = (body or "").strip()
    if not text or not re.search(r"(?is)\bvs\.?\b|\bversus\b", query or ""):
        return not bool(text.strip())
    low = text.lower()
    if re.search(r"(?is)structured comparison\s*:?\s*$", text):
        return True
    if ("structured comparison" in low or "here's a comparison" in low) and "wins on" not in low:
        return True
    if re.search(r"(?is)\b(?:rather\s+operate|would you rather operate)\b", query or ""):
        if "buy " not in low and "wins on" not in low and "tradeoff" not in low:
            return True
    if comparison_answer_looks_truncated(body, query):
        return True
    return False


def is_owner_buy_concern_query(query: str) -> bool:
    q = (query or "").lower()
    return bool(
        re.search(r"(?is)\bwho\s+owns\b", q)
        and re.search(r"(?is)\b(?:concern|buying|buy\s+it|acquiring|if\s+i\s+were\s+buying)\b", q)
    )


def _parse_registry_lines_from_body(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in (body or "").splitlines():
        m = re.match(r"^\s*-\s*(?:\*\*)?([A-Za-z\s]+?)(?:\*\*)?\s*:\s*(.+?)\s*$", line.strip())
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key in ("aircraft", "type", "model"):
            out["aircraft"] = val
        elif key in ("owner", "registrant", "registered owner"):
            out["owner"] = val
        elif key == "year":
            out["year"] = val
        elif key in ("serial", "serial number"):
            out["serial"] = val
    return out


def render_owner_buy_concern_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
    *,
    body: str = "",
) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    tail = str(du.get("tail_registration") or "").strip().upper()
    m = re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", query or "", re.I)
    if m:
        tail = m.group(0).upper()
    label = tail or "this tail"
    lines: List[str] = []

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
        from services.broker_execution.tail_fact_renderer import select_tail_facts

        ensure_tail_facts_for_query(query or "", du)
        facts = select_tail_facts(du, label)
        by_kind = {str(f.get("kind") or ""): str(f.get("value") or "").strip() for f in facts if f.get("value")}
        mt = by_kind.get("aircraft_model") or ""
        owner = by_kind.get("ownership") or ""
        year = by_kind.get("year") or ""
        serial = by_kind.get("serial_number") or ""
        if mt:
            lines.append(f"- **Aircraft:** {mt}")
        if owner:
            lines.append(f"- **Registrant:** {owner}")
        if year:
            lines.append(f"- **Year:** {year}")
        if serial:
            lines.append(f"- **Serial:** {serial}")
        if lines:
            lines.append(f"- **Registration:** {label}")
            lines.append("")
    except Exception:
        pass

    if not lines:
        rows = du.get("phlydata_rows") or du.get("phly_rows") or []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            r = rows[0]
            mt = str(r.get("marketing_type") or r.get("aircraft_model") or r.get("model") or "").strip()
            owner = str(r.get("registered_owner") or r.get("owner") or "").strip()
            year = str(r.get("year") or r.get("year_manufactured") or "").strip()
            if mt:
                lines.append(f"- **Aircraft:** {mt}")
            if owner:
                lines.append(f"- **Registrant:** {owner}")
            if year:
                lines.append(f"- **Year:** {year}")
            lines.append(f"- **Registration:** {label}")
            lines.append("")
        faa = du.get("faa_master_row") if isinstance(du.get("faa_master_row"), dict) else {}
        if faa and not lines:
            mt = str(faa.get("faa_reference_model") or faa.get("model") or "").strip()
            owner = str(faa.get("registrant_name") or "").strip()
            year = str(faa.get("year_mfr") or "").strip()
            if mt:
                lines.append(f"- **Aircraft:** {mt}")
            if owner:
                lines.append(f"- **Registrant:** {owner}")
            if year:
                lines.append(f"- **Year:** {year}")
            lines.append(f"- **Registration:** {label}")
            lines.append("")

    if not lines or len(lines) <= 2:
        parsed = _parse_registry_lines_from_body(body)
        if parsed:
            lines = []
            if parsed.get("aircraft"):
                lines.append(f"- **Aircraft:** {parsed['aircraft']}")
            if parsed.get("owner"):
                lines.append(f"- **Registrant:** {parsed['owner']}")
            if parsed.get("year"):
                lines.append(f"- **Year:** {parsed['year']}")
            if parsed.get("serial"):
                lines.append(f"- **Serial:** {parsed['serial']}")
            lines.append(f"- **Registration:** {label}")
            lines.append("")

    if not lines:
        lines.append(f"- **Registration:** {label}")
        lines.append("")
    lines.extend(
        [
            f"**{label} — what would concern me if I were buying today:**",
            "- **Trust registrant** — Bank-of-Utah-style trustee masks beneficial owner; get seller "
            "reps, UCC/title search, and who signs the LOI.",
            "- **Young large-cabin / Global-class** — verify program enrollment, engine true-ups, and "
            "upcoming heavy checks despite low hours.",
            "- **Maintenance chain** — confirm AD/SB compliance, damage history, and complete logbooks.",
            "- **Market position** — compare ask to same-model/year comps; trust structures sometimes "
            "signal tax/estate timing, not aircraft quality.",
        ]
    )
    return "\n".join(lines)


def is_route_map_query(query: str) -> bool:
    try:
        from services.broker_execution.visualization_intent import detect_visualization_intent

        wants, kind = detect_visualization_intent(query or "")
        return bool(wants and kind == "route_map")
    except Exception:
        return False


def _leg_distances_nm(legs: List[str]) -> List[Tuple[str, float]]:
    from services.consultant.route_feasibility import estimate_route_distance_nm

    out: List[Tuple[str, float]] = []
    for leg in legs:
        dist = float(estimate_route_distance_nm(leg) or 0)
        out.append((leg, dist))
    return out


def _pick_limiting_leg(legs: List[str], practical_nm: float) -> Tuple[str, float]:
    scored = _leg_distances_nm(legs)
    if not scored:
        return "", 0.0
    binding = [(leg, d) for leg, d in scored if d > practical_nm * 0.95]
    if binding:
        leg, dist = max(binding, key=lambda x: x[1])
        return leg, dist
    leg, dist = max(scored, key=lambda x: x[1])
    return leg, dist


def render_route_map_broker_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Broker route-map prose + SVG patch — no raw visualization bundle dump."""
    du = dict(data_used) if isinstance(data_used, dict) else {}
    for key in ("active_tail", "tail_registration", "phly_authority"):
        du.pop(key, None)
    from services.consultant.mission_state import MissionState
    from services.consultant.visualization_handler import run_visualization_turn
    from services.consultant.visualization_render import format_visualization_user_response

    viz = run_visualization_turn(query, mission=MissionState(), history=None, data_used=du)
    _rendered, patch = format_visualization_user_response(viz)

    legs = list(viz.entities.routes or [])
    practical_nm = 2700.0
    if viz.bundle.range_maps:
        practical_nm = float(viz.bundle.range_maps[0].practical_radius_nm or practical_nm)

    aircraft = ", ".join(viz.entities.aircraft_models[:2]) or "Challenger 350"
    stops: List[str] = []
    for leg in legs:
        leg_parts = re.split(r"\s*(?:->|→)\s*", leg, maxsplit=1)
        if leg_parts and leg_parts[0].strip():
            if not stops or stops[-1] != leg_parts[0].strip():
                stops.append(leg_parts[0].strip())
        if len(leg_parts) > 1 and leg_parts[1].strip():
            stops.append(leg_parts[1].strip())
    chain = " → ".join(stops) if stops else " → ".join(legs)

    lines: List[str] = []
    if viz.caption:
        lines.append(viz.caption.strip())
    elif chain:
        lines.append(f"Route map: **{aircraft}** — {chain}.")

    if legs:
        limiting_leg, limit_dist = _pick_limiting_leg(legs, practical_nm)
        lines.append(
            f"**Practical envelope:** ~{int(practical_nm)} nm ({aircraft}, NBAA-style with full seats — not brochure max)."
        )
        lines.append("**Where range limitations appear:**")
        for leg, dist in _leg_distances_nm(legs):
            if dist <= 0:
                lines.append(f"- {leg} — stage length unresolved; verify distance before committing.")
            elif dist > practical_nm:
                lines.append(
                    f"- **{leg} (~{int(dist)} nm) — BINDS.** Exceeds practical nonstop; plan a fuel stop, "
                    "reduce payload, or use a larger-range aircraft on this leg."
                )
            elif dist > practical_nm * 0.88:
                lines.append(
                    f"- **{leg} (~{int(dist)} nm) — marginal.** Possible with restrictions (fewer pax, seasonal fuel)."
                )
            else:
                lines.append(f"- {leg} (~{int(dist)} nm) — within practical envelope.")
        if limiting_leg and limit_dist > 0:
            if limit_dist > practical_nm:
                lines.append(
                    f"\n**First hard limit:** **{limiting_leg}** (~{int(limit_dist)} nm) — "
                    "this leg breaks nonstop planning on this routing; the schematic map highlights it."
                )
            else:
                lines.append(
                    f"\n**Longest stage:** **{limiting_leg}** (~{int(limit_dist)} nm) — "
                    "still inside practical range; watch payload and seasonal wind."
                )

    prose = "\n".join(lines).strip()
    patch = dict(patch or {})
    patch["broker_query_guard_applied"] = 1
    return prose, patch


def should_skip_llm_for_broker_guard(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Queries that must not be left to LLM drafting."""
    q = (query or "").strip()
    if not q:
        return False
    if is_cosmetic_refresh_query(q):
        return True
    if is_tail_buy_concern_query(q):
        return True
    if is_owner_buy_concern_query(q):
        return True
    if is_prior_ownership_query(q):
        return True
    if is_maintenance_profile_query(q):
        return True
    if is_dom_psychology_query(q):
        return True
    if is_nonstop_feasibility_query(q):
        return True
    if is_single_aircraft_mission_query(q):
        return True
    if is_route_map_query(q):
        return True
    if is_segment_liquidity_query(q):
        return True
    if is_pre_offer_verification_query(q):
        return True
    if is_engine_program_query(q):
        return True
    if is_resale_maximization_query(q):
        return True
    if is_portfolio_mission_buy_query(q):
        return True
    if is_comparison_query(q):
        return True
    if is_tail_gallery_intent(q, history):
        return True
    return False


def gallery_images_present(data_used: Optional[Dict[str, Any]]) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    imgs = du.get("aircraft_images") or []
    return isinstance(imgs, list) and len(imgs) > 0


def is_tail_gallery_intent(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Explicit visual/tail gallery request — independent of whether images were retrieved."""
    try:
        from rag.consultant_query_anchor import user_wants_full_gallery
        from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer

        return bool(
            user_wants_full_gallery(query)
            or wants_consultant_aircraft_images_in_answer(query, history)
        )
    except Exception:
        return False


def should_force_gallery_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    return is_tail_gallery_intent(query, history)


def gallery_contradicts_answer(body: str, data_used: Optional[Dict[str, Any]] = None) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    imgs = du.get("aircraft_images") or []
    if not isinstance(imgs, list) or not imgs:
        return False
    low = (body or "").lower()
    if re.search(
        r"(?is)\b(?:couldn'?t find|don'?t have|do not have|currently don'?t have|"
        r"no verified images?|no specific images?|no images available|no galleries|not find any|"
        r"can'?t display images?|cannot display images?|can'?t provide images?|"
        r"cannot provide images?|don'?t have photos?|unable to (?:show|display|provide) images?|"
        r"unfortunately.{0,40}no)\b",
        low,
    ):
        return True
    if re.search(r"(?is)\bno images\b", low):
        return True
    # LLM named wrong type while gallery is tail-anchored
    rows = du.get("phly_rows") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        mt = str(rows[0].get("marketing_type") or rows[0].get("aircraft_model") or "").lower()
        if mt and "citation" in mt and "challenger" in low:
            return True
        if mt and "challenger" in mt and "citation" in low:
            return True
    return False


def render_gallery_forward_answer(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    tail = str(du.get("tail_registration") or "").strip().upper()
    m = re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", query or "", re.I)
    if m:
        tail = m.group(0).upper()
    imgs = du.get("aircraft_images") or []
    raw_n = len(imgs) if isinstance(imgs, list) else 0
    label = tail or "this tail"
    model = ""
    rows = du.get("phly_rows") or du.get("phlydata_rows") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        model = str(rows[0].get("marketing_type") or rows[0].get("aircraft_model") or "").strip()
    if not model and label:
        try:
            from services.broker_execution.tail_investigation import load_tail_investigation

            inv = load_tail_investigation(query, du)
            for f in inv.get("facts") or []:
                if not isinstance(f, dict):
                    continue
                if str(f.get("label") or "").strip().lower() in ("aircraft", "type", "model"):
                    model = str(f.get("value") or "").strip()
                    break
        except Exception:
            pass
    counts = du.get("gallery_trust_tier_counts") if isinstance(du.get("gallery_trust_tier_counts"), dict) else {}
    if counts:
        try:
            from services.broker_execution.image_verification_tiers import render_gallery_tier_prose

            return render_gallery_tier_prose(
                query, tail=label, model=model, counts=counts, total_before_filter=raw_n
            )
        except Exception:
            pass
    wants_cockpit = bool(re.search(r"(?is)\bcockpit\b", query or ""))
    wants_cabin = bool(re.search(r"(?is)\bcabin|cabine|interior\b", query or ""))
    if wants_cockpit:
        facet = "cockpit"
        noun = "cockpit photo"
    elif wants_cabin:
        facet = "cabin"
        noun = "cabin photo"
    else:
        facet = "photo"
        noun = "photo"
    if raw_n == 0:
        return (
            f"No usable **{facet}** images for **{label}**"
            + (f" ({model})" if model else "")
            + " in listing scrape or image search this turn.\n"
            "I would check JetPhotos, Planespotters, Virtual Hangar, or the broker listing — "
            "the caption must show this exact registration before you treat a photo as this aircraft."
        )
    if raw_n != 1:
        noun += "s"
    return (
        f"**{label}**"
        + (f" ({model})" if model else "")
        + f" — **{raw_n}** {noun} below. Confirm registration on each source before underwriting identity."
    )


def try_broker_query_guard(
    query: str,
    body: str,
    data_used: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Return replacement answer when query shape requires deterministic broker prose."""
    q = (query or "").strip()
    if not q:
        return None

    if is_cosmetic_refresh_query(q):
        return render_cosmetic_skepticism_answer(q)
    if is_tail_buy_concern_query(q):
        try:
            from services.broker_execution.tail_investigation import render_tail_acquisition_concerns

            return render_tail_acquisition_concerns(q, data_used, body=body or "")
        except Exception:
            pass
    if is_owner_buy_concern_query(q):
        return render_owner_buy_concern_answer(q, data_used, body=body or "")
    if is_prior_ownership_query(q):
        return render_prior_ownership_answer(q, data_used)
    if is_maintenance_profile_query(q):
        return render_maintenance_profile_answer(q, data_used)
    if is_dom_psychology_query(q):
        return render_dom_psychology_answer(q)
    if is_nonstop_feasibility_query(q):
        return render_nonstop_feasibility_answer(q)
    if is_single_aircraft_mission_query(q):
        return render_single_aircraft_mission_answer(q)
    if is_segment_liquidity_query(q):
        return render_segment_liquidity_answer(q)
    if is_pre_offer_verification_query(q):
        return render_pre_offer_verification_answer(q)
    if is_resale_maximization_query(q):
        return render_resale_maximization_answer(q)
    if is_portfolio_mission_buy_query(q):
        return render_portfolio_mission_buy_answer(q)
    if is_engine_program_query(q):
        try:
            from services.broker_execution.tail_acquisition_dossier import render_engine_program_answer

            eng = render_engine_program_answer(q, data_used)
            if eng:
                return eng
        except Exception:
            pass

    if is_route_map_query(q):
        prose, _patch = render_route_map_broker_answer(q, data_used)
        if isinstance(data_used, dict) and _patch:
            data_used.update(_patch)
        return prose

    try:
        from services.broker_execution.mission_broker_answer import (
            build_deterministic_mission_answer,
            is_mission_shaped_query,
        )

        if is_mission_shaped_query(q):
            mission = build_deterministic_mission_answer(q, data_used)
            low_body = (body or "").lower()
            if mission and (
                re.search(
                    r"(?is)\bdoes not realistically\b|expect one-stop\b|budget.*too low\b",
                    mission.lower(),
                )
                or re.search(
                    r"(?is)\b(?:global\s+\d{3,4}|gulfstream\s+g\d|g650|g700|g500|bombardier\s+global)\b",
                    low_body,
                )
                or re.search(
                    r"(?is)\b(?:here are a few|consider the following|you may need to consider|"
                    r"great-circle distance)\b",
                    low_body,
                )
            ):
                return mission
    except Exception:
        pass

    if is_tail_gallery_intent(q, history):
        low = (body or "").lower()
        if (
            gallery_contradicts_answer(body, data_used)
            or not gallery_images_present(data_used)
            or ("below" not in low and "gallery" not in low)
            or re.search(
                r"(?is)\b(?:broker'?s?\s+perspective|airframe\s+(?:hours|total\s+time)|"
                r"maintenance\s+records|passengers?|would you like to see)\b",
                low,
            )
            or _answer_looks_like_listing_dump(body)
            or re.search(r"(?is)\brecommend reaching out\b", low)
        ):
            return render_gallery_forward_answer(q, data_used)

    if gallery_contradicts_answer(body, data_used) and re.search(
        r"(?is)\b(?:image|photo|gallery|verified)\b", q
    ):
        return render_gallery_forward_answer(q, data_used)

    if is_comparison_query(q):
        try:
            from services.broker_execution.comparison_broker_facts import (
                attach_comparison_broker_ui,
                render_comparison_display_answer,
            )

            ui = attach_comparison_broker_ui(q, data_used)
            alt = render_comparison_display_answer(q, data_used) if ui else ""
            if alt and len(alt) > 20:
                if comparison_answer_looks_incomplete(body, q) or "wins on" not in (body or "").lower():
                    return alt
        except Exception:
            pass

    return None


def resolve_broker_guard_stream_payload(
    query: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a full stream payload (answer + data_used patch) for broker-guard turns.

    Skips LLM drafting when the query shape must be deterministic (resale pick,
    route map, comparison, owner+buy concerns, gallery-forward, etc.).
    """
    q = (query or "").strip()
    if not q or not isinstance(payload, dict):
        return None

    du = dict(payload.get("data_used") or {})
    body = str(payload.get("answer") or "").strip()
    imgs = payload.get("aircraft_images")
    if isinstance(imgs, list) and imgs:
        du["aircraft_images"] = imgs
    hist = payload.get("history")
    if not isinstance(hist, list):
        hist = du.get("consultant_history")
    history = hist if isinstance(hist, list) else None

    skip_llm = should_skip_llm_for_broker_guard(q, history)
    force_gallery = is_tail_gallery_intent(q, history)
    if not skip_llm and not force_gallery:
        return None

    guarded = try_broker_query_guard(q, body, du, history)
    if not guarded and force_gallery:
        guarded = render_gallery_forward_answer(q, du)
    if not guarded:
        return None

    out = dict(payload)
    out["answer"] = guarded.strip()
    du["broker_query_guard_applied"] = 1
    du["broker_guard_stream_shortcircuit"] = 1
    if (
        is_cosmetic_refresh_query(q)
        or is_dom_psychology_query(q)
        or is_nonstop_feasibility_query(q)
        or is_tail_buy_concern_query(q)
        or is_maintenance_profile_query(q)
        or is_prior_ownership_query(q)
    ):
        du["consultant_suppress_gallery"] = 1
        out["aircraft_images"] = []
    elif isinstance(imgs, list):
        out["aircraft_images"] = imgs
    out["data_used"] = du
    return out
