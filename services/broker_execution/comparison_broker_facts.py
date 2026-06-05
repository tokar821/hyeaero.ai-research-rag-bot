"""
Broker-depth comparison facts — wins, tradeoffs, buy-if guidance for the fact pack.

Presentation / LLM context only; does not change comparison_v2 ranking or dispatch.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Display spec hints (shared with compression formatters).
_COMPARISON_SPEC_ROWS: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {
    ("gulfstream g280", "citation longitude"): [
        ("Range (nm)", "~3,600", "~3,500"),
        ("Cabin", "Large super-mid", "Super-mid stand-up"),
        ("Operating cost", "Higher acquisition", "Mid super-mid OPEX"),
        ("Runway / field", "Good hot/high", "Strong field performance"),
        ("Market liquidity", "Moderate Gulfstream", "Strong Textron"),
    ],
    ("praetor 600", "citation longitude"): [
        ("Range (nm)", "~4,000", "~3,500"),
        ("Cabin", "Super-mid efficient", "Super-mid stand-up"),
        ("Operating cost", "Lower Embraer band", "Mid Textron band"),
        ("Runway / field", "Strong field flex", "Strong field performance"),
        ("Market liquidity", "Growing Praetor", "Strong Longitude"),
    ],
    ("g650", "g700"): [
        ("Range (nm)", "~7,500", "~7,500+"),
        ("Cabin", "Ultra-large proven", "Ultra-large newest"),
        ("Operating cost", "Very high", "Very high"),
        ("Market liquidity", "Moderate", "Moderate/newer"),
    ],
    ("challenger 350", "praetor 500"): [
        ("Range (nm)", "~3,200", "~3,340"),
        ("Cabin", "Large super-mid", "Super-mid"),
        ("Operating cost", "Higher large-cabin", "Lower Embraer band"),
        ("Field performance", "Moderate", "Strong"),
        ("Market liquidity", "Strong Bombardier", "Growing Praetor"),
    ],
    ("challenger 350", "praetor 600"): [
        ("Range (nm)", "~3,200", "~4,000"),
        ("Cabin", "Large super-mid", "Super-mid efficient"),
        ("Operating cost", "Higher large-cabin", "Lower Embraer band"),
        ("Field performance", "Moderate", "Strong"),
        ("Market liquidity", "Strong Bombardier", "Growing Praetor"),
    ],
    ("challenger 350", "citation latitude"): [
        ("Speed (ktas)", "~470", "~440"),
        ("Range (nm)", "~3,200", "~2,700"),
        ("Cabin width", "Wider stand-up", "Midsize stand-up"),
        ("Baggage", "Strong", "Adequate midsize"),
        ("Support network", "Bombardier global", "Textron dense U.S."),
        ("Resale", "Strong large-cabin", "Solid midsize liquidity"),
        ("Operating economics", "Higher OPEX", "Lower midsize OPEX"),
        ("Dispatch reliability", "Mature fleet", "Mature Textron"),
    ],
    ("falcon 2000lxs", "praetor 600"): [
        ("Range (nm)", "~4,000", "~4,000"),
        ("Cabin", "Large-cabin Dassault", "Super-mid Embraer"),
        ("Operating cost", "Higher Dassault OPEX", "Lower Embraer band"),
        ("Runway / field", "Good", "Strong field flex"),
        ("Market liquidity", "Dassault large-cabin", "Growing Praetor"),
    ],
    ("falcon 2000lxs", "challenger 350"): [
        ("Speed (ktas)", "~470", "~470"),
        ("Range (nm)", "~4,000", "~3,200"),
        ("Cabin", "Large-cabin Dassault", "Large super-mid Bombardier"),
        ("Operating economics", "Higher Dassault OPEX", "Strong Bombardier liquidity"),
        ("Field performance", "Good", "Moderate hot/high"),
        ("Resale", "Dassault large-cabin", "Deep Bombardier pool"),
    ],
    ("gulfstream g280", "citation longitude"): [
        ("Speed (ktas)", "~480", "~440"),
        ("Range (nm)", "~3,600", "~3,500"),
        ("Cabin", "Large super-mid Gulfstream", "Textron super-mid"),
        ("Support", "Gulfstream network", "Textron density"),
        ("Resale", "Gulfstream depth", "Textron liquidity"),
    ],
    ("gulfstream g280", "praetor 600"): [
        ("Speed (ktas)", "~480", "~466"),
        ("Range (nm)", "~3,600", "~4,000"),
        ("Cabin width", "Large super-mid", "Super-mid efficient"),
        ("Baggage", "Strong", "Good super-mid"),
        ("Support network", "Gulfstream", "Embraer growing"),
        ("Resale", "Gulfstream depth", "Growing Praetor"),
        ("Operating economics", "Higher acquisition/OPEX", "Lower Embraer band"),
        ("Dispatch reliability", "Mature G280", "Strong new-generation"),
    ],
}

_COMPARISON_BROKER_GUIDANCE: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("gulfstream g280", "citation longitude"): {
        "a_wins": ["cruise speed", "Gulfstream resale depth", "transcon range headroom"],
        "b_wins": ["cabin volume per dollar", "Textron support density", "predictable super-mid OPEX"],
        "tradeoffs": [
            "G280 wins U.S. coast-to-coast speed and hot/high flexibility.",
            "Longitude wins stand-up cabin comfort and lower super-mid operating cost band.",
        ],
        "buy_a_if": "Speed and Gulfstream pedigree matter more than lowest OPEX.",
        "buy_b_if": "Cabin comfort and Textron fleet familiarity matter more than top-end cruise.",
    },
    ("praetor 600", "citation longitude"): {
        "a_wins": ["range per dollar", "field performance", "Embraer efficiency story"],
        "b_wins": ["cabin headroom", "Textron resale liquidity", "U.S. support footprint"],
        "tradeoffs": [
            "Praetor 600 stretches super-mid range with strong runway flexibility.",
            "Longitude trades a bit of range efficiency for a more spacious Textron cabin.",
        ],
        "buy_a_if": "You want maximum super-mid range and runway flexibility at lower acquisition.",
        "buy_b_if": "You prioritize Textron ecosystem and cabin volume on typical U.S. missions.",
    },
    ("g650", "g700"): {
        "a_wins": ["proven ULR dispatch", "mature resale", "global support depth"],
        "b_wins": ["newest Gulfstream cabin", "latest avionics", "flagship positioning"],
        "tradeoffs": [
            "G650ER is the proven ULR workhorse with deep support.",
            "G700 is the newer flagship with similar mission but less fleet history.",
        ],
        "buy_a_if": "Proven dispatch maturity and resale liquidity are paramount.",
        "buy_b_if": "You want the newest Gulfstream cabin and are comfortable with newer fleet data.",
    },
    ("challenger 350", "praetor 500"): {
        "a_wins": ["cabin volume", "large-cabin comfort", "Bombardier resale depth"],
        "b_wins": ["operating economics", "field performance", "Embraer efficiency"],
        "tradeoffs": [
            "Challenger 350 wins cabin size and U.S. large-cabin comfort.",
            "Praetor 500 wins efficiency and runway flexibility on shorter legs.",
        ],
        "buy_a_if": "Cabin volume and large-cabin comfort matter more than lowest OPEX.",
        "buy_b_if": "Efficiency and field performance matter more than maximum cabin size.",
    },
    ("challenger 350", "praetor 600"): {
        "a_wins": ["cabin volume", "large-cabin comfort", "Bombardier U.S. support"],
        "b_wins": ["range per dollar", "field performance", "Embraer operating economics"],
        "tradeoffs": [
            "Challenger 350 wins stand-up large-cabin comfort on transcon missions.",
            "Praetor 600 wins range and efficiency for the same mission budget.",
        ],
        "buy_a_if": "Maximum cabin volume and large-cabin comfort are the priority.",
        "buy_b_if": "Range, efficiency, and runway flexibility outweigh cabin size.",
    },
    ("challenger 350", "citation latitude"): {
        "a_wins": ["cabin volume", "range", "large-cabin comfort", "hot/high runway margin"],
        "b_wins": ["operating economics", "Textron support density", "lower acquisition"],
        "tradeoffs": [
            "Challenger 350 is objectively stronger on cabin, range, and runway for demanding missions.",
            "Latitude wins on OPEX and U.S. fleet familiarity for typical midsize trips.",
        ],
        "buy_a_if": "You need maximum cabin and runway performance — ignore purchase price.",
        "buy_b_if": "Predictable midsize economics and Textron ecosystem matter more than cabin size.",
    },
    ("falcon 2000lxs", "praetor 600"): {
        "a_wins": ["large-cabin comfort", "Dassault pedigree", "Atlantic/transcon headroom"],
        "b_wins": ["operating economics", "field performance", "lower acquisition band"],
        "tradeoffs": [
            "Falcon 2000LXS wins cabin class and comfort for owner-flown large-cabin missions.",
            "Praetor 600 wins efficiency and runway flexibility when you do not need Dassault cabin scale.",
        ],
        "buy_a_if": "You operate like a large-cabin owner and value Dassault comfort over OPEX.",
        "buy_b_if": "You want super-mid range and economics without large-cabin carrying cost.",
    },
    ("falcon 2000lxs", "challenger 350"): {
        "a_wins": ["range headroom", "Dassault large-cabin comfort", "transatlantic credibility"],
        "b_wins": ["U.S. support density", "Bombardier resale liquidity", "lower acquisition band"],
        "tradeoffs": [
            "Falcon 2000LXS wins range and large-cabin comfort on long U.S. and Atlantic missions.",
            "Challenger 350 wins U.S. fleet familiarity and typically lower operating cost band.",
        ],
        "buy_a_if": "You want Dassault large-cabin range and comfort on longer legs.",
        "buy_b_if": "You want Bombardier large-cabin comfort with stronger U.S. support and liquidity.",
    },
    ("challenger 3500", "falcon 2000lxs"): {
        "a_wins": ["newest Bombardier cabin", "connectivity", "super-mid operating economics"],
        "b_wins": ["Dassault pedigree", "large-cabin scale", "Atlantic/transcon headroom"],
        "tradeoffs": [
            "Challenger 3500 wins modern cabin tech and lower operating band for super-mid missions.",
            "Falcon 2000LXS wins true large-cabin presence and Dassault range comfort when range matters.",
        ],
        "buy_a_if": "You want the newest super-mid cabin and U.S. operating economics without large-cabin carrying cost.",
        "buy_b_if": "You want Dassault large-cabin comfort and pedigree even if range were not the deciding factor.",
        "operate_pick": "Challenger 3500",
        "operate_why": "day-to-day U.S. owner-operator economics and cabin tech beat large-cabin carrying cost when range is off the table.",
    },
    ("gulfstream g280", "praetor 600"): {
        "a_wins": ["cruise speed", "Gulfstream brand/resale", "hot/high flexibility"],
        "b_wins": ["range per dollar", "cabin efficiency", "Embraer field performance"],
        "tradeoffs": [
            "G280 fits corporate flight departments prioritizing speed and Gulfstream pedigree.",
            "Praetor 600 fits departments wanting super-mid range with lower operating band.",
        ],
        "buy_a_if": "Corporate department values Gulfstream dispatch maturity and transcon speed.",
        "buy_b_if": "Department wants maximum super-mid range and runway flexibility per dollar.",
    },
}


def _norm_model(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    x, y = _norm_model(a), _norm_model(b)
    return (x, y) if x <= y else (y, x)


def _normalize_comparison_model_name(name: str) -> str:
    raw = re.sub(r"\s+", " ", (name or "").strip())
    low = raw.lower()
    if re.search(r"falcon\s*2000\s*lxs", low):
        return "Falcon 2000LXS"
    if re.search(r"falcon\s*2000\b", low):
        return "Falcon 2000LXS"
    if re.search(r"praetor\s*600", low):
        return "Praetor 600"
    if re.search(r"challenger\s*3500", low):
        return "Challenger 3500"
    if re.search(r"challenger\s*350", low):
        return "Challenger 350"
    if re.search(r"citation\s+longitude", low):
        return "Citation Longitude"
    if re.search(r"gulfstream\s*g\s*280|\bg280\b", low):
        return "Gulfstream G280"
    return raw


def _resolve_comparison_models(query: str, data_used: dict) -> Tuple[str, str]:
    du = data_used if isinstance(data_used, dict) else {}
    cv2 = du.get("comparison_v2")
    if isinstance(cv2, dict):
        models = list(cv2.get("models") or [])
        if len(models) >= 2:
            return (
                _normalize_comparison_model_name(str(models[0])),
                _normalize_comparison_model_name(str(models[1])),
            )
    parts = re.split(r"(?is)\bvs\.?\b|\bversus\b", query or "", maxsplit=1)
    if len(parts) == 2:
        left = re.split(r"[.?!]", parts[0], maxsplit=1)[0].strip()
        right = re.split(r"[.?!]", parts[1], maxsplit=1)[0].strip()
        return (
            _normalize_comparison_model_name(left),
            _normalize_comparison_model_name(right),
        )
    or_m = re.search(
        r"(?is)\b(?:rather\s+operate|would\s+you\s+rather\s+operate|operate)\s+(?:a\s+)?(.+?)\s+or\s+(?:a\s+)?(.+?)(?:\?|$)",
        query or "",
    )
    if or_m:
        return (
            _normalize_comparison_model_name(or_m.group(1)),
            _normalize_comparison_model_name(or_m.group(2)),
        )
    try:
        from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison

        res = soft_resolve_comparison(query)
        if res and len(res.models) >= 2:
            return (
                _normalize_comparison_model_name(str(res.models[0])),
                _normalize_comparison_model_name(str(res.models[1])),
            )
    except Exception:
        pass
    return "Aircraft A", "Aircraft B"


def _lookup_pair_entry(table: Dict[Tuple[str, str], Any], a: str, b: str) -> Tuple[Optional[Any], Tuple[str, str], bool]:
    """Return (entry, canonical_key, swapped_vs_input)."""
    key = _pair_key(a, b)
    if key in table:
        swapped = _norm_model(a) != key[0]
        return table[key], key, swapped
    rev = (key[1], key[0])
    if rev in table:
        swapped = _norm_model(a) != rev[0]
        return table[rev], rev, swapped
    return None, key, False


def _catalog_spec_rows(a: str, b: str) -> List[Tuple[str, str, str]]:
    """Pull indicative specs from verified catalog when pair table has no entry."""
    try:
        from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

        ra = get_aircraft_authority_record(aircraft_model=a)
        rb = get_aircraft_authority_record(aircraft_model=b)
        if not ra or not rb:
            return []
        rows: List[Tuple[str, str, str]] = []
        if ra.nbaa_range_nm and rb.nbaa_range_nm:
            rows.append(("Range (nm)", f"~{int(ra.nbaa_range_nm):,}", f"~{int(rb.nbaa_range_nm):,}"))
        if ra.cabin_width and rb.cabin_width:
            rows.append(("Cabin width (ft)", f"~{ra.cabin_width}", f"~{rb.cabin_width}"))
        if ra.max_cruise_speed and rb.max_cruise_speed:
            rows.append(("Speed (ktas)", f"~{int(ra.max_cruise_speed)}", f"~{int(rb.max_cruise_speed)}"))
        cat_a = str(getattr(ra, "category", "") or "").strip()
        cat_b = str(getattr(rb, "category", "") or "").strip()
        if cat_a or cat_b:
            rows.append(("Category", cat_a or "—", cat_b or "—"))
        return rows
    except Exception:
        return []


def _lookup_table(a: str, b: str) -> List[Tuple[str, str, str]]:
    entry, key, swapped = _lookup_pair_entry(_COMPARISON_SPEC_ROWS, a, b)
    if entry:
        rows = entry
        if swapped:
            return [(r[0], r[2], r[1]) for r in rows]
        return rows
    catalog = _catalog_spec_rows(a, b)
    if catalog:
        return catalog
    return [
        ("Range (nm)", "Compare verified NBAA range", "Compare verified NBAA range"),
        ("Cabin", "Compare stand-up cabin class", "Compare stand-up cabin class"),
        ("Operating economics", "Model-dependent OPEX band", "Model-dependent OPEX band"),
        ("Dispatch / support", "Compare OEM support footprint", "Compare OEM support footprint"),
    ]


def _lookup_guidance(a: str, b: str) -> Optional[Dict[str, Any]]:
    raw, _key, swapped = _lookup_pair_entry(_COMPARISON_BROKER_GUIDANCE, a, b)
    if not raw:
        return None
    if not swapped:
        return raw
    out: Dict[str, Any] = {
        "a_wins": list(raw.get("b_wins") or []),
        "b_wins": list(raw.get("a_wins") or []),
        "tradeoffs": list(raw.get("tradeoffs") or []),
        "buy_a_if": raw.get("buy_b_if") or "",
        "buy_b_if": raw.get("buy_a_if") or "",
    }
    if raw.get("operate_pick"):
        out["operate_pick"] = raw.get("operate_pick")
        out["operate_why"] = raw.get("operate_why") or ""
    return out


def _intelligence_lines(data_used: dict, a: str, b: str) -> List[str]:
    cv2 = data_used.get("comparison_v2")
    if not isinstance(cv2, dict):
        return []
    try:
        from services.comparison.comparison_intelligence import enrich_comparison_payload

        enriched = enrich_comparison_payload(dict(cv2))
    except Exception:
        enriched = cv2
    rows = enriched.get("comparison_rows") or []
    if not rows:
        return []
    lines: List[str] = []
    for model_name in (a, b):
        low = model_name.lower()
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("aircraft_id") or "").lower()
            if not label or (low not in label and label not in low):
                continue
            dims = []
            for k in (
                "maintenance_ecosystem",
                "airport_flexibility",
                "dispatch_maturity",
                "cabin_usability",
            ):
                v = row.get(k)
                if v and str(v) != "standard":
                    dims.append(f"{k.replace('_', ' ')}: {v}")
            if dims:
                lines.append(f"• {model_name}: " + "; ".join(dims[:3]))
            break
    return lines


def build_comparison_broker_facts_block(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """Structured broker comparison block for the LLM fact pack."""
    du = data_used if isinstance(data_used, dict) else {}
    a, b = _resolve_comparison_models(query, du)
    if a == "Aircraft A" and b == "Aircraft B":
        q = (query or "").lower()
        if "vs" not in q and "versus" not in q and not du.get("comparison_v2"):
            return ""

    guidance = _lookup_guidance(a, b)
    rows = _lookup_table(a, b)
    lines = [
        f"[COMPARISON BROKER FACTS — {a} vs {b}]",
        "Narrate in prose: lead with which aircraft wins for the user's stated mission or buyer profile.",
        "",
        "Spec snapshot (indicative — do not invent beyond this):",
    ]
    for dim, va, vb in rows:
        lines.append(f"• {dim}: {a} {va}; {b} {vb}")

    if guidance:
        lines.append("")
        lines.append(f"{a} wins on: " + ", ".join(guidance.get("a_wins") or []))
        lines.append(f"{b} wins on: " + ", ".join(guidance.get("b_wins") or []))
        for t in guidance.get("tradeoffs") or []:
            lines.append(f"• Tradeoff: {t}")
        buy_a = str(guidance.get("buy_a_if") or "").strip()
        buy_b = str(guidance.get("buy_b_if") or "").strip()
        if buy_a:
            lines.append(f"Buy {a} if: {buy_a}")
        if buy_b:
            lines.append(f"Buy {b} if: {buy_b}")

    intel = _intelligence_lines(du, a, b)
    if intel:
        lines.append("")
        lines.append("Operational intelligence:")
        lines.extend(intel)

    cv2 = du.get("comparison_v2")
    if isinstance(cv2, dict) and cv2.get("rows"):
        try:
            snippet = json.dumps(cv2.get("rows"), default=str)[:800]
            lines.append("")
            lines.append(f"Verified comparison rows: {snippet}")
        except Exception:
            pass

    body = "\n".join(lines).strip()
    return body if len(body) > 120 else ""


def render_comparison_client_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Client-facing comparison prose without internal fact-pack headers."""
    du = data_used if isinstance(data_used, dict) else {}
    a, b = _resolve_comparison_models(query, du)
    block = build_comparison_broker_facts_block(query, data_used)
    if not block:
        return ""
    skip_prefixes = (
        "[comparison",
        "narrate in prose",
        "spec snapshot",
        "verified comparison rows",
        "operational intelligence:",
    )
    lines: List[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(low.startswith(p) for p in skip_prefixes):
            continue
        lines.append(line)
    if not lines:
        guidance = _lookup_guidance(a, b)
        if guidance:
            lines.append(
                f"Comparing {_normalize_comparison_model_name(a)} vs {_normalize_comparison_model_name(b)}:"
            )
            lines.append(f"{a} wins on: " + ", ".join(guidance.get("a_wins") or []))
            lines.append(f"{b} wins on: " + ", ".join(guidance.get("b_wins") or []))
            for t in guidance.get("tradeoffs") or []:
                lines.append(f"• {t}")
            buy_a = str(guidance.get("buy_a_if") or "").strip()
            buy_b = str(guidance.get("buy_b_if") or "").strip()
            if buy_a:
                lines.append(f"Buy {a} if: {buy_a}")
            if buy_b:
                lines.append(f"Buy {b} if: {buy_b}")
    guidance = _lookup_guidance(a, b)
    if re.search(r"(?is)\b(?:rather\s+operate|would\s+you\s+rather)\b", query or "") and guidance:
        pick = str(guidance.get("operate_pick") or "").strip()
        why = str(guidance.get("operate_why") or "").strip()
        if not pick:
            pick = a if len(guidance.get("a_wins") or []) >= len(guidance.get("b_wins") or []) else b
        lead = [f"**I would operate the {pick}** — {why or 'better fit for owner-operator day-to-day economics and cabin experience.'}"]
        lead.extend(lines[:12])
        body = "\n".join(lead).strip()
        return body if len(body) > 60 else ""
    body = "\n".join(lines[:16]).strip()
    return body if len(body) > 60 else ""


__all__ = [
    "build_comparison_broker_facts_block",
    "render_comparison_client_answer",
    "_COMPARISON_SPEC_ROWS",
]
