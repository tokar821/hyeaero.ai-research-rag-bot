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
    ("challenger 300", "citation latitude"): [
        ("Speed (ktas)", "~459", "~440"),
        ("Range (nm)", "~3,100", "~2,700"),
        ("Cabin width", "Super-mid stand-up", "Midsize stand-up"),
        ("Baggage", "Strong super-mid", "Adequate midsize"),
        ("Support network", "Bombardier global", "Textron dense U.S."),
        ("Resale", "Mature super-mid pool", "Solid midsize liquidity"),
        ("Operating economics", "Higher super-mid OPEX", "Lower midsize OPEX"),
        ("Dispatch reliability", "Mature CL300 fleet", "Mature Textron"),
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
    ("challenger 300", "citation latitude"): {
        "a_wins": ["cabin volume", "range", "super-mid comfort", "runway margin"],
        "b_wins": ["operating economics", "Textron support density", "lower acquisition"],
        "tradeoffs": [
            "Challenger 300 wins cabin scale, range, and runway for heavier super-mid missions.",
            "Latitude wins on OPEX and Textron fleet familiarity for typical midsize trips.",
        ],
        "buy_a_if": "You want super-mid cabin and range without stepping to a 350 acquisition band.",
        "buy_b_if": "Predictable midsize economics matter more than super-mid cabin headroom.",
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


def _strip_compare_leadin(text: str) -> str:
    return re.sub(
        r"(?is)^(?:compare|comparing|difference\s+between|which\s+is\s+better)\s+",
        "",
        (text or "").strip(),
    )


def _normalize_comparison_model_name(name: str) -> str:
    raw = re.sub(r"\s+", " ", _strip_compare_leadin(name or "")).strip()
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
    if re.search(r"challenger\s*300", low):
        return "Challenger 300"
    if re.search(r"citation\s+latitude", low):
        return "Citation Latitude"
    if re.search(r"citation\s+longitude", low):
        return "Citation Longitude"
    if re.search(r"gulfstream\s*g\s*280|\bg280\b", low):
        return "Gulfstream G280"
    return raw


def _resolve_comparison_models(query: str, data_used: dict) -> Tuple[str, str]:
    du = data_used if isinstance(data_used, dict) else {}
    parts = re.split(r"(?is)\bvs\.?\b|\bversus\b", query or "", maxsplit=1)
    if len(parts) == 2:
        left = re.split(r"[.?!]", parts[0], maxsplit=1)[0].strip()
        right = re.split(r"[.?!]", parts[1], maxsplit=1)[0].strip()
        a = _normalize_comparison_model_name(left)
        b = _normalize_comparison_model_name(right)
        if a != "Aircraft A" and b != "Aircraft B":
            return a, b
    cv2 = du.get("comparison_v2")
    if isinstance(cv2, dict):
        models = list(cv2.get("models") or [])
        if len(models) >= 2:
            return (
                _normalize_comparison_model_name(str(models[0])),
                _normalize_comparison_model_name(str(models[1])),
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


def _fmt_range_nm(nm: Optional[float]) -> str:
    if not nm or nm <= 0:
        return "—"
    return f"~{int(round(nm)):,} nm"


def _fmt_speed_ktas(ktas: Optional[float]) -> str:
    if not ktas or ktas <= 0:
        return "—"
    return f"~{int(round(ktas))} ktas"


def _fmt_passengers(rec: Any) -> str:
    pmin = int(getattr(rec, "passenger_capacity_min", 0) or 0)
    pmax = int(getattr(rec, "passenger_capacity_max", 0) or 0)
    if pmax <= 0:
        return "—"
    if pmin > 0 and pmin != pmax:
        return f"Up to {pmax} ({pmin} typical)"
    return f"Up to {pmax}"


def _fmt_takeoff_ft(ft: Optional[int]) -> str:
    if not ft or ft <= 0:
        return "—"
    return f"~{int(ft):,} ft"


def _fmt_cabin_dim(ft_val: Optional[float]) -> str:
    if not ft_val or ft_val <= 0:
        return "—"
    feet = int(ft_val)
    inches = int(round((ft_val - feet) * 12))
    if inches >= 12:
        feet += inches // 12
        inches = inches % 12
    if inches:
        return f"{feet} ft {inches} in"
    return f"{feet} ft"


_PLACEHOLDER_SPEC_MARKERS = (
    "compare verified",
    "compare certified",
    "compare published",
    "compare oem",
    "model-dependent",
)


def _pair_authority_records(
    a: str, b: str
) -> Tuple[Optional[Any], Optional[Any], List[str]]:
    try:
        from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

        ra = get_aircraft_authority_record(aircraft_model=a)
        rb = get_aircraft_authority_record(aircraft_model=b)
    except Exception:
        return None, None, [a, b]
    missing: List[str] = []
    if not ra:
        missing.append(a)
    if not rb:
        missing.append(b)
    return ra, rb, missing


def _spec_rows_from_records(ra: Any, rb: Any) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    if (ra.nbaa_range_nm or 0) > 0 or (rb.nbaa_range_nm or 0) > 0:
        rows.append(("Max range", _fmt_range_nm(ra.nbaa_range_nm), _fmt_range_nm(rb.nbaa_range_nm)))
    if (ra.passenger_capacity_max or 0) > 0 or (rb.passenger_capacity_max or 0) > 0:
        rows.append(("Max passengers", _fmt_passengers(ra), _fmt_passengers(rb)))
    if ra.max_cruise_speed or rb.max_cruise_speed:
        rows.append(
            ("Max speed", _fmt_speed_ktas(ra.max_cruise_speed), _fmt_speed_ktas(rb.max_cruise_speed))
        )
    if ra.takeoff_distance_ft or rb.takeoff_distance_ft:
        rows.append(
            (
                "Takeoff distance",
                _fmt_takeoff_ft(ra.takeoff_distance_ft),
                _fmt_takeoff_ft(rb.takeoff_distance_ft),
            )
        )
    if ra.cabin_length or rb.cabin_length:
        rows.append(
            ("Cabin length", _fmt_cabin_dim(ra.cabin_length), _fmt_cabin_dim(rb.cabin_length))
        )
    if ra.cabin_height or rb.cabin_height:
        rows.append(
            ("Cabin height", _fmt_cabin_dim(ra.cabin_height), _fmt_cabin_dim(rb.cabin_height))
        )
    if ra.cabin_width or rb.cabin_width:
        rows.append(
            ("Cabin width", _fmt_cabin_dim(ra.cabin_width), _fmt_cabin_dim(rb.cabin_width))
        )
    cat_a = str(getattr(ra, "aircraft_category", "") or "").strip()
    cat_b = str(getattr(rb, "aircraft_category", "") or "").strip()
    if cat_a or cat_b:
        rows.append(("Category", cat_a or "—", cat_b or "—"))
    return rows


def _catalog_spec_rows(a: str, b: str) -> List[Tuple[str, str, str]]:
    """Google-style numeric specs from verified catalog + AKAL enrichment (catalog-first)."""
    ra, rb, missing = _pair_authority_records(a, b)
    if missing or not ra or not rb:
        return []
    try:
        return _spec_rows_from_records(ra, rb)
    except Exception:
        return []


def _is_placeholder_spec_value(val: str) -> bool:
    low = (val or "").strip().lower()
    return any(m in low for m in _PLACEHOLDER_SPEC_MARKERS)


def _rows_are_placeholders(rows: List[Tuple[str, str, str]]) -> bool:
    if not rows:
        return True
    for _dim, va, vb in rows:
        if _is_placeholder_spec_value(va) or _is_placeholder_spec_value(vb):
            return True
    return False


def _build_catalog_comparison_guidance(
    a: str,
    b: str,
    ra: Any,
    rb: Any,
) -> Dict[str, Any]:
    """Broker verdict synthesized from verified authority records — no LLM, no invention."""
    a_wins: List[str] = []
    b_wins: List[str] = []
    commentary: List[str] = []

    range_a = float(ra.nbaa_range_nm or 0)
    range_b = float(rb.nbaa_range_nm or 0)
    if range_a > 0 and range_b > 0:
        diff = int(abs(range_a - range_b))
        if diff >= 120:
            if range_a > range_b:
                a_wins.append("NBAA range")
                commentary.append(
                    f"{a} carries about {diff:,} nm more verified range than {b} — "
                    "fewer fuel stops on longer U.S. and Atlantic missions."
                )
            else:
                b_wins.append("NBAA range")
                commentary.append(
                    f"{b} carries about {diff:,} nm more verified range than {a} — "
                    "fewer fuel stops on longer U.S. and Atlantic missions."
                )

    pax_a = int(ra.passenger_capacity_max or 0)
    pax_b = int(rb.passenger_capacity_max or 0)
    if pax_a > 0 and pax_b > 0 and pax_a != pax_b:
        if pax_a > pax_b:
            a_wins.append("passenger capacity")
            commentary.append(
                f"{a} seats one more certified passenger than {b} in typical layouts — "
                "meaningful only if you routinely fly full cabins."
            )
        else:
            b_wins.append("passenger capacity")
            commentary.append(
                f"{b} seats more certified passengers than {a} — "
                "meaningful only if you routinely fly full cabins."
            )

    spd_a = float(ra.max_cruise_speed or 0)
    spd_b = float(rb.max_cruise_speed or 0)
    if spd_a > 0 and spd_b > 0 and abs(spd_a - spd_b) >= 8:
        if spd_a > spd_b:
            a_wins.append("cruise speed")
            commentary.append(
                f"{a} is faster by roughly {int(spd_a - spd_b)} ktas — "
                "you buy schedule margin on long legs, not just brochure bragging rights."
            )
        else:
            b_wins.append("cruise speed")
            commentary.append(
                f"{b} is faster by roughly {int(spd_b - spd_a)} ktas — "
                "you buy schedule margin on long legs, not just brochure bragging rights."
            )

    to_a = int(ra.takeoff_distance_ft or 0)
    to_b = int(rb.takeoff_distance_ft or 0)
    if to_a > 0 and to_b > 0 and abs(to_a - to_b) >= 200:
        if to_a < to_b:
            a_wins.append("runway flexibility")
            commentary.append(
                f"{a} publishes shorter takeoff distance (~{to_a:,} ft vs ~{to_b:,} ft) — "
                "better for shorter fields and hot-and-high margins."
            )
        else:
            b_wins.append("runway flexibility")
            commentary.append(
                f"{b} publishes shorter takeoff distance (~{to_b:,} ft vs ~{to_a:,} ft) — "
                "better for shorter fields and hot-and-high margins."
            )

    for attr, label in (
        ("cabin_width", "cabin width"),
        ("cabin_length", "cabin length"),
        ("cabin_height", "cabin height"),
    ):
        va = float(getattr(ra, attr, 0) or 0)
        vb = float(getattr(rb, attr, 0) or 0)
        if va > 0 and vb > 0 and abs(va - vb) >= 0.15:
            if va > vb:
                a_wins.append(label)
                commentary.append(
                    f"{a} has more {label} than {b} — owner comfort and cabin presence favor the larger box."
                )
            else:
                b_wins.append(label)
                commentary.append(
                    f"{b} has more {label} than {a} — owner comfort and cabin presence favor the larger box."
                )

    cat_a = str(getattr(ra, "aircraft_category", "") or "").strip()
    cat_b = str(getattr(rb, "aircraft_category", "") or "").strip()
    if cat_a and cat_b and cat_a != cat_b:
        commentary.append(
            f"Class split: {a} is cataloged {cat_a}; {b} is {cat_b} — "
            "do not compare acquisition or OPEX as if they are the same cabin band."
        )
        if cat_a in ("large-cabin", "ultra-long") and cat_b not in ("large-cabin", "ultra-long"):
            a_wins.append("cabin class")
        elif cat_b in ("large-cabin", "ultra-long") and cat_a not in ("large-cabin", "ultra-long"):
            b_wins.append("cabin class")

    opex_a = float(getattr(ra, "confidence", 1) or 1)
    _ = opex_a  # reserved — OPEX from authority when variable_cost wired
    if cat_a == cat_b and cat_a in ("light", "super-midsize"):
        if range_a > 0 and range_b > 0 and range_a < range_b:
            a_wins.append("operating economics band")
            commentary.append(
                f"{a} is the lighter airframe in the same band — "
                f"typically lower direct operating cost than {b} on shorter missions."
            )
        elif range_b > 0 and range_a > 0 and range_b < range_a:
            b_wins.append("operating economics band")
            commentary.append(
                f"{b} is the lighter airframe in the same band — "
                f"typically lower direct operating cost than {a} on shorter missions."
            )

    tradeoffs: List[str] = []
    if a_wins and b_wins:
        tradeoffs.append(
            f"{a} leads on {', '.join(a_wins[:3])}; "
            f"{b} counters on {', '.join(b_wins[:3])}."
        )
    elif a_wins:
        tradeoffs.append(f"On verified specs alone, {a} is stronger across {', '.join(a_wins[:3])}.")
    elif b_wins:
        tradeoffs.append(f"On verified specs alone, {b} is stronger across {', '.join(b_wins[:3])}.")

    buy_a_if = ""
    buy_b_if = ""
    if a_wins:
        buy_a_if = f"You would favor {a} when {a_wins[0]} drives the mission more than {b}'s strengths."
    if b_wins:
        buy_b_if = f"You would favor {b} when {b_wins[0]} matters more than {a}'s advantages."

    summary_parts: List[str] = []
    if commentary:
        summary_parts.append(commentary[0])
    if len(commentary) > 1:
        summary_parts.append(commentary[1])
    if not summary_parts and tradeoffs:
        summary_parts.append(tradeoffs[0])
    broker_summary = " ".join(summary_parts).strip()

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "tradeoffs": tradeoffs,
        "commentary": commentary,
        "buy_a_if": buy_a_if,
        "buy_b_if": buy_b_if,
        "broker_summary": broker_summary,
    }


def _swap_spec_rows(rows: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    return [(r[0], r[2], r[1]) for r in rows]


def _lookup_table(a: str, b: str) -> List[Tuple[str, str, str]]:
    """Catalog / authority specs first; curated pair table is fallback only — never placeholders."""
    ra, rb, missing = _pair_authority_records(a, b)
    if not missing and ra and rb:
        catalog = _spec_rows_from_records(ra, rb)
        if len(catalog) >= 3:
            return catalog
    catalog = _catalog_spec_rows(a, b)
    if len(catalog) >= 3:
        return catalog
    entry, _key, swapped = _lookup_pair_entry(_COMPARISON_SPEC_ROWS, a, b)
    if entry:
        return _swap_spec_rows(entry) if swapped else entry
    if catalog:
        return catalog
    return []


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


def _commentary_beyond_summary(commentary: List[str], summary: str) -> List[str]:
    """Drop lines already spoken in the chat bubble lead-in."""
    low = (summary or "").lower()
    out: List[str] = []
    for raw in commentary:
        line = str(raw).strip()
        if not line:
            continue
        if line.lower() in low:
            continue
        out.append(line)
    return out


def build_comparison_broker_ui_payload(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Structured comparison card for the consultant chat UI (spec table + verdict)."""
    du = data_used if isinstance(data_used, dict) else {}
    a, b = _resolve_comparison_models(query, du)
    if a in ("Aircraft A", "") or b in ("Aircraft B", ""):
        return None

    ra, rb, missing = _pair_authority_records(a, b)
    if len(missing) == 2:
        return None

    if len(missing) == 1:
        unverified = missing[0]
        verified_rec = ra if ra else rb
        verified_name = a if ra else b
        if not verified_rec:
            return None
        cat = str(getattr(verified_rec, "aircraft_category", "") or "aircraft").strip()
        return {
            "model_a": a,
            "model_b": b,
            "specs": [],
            "verification_status": "partial",
            "missing_models": missing,
            "broker_notice": (
                f"{unverified} is not in our verified aircraft catalog — "
                "I will not quote side-by-side comparison specs for it."
            ),
            "broker_summary": (
                f"I can speak to verified {verified_name} data ({cat}, "
                f"~{int(verified_rec.nbaa_range_nm or 0):,} nm range). "
                f"For {unverified}, name the exact variant or pick a verified alternative."
            ),
            "commentary": [
                (
                    f"Verified {verified_name}: "
                    f"{_fmt_passengers(verified_rec)}; "
                    f"{_fmt_range_nm(verified_rec.nbaa_range_nm)} NBAA range."
                ),
            ],
        }

    rows = _lookup_table(a, b)
    if _rows_are_placeholders(rows):
        rows = []
    guidance = _lookup_guidance(a, b)
    if ra and rb and not guidance:
        guidance = _build_catalog_comparison_guidance(a, b, ra, rb)
    if not rows and not guidance:
        return None

    payload: Dict[str, Any] = {
        "model_a": a,
        "model_b": b,
        "specs": [{"dimension": dim, "a": va, "b": vb} for dim, va, vb in rows],
        "verification_status": "verified",
    }
    if guidance:
        summary = str(guidance.get("broker_summary") or "").strip()
        raw_tradeoffs = list(guidance.get("tradeoffs") or [])
        if not summary and raw_tradeoffs:
            summary = str(raw_tradeoffs[0]).strip()
        raw_commentary = list(guidance.get("commentary") or [])
        payload.update(
            {
                "a_wins": list(guidance.get("a_wins") or []),
                "b_wins": list(guidance.get("b_wins") or []),
                "tradeoffs": _commentary_beyond_summary(raw_tradeoffs, summary),
                "commentary": _commentary_beyond_summary(raw_commentary, summary),
                "buy_a_if": str(guidance.get("buy_a_if") or "").strip(),
                "buy_b_if": str(guidance.get("buy_b_if") or "").strip(),
                "broker_summary": summary,
            }
        )
        if guidance.get("operate_pick"):
            payload["operate_pick"] = str(guidance.get("operate_pick") or "").strip()
            payload["operate_why"] = str(guidance.get("operate_why") or "").strip()
    return payload


def attach_comparison_broker_ui(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Write ``comparison_broker_ui`` into ``data_used`` when a pair resolves."""
    if not isinstance(data_used, dict):
        return None
    payload = build_comparison_broker_ui_payload(query, data_used)
    if payload:
        data_used["comparison_broker_ui"] = payload
        data_used["broker_execution_category"] = "comparison"
    return payload


def render_comparison_full_text(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Plain-text comparison for copy/PDF — includes specs and verdict."""
    ui = build_comparison_broker_ui_payload(query, data_used)
    if not ui:
        return render_comparison_client_answer(query, data_used)
    a = str(ui.get("model_a") or "")
    b = str(ui.get("model_b") or "")
    lines: List[str] = []
    summary = str(ui.get("broker_summary") or "").strip()
    notice = str(ui.get("broker_notice") or "").strip()
    if summary:
        lines.append(summary)
    elif notice:
        lines.append(notice)
    else:
        lines.append(f"{a} vs {b}")
    for c in ui.get("commentary") or []:
        c = str(c).strip()
        if c and c not in lines:
            lines.append(c)
    for row in ui.get("specs") or []:
        if not isinstance(row, dict):
            continue
        dim = str(row.get("dimension") or "").strip()
        if not dim:
            continue
        lines.append(f"- {dim}: {a} {row.get('a', '')}; {b} {row.get('b', '')}")
    a_wins = ui.get("a_wins") or []
    b_wins = ui.get("b_wins") or []
    if a_wins:
        lines.append(f"{a} wins on: {', '.join(a_wins)}")
    if b_wins:
        lines.append(f"{b} wins on: {', '.join(b_wins)}")
    tradeoff_lead = summary or (str((ui.get("tradeoffs") or [""])[0]).strip() if ui.get("tradeoffs") else "")
    for t in _commentary_beyond_summary(list(ui.get("tradeoffs") or []), tradeoff_lead):
        lines.append(f"- {t}")
    buy_a = str(ui.get("buy_a_if") or "").strip()
    buy_b = str(ui.get("buy_b_if") or "").strip()
    if buy_a:
        lines.append(f"Buy {a} if: {buy_a}")
    if buy_b:
        lines.append(f"Buy {b} if: {buy_b}")
    return "\n".join(lines).strip()


def render_comparison_display_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Broker lead-in; spec table and commentary render separately in the chat UI."""
    ui = build_comparison_broker_ui_payload(query, data_used)
    if not ui:
        return render_comparison_client_answer(query, data_used)
    a = str(ui.get("model_a") or "")
    b = str(ui.get("model_b") or "")
    if ui.get("verification_status") == "partial":
        return str(ui.get("broker_summary") or ui.get("broker_notice") or f"{a} vs {b}.").strip()
    if re.search(r"(?is)\b(?:rather\s+operate|would\s+you\s+rather)\b", query or ""):
        pick = str(ui.get("operate_pick") or "").strip()
        why = str(ui.get("operate_why") or "").strip()
        if not pick:
            a_wins = ui.get("a_wins") or []
            b_wins = ui.get("b_wins") or []
            pick = a if len(a_wins) >= len(b_wins) else b
        return (
            f"**I would operate the {pick}** — "
            f"{why or 'better fit for owner-operator day-to-day economics and cabin experience.'}"
        ).strip()
    summary = str(ui.get("broker_summary") or "").strip()
    if summary:
        return summary
    tradeoffs = ui.get("tradeoffs") or []
    if tradeoffs:
        return str(tradeoffs[0]).strip()
    a_wins = ui.get("a_wins") or []
    b_wins = ui.get("b_wins") or []
    if a_wins or b_wins:
        bits: List[str] = []
        if a_wins:
            bits.append(f"{a} leads on {', '.join(a_wins[:4])}")
        if b_wins:
            bits.append(f"{b} leads on {', '.join(b_wins[:4])}")
        return ". ".join(bits) + "."
    return f"{a} vs {b}."


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
    "attach_comparison_broker_ui",
    "build_comparison_broker_facts_block",
    "build_comparison_broker_ui_payload",
    "render_comparison_client_answer",
    "render_comparison_display_answer",
    "render_comparison_full_text",
    "_COMPARISON_SPEC_ROWS",
]
