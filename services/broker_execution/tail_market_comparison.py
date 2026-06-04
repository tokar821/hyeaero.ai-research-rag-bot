"""
Tail vs market comparison — broker answer when user compares a registration to comps / fleet / typical.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_execution.tail_acquisition_dossier import _fmt_price, _hours_context, _phly_row, _reg


_TAIL_MARKET_QUERY_RE = re.compile(
    r"(?is)\b(?:"
    r"vs\.?\s+(?:the\s+)?(?:market|typical|comps?|fleet|similar|active\s+inventory|type\s+average)|"
    r"versus\s+(?:the\s+)?(?:market|typical|comps?|fleet)|"
    r"compare(?:d)?\s+(?:to|with|against)\s+(?:the\s+)?(?:market|typical|comps?|fleet|similar)|"
    r"against\s+(?:the\s+)?(?:market|typical|comps?|fleet|similar)|"
    r"how\s+does\s+.{0,60}\s+stack\s+up|"
    r"on\s+the\s+market|market\s+position\s+of|"
    r"vs\.?\s+typical\s+\w"
    r")\b"
)


def is_tail_market_comparison_query(query: str, data_used: Optional[Dict[str, Any]] = None) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    du = data_used if isinstance(data_used, dict) else {}
    reg = _reg(q, du)
    if not reg:
        return False
    if not _TAIL_MARKET_QUERY_RE.search(q):
        return False
    try:
        from services.broker_execution.comparison_broker_facts import _resolve_comparison_models

        a, b = _resolve_comparison_models(q, du)
        if a and b and a != "Aircraft A" and b != "Aircraft B":
            low = q.lower()
            if re.search(r"(?is)\bvs\.?|\bversus\b", low):
                if reg.lower() not in a.lower() and reg.lower() not in b.lower():
                    return False
    except Exception:
        pass
    return True


def render_tail_market_comparison_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Client-facing: this tail vs type market / comps (not model-vs-model table)."""
    du = data_used if isinstance(data_used, dict) else {}
    if not is_tail_market_comparison_query(query, du):
        return ""

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query

        ensure_tail_facts_for_query(query, du)
    except Exception:
        pass

    reg = _reg(query, du)
    phly = _phly_row(du, reg)
    if not reg:
        return ""

    mm = " ".join(
        x
        for x in (
            (phly.get("manufacturer") or "").strip(),
            (phly.get("model") or "").strip(),
        )
        if x
    ).strip()
    opener = f"{reg}" + (f" ({mm})" if mm else "") + " vs the active market for this type:"

    bullets: List[str] = []
    ask = _fmt_price(phly.get("ask_price"))
    status = (phly.get("aircraft_status") or "").strip()
    hours = phly.get("airframe_total_time") or phly.get("total_time")
    year = phly.get("year") or phly.get("year_mfr")
    eng = (phly.get("engine_program") or "").strip()
    dom = phly.get("days_on_market") or phly.get("dom")

    if ask or status:
        bullets.append(
            f"• Ask / status: {status or 'listed'}{f' at {ask}' if ask else ''} — "
            "benchmark against recent same-model closings and current ask distribution."
        )
    else:
        bullets.append("• Ask / status: confirm live ask and listing status before anchoring comps.")

    mr = du.get("market_reality")
    if isinstance(mr, dict) and mr:
        band = mr.get("band_mid_usd") or mr.get("mid_band_usd")
        if band is not None:
            try:
                mid_m = float(band) / 1_000_000
                bullets.append(
                    f"• Market band: ~${mid_m:.1f}M mid for the type — classify this tail as premium, "
                    "in-line, or discount vs band before LOI."
                )
            except (TypeError, ValueError):
                bullets.append(f"• Market band: {band} — position ask vs mid-band comps.")
        note = str(mr.get("brief") or mr.get("summary") or "").strip()
        if note and len(note) < 280:
            bullets.append(f"• Market read: {note}")

    bullets.append(f"• Utilization: {_hours_context(hours, year)}")

    if eng:
        bullets.append(
            f"• Programs: {eng} — enrolled tails trade differently; verify transfer at closing."
        )
    else:
        bullets.append("• Programs: no engine program on file — treat as a pricing and resale discount vs enrolled comps.")

    if dom not in (None, ""):
        bullets.append(f"• Time on market: {dom} days — long DOM vs peers signals ask resistance or missing data.")
    else:
        bullets.append("• Time on market: confirm DOM; stale listings often need ask reset or seller motivation check.")

    bullets.append(
        "• Liquidity: same-model inventory is the comp set — don't average in unrelated segments "
        "(different year, program state, or damage history)."
    )
    bullets.append(
        "• Broker read: underwrite to **closed** comps and program transferability, not brochure ask; "
        "if ask is above band with high hours, negotiate on engine reserves and upcoming inspections."
    )

    return opener + "\n" + "\n".join(bullets[:8])


def build_tail_market_comparison_block(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Fact-pack block for LLM context."""
    body = render_tail_market_comparison_answer(query, data_used)
    if not body:
        return ""
    return "[TAIL VS MARKET — broker authority]\n" + body


__all__ = [
    "build_tail_market_comparison_block",
    "is_tail_market_comparison_query",
    "render_tail_market_comparison_answer",
]
