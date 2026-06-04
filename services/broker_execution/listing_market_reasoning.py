"""
Listing price reasoning — broker-style comps narrative (not bare median).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def render_listing_price_reasoning(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """
    Client-facing price opinion when model/year/ask are in the query.
    """
    du = data_used if isinstance(data_used, dict) else {}
    audit = du.get("listing_parse_audit") or {}
    if not isinstance(audit, dict) or not audit.get("parse_success"):
        try:
            from services.broker_execution.listing_parse_audit import build_listing_parse_audit

            audit = build_listing_parse_audit(query)
        except Exception:
            return ""

    model = str(audit.get("detected_model") or "").strip()
    year = audit.get("detected_year")
    price = audit.get("detected_price")
    if not model or price is None:
        return ""

    if not re.search(r"(?is)\b(?:aggressive|fair|cheap|overpriced|underpriced|realistic)\b", query or ""):
        return ""

    mr = du.get("market_reality") if isinstance(du.get("market_reality"), dict) else {}
    band_lo = mr.get("band_low_usd") or mr.get("low_band_usd")
    band_hi = mr.get("band_high_usd") or mr.get("high_band_usd")
    band_mid = mr.get("band_mid_usd") or mr.get("mid_band_usd")

    lines = [
        f"On a {year or '—'} {model} listed around ${price:.1f}M, frame the ask against comps — not a single median:",
        "",
        "• Hours and cycles vs peers",
        "• Engine/APU program enrollment and transferability",
        "• Pedigree (corporate, 135, owner-flown) and maintenance tracking",
        "• Damage, AD compliance, and avionics vintage",
    ]
    if band_lo and band_hi:
        try:
            lo, hi = float(band_lo) / 1_000_000, float(band_hi) / 1_000_000
            lines.append(f"• Indicative synced band for this model/year: roughly ${lo:.1f}M–${hi:.1f}M")
            if price > hi * 1.05:
                lines.append(f"At ${price:.1f}M I would call this aggressive unless programs and pedigree are exceptional.")
            elif price < lo * 0.95:
                lines.append(f"At ${price:.1f}M this looks cheap — verify damage, times, and program gaps.")
            else:
                lines.append(f"At ${price:.1f}M this is fair if hours/programs match the band; verify the specific tail.")
        except (TypeError, ValueError):
            if band_mid:
                lines.append(f"• Market mid reference (indicative): ${float(band_mid)/1_000_000:.1f}M")
    else:
        lines.append(
            f"• Without live comps in sync, treat ${price:.1f}M as conditional — pull recent sales and program status before a verdict."
        )

    return "\n".join(lines).strip()
