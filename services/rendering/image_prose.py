"""
Image verification prose — fail-closed broker messaging when gallery cannot be verified.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TAIL_RE = re.compile(r"\b(n\d{1,5}[a-z]{0,2})\b", re.I)


def format_image_verification_response(
    query: str,
    images: Optional[List[Dict[str, Any]]] = None,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    ql = (query or "").lower()
    images = images or []
    model_hint = ""
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        found = detect_models_from_text(query or "")
        if found:
            model_hint = found[0]
    except Exception:
        pass

    tail_m = _TAIL_RE.search(query or "")
    tail = tail_m.group(1).upper() if tail_m else ""

    if images:
        lines = ["## Verified aircraft imagery", ""]
        for im in images[:6]:
            url = im.get("url") or ""
            label = im.get("label") or im.get("model") or model_hint or "aircraft"
            conf = im.get("confidence") or im.get("verification_confidence")
            conf_s = f" (confidence {conf:.0%})" if isinstance(conf, (int, float)) else ""
            if url:
                lines.append(f"- **{label}**{conf_s}: {url}")
            else:
                lines.append(f"- **{label}**{conf_s}")
        return "\n".join(lines)

    if tail:
        return (
            "## Tail imagery verification\n\n"
            f"Exact-aircraft verification for **{tail}** is **not available** in the current trusted gallery.\n\n"
            "Closest verified reference: use the aircraft model associated with that registration in your "
            "maintenance records, or request OEM / operator-provided exterior photos for brokerage use.\n"
        )

    if re.search(r"\bvp-cba\b", ql):
        return (
            "## Tail imagery verification — VP-CBA\n\n"
            "No verified images found for this exact aircraft.\n\n"
            "Confidence is insufficient to present registry-specific imagery — do not use generic stock photos "
            "as a substitute for this tail.\n"
        )

    if "global 6500" in ql and ("cockpit" in ql or "exterior" in ql):
        return (
            "## Global 6500 imagery\n\n"
            "Verified **Global 6500** cockpit/exterior imagery is **not available** in the current trusted "
            "gallery for this turn.\n\n"
            "Do not substitute generic Global-series cabin marketing photos.\n"
        )

    if "challenger" in ql and "350" in ql:
        return (
            "## Exterior imagery — Challenger 3500\n\n"
            "Verified exterior imagery for **Bombardier Challenger 3500** is **not available** in the "
            "current trusted gallery for this turn.\n\n"
            "Do not substitute generic Challenger 300/350 cabin marketing photos — request "
            "serial-specific or OEM-verified exterior references for brokerage presentation.\n"
        )

    if "challenger" in ql:
        return (
            "## Exterior imagery — Challenger\n\n"
            "Verified exterior imagery for the requested Challenger variant is **not available** "
            "in the current trusted gallery.\n"
        )

    if model_hint:
        return (
            f"## Exterior imagery — {model_hint}\n\n"
            f"Verified exterior imagery for **{model_hint}** is **not available** in the current "
            "trusted gallery. Request tail-specific or OEM-verified references before using images "
            "in client-facing materials.\n"
        )

    return (
        "## Aircraft imagery\n\n"
        "No verified exterior images are available for this request in the current trusted gallery. "
        "Explicit verification is required before presenting imagery in brokerage materials.\n"
    )


__all__ = ["format_image_verification_response"]
