"""
Aircraft cabin-class ladder hints for refinement (not a complete type database).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .schemas import AircraftCategory


def categorize_model_hint(model_name: Optional[str]) -> AircraftCategory:
    """Best-effort class from marketing name tokens."""
    if not model_name:
        return AircraftCategory.UNKNOWN
    m = (model_name or "").lower()
    if re.search(r"\b(phenom\s*100|mustang|eclipse|cj1|citation\s*jet|m2)\b", m):
        return AircraftCategory.VLJ if "100" in m else AircraftCategory.LIGHT
    if re.search(r"\b(phenom\s*300|lear\s*jet|cj3|citation\s*cj|encore|beechjet|hawker\s*400)\b", m):
        return AircraftCategory.LIGHT if re.search(r"(cj3|citation|cj2)", m) else AircraftCategory.MIDSIZE
    if re.search(
        r"\b(latitude|longitude|sovereign|challenger\s*350|praetor|falcon\s*50|hawker\s*[^4]|g280)\b",
        m,
        re.I,
    ):
        return AircraftCategory.SUPER_MID
    if re.search(r"\b(falcon\s*7x|falcon\s*8x|global\s*|g\s*550|g\s*650|g700|7500)\b", m, re.I):
        return AircraftCategory.ULR if re.search(r"(7500|g700|6500|650er|550)", m, re.I) else AircraftCategory.LARGE
    if re.search(r"\b(challenger\s*6|605|850)\b", m, re.I):
        return AircraftCategory.LARGE
    return AircraftCategory.UNKNOWN


def evolution_hint_for_upgrade(
    prev_model: Optional[str],
    *,
    max_budget_usd: Optional[float] = None,
    refinement_query: str = "",
) -> str:
    """Natural-language suffix for retrieval when user asks for bigger / more cabin."""
    rq = (refinement_query or "").strip()
    if re.match(r"^\s*bigger\s*[\.\!]?\s*$", rq, re.I):
        try:
            from services.intent_persistence.pivot import bigger_modern_cabin_models

            models = bigger_modern_cabin_models()
            return (
                f" Upscale cabin volume while keeping modern lounge feel: "
                f"{', '.join(models)}."
            )
        except Exception:
            pass
    if not (prev_model or "").strip():
        return ""
    cls, hint = suggest_next_class(prev_model, max_budget_usd=max_budget_usd)
    if max_budget_usd is not None and max_budget_usd <= 12_000_000:
        return (
            f" Larger cabin than {prev_model.strip()} within ~${max_budget_usd/1_000_000:.0f}M — "
            f"super-midsize (e.g. Challenger 350, Citation Latitude, Legacy 650), not ULR flagships. "
        )
    return f" Larger cabin than {prev_model.strip()} targeting {cls.value.replace('_', ' ')} class or higher. {hint}"


def suggest_next_class(
    model_name: Optional[str],
    *,
    max_budget_usd: Optional[float] = None,
) -> Tuple[AircraftCategory, str]:
    cur = categorize_model_hint(model_name)
    order = [
        AircraftCategory.VLJ,
        AircraftCategory.LIGHT,
        AircraftCategory.MIDSIZE,
        AircraftCategory.SUPER_MID,
        AircraftCategory.LARGE,
        AircraftCategory.ULR,
    ]
    try:
        idx = order.index(cur)
    except ValueError:
        idx = 2  # default midsize band
    nxt = order[min(idx + 1, len(order) - 1)]
    if max_budget_usd is not None and max_budget_usd <= 12_000_000:
        if nxt in (AircraftCategory.LARGE, AircraftCategory.ULR):
            nxt = AircraftCategory.SUPER_MID
    elif max_budget_usd is not None and max_budget_usd <= 22_000_000:
        if nxt == AircraftCategory.ULR:
            nxt = AircraftCategory.LARGE
    return nxt, f"Prefer {nxt.value.replace('_', ' ')}"
