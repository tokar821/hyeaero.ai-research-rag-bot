"""Deterministic refinement interpreter + deictic query reinforcement."""

from __future__ import annotations

import re
from typing import List, Optional

from .schemas import RefinementInterpretation


_RESET_RE = re.compile(
    r"\b(start\s+over|new\s+topic|forget\s+(?:everything|that)|different\s+(?:subject|topic)|reset\b)\b",
    re.I,
)


def interpret_refinement(query: str, *, prev_aircraft: Optional[str], prev_tail: Optional[str]) -> RefinementInterpretation:
    ql = (query or "").strip().lower()
    if _RESET_RE.search(ql):
        return RefinementInterpretation(type="explicit_reset", inherit_entity=False, notes="User asked to reset thread")

    if re.search(
        r"\bactually\b.*\b(bigger|larger|more\s+(?:space|room|cabin))\b|"
        r"\b(?:something\s+)?bigger\b|\b(?:step|move)\s*up\b|"
        r"\byounger\s+feeling\b",
        ql,
    ):
        return RefinementInterpretation(
            type="size_upgrade",
            reference_aircraft=prev_aircraft,
            reference_tail=prev_tail,
            preserve_traits=[],
            notes="Upsize versus prior focal aircraft",
        )

    if re.search(
        r"\b(smaller|cheaper\b.*jet|cheaper\b| tighter\s+budget)\b|\b(?:downsize|step\s+down)\b",
        ql,
    ):
        bd = RefinementInterpretation(
            type="size_or_budget_down",
            reference_aircraft=prev_aircraft,
            reference_tail=prev_tail,
            notes="User wants smaller or cheaper",
        )
        return bd

    if re.search(r"\b(not\s+)?that\s+expensive\b|\b(can't\s+go)\s+full\b|\bbe[- ]reasonable\b|\b(value|budget)\s+(?:friendly|conscious)\b", ql):
        return RefinementInterpretation(
            type="budget_shift",
            remove_traits=["ultra-premium"],
            add_traits=["value-conscious"],
            notes="Budget sensitivity",
        )

    if re.search(
        r"\bless\s+corporate\b|\b(?:not\s+)?corporate\b|\b(relaxed|softer|hotel|residential)\b|"
        r"\bold[- ]money\b|\bbanker\s+vibe\b",
        ql,
    ):
        return RefinementInterpretation(
            type="style_shift",
            remove_traits=["corporate"],
            add_traits=["residential aesthetic", "lifestyle cabin"],
            notes="Corporate-averse framing",
        )

    if re.search(r"\bmore\s+modern\b|\b(updated|minimal|clean\s+lines|contemporary)\b", ql):
        return RefinementInterpretation(
            type="style_shift",
            add_traits=["modern", "contemporary"],
            notes="Modernization preference",
        )

    if re.search(r"\binfluencers?\s+rent\b|\binstagram\b|\bwow\s+factor\b|\b(statement|presence)\s+jet\b", ql):
        return RefinementInterpretation(
            type="lifestyle_inference",
            inferred_style_tags=["lifestyle-charter vibe", "head-turning visuals", "prestige interior"],
            add_traits=["media-friendly cabin", "strong visual runway presence"],
            notes="Likely influencer / prestige charter aesthetic",
        )

    m_vs = re.search(
        r"(?:\bcompare\s+)?(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[\.\!]?\s*$",
        ql,
        re.I,
    )
    if m_vs:
        left, right = m_vs.group(1).strip(), m_vs.group(2).strip()
        left = re.sub(r"^compare\s+", "", left, flags=re.I).strip()
        if left and right:
            return RefinementInterpretation(
                type="comparison_anchor",
                reference_aircraft=f"{left} vs {right}",
                add_traits=["comparison shopping"],
                notes="Explicit model comparison",
            )

    # Deictic comparison: "Compare that preference to a Gulfstream G650 cabin"
    if re.search(r"\bcompare\b", ql) and re.search(r"\b(that|this)\b", ql) and re.search(
        r"\b(preference|interior|cabin|cockpit|feel|vibe)\b", ql
    ):
        return RefinementInterpretation(
            type="comparison_anchor",
            reference_aircraft=prev_aircraft or "",
            add_traits=["comparison shopping"],
            notes="Deictic comparison anchored to prior preference",
        )

    if re.search(r"\bcompare\b", ql) and re.search(
        r"\bg650\b|\bgulfstream\b", ql
    ) and re.search(r"\b(cheaper|less expensive|vs\.?|versus|alternative)\b", ql):
        return RefinementInterpretation(
            type="comparison_anchor",
            reference_aircraft="Gulfstream G650 family",
            add_traits=["comparison shopping"],
            notes="Compare against G650 / ULR flagship",
        )

    if re.search(r"\bbox\s+spring\b|\bbedroom\b|\bberth\b|\bdivan\b", ql):
        return RefinementInterpretation(
            type="sleeping_configuration",
            requested_view="cabin berth / divan",
            inherit_entity=True,
        )

    if re.search(r"\b(old|dated|retro|tired\s+looking)\b", ql):
        return RefinementInterpretation(
            type="style_shift",
            remove_traits=["classic traditional"],
            add_traits=["renewed completions", "contemporary refurbishment"],
            notes="Reject dated finishes",
        )

    if re.search(
        r"\b(?:now\s+)?show\s+cockpit\b|\bcockpit\s+too\b|\band\s+cockpit\b|\bnow\s+cockpit\b|\bflight\s+deck\b",
        ql,
    ):
        return RefinementInterpretation(
            type="view_change",
            requested_view="cockpit",
            inherit_entity=True,
        )

    if re.search(r"^(?:show\s+me|show\s+us|pics?|photos?|images?)\s*$|\b(show\s+(?:me\s+)?(?:that|this|more)?)\b|\b(let\s+(?:me|us)\s+see)\b", ql):
        return RefinementInterpretation(type="ambiguous_followup", inherit_entity=True, notes="Likely gallery / carry-forward intent")

    if len(ql) < 140 and re.search(r"\b(that|this|those|these)\s+(looks?|looks like|feel)\b|\bsame\s+(thing|plane|jet)\b", ql):
        return RefinementInterpretation(type="ambiguous_followup", inherit_entity=True)

    return RefinementInterpretation(type="none", inherit_entity=True)


def merge_traits(base: List[str], add: List[str], remove: List[str]) -> List[str]:
    out = [str(x).strip() for x in base if str(x).strip()]
    rem = {r.lower() for r in remove}
    out = [x for x in out if x.lower() not in rem]
    for a in add:
        s = str(a).strip()
        if s and all(s.lower() != o.lower() for o in out):
            out.append(s)
    return out[-48:]


def reinforce_query_with_context(
    query: str,
    *,
    interpretation: RefinementInterpretation,
    locked_tail: Optional[str],
    locked_model: Optional[str],
    augment_size: bool,
    size_augment_fragment: str,
) -> str:
    """Attach implicit references for retrieval-only / short lines."""
    q = (query or "").strip()
    if not q:
        return q
    extra: List[str] = []

    if locked_tail:
        ql = q.lower()
        if locked_tail.lower() not in ql and (
            interpretation.type
            in (
                "view_change",
                "ambiguous_followup",
                "sleeping_configuration",
                "none",
                "style_shift",
            )
            or re.search(r"\b(show|see|photos?|interior|cabin|cockpit|inside|gallery)\b", ql)
        ):
            extra.append(f"tail {locked_tail}")

    elif locked_model and interpretation.inherit_entity:
        ql = q.lower()
        if (locked_model or "").lower() not in ql and interpretation.type in (
            "view_change",
            "ambiguous_followup",
            "style_shift",
            "none",
            "comparison_anchor",
        ):
            extra.append(str(locked_model))

    if augment_size and size_augment_fragment and interpretation.type == "size_upgrade":
        extra.append(size_augment_fragment.strip())

    if not extra:
        return q
    return (q + " " + " ".join(extra)).strip()
