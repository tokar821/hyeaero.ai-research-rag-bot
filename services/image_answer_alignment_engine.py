"""
HyeAero.AI **image + answer alignment**: structured plan so the reply LLM keeps
gallery copy 1:1 with aircraft headings and visually grounded (no orphan “luxury” claims).

Deterministic clustering from titles/URLs + Phly/marketing candidates. Optional env disables.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def consultant_image_answer_alignment_enabled() -> bool:
    return (os.getenv("CONSULTANT_IMAGE_ANSWER_ALIGNMENT") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _phly_model_candidates(
    phly_rows: List[Dict[str, Any]],
    *,
    marketing_hint: Optional[str],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for r in phly_rows or []:
        if not isinstance(r, dict):
            continue
        man = str(r.get("manufacturer") or "").strip()
        mdl = str(r.get("model") or "").strip()
        try:
            from services.searchapi_aircraft_images import compose_manufacturer_model_phrase, normalize_aircraft_name

            mm = compose_manufacturer_model_phrase(man, mdl).strip()
            mm = normalize_aircraft_name(mm) if mm else ""
        except Exception:
            mm = " ".join(x for x in (man, mdl) if x).strip()
        if not mm or len(mm) < 3:
            continue
        k = mm.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append({"model": mm, "reason": "aircraft_record_context", "fit_score": 0.88})
        if len(out) >= limit:
            break
    if not out and (marketing_hint or "").strip():
        mh = str(marketing_hint).strip()
        out.append({"model": mh, "reason": "gallery_marketing_anchor", "fit_score": 0.74})
    return out[:limit]


def _blob_for_image(row: Dict[str, Any]) -> str:
    return f"{row.get('url') or ''} {row.get('description') or ''} {row.get('page_url') or ''}".lower()


def _best_matching_candidate(blob: str, candidates: List[str]) -> Optional[str]:
    best: Optional[str] = None
    best_len = 0
    for c in candidates:
        cl = c.lower()
        if cl in blob:
            if len(cl) > best_len:
                best_len = len(cl)
                best = c
        toks = [t for t in re.split(r"[^\w]+", cl) if len(t) >= 3]
        if any(t in blob for t in toks):
            if len(cl) > best_len:
                best_len = len(cl)
                best = c
    return best


def _group_images_by_aircraft(
    aircraft_images: List[Dict[str, Any]],
    candidate_models: List[str],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    fallback = candidate_models[0] if candidate_models else "Aircraft"
    for row in aircraft_images or []:
        if not str(row.get("url") or "").strip():
            continue
        b = _blob_for_image(row)
        m = _best_matching_candidate(b, candidate_models) or fallback
        groups[m].append(row)
    out = [{"model": k, "images": v, "count": len(v)} for k, v in groups.items()]
    out.sort(key=lambda g: -g["count"])
    return out[:3]


def _global_image_confidence(gallery_meta: Dict[str, Any]) -> float:
    gm = gallery_meta or {}
    eng = gm.get("image_query_engine") or {}
    try:
        qconf = float(eng.get("confidence") or 1.0)
    except (TypeError, ValueError):
        qconf = 1.0
    rk = gm.get("image_rank_filter_engine") or {}
    try:
        rconf = float(rk.get("confidence") or 1.0)
    except (TypeError, ValueError):
        rconf = 1.0
    return max(0.0, min(1.0, min(qconf, rconf)))


def _premium_intent_section_pass_rate(
    aircraft_images: List[Dict[str, Any]],
    premium_intent: Dict[str, Any],
) -> float:
    """Share of gallery rows that pass premium facet/tail rules when intent constrains visuals."""
    if not aircraft_images:
        return 0.0
    try:
        from services.consultant_image_search_orchestrator import premium_image_row_passes_validation
    except Exception:
        return 1.0
    ok = 0
    for im in aircraft_images:
        if not str(im.get("url") or "").strip():
            continue
        row = {
            "title": str(im.get("description") or ""),
            "snippet": "",
            "_source_page": str(im.get("page_url") or ""),
            "url": str(im.get("url") or ""),
        }
        if premium_image_row_passes_validation(row, premium_intent):
            ok += 1
    n = len([x for x in aircraft_images if str(x.get("url") or "").strip()])
    if not n:
        return 0.0
    return ok / float(n)


def build_image_answer_alignment_plan(
    *,
    user_query: str,
    aircraft_images: List[Dict[str, Any]],
    phly_rows: List[Dict[str, Any]],
    gallery_meta: Dict[str, Any],
    marketing_type_hint: Optional[str],
) -> Dict[str, Any]:
    """
    Build JSON-serializable alignment plan + a short **llm_directives** string for the system prompt.
    """
    candidates = _phly_model_candidates(phly_rows, marketing_hint=marketing_type_hint, limit=3)
    cand_models = [str(c["model"]) for c in candidates]
    if not cand_models and (marketing_type_hint or "").strip():
        cand_models = [str(marketing_type_hint).strip()]
    gconf = _global_image_confidence(gallery_meta)
    premium = gallery_meta.get("consultant_premium_intent") if isinstance(gallery_meta, dict) else None
    if not isinstance(premium, dict):
        premium = {}
    sec_rate = _premium_intent_section_pass_rate(aircraft_images, premium)

    groups = _group_images_by_aircraft(aircraft_images, cand_models or ["Aircraft"])

    low_conf = gconf < 0.7
    bad_section = sec_rate < 0.5 and (
        bool(str(premium.get("image_type") or "").strip())
        or (isinstance(premium.get("image_facets"), list) and len(premium.get("image_facets") or []) > 1)
        or bool(str(premium.get("tail_number") or "").strip())
    )
    weak = low_conf or not aircraft_images or bad_section
    alignment_ok = not weak and bool(groups) and all(g["count"] >= 1 for g in groups)

    weak_msg = None
    if weak:
        weak_msg = (
            "I cannot find reliable interior images for this aircraft. "
            "Here are the closest accurate references:"
        )

    itype = str(premium.get("image_type") or "").strip().lower()
    facets = premium.get("image_facets") if isinstance(premium.get("image_facets"), list) else []
    if itype in ("cockpit", "flight deck", "flightdeck"):
        section_label = "cockpit"
    elif itype in ("cabin", "interior", "salon"):
        section_label = "cabin"
    elif itype == "exterior":
        section_label = "exterior"
    elif len(facets) >= 2:
        section_label = "multi_facet"
    else:
        section_label = itype or "cabin"

    selected_flat: List[Dict[str, Any]] = []
    for g in groups:
        for im in g.get("images") or []:
            u = str(im.get("url") or "").strip()
            if not u:
                continue
            selected_flat.append(
                {
                    "url": u,
                    "score": round(gconf, 3),
                    "aircraft_detected": g.get("model"),
                    "section": section_label,
                }
            )

    best_model = ""
    if groups:
        best_model = str(groups[0]["model"] or "")

    directives = _format_llm_directives(
        user_intent=(user_query or "").strip()[:500],
        candidates=candidates,
        groups=groups,
        global_confidence=gconf,
        section_pass_rate=sec_rate,
        alignment_ok=alignment_ok,
        weak_message=weak_msg,
        best_model=best_model,
    )

    return {
        "user_intent": (user_query or "").strip()[:800],
        "aircraft_candidates": candidates,
        "image_groups": groups,
        "selected_images": selected_flat,
        "global_image_confidence": round(gconf, 4),
        "section_alignment_rate": round(sec_rate, 4),
        "alignment_ok": alignment_ok,
        "weak_images_message": weak_msg,
        "best_match_model": best_model or None,
        "llm_directives": directives,
    }


def _format_llm_directives(
    *,
    user_intent: str,
    candidates: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    global_confidence: float,
    section_pass_rate: float,
    alignment_ok: bool,
    weak_message: Optional[str],
    best_model: str,
) -> str:
    lines = [
        "**[IMAGE ANSWER ALIGNMENT — this turn]**",
        "Images are **evidence**; your text is the **argument**. They must match.",
    ]
    if user_intent:
        lines.append(f"- **User intent (short):** {user_intent}")
    if not alignment_ok and weak_message:
        lines.append(
            f"- **Weak or uncertain gallery** (confidence {global_confidence:.2f}, visual-facet match rate "
            f"{section_pass_rate:.0%}): open with exactly: *{_quote_user_phrase(weak_message)}*"
        )
        lines.append("- Do **not** invent cabin details or claim a perfect match to a specific tail/model.")
    else:
        lines.append(
            f"- **Gallery / engine confidence (combined):** {global_confidence:.2f}; "
            f"**visual-facet match rate:** {section_pass_rate:.0%} — stay visually honest."
        )
    if candidates:
        cand_txt = "; ".join(str(c.get("model")) for c in candidates[:3])
        lines.append(f"- **Aircraft candidates (mission context):** {cand_txt}")
    if groups:
        lines.append("- **Group images 1:1 with headings** — use exactly this structure order:")
        for g in groups[:4]:
            lines.append(
                f"  - **{g['model']}** ({g['count']} image(s) in app gallery) — "
                "write the explanation **only** for these URLs’ visible cues (use image titles/snippets; do not invent)."
            )
    lines.extend(
        [
            "- Under each aircraft, describe **only** what is plausible from **image titles + visible product context** "
            "(layout hints, club seating, galley line, window line, materials **if** titles/snippet imply them). "
            "**Forbidden** unless clearly grounded in those cues: *spacious*, *luxury*, *stunning*, *world-class*.",
            "- **Never** describe aircraft A using images grouped under B.",
            "- End with **Best Match:** one model + **Reason:** one line tying **mission fit** + **visual support** "
            "(what you can infer from the grouped image titles/snippets).",
        ]
    )
    if best_model and alignment_ok:
        lines.append(f"- Default **Best Match** lean (if ties): **{best_model}** (largest image group in this turn).")
    return "\n".join(lines)


def _quote_user_phrase(s: str) -> str:
    t = (s or "").strip()
    if len(t) > 160:
        return t[:157] + "..."
    return t


def format_alignment_block_for_layered_context(plan: Dict[str, Any], *, max_chars: int = 2800) -> str:
    """Compact block for RAG layered context (not full JSON dump)."""
    if not isinstance(plan, dict) or not plan.get("llm_directives"):
        return ""
    body = str(plan.get("llm_directives") or "").strip()
    groups = plan.get("image_groups") or []
    extra: List[str] = []
    for g in groups[:5]:
        extra.append(f"### {g.get('model')}")
        for im in (g.get("images") or [])[:4]:
            d = (im.get("description") or "").strip() or "(no title)"
            u = str(im.get("url") or "")[:120]
            extra.append(f"- {d} — {u}")
    tail = "\n".join(extra)
    out = f"{body}\n\n**Gallery rows (for grounding):**\n{tail}".strip()
    if len(out) > max_chars:
        return out[: max_chars - 3] + "..."
    return out
