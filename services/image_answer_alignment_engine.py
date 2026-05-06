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
    # Latest-turn marketing anchor (from explicit query inference) must lead grouping — stale Phly
    # rows from earlier purchase threads must not reorder gallery headings.
    mh0 = (marketing_hint or "").strip()
    if mh0 and len(mh0) >= 3:
        seen.add(mh0.lower())
        out.append({"model": mh0, "reason": "latest_query_marketing_anchor", "fit_score": 0.92})
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


def _alignment_image_blob(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("title", "description", "url", "source_domain", "page_url", "source")
    ).lower()


def _answer_mentions_models(answer: str, candidates: Optional[List[str]]) -> List[str]:
    """Models explicitly grounded in answer text (subset of candidates when possible)."""
    al = (answer or "").lower()
    out: List[str] = []
    seen: set[str] = set()
    for c in candidates or []:
        cs = str(c).strip()
        if not cs or len(cs) < 3:
            continue
        cl = cs.lower()
        if cl in al:
            if cl not in seen:
                seen.add(cl)
                out.append(cs)
            continue
        toks = [t for t in re.split(r"[^\w]+", cl) if len(t) >= 2 and t not in ("g", "iv", "lx", "ex")]
        if len(toks) >= 2 and all(t in al for t in toks[-2:]):
            if cl not in seen:
                seen.add(cl)
                out.append(cs)
    return out


def _exterior_only_blob(blob: str) -> bool:
    if re.search(
        r"\b(ramp|taxi|takeoff|landing|exterior|parked|gear\s*down|rotate|airside)\b",
        blob,
        re.I,
    ) and not re.search(
        r"\b(cabin|interior|seating|windows?|galley|divan|cockpit|flight\s*deck)\b",
        blob,
        re.I,
    ):
        return True
    return False


def _dominant_model_for_row(blob: str, candidates: List[str]) -> Optional[str]:
    best = None
    best_len = 0
    for c in candidates:
        cl = c.lower()
        if cl in blob and len(cl) > best_len:
            best_len = len(cl)
            best = c
    return best


def _claim_style_terms(answer: str) -> List[str]:
    al = (answer or "").lower()
    terms: List[str] = []
    for phrase, toks in (
        ("modern interior", ("modern", "interior")),
        ("ambient lighting", ("ambient", "lighting")),
        ("luxury", ("luxury",)),
    ):
        if phrase.replace(" ", "") in al.replace(" ", "") or all(t in al for t in toks):
            terms.extend(list(toks))
    return list(dict.fromkeys(terms))


def _blob_supports_claims(blob: str, claims: List[str]) -> bool:
    if not claims:
        return True
    for t in claims:
        if t in blob:
            return True
    return False


def _row_matches_answer_aircraft(blob: str, mentioned: List[str]) -> bool:
    """True only if blob meets **≥ 0.65** aircraft similarity to an allowed model mention."""
    if not mentioned:
        return True
    for m in mentioned:
        ml = m.lower()
        if ml in blob:
            return True
        try:
            from services.aviation_image_rank_filter_engine import _aircraft_match_score

            if float(_aircraft_match_score(m, blob)) >= 0.65:
                return True
        except Exception:
            continue
    return False


def _row_matches_aircraft_candidates_strict(blob: str, candidates: List[str]) -> bool:
    """Every retained row must satisfy the candidate lock."""
    if not candidates:
        return True
    best = 0.0
    try:
        from services.aviation_image_rank_filter_engine import _aircraft_match_score

        for c in candidates:
            best = max(best, float(_aircraft_match_score(c, blob)))
    except Exception:
        return False
    return best >= 0.65


def align_images_with_consultant_answer(
    *,
    answer_text: str,
    normalized_intent: Optional[Dict[str, Any]],
    selected_images: List[Dict[str, Any]],
    aircraft_candidates: Optional[List[str]] = None,
    image_pool: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Align ranked/selected images with the **final** consultant answer + intent.

    Each input row should include at least ``url`` plus text fields (``title`` / ``description``)
    so aircraft / visual / claim checks can run. Ranking-engine fields (``score``, …) are preserved.

    Returns:
        ``final_images``, ``alignment_score`` (0–1), ``issues`` (strings), ``fix_applied`` (bool).

    Optional ``image_pool`` supplies extra rows for claim-gap replacement or minimum-count fallback
    (re-ranked with a single strict aircraft when needed). No user-facing prose is added here.
    """
    issues: List[str] = []
    fix_applied = False
    norm = normalized_intent if isinstance(normalized_intent, dict) else {}
    cands = [str(x).strip() for x in (aircraft_candidates or []) if str(x).strip()]

    rows = [dict(x) for x in (selected_images or []) if str(x.get("url") or "").strip()]
    rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

    mentioned = _answer_mentions_models(answer_text or "", cands)
    vf = str(norm.get("visual_focus") or "").strip().lower()
    it = str(norm.get("intent_type") or "").strip().lower()
    # Interior/cabin user ask → strip exterior-only rows (cockpit intent handled separately).
    no_exterior_when_cabin_intent = it in ("interior_visual", "cabin_search") or vf in (
        "interior",
        "cabin",
        "bedroom",
        "galley",
    )

    claims = _claim_style_terms(answer_text or "")

    # --- 0) Mandatory candidate lock ---
    kept0: List[Dict[str, Any]] = []
    for r in rows:
        b = _alignment_image_blob(r)
        if not _row_matches_aircraft_candidates_strict(b, cands):
            issues.append(f"removed_image_candidate_lock_aircraft<{r.get('url','')[:48]}>")
            fix_applied = True
            continue
        kept0.append(r)
    rows = kept0

    # --- 1) Aircraft wording vs answer-derived mentions ---
    if mentioned:
        kept: List[Dict[str, Any]] = []
        for r in rows:
            b = _alignment_image_blob(r)
            if not _row_matches_answer_aircraft(b, mentioned):
                issues.append(f"removed_image_not_matching_answer_aircraft:{r.get('url','')[:48]}")
                fix_applied = True
                continue
            kept.append(r)
        rows = kept

    # --- 4) Interior / cabin → no exterior-only rows ---
    if no_exterior_when_cabin_intent and vf != "exterior":
        kept2: List[Dict[str, Any]] = []
        for r in rows:
            b = _alignment_image_blob(r)
            if vf not in ("exterior",) and _exterior_only_blob(b):
                issues.append("removed_exterior_only_under_interior_intent")
                fix_applied = True
                continue
            kept2.append(r)
        rows = kept2

    # --- 3) Single aircraft cluster ---
    if len(cands) >= 2 and len(rows) >= 2:
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            b = _alignment_image_blob(r)
            dm = _dominant_model_for_row(b, cands) or (cands[0] if cands else "")
            clusters[dm or "_"].append(r)
        best_key = max(clusters.keys(), key=lambda k: len(clusters[k]))
        if len(clusters) > 1:
            merged = clusters[best_key]
            if len(merged) < len(rows):
                issues.append("removed_mixed_aircraft_cluster_outliers")
                fix_applied = True
            rows = merged

    # --- Claims: strict per-row; no pool backfill ---
    if claims:
        kept_claims: List[Dict[str, Any]] = []
        for r in rows:
            b = _alignment_image_blob(r)
            if not _blob_supports_claims(b, claims):
                issues.append(f"removed_claim_mismatch<{r.get('url','')[:48]}>")
                fix_applied = True
                continue
            kept_claims.append(r)
        rows = kept_claims

    rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    rows = rows[:5]

    if len(rows) < 3:
        issues.append("below_three_strict_aligned_images_returning_empty")
        fix_applied = True
        rows = []

    # Alignment score
    if not rows:
        align = 0.15
    else:
        base = sum(float(r.get("score") or 0.72) for r in rows) / max(1, len(rows))
        pen = min(0.45, 0.08 * len(issues))
        align = max(0.0, min(1.0, base - pen))

    # Strip internal-only keys for output
    final: List[Dict[str, Any]] = []
    for r in rows[:6]:
        out_row = {k: v for k, v in r.items() if not str(k).startswith("_")}
        final.append(out_row)

    return {
        "final_images": final,
        "alignment_score": round(float(align), 4),
        "issues": issues,
        "fix_applied": bool(fix_applied),
    }


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
