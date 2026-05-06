"""
Aviation Image Ranking Engine — scores gallery rows vs **normalized intent** + **aircraft_candidates**.

Weights: aircraft 40%, visual 30%, source 20%, semantic alignment 10%.
Hard gate: ``aircraft_match < 0.65`` or **visual mismatch** → final score 0 (reject). No fractional
visual penalties. Output sorted by score descending; keeps scores ≥ ``_MIN_FINAL_SCORE``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from services.aviation_image_rank_filter_engine import _aircraft_match_score

_MIN_FINAL_SCORE = 0.65
_AIRCRAFT_MATCH_HARD_REJECT = 0.65
_WEIGHT_AIR = 0.40
_WEIGHT_VIS = 0.30
_WEIGHT_SRC = 0.20
_WEIGHT_SEM = 0.10

_INTERIOR_CUES = re.compile(
    r"\b(cabin|interior|seating|windows?|aisle|divan|galley|lav|berth|"
    r"club\s*seating|vip|main\s+cabin)\b",
    re.I,
)
_EXTERIOR_CUES = re.compile(
    r"\b(ramp|taxi|takeoff|landing|exterior|airside|parked|gear\s*down|rotate)\b",
    re.I,
)
_COCKPIT_CUES = re.compile(
    r"\b(cockpit|flight\s*deck|pfd|mfd|fms|glass\s*cockpit|avionics|pedestal|throttle)\b",
    re.I,
)
_BEDROOM_CUES = re.compile(
    r"\b(bed|berth|divan|sleeping|aft\s*cabin|state\s*room)\b",
    re.I,
)

_HIGH_TRUST = re.compile(
    r"(?is)("
    r"gulfstream\.com|bombardier\.com|dassault-aviation\.com|embraer\.com|airbus\.com|boeing\.com|"
    r"controller\.com|globalair\.com|avbuyer\.com|jetcraft\.com|amjet\.com|"
    r"jetphotos\.net|planespotters\.net|airliners\.net|flightaware\.com|"
    r"ainonline\.com|flightglobal\.com|aviationweek\.com|bjtonline\.com"
    r")",
)
_MED_TRUST = re.compile(
    r"(?is)(wikipedia\.org|simpleflying\.com|onemileatatime|thepointsguy|"
    r"skybrary\.aero|easa\.|faa\.gov)",
)

_LOW_TRUST = re.compile(
    r"(?is)(reddit\.|redd\.it|imgflip|pinterest|facebook\.com|instagram|"
    r"blogspot\.|wordpress\.com|tumblr\.)",
)


def _image_blob(im: Dict[str, Any]) -> str:
    meta = im.get("metadata")
    if isinstance(meta, (dict, list)):
        meta_s = json.dumps(meta, default=str)[:2000]
    else:
        meta_s = str(meta or "")
    parts = [
        str(im.get("title") or ""),
        str(im.get("description") or ""),
        str(im.get("url") or ""),
        str(im.get("source_domain") or ""),
        str(im.get("source") or ""),
        meta_s,
    ]
    return " ".join(parts).strip().lower()


def _domain_blob(im: Dict[str, Any]) -> str:
    d = str(im.get("source_domain") or "").strip().lower()
    if d:
        return d
    u = str(im.get("url") or "").lower()
    m = re.search(r"://([^/]+)/", u)
    return (m.group(1) if m else u)[:200]


def _candidates_list(
    aircraft_candidates: Optional[List[str]],
    normalized_intent: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Explicit ``[]`` locks to an empty roster (caller forces **no** intent-aircraft augmentation).
    ``None`` still allows intent.aircraft augmentation for backward compatibility.
    """
    if aircraft_candidates is not None:
        return [str(x).strip() for x in aircraft_candidates if str(x).strip()]
    out: List[str] = []
    if isinstance(normalized_intent, dict):
        a = str(normalized_intent.get("aircraft") or "").strip()
        if a:
            out.append(a)
    return out


def _aircraft_match_component(blob: str, candidates: List[str]) -> float:
    """0–1; best match across candidate aircraft strings."""
    if not candidates:
        return 0.0
    best = 0.0
    for c in candidates:
        best = max(best, float(_aircraft_match_score(c, blob)))
    # Boost exact substring (title often has full marketing name)
    low = blob
    for c in candidates:
        cl = c.lower()
        if len(cl) >= 6 and cl in low:
            best = max(best, 1.0)
        # "same family" shorthand: e.g. falcon 7x vs "falcon"
        fam = re.split(r"[^\w]+", cl)
        fam = [f for f in fam if len(f) >= 4]
        for f in fam[:2]:
            if f in low and len(f) >= 5:
                best = max(best, 0.85)
    return max(0.0, min(1.0, best))


def _visual_match_component(normalized_intent: Optional[Dict[str, Any]], blob: str) -> Tuple[float, bool]:
    """
    Returns (score 0–1, mismatch_flag).
    mismatch: e.g. exterior-only when interior requested.
    """
    if not isinstance(normalized_intent, dict):
        return 0.55, False
    vf = str(normalized_intent.get("visual_focus") or "").strip().lower()
    it = str(normalized_intent.get("intent_type") or "").strip().lower()
    if vf in ("", "null"):
        vf = ""
    if it == "cockpit" or vf == "cockpit":
        if _COCKPIT_CUES.search(blob):
            return 0.95, False
        if _INTERIOR_CUES.search(blob) and not _COCKPIT_CUES.search(blob):
            return 0.12, True
        if _EXTERIOR_CUES.search(blob) and not _COCKPIT_CUES.search(blob):
            return 0.15, True
        return 0.45, False
    if vf == "exterior" or it == "generic_visual":
        if _EXTERIOR_CUES.search(blob):
            return 0.92, False
        if _INTERIOR_CUES.search(blob) and not _EXTERIOR_CUES.search(blob):
            return 0.25, True
        return 0.5, False
    if vf in ("bedroom",):
        if _BEDROOM_CUES.search(blob):
            return 0.9, False
        if _EXTERIOR_CUES.search(blob) and not _INTERIOR_CUES.search(blob):
            return 0.1, True
        return 0.5, False
    # interior / cabin / default
    if _EXTERIOR_CUES.search(blob) and not _INTERIOR_CUES.search(blob):
        return 0.08, True
    if _INTERIOR_CUES.search(blob):
        return 0.92, False
    if _COCKPIT_CUES.search(blob):
        return 0.35, False
    return 0.48, False


def _source_quality_score(im: Dict[str, Any], blob: str) -> float:
    d = _domain_blob(im)
    blob_d = f"{d} {blob}"
    if _LOW_TRUST.search(blob_d):
        return 0.18
    if _HIGH_TRUST.search(blob_d):
        return 0.95
    if _MED_TRUST.search(blob_d):
        return 0.72
    if re.search(r"\.(aero|gov)\b", blob_d):
        return 0.88
    return 0.52


def _semantic_score(normalized_intent: Optional[Dict[str, Any]], blob: str) -> float:
    if not isinstance(normalized_intent, dict):
        return 0.5
    cons = normalized_intent.get("constraints")
    style = ""
    if isinstance(cons, dict):
        style = str(cons.get("style") or "").lower()
    tokens = [t for t in re.split(r"[^\w]+", style) if len(t) >= 3]
    boost = 0.55
    for t in tokens:
        if t in blob:
            boost += 0.12
    for kw in ("modern", "ambient", "lighting", "luxury", "minimal", "elegant"):
        if kw in style and kw in blob:
            boost += 0.1
        elif kw in blob:
            boost += 0.05
    return max(0.0, min(1.0, boost))


def _image_id(im: Dict[str, Any], idx: int) -> str:
    if str(im.get("image_id") or "").strip():
        return str(im.get("image_id")).strip()
    u = str(im.get("url") or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(u).hexdigest()[:16] if u else f"idx_{idx}"


def _dominant_aircraft_key(blob: str, candidates: List[str]) -> str:
    for c in sorted(candidates, key=len, reverse=True):
        if c.lower() in blob:
            return c.lower()[:48]
    m = re.search(
        r"\b(gulfstream|falcon|citation|challenger|global|learjet|embraer|bombardier)\s*[\w-]+",
        blob,
        re.I,
    )
    return (m.group(0).lower() if m else "")[:48]


def rank_aviation_images_for_intent(
    *,
    normalized_intent: Optional[Dict[str, Any]],
    aircraft_candidates: Optional[List[str]],
    images: List[Dict[str, Any]],
    min_score: float = _MIN_FINAL_SCORE,
    max_keep: int = 6,
) -> List[Dict[str, Any]]:
    """
    Rank ``images`` (title, url, source_domain, description, optional metadata / image_id).

    Returns a list of dicts:
    ``image_id``, ``score``, ``aircraft_match``, ``visual_match``, ``source_quality``, ``reason``.
    Sorted by ``score`` descending; only ``score >= min_score``; at most ``max_keep`` items.
    """
    cands = _candidates_list(aircraft_candidates, normalized_intent)
    rows: List[Dict[str, Any]] = []

    for idx, im in enumerate(images or []):
        if not isinstance(im, dict):
            continue
        blob = _image_blob(im)
        if len(blob) < 4:
            continue

        am = _aircraft_match_component(blob, cands)
        vm, vis_bad = _visual_match_component(normalized_intent, blob)
        sq = _source_quality_score(im, blob)
        sem = _semantic_score(normalized_intent, blob)
        dk = _dominant_aircraft_key(blob, cands)

        if am < float(_AIRCRAFT_MATCH_HARD_REJECT):
            total = 0.0
            reason = "aircraft_match_below_0p65_hard_reject"
        elif vis_bad:
            total = 0.0
            reason = "visual_focus_mismatch_hard_reject"
        else:
            total = (
                _WEIGHT_AIR * am
                + _WEIGHT_VIS * vm
                + _WEIGHT_SRC * sq
                + _WEIGHT_SEM * sem
            )
            reason = "; ".join(
                [
                    f"aircraft={am:.2f}",
                    f"visual={vm:.2f}",
                    f"src={sq:.2f}",
                    f"sem={sem:.2f}",
                ]
            )

        total = max(0.0, min(1.0, float(total)))
        rows.append(
            {
                "image_id": _image_id(im, idx),
                "score": float(total),
                "aircraft_match": round(am, 4),
                "visual_match": round(vm, 4),
                "source_quality": round(sq, 4),
                "reason": reason[:240],
                "_dk": dk,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)

    out: List[Dict[str, Any]] = []
    for r in rows:
        if r["score"] < float(min_score):
            continue
        r["score"] = round(float(r["score"]), 4)
        r.pop("_dk", None)
        out.append(r)
        if len(out) >= int(max_keep):
            break
    return out
