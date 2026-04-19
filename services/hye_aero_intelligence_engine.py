"""
HyeAero.AI — production aircraft **intelligence engine** (non-conversational).

Orchestrates deterministic identity checks (FAA-oriented U.S. logic + ICAO-style format classes),
optional registry snapshot, market listing slice, hybrid valuation, strict visual intelligence,
and acquisition decision JSON.

**No LLM** inside this module; unknowns are explicit nulls / empty lists. Nothing invents existence beyond
structured DB/API outputs already used elsewhere in the product.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from rag.aviation_tail import (
    find_strict_tail_candidates_in_text,
    is_invalid_placeholder_us_n_tail,
    normalize_tail_token,
    registration_format_kind,
)
from rag.consultant_query_expand import _detect_manufacturers, _detect_models
from rag.consultant_validity import validate_aircraft_model
from services.aircraft_decision_engine import public_decision_payload, run_aircraft_decision_engine
from services.image_intelligence_engine import resolve_aircraft_identity, run_aircraft_image_intelligence
from services.market_comparison import run_comparison
from services.price_estimate import estimate_value_hybrid
from services.searchapi_aircraft_images import compose_manufacturer_model_phrase, normalize_aircraft_name

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def _loose_us_n_from_subject(subject: str) -> str:
    for m in re.finditer(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", subject or "", re.I):
        return normalize_tail_token(m.group(0))
    return ""


def _primary_registration(subject: str) -> Tuple[str, str]:
    """
    Returns ``(normalized_mark, detection_channel)`` where channel is ``strict`` | ``loose`` | ``none``.
    """
    strict = find_strict_tail_candidates_in_text(subject or "")
    if strict:
        return normalize_tail_token(strict[0]), "strict"
    loose = _loose_us_n_from_subject(subject or "")
    if loose:
        return loose, "loose"
    for pat in (
        r"\b(?:G|D|F|VH|TC|PR|PP|PT|C-[FGI])-[A-Z0-9]{2,5}\b",
        r"\bC-[FGI][A-Z0-9]{2,4}\b",
    ):
        m = re.search(pat, subject or "", re.I)
        if m:
            return normalize_tail_token(m.group(0).replace(" ", "")), "international_token"
    return "", "none"


def _slim_faa_public_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """FAA / registry-style facts suitable for external JSON (no street address)."""
    return {
        "n_number": r.get("n_number"),
        "serial_number": r.get("serial_number"),
        "year_mfr": r.get("year_mfr"),
        "faa_reference_model": r.get("faa_reference_model"),
        "status_code": r.get("status_code"),
        "type_aircraft": r.get("type_aircraft"),
        "registrant_name": r.get("registrant_name"),
        "city": r.get("city"),
        "state": r.get("state"),
        "country": r.get("country") or r.get("region"),
    }


def _valuation_slim(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "estimated_value_millions": raw.get("estimated_value_millions"),
        "range_low_millions": raw.get("range_low_millions"),
        "range_high_millions": raw.get("range_high_millions"),
        "confidence_pct": raw.get("confidence_pct"),
        "market_demand": raw.get("market_demand"),
        "time_to_sale_days": raw.get("time_to_sale_days"),
        "error": raw.get("error"),
        "message": raw.get("message"),
    }


def _market_slim(rows: List[Dict[str, Any]], *, limit: int = 12) -> Dict[str, Any]:
    slim: List[Dict[str, Any]] = []
    for r in (rows or [])[:limit]:
        if not isinstance(r, dict):
            continue
        slim.append(
            {
                "listing_id": r.get("listing_id"),
                "source_platform": r.get("source_platform"),
                "ask_price": r.get("ask_price"),
                "manufacturer_year": r.get("manufacturer_year"),
                "airframe_total_time": r.get("airframe_total_time"),
                "manufacturer": r.get("manufacturer"),
                "model": r.get("model"),
                "location": r.get("location"),
            }
        )
    return {"row_count": len(rows or []), "sample": slim}


def _resolve_marketing_and_parts(subject: str) -> Tuple[str, str, str]:
    low = (subject or "").lower()
    mans = _detect_manufacturers(low)
    mdls = _detect_models(subject or "")
    mm = ""
    if mans and mdls:
        mm = compose_manufacturer_model_phrase(mans[0], mdls[0])
    elif mdls:
        mm = mdls[0]
    elif mans:
        mm = mans[0]
    mm = normalize_aircraft_name(mm.strip()) if mm else ""
    return mm, (mans[0] if mans else ""), (mdls[0] if mdls else "")


def run_hye_aero_aircraft_intelligence(
    subject: str,
    *,
    db: Any,
    region: Optional[str] = None,
    embedding_service: Any = None,
    pinecone_client: Any = None,
    include_visual_intel: bool = True,
    include_market: bool = True,
    include_valuation: bool = True,
    include_acquisition_scores: bool = True,
    market_limit: int = 40,
) -> Dict[str, Any]:
    """
    Single-call broker intelligence bundle. Returns a JSON-serializable dict only.

    ``engine_status``:
      - ``OK`` — pipeline ran (individual subsystems may still report empty data).
      - ``INVALID_MODEL`` — blocked fake marketing label.
      - ``INVALID_REGISTRATION`` — placeholder U.S. mark.
      - ``DATABASE_UNAVAILABLE`` — ``db`` missing / broken.
    """
    rid = str(uuid.uuid4())
    sub = (subject or "").strip()
    base: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request_id": rid,
        "subject": sub,
        "engine_status": "OK",
        "warnings": [],
    }

    inv = validate_aircraft_model(sub)
    if inv:
        base["engine_status"] = "INVALID_MODEL"
        base["validity"] = {
            "status": inv.status,
            "message": inv.message,
            "suggestions": list(inv.suggestions or []),
        }
        base["identity"] = {
            "primary_registration": "",
            "registration_detection": "none",
            "registration_format": "UNKNOWN",
            "canonical_aircraft": "",
            "registry_authoritative": False,
            "identity_resolution_reason": "invalid_model_string",
        }
        base["faa_registry_snapshot"] = None
        base["market"] = {"row_count": 0, "sample": []}
        base["valuation"] = _valuation_slim({})
        base["visual_intelligence"] = {
            "aircraft": "",
            "image_type": "cabin",
            "images": [],
            "insight": "Invalid or non-existent aircraft marketing label — visual and valuation subsystems not run.",
        }
        base["acquisition_decision"] = None
        return base

    reg, reg_ch = _primary_registration(sub)
    reg_fmt = registration_format_kind(reg) if reg else "EMPTY"

    identity_block: Dict[str, Any] = {
        "primary_registration": reg,
        "registration_detection": reg_ch,
        "registration_format": reg_fmt,
        "canonical_aircraft": "",
        "registry_authoritative": False,
        "identity_resolution_reason": "",
    }

    if reg and is_invalid_placeholder_us_n_tail(reg):
        base["engine_status"] = "INVALID_REGISTRATION"
        identity_block["identity_resolution_reason"] = "placeholder_us_n_tail"
        base["identity"] = identity_block
        base["validity"] = {"status": "invalid_registration", "message": "Placeholder U.S. registration pattern.", "suggestions": []}
        base["faa_registry_snapshot"] = None
        base["market"] = {"row_count": 0, "sample": []}
        base["valuation"] = _valuation_slim({})
        base["visual_intelligence"] = {
            "aircraft": "",
            "image_type": "cockpit" if "cockpit" in sub.lower() else "cabin",
            "images": [],
            "insight": "Registration withheld: all-zero U.S. marks are not used for retrieval or imagery.",
        }
        base["acquisition_decision"] = None
        return base

    if db is None:
        base["engine_status"] = "DATABASE_UNAVAILABLE"
        base["warnings"].append("PostgreSQL client unavailable — registry, market, and hybrid valuation skipped.")
        identity_block["identity_resolution_reason"] = "no_database"
        base["identity"] = identity_block
        base["validity"] = {"status": "unknown", "message": "", "suggestions": []}
        base["faa_registry_snapshot"] = None
        base["market"] = {"row_count": 0, "sample": []}
        base["valuation"] = _valuation_slim({})
        if include_visual_intel:
            base["visual_intelligence"] = run_aircraft_image_intelligence(sub, db=None)
        else:
            base["visual_intelligence"] = None
        base["acquisition_decision"] = (
            public_decision_payload(run_aircraft_decision_engine(sub, db=None))
            if include_acquisition_scores
            else None
        )
        return base

    canonical = ""
    auth = False
    reason = ""
    if reg and reg_fmt in ("US_N_STRICT", "US_N_LOOSE"):
        canonical, auth, reason = resolve_aircraft_identity(tail=reg, db=db)
    identity_block["canonical_aircraft"] = canonical if (auth or (not reg and bool(canonical))) else ""
    identity_block["registry_authoritative"] = bool(auth and canonical)
    identity_block["identity_resolution_reason"] = reason or ("resolved" if canonical else "unresolved_or_tail_only")

    mm, mfr_token, mdl_token = _resolve_marketing_and_parts(sub)
    faa_rows: List[Dict[str, Any]] = []
    faa_kind: Optional[str] = None
    if reg and reg_fmt in ("US_N_STRICT", "US_N_LOOSE") and not is_invalid_placeholder_us_n_tail(reg):
        try:
            from services.faa_master_lookup import fetch_faa_master_owner_rows

            faa_rows, faa_kind = fetch_faa_master_owner_rows(db, serial="", model=None, registration=reg)
        except Exception as e:
            logger.debug("FAA snapshot skipped: %s", e)
            base["warnings"].append("faa_master lookup failed for this subject.")

    faa_snapshot = None
    if faa_rows:
        faa_snapshot = {"match_kind": faa_kind, "record": _slim_faa_public_row(dict(faa_rows[0]))}

    market_model = (identity_block["canonical_aircraft"] or mm or "").strip()
    if not market_model and faa_snapshot and (faa_snapshot["record"].get("faa_reference_model") or "").strip():
        market_model = normalize_aircraft_name(str(faa_snapshot["record"]["faa_reference_model"]).strip())

    market_out: Dict[str, Any] = {"row_count": 0, "sample": []}
    if include_market and db is not None and len(market_model) >= 2:
        try:
            mc = run_comparison(db=db, models=[market_model], region=region or "Global", limit=int(market_limit))
            if not mc.get("error"):
                market_out = _market_slim(list(mc.get("rows") or []))
            else:
                base["warnings"].append(f"market_comparison: {mc.get('error')}")
        except Exception as e:
            logger.debug("market_comparison skipped: %s", e)
            base["warnings"].append("market_comparison raised an exception.")

    val_slim = _valuation_slim({})
    if include_valuation and db is not None:
        vm = mdl_token or market_model or ""
        try:
            raw_v = estimate_value_hybrid(
                db=db,
                embedding_service=embedding_service,
                pinecone_client=pinecone_client,
                manufacturer=(mfr_token or None) if mfr_token else None,
                model=vm or None,
                region=region,
            )
            val_slim = _valuation_slim(raw_v if isinstance(raw_v, dict) else {})
        except Exception as e:
            logger.debug("valuation skipped: %s", e)
            base["warnings"].append("valuation engine failed for this subject.")

    vis = None
    if include_visual_intel:
        try:
            vis = run_aircraft_image_intelligence(sub, db=db)
        except Exception as e:
            logger.debug("visual intelligence skipped: %s", e)
            base["warnings"].append("visual_intelligence engine failed.")
            vis = {"aircraft": "", "image_type": "cabin", "images": [], "insight": "Visual subsystem error."}

    acq = None
    if include_acquisition_scores:
        try:
            acq = public_decision_payload(run_aircraft_decision_engine(sub, db=db, region=region))
        except Exception as e:
            logger.debug("acquisition decision skipped: %s", e)
            base["warnings"].append("acquisition_decision engine failed.")

    base["identity"] = identity_block
    base["validity"] = {"status": "ok", "message": "", "suggestions": []}
    base["faa_registry_snapshot"] = faa_snapshot
    base["market"] = market_out
    base["valuation"] = val_slim
    base["visual_intelligence"] = vis
    base["acquisition_decision"] = acq
    base["detected_marketing_guess"] = mm or None
    return base
