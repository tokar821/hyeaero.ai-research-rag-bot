"""
Lightweight Pinecone vector metadata: normalized identity fields for filtering and debugging.

Keeps payloads small (no descriptions, JSON blobs, or long text). See :func:`sanitize_pinecone_metadata_dict`.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Optional

from rag.aircraft_normalization import normalize_manufacturer

logger = logging.getLogger(__name__)

# Pinecone string fields: keep each short so total metadata stays well under 1 KB.
_MAX_SHORT = 128
_MAX_ENTITY_ID = 64

SOURCE_TABLE_BY_ENTITY_TYPE: Dict[str, str] = {
    "aircraft": "aircraft",
    "aircraft_listing": "aircraft_listings",
    "aircraft_sale": "aircraft_sales",
    "faa_registration": "faa_registrations",
    "aviacost_aircraft_detail": "aviacost_aircraft_details",
    "aircraftpost_fleet_aircraft": "aircraftpost_fleet_aircraft",
    "document": "documents",
    "phlydata_aircraft": "phlydata_aircraft",
}

# Logical bucket for intent-shaped retrieval (mirrors consultant ``doc_type`` filters).
_ENTITY_TYPE_TO_DOC_TYPE: Dict[str, str] = {
    "aircraft": "aircraft_model",
    "aviacost_aircraft_detail": "aircraft_model",
    "aircraftpost_fleet_aircraft": "aircraft_model",
    "phlydata_aircraft": "aircraft_model",
    "document": "document",
    "aircraft_listing": "aircraft_listing",
    "aircraft_sale": "market_data",
    "faa_registration": "registry",
}

# Only these keys are written on new upserts (keeps payloads small and consistent).
_PINECONE_UPSERT_ALLOWLIST = frozenset(
    {
        "entity_type",
        "entity_id",
        "doc_type",
        "aircraft_model",
        "manufacturer",
        "serial_number",
        "tail_number",
        "year",
        "source_table",
        "chunk_index",
        "total_chunks",
        "chunking_strategy",
    }
)


def normalize_serial_for_metadata(value: Any) -> Optional[str]:
    """Strip SN-/Serial prefixes; keep alphanumeric core uppercase (e.g. SN-6201 -> 6201)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"^(?:SN|S/N|SERIAL)[\s:\-#]*", "", s, flags=re.IGNORECASE)
    core = re.sub(r"[^A-Za-z0-9]", "", s)
    if len(core) < 1:
        return None
    return core.upper()


def normalize_tail_for_metadata(value: Any) -> Optional[str]:
    """Registration / tail: uppercase, strip internal spaces."""
    if value is None:
        return None
    s = re.sub(r"\s+", "", str(value).strip().upper())
    if len(s) < 2:
        return None
    return s


def normalize_aircraft_model_metadata(value: Any) -> Optional[str]:
    """Trim and UPPERCASE model for consistent metadata filters (e.g. G650, CITATION X)."""
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value).strip())
    if not s:
        return None
    return s.upper()[:_MAX_SHORT]


def normalize_manufacturer_metadata(value: Any) -> Optional[str]:
    """Trim and canonicalize manufacturer names (aliases), no forced all-caps."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    c = normalize_manufacturer(s)
    out = (c or s).strip()
    return out[:_MAX_SHORT] if out else None


def _int_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        y = int(value)
        if 1900 <= y <= 2100:
            return y
    except (TypeError, ValueError):
        pass
    return None


def _pick_year(record: Dict[str, Any], entity_type: str) -> Optional[int]:
    if entity_type == "faa_registration":
        return _int_year(record.get("year_mfr"))
    if entity_type == "aircraftpost_fleet_aircraft":
        return _int_year(record.get("mfr_year"))
    y = _int_year(record.get("manufacturer_year"))
    if y is not None:
        return y
    return _int_year(record.get("delivery_year"))


def _pick_tail(record: Dict[str, Any], entity_type: str) -> Optional[str]:
    if entity_type == "faa_registration":
        return normalize_tail_for_metadata(record.get("n_number"))
    v = record.get("registration_number")
    if v is None and entity_type == "aircraft_listing":
        v = record.get("tail") or record.get("tail_number")
    return normalize_tail_for_metadata(v)


def _pick_serial(record: Dict[str, Any]) -> Optional[str]:
    return normalize_serial_for_metadata(record.get("serial_number"))


def _pick_model(record: Dict[str, Any], entity_type: str) -> Optional[str]:
    if entity_type == "aviacost_aircraft_detail":
        return normalize_aircraft_model_metadata(record.get("name"))
    if entity_type == "aircraftpost_fleet_aircraft":
        return normalize_aircraft_model_metadata(record.get("make_model_name"))
    return normalize_aircraft_model_metadata(record.get("model"))


def _pick_manufacturer(record: Dict[str, Any], entity_type: str) -> Optional[str]:
    if entity_type == "aviacost_aircraft_detail":
        return normalize_manufacturer_metadata(record.get("manufacturer_name"))
    return normalize_manufacturer_metadata(record.get("manufacturer"))


def _pick_entity_id(record: Dict[str, Any], entity_type: str, entity_id_override: Optional[str]) -> str:
    if entity_id_override is not None:
        return str(entity_id_override).strip()[:_MAX_ENTITY_ID]
    if entity_type == "phlydata_aircraft":
        aid = record.get("aircraft_id")
        if aid is not None:
            return str(aid).strip()[:_MAX_ENTITY_ID]
    rid = record.get("id")
    return str(rid).strip()[:_MAX_ENTITY_ID] if rid is not None else ""


def build_vector_metadata(
    entity_type: str,
    record: Dict[str, Any],
    *,
    entity_id_override: Optional[str] = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> Dict[str, Any]:
    """
    Build the standard metadata dict for one Pinecone vector.

    Omits keys when values are missing (Pinecone: omit rather than null).
    """
    et = (entity_type or "").strip()
    st = SOURCE_TABLE_BY_ENTITY_TYPE.get(et)
    if not st:
        st = et or "unknown"

    eid = _pick_entity_id(record, et, entity_id_override)
    meta: Dict[str, Any] = {
        "entity_type": et,
        "entity_id": eid,
        "source_table": st[:_MAX_SHORT],
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
    }
    doc_t = _ENTITY_TYPE_TO_DOC_TYPE.get(et)
    if doc_t:
        meta["doc_type"] = doc_t

    mfr = _pick_manufacturer(record, et)
    if mfr:
        meta["manufacturer"] = mfr

    model = _pick_model(record, et)
    if model:
        meta["aircraft_model"] = model

    serial = _pick_serial(record)
    if serial:
        meta["serial_number"] = serial

    tail = _pick_tail(record, et)
    if tail:
        meta["tail_number"] = tail

    year = _pick_year(record, et)
    if year is not None:
        meta["year"] = year

    return meta


def sanitize_pinecone_metadata_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pinecone-safe: no null/empty, strings truncated, estimated total size under ~1 KB.
    Drops keys not in the allowlist except we only pass through known keys when cleaning upserts.
    """
    out: Dict[str, Any] = {}
    approx = 0
    for k, v in meta.items():
        if k not in _PINECONE_UPSERT_ALLOWLIST:
            continue
        if v is None or v == "":
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if isinstance(v, bool):
            out[k] = v
            approx += 8
        elif isinstance(v, int):
            out[k] = v
            approx += 12
        elif isinstance(v, float):
            out[k] = v
            approx += 12
        elif isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            cap = _MAX_ENTITY_ID if k == "entity_id" else _MAX_SHORT
            if len(s) > cap:
                s = s[: cap - 3] + "..."
            out[k] = s
            approx += len(s) + len(k)
        else:
            s = str(v).strip()
            if not s:
                continue
            if len(s) > _MAX_SHORT:
                s = s[: _MAX_SHORT - 3] + "..."
            out[k] = s
            approx += len(s) + len(k)
    if approx > 900:
        logger.warning("Pinecone metadata approx size %s bytes — truncating entity_id if needed", approx)
    return out


def infer_pinecone_entity_filter(query: str) -> Optional[Dict[str, Any]]:
    """
    Infer metadata filter for Pinecone ``query`` from lightweight heuristics.

    Returns Pinecone filter dict or None for unrestricted search.
    """
    q = (query or "").strip()
    if not q:
        return None
    ql = q.lower()

    # FAA / registration style
    if re.search(r"\bn[\-\s]?[a-z0-9]{1,8}\b", q, re.IGNORECASE) and any(
        t in ql for t in ("faa", "registration", "registrant", "certificate", "n-number", "n number")
    ):
        return {"entity_type": {"$in": ["faa_registration"]}}

    if any(t in ql for t in ("faa registration", "faa master", "federal aviation")):
        return {"entity_type": {"$in": ["faa_registration"]}}

    # Listings / marketplace
    if any(
        t in ql
        for t in (
            "listing",
            "for sale",
            "on the market",
            "asking price",
            "seller",
            "broker listing",
            "controller",
            "jetnet listing",
        )
    ):
        return {"entity_type": {"$in": ["aircraft_listing"]}}

    # Type specs / performance (reference + fleet stats)
    if any(
        t in ql
        for t in (
            "specs",
            "spec ",
            " specification",
            "cruise speed",
            "max cruise",
            "range nm",
            "range at",
            "fuel burn",
            "operating cost",
            "cost per hour",
            "seats",
            "passenger capacity",
        )
    ):
        return {
            "entity_type": {
                "$in": [
                    "aviacost_aircraft_detail",
                    "aircraft",
                    "aircraftpost_fleet_aircraft",
                ]
            }
        }

    return None


def legacy_meta_aircraft_model(meta: Dict[str, Any]) -> str:
    """Read model from new or legacy Pinecone metadata."""
    return str(
        meta.get("aircraft_model") or meta.get("model") or ""
    ).strip()
