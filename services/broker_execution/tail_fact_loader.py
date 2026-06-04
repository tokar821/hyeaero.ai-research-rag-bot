"""
Phase 56 — load registry / Phly tail facts into data_used before template rendering.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_registration(query: str) -> Optional[str]:
    try:
        from rag.aviation_tail import primary_registration_from_query

        return primary_registration_from_query(query or "")
    except Exception:
        return None


def _get_db(data_used: dict):
    db = data_used.get("db")
    if db is not None:
        return db
    try:
        from api.main import get_db

        return get_db()
    except Exception:
        return None


def ensure_tail_facts_for_query(query: str, data_used: dict) -> bool:
    """
    Populate ``data_used`` with FAA / Phly registry rows when a tail is present.
    Returns True if any authoritative row was loaded.
    """
    if not isinstance(data_used, dict):
        return False
    if data_used.get("tail_facts_loaded"):
        return bool(data_used.get("tail_facts"))

    reg = _extract_registration(query)
    if not reg:
        data_used["tail_facts_loaded"] = True
        return False

    data_used["tail_registration"] = reg
    loaded = False
    db = _get_db(data_used)

    rows: List[Dict[str, Any]] = list(data_used.get("phlydata_rows") or data_used.get("phly_rows") or [])
    if not rows and db is not None:
        try:
            from rag.phlydata_consultant_lookup import lookup_phlydata_aircraft_rows

            rows = lookup_phlydata_aircraft_rows(db, [reg])
            if rows:
                data_used["phlydata_rows"] = rows
                data_used["phly_rows"] = rows
                loaded = True
        except Exception as exc:
            logger.debug("phly tail lookup skipped: %s", exc)

    faa_row = data_used.get("faa_master_row")
    if not isinstance(faa_row, dict) and db is not None:
        try:
            from services.faa_master_lookup import fetch_faa_master_owner_rows

            faa_rows, kind = fetch_faa_master_owner_rows(
                db, serial="", model=None, registration=reg
            )
            if faa_rows:
                faa_row = dict(faa_rows[0])
                data_used["faa_master_row"] = faa_row
                data_used.setdefault("phly_meta", {})["faa_master_owner_rows"] = 1
                data_used["phly_meta"]["faa_master_match_kind"] = kind
                loaded = True
        except Exception as exc:
            logger.debug("faa tail lookup skipped: %s", exc)

    if not faa_row and not rows and db is not None:
        try:
            from rag.phlydata_consultant_lookup import faa_master_standalone_authority_for_tokens
            from services.faa_master_lookup import fetch_faa_master_owner_rows

            auth, meta, fr = faa_master_standalone_authority_for_tokens(
                db, [reg], fetch_faa_master_owner_rows
            )
            if fr:
                data_used["faa_master_row"] = dict(fr)
                data_used["phly_authority"] = auth
                data_used["phly_meta"] = meta if isinstance(meta, dict) else {}
                loaded = True
        except Exception as exc:
            logger.debug("faa standalone tail lookup skipped: %s", exc)

    facts = []
    try:
        from services.broker_execution.tail_fact_renderer import select_tail_facts

        facts = select_tail_facts(data_used, reg)
        data_used["tail_selected_facts"] = facts
    except Exception:
        pass

    data_used["tail_facts"] = facts
    data_used["tail_facts_loaded"] = True
    return loaded or bool(facts)


__all__ = ["ensure_tail_facts_for_query"]
