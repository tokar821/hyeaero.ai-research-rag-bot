"""Deterministic exact search for Ask Consultant.

Goal: for questions that require exact aggregates/lists (counts, for-sale rate,
serial numbers, model lists), compute results directly from PostgreSQL instead
of relying on the LLM to aggregate small Pinecone snippet sets.

This is intentionally conservative: only triggers on clearly "aggregate/list"
intent patterns so normal narrative questions continue to use RAG+LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _extract_for_phrase_model(query: str) -> Optional[str]:
    """
    Extract a model phrase from things like:
      - "For Embraer Phenom 100, ..."
      - "for Gulfstream G550"
    Returns the phrase as-is (original casing), not normalized.
    """
    # Capture after "for " up to a comma or question mark.
    m = re.search(r"\bfor\s+(.+?)(?:\?|,|\n|$)", query, flags=re.IGNORECASE)
    if not m:
        return None
    phrase = (m.group(1) or "").strip()
    if not phrase:
        return None
    # Avoid obviously generic phrases
    if len(phrase) < 3:
        return None
    return phrase


def _extract_serial_like(query: str) -> Optional[str]:
    """Extract a serial-like token from the query (e.g. 'serial 6013')."""
    m = re.search(r"\bserial(?:\s*number)?\s*[:=]?\s*([0-9A-Za-z\-]+)\b", query, flags=re.IGNORECASE)
    if not m:
        return None
    return (m.group(1) or "").strip()


def _extract_source_filter(query: str) -> Tuple[bool, bool]:
    """
    Detect a "for sale = true only" constraint.
    Returns (for_sale_requested, for_sale_true_only).
    """
    q = _norm_text(query)
    for_sale_requested = "for sale" in q or "for-sale" in q or "forsale" in q
    for_sale_true_only = ("forsale=true" in q) or ("for sale=true" in q) or ("for sale only" in q)
    if not for_sale_true_only and re.search(r"with\s+for\s*sale\s*=?\s*true\b", q):
        for_sale_true_only = True
    return for_sale_requested, for_sale_true_only


def try_exact_aircraft_answer(db: PostgresClient, query: str) -> Optional[Dict[str, Any]]:
    """
    If the query is a supported aggregate/list intent, return an exact answer payload.
    Otherwise return None.
    """
    qn = _norm_text(query)

    # Trigger scope: aggregate/list-like questions
    is_list_or_agg = any(
        k in qn
        for k in (
            "for-sale rate",
            "for sale rate",
            "for-sale",
            "for sale",
            "how many",
            "number of",
            "records included",
            "serial numbers",
            "serial number",
            "models of",
            "models",
            "model list",
            "list",
            "included",
            "gimme",
            "give me",
        )
    )
    if not is_list_or_agg:
        return None

    # AircraftPost intents
    if "aircraftpost" in qn or "aircraft post" in qn:
        return _exact_aircraftpost_answer(db, query)

    # Aviacost intents (limited deterministic subset)
    if "aviacost" in qn:
        return _exact_aviacost_answer(db, query)

    # FAA intents (limited deterministic subset)
    if "faa" in qn:
        return _exact_faa_answer(db, query)

    return None


def _exact_aircraftpost_answer(db: PostgresClient, query: str) -> Optional[Dict[str, Any]]:
    phrase = _extract_for_phrase_model(query)
    for_sale_requested, for_sale_true_only = _extract_source_filter(query)
    qn = _norm_text(query)

    # Models list
    if "models" in qn and "serial" not in qn and ("aircraftpost" in qn or "aircraft post" in qn):
        want_for_sale_models = for_sale_requested and for_sale_true_only
        sql_base = """
            FROM aircraftpost_fleet_aircraft
            WHERE make_model_name IS NOT NULL AND TRIM(make_model_name) <> ''
        """
        params: List[Any] = []
        if want_for_sale_models:
            sql_base += " AND for_sale IS TRUE "

        total_models_sql = f"SELECT COUNT(DISTINCT TRIM(make_model_name)) AS n {sql_base};"
        total_models = db.execute_query(total_models_sql, tuple(params))
        n_models = int(total_models[0]["n"]) if total_models else 0

        models_rows = db.execute_query(
            f"""
            SELECT make_model_name, COUNT(*) AS n
            {sql_base}
            GROUP BY make_model_name
            ORDER BY n DESC, make_model_name ASC
            LIMIT 50
            """,
            tuple(params),
        )

        model_lines = [f"- {r['make_model_name']} ({r['n']})" for r in models_rows]
        answer_lines = ["AircraftPost model list (distinct make_model_name):", *model_lines, f"Total distinct models: {n_models}"]
        if want_for_sale_models:
            answer_lines.append("Filter: for_sale = true only.")

        return {
            "answer": "\n".join(answer_lines),
            "sources": [],
            "data_used": {"source": "aircraftpost_fleet_aircraft", "filter": "for_sale=true" if want_for_sale_models else "all"},
            "error": None,
        }

    # Serial numbers
    if "serial" in qn and ("aircraftpost" in qn or "aircraft post" in qn):
        if not phrase:
            return {
                "answer": "To list AircraftPost serial numbers, provide a model phrase, e.g. “For Embraer Phenom 100, give me AircraftPost serial numbers”.",
                "sources": [],
                "data_used": {},
                "error": None,
            }

        where = "WHERE make_model_name ILIKE %s AND serial_number IS NOT NULL AND TRIM(serial_number) <> ''"
        params2: List[Any] = [f"%{phrase}%"]
        if for_sale_requested and for_sale_true_only:
            where += " AND for_sale IS TRUE"

        serial_rows = db.execute_query(
            f"""
            SELECT serial_number, registration_number, mfr_year, for_sale
            FROM aircraftpost_fleet_aircraft
            {where}
            ORDER BY ingestion_date DESC, updated_at DESC
            LIMIT 50
            """,
            tuple(params2),
        )
        if not serial_rows:
            return {
                "answer": f"No AircraftPost fleet rows found matching model phrase: {phrase}.",
                "sources": [],
                "data_used": {},
                "error": None,
            }

        lines = ["AircraftPost serial numbers (latest first):"]
        for r in serial_rows:
            sale_tag = "For sale" if r.get("for_sale") is True else ("Not for sale" if r.get("for_sale") is False else "Unknown sale status")
            lines.append(
                f"- {r['serial_number']} (Reg: {r.get('registration_number') or '—'}, Year: {r.get('mfr_year') or '—'}, {sale_tag})"
            )

        return {
            "answer": "\n".join(lines),
            "sources": [],
            "data_used": {"source": "aircraftpost_fleet_aircraft", "make_model_phrase": phrase},
            "error": None,
        }

    # For-sale rate and record count
    if ("for-sale rate" in qn or "for sale rate" in qn) and phrase:
        totals_rows = db.execute_query(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN for_sale IS TRUE THEN 1 ELSE 0 END) AS for_sale_true,
              SUM(CASE WHEN for_sale IS FALSE THEN 1 ELSE 0 END) AS for_sale_false
            FROM aircraftpost_fleet_aircraft
            WHERE make_model_name ILIKE %s
              AND serial_number IS NOT NULL
              AND TRIM(serial_number) <> ''
            """,
            (f"%{phrase}%",),
        )
        r = totals_rows[0] if totals_rows else None
        total = int(r["total"]) if r and r.get("total") is not None else 0
        for_sale_true = int(r["for_sale_true"]) if r and r.get("for_sale_true") is not None else 0
        rate = (for_sale_true / total) if total else None

        answer_lines = [
            f"AircraftPost fleet for-sale rate for: {phrase}",
            f"- Total fleet records: {total}",
            f"- For-sale records: {for_sale_true}",
            f"- For-sale rate: {('%.2f' % (rate * 100)) + '%' if rate is not None else '—'}",
        ]

        return {
            "answer": "\n".join(answer_lines),
            "sources": [],
            "data_used": {"source": "aircraftpost_fleet_aircraft", "make_model_phrase": phrase},
            "error": None,
        }

    return None


def _exact_aviacost_answer(db: PostgresClient, query: str) -> Optional[Dict[str, Any]]:
    phrase = _extract_for_phrase_model(query)
    qn = _norm_text(query)
    if not phrase:
        return None

    tokens = phrase.split()
    if len(tokens) >= 2:
        mfr = tokens[0]
        mdl = " ".join(tokens[1:])
    else:
        mfr = None
        mdl = phrase

    where_parts: List[str] = []
    params: List[Any] = []
    if mdl:
        where_parts.append("name ILIKE %s")
        params.append(f"%{mdl}%")
    if mfr:
        where_parts.append("(manufacturer_name ILIKE %s OR name ILIKE %s)")
        params.extend([f"%{mfr}%", f"%{mfr}%"])

    if not where_parts:
        return None

    rows = db.execute_query(
        f"""
        SELECT name, manufacturer_name, category_name,
               variable_cost_per_hour, average_pre_owned_price,
               fuel_gallons_per_hour, normal_cruise_speed_kts
        FROM aviacost_aircraft_details
        WHERE {" AND ".join(where_parts)}
        ORDER BY LENGTH(COALESCE(name,'')) ASC
        LIMIT 3
        """,
        tuple(params),
    )
    if not rows:
        return {
            "answer": f"Based on Aviacost data, I couldn't find a match for: {phrase}.",
            "sources": [],
            "data_used": {"source": "aviacost_aircraft_details", "phrase": phrase},
            "error": None,
        }

    r = rows[0]
    answer_lines = [f"Aviacost operating cost reference for: {phrase}"]
    answer_lines.append(f"- Name: {r.get('name') or '—'}")
    answer_lines.append(f"- Variable cost/hr: {r.get('variable_cost_per_hour') if r.get('variable_cost_per_hour') is not None else '—'}")
    answer_lines.append(
        f"- Avg pre-owned price: {r.get('average_pre_owned_price') if r.get('average_pre_owned_price') is not None else '—'}"
    )
    answer_lines.append(f"- Fuel (gal/hr): {r.get('fuel_gallons_per_hour') if r.get('fuel_gallons_per_hour') is not None else '—'}")
    answer_lines.append(f"- Cruise speed (kts): {r.get('normal_cruise_speed_kts') if r.get('normal_cruise_speed_kts') is not None else '—'}")

    return {
        "answer": "\n".join(answer_lines),
        "sources": [],
        "data_used": {"source": "aviacost_aircraft_details", "phrase": phrase},
        "error": None,
    }


def _exact_faa_answer(db: PostgresClient, query: str) -> Optional[Dict[str, Any]]:
    serial = _extract_serial_like(query)
    if not serial:
        return None

    rows = db.execute_query(
        """
        SELECT serial_number, registrant_name, street, street2, city, state, zip_code, country
        FROM faa_registrations
        WHERE serial_number = %s
        ORDER BY updated_at DESC
        LIMIT 5
        """,
        (serial,),
    )
    if not rows:
        return {
            "answer": f"FAA registry: no match found for serial {serial}.",
            "sources": [],
            "data_used": {"source": "faa_registrations", "serial": serial},
            "error": None,
        }

    r = rows[0]
    addr = ", ".join(
        [x for x in [r.get("street"), r.get("street2"), r.get("city"), r.get("state"), r.get("zip_code"), r.get("country")] if x]
    )
    answer_lines = [
        f"FAA registry (serial {serial}):",
        f"- Registrant: {r.get('registrant_name') or '—'}",
        f"- Address: {addr or '—'}",
    ]
    return {
        "answer": "\n".join(answer_lines),
        "sources": [],
        "data_used": {"source": "faa_registrations", "serial": serial},
        "error": None,
    }

