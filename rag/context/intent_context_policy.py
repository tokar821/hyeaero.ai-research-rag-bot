"""Intent-gated consultant context: section buckets, phly splitting, token budget."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rag.intent.schemas import AviationIntent, ConsultantIntent, IntentClassification

SECTION_AIRCRAFT_SPECS = "AIRCRAFT_SPECS"
SECTION_OPERATIONAL_DATA = "OPERATIONAL_DATA"
SECTION_MARKET_DATA = "MARKET_DATA"
SECTION_REGISTRY_DATA = "REGISTRY_DATA"

SECTION_ORDER: Tuple[str, ...] = (
    SECTION_AIRCRAFT_SPECS,
    SECTION_OPERATIONAL_DATA,
    SECTION_MARKET_DATA,
    SECTION_REGISTRY_DATA,
)

_CHARS_PER_TOKEN = int((os.getenv("CONSULTANT_CONTEXT_CHARS_PER_TOKEN") or "4").strip() or "4")
_DEFAULT_MAX_TOKENS = int((os.getenv("CONSULTANT_CONTEXT_MAX_TOKENS") or "1500").strip() or "1500")


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def intent_filter_enabled() -> bool:
    return not _env_truthy("CONSULTANT_CONTEXT_INTENT_FILTER_DISABLE")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // max(1, _CHARS_PER_TOKEN))


def consultant_context_token_budget(user_max_tokens: Optional[int] = None) -> int:
    if user_max_tokens is not None and user_max_tokens > 0:
        return user_max_tokens
    return max(256, _DEFAULT_MAX_TOKENS)


def effective_context_char_cap(*, max_context_chars: int, max_context_tokens: Optional[int]) -> int:
    tok = consultant_context_token_budget(max_context_tokens)
    token_chars = max(256, tok * _CHARS_PER_TOKEN)
    return min(max(1, max_context_chars), token_chars)


def _section_mask_for_aviation(av: AviationIntent) -> Dict[str, bool]:
    # Default: allow all
    all_on = {s: True for s in SECTION_ORDER}
    comparison = {
        SECTION_AIRCRAFT_SPECS: True,
        SECTION_OPERATIONAL_DATA: True,
        SECTION_MARKET_DATA: False,
        SECTION_REGISTRY_DATA: False,
    }
    specs_mission = comparison
    if av == AviationIntent.AIRCRAFT_COMPARISON:
        return comparison
    if av == AviationIntent.AIRCRAFT_SPECS:
        return specs_mission
    if av == AviationIntent.MISSION_FEASIBILITY:
        return specs_mission
    if av == AviationIntent.SERIAL_LOOKUP:
        return {
            SECTION_AIRCRAFT_SPECS: True,
            SECTION_OPERATIONAL_DATA: True,
            SECTION_MARKET_DATA: False,
            SECTION_REGISTRY_DATA: False,
        }
    if av == AviationIntent.REGISTRATION_LOOKUP:
        return {
            SECTION_AIRCRAFT_SPECS: True,
            SECTION_OPERATIONAL_DATA: True,
            SECTION_MARKET_DATA: False,
            SECTION_REGISTRY_DATA: True,
        }
    if av == AviationIntent.OPERATOR_LOOKUP:
        return {
            SECTION_AIRCRAFT_SPECS: True,
            SECTION_OPERATIONAL_DATA: True,
            SECTION_MARKET_DATA: False,
            SECTION_REGISTRY_DATA: True,
        }
    if av == AviationIntent.AIRCRAFT_FOR_SALE:
        return all_on
    if av == AviationIntent.MARKET_PRICE:
        return {
            SECTION_AIRCRAFT_SPECS: True,
            SECTION_OPERATIONAL_DATA: True,
            SECTION_MARKET_DATA: True,
            SECTION_REGISTRY_DATA: False,
        }
    if av == AviationIntent.GENERAL_QUESTION:
        return all_on
    return all_on


def _section_mask_for_primary(primary: ConsultantIntent) -> Dict[str, bool]:
    all_on = {s: True for s in SECTION_ORDER}
    if primary == ConsultantIntent.REGISTRATION_LOOKUP:
        return _section_mask_for_aviation(AviationIntent.REGISTRATION_LOOKUP)
    if primary == ConsultantIntent.MARKET_PRICING:
        return _section_mask_for_aviation(AviationIntent.MARKET_PRICE)
    if primary == ConsultantIntent.TECHNICAL_SPEC:
        return _section_mask_for_aviation(AviationIntent.AIRCRAFT_SPECS)
    if primary == ConsultantIntent.AIRCRAFT_IDENTITY:
        return {
            SECTION_AIRCRAFT_SPECS: True,
            SECTION_OPERATIONAL_DATA: True,
            SECTION_MARKET_DATA: False,
            SECTION_REGISTRY_DATA: True,
        }
    if primary == ConsultantIntent.GENERAL_AVIATION:
        return all_on
    return all_on


def section_mask_for_intent(ic: Optional[IntentClassification]) -> Dict[str, bool]:
    all_on = {s: True for s in SECTION_ORDER}
    if ic is None:
        return all_on
    if not intent_filter_enabled():
        return all_on
    if ic.aviation_intent is not None:
        return _section_mask_for_aviation(ic.aviation_intent)
    return _section_mask_for_primary(ic.primary)


def _is_registry_line(line: str) -> bool:
    s = line
    if "FOR USER REPLY — U.S. legal registrant" in s:
        return True
    if re.search(r"FAA MASTER registrant\s*\(faa_master\)\s*:", s):
        return True
    if re.search(r"^\s*-\s*FAA mailing street\s*:", s):
        return True
    if re.search(r"^\s*-\s*FAA mailing location\s*:", s):
        return True
    if re.search(r"^\s*Registrant name\s*:", s, re.I):
        return True
    if re.search(r"^\s*Mailing street\s*:", s, re.I):
        return True
    if re.search(r"^\s*Mailing city/state/ZIP/country\s*:", s, re.I):
        return True
    return False


def _is_market_line(line: str) -> bool:
    s = line.lower()
    if "for user reply — phlydata" in s:
        return True
    if re.search(
        r"\b(ask_price|take_price|sold_price|aircraft_status)\s*:", s
    ):
        return True
    if "aircraft_status —" in s or "ask price —" in s or "take price" in s or "sold price" in s:
        return True
    if "seller broker" in s or "buyer broker" in s:
        return True
    if re.search(r"^\s*-\s*seller\s*:", s) or re.search(r"^\s*-\s*buyer\s*:", s):
        return True
    if "date listed" in s and "phlydata" in s:
        return True
    return False


def _is_specs_line(line: str) -> bool:
    s = line
    if "FOR USER REPLY — public.aircraft" in s:
        return True
    if "FOR USER REPLY — FAA aircraft type / year" in s:
        return True
    if re.match(r"^-\s*Aircraft\s+\d+\s*:\s*$", s.strip()):
        return True
    if re.search(r"^\s*-\s*Serial\s*:", s):
        return True
    if re.search(r"^\s*-\s*Registration\s*\(tail\)\s*:", s):
        return True
    if re.search(r"^\s*-\s*Make / model\s*:", s):
        return True
    if re.search(r"^\s*-\s*Year\s*:", s):
        return True
    if re.search(r"^\s*-\s*Category\s*:", s):
        return True
    if re.search(r"^\s*registration_number\s*:", s, re.I):
        return True
    if re.search(r"^\s*serial_number\s*:", s, re.I):
        return True
    if "manufacturer / model" in s.lower():
        return True
    if re.search(r"^\s*manufacturer_year\b", s, re.I):
        return True
    if re.search(r"^\s*category\s*:", s, re.I) and "aircraft" in s.lower():
        return True
    if s.strip().startswith("PhlyData — Hye Aero"):
        return True
    if s.strip().startswith("Evaluation order:"):
        return True
    if "Registration token matched:" in s:
        return True
    if "FAA match kind:" in s:
        return True
    if "[FAA aircraft identity from MASTER" in s:
        return True
    if re.match(r"^\s*-\s*faa_", s.lower()):
        return True
    if re.match(r"^\s*-\s*year_mfr\b", s.lower()):
        return True
    return False


def _classify_phly_line(line: str) -> str:
    if _is_registry_line(line):
        return SECTION_REGISTRY_DATA
    if _is_market_line(line):
        return SECTION_MARKET_DATA
    if _is_specs_line(line):
        return SECTION_AIRCRAFT_SPECS
    return SECTION_OPERATIONAL_DATA


def partition_phly_authority(phly: str) -> Dict[str, List[str]]:
    """Split Phly+FAA authority text into section buckets (line lists)."""
    out: Dict[str, List[str]] = {s: [] for s in SECTION_ORDER}
    if not (phly or "").strip():
        return out
    in_us_reg_verbatim = False
    for line in phly.splitlines():
        ls = line.strip()
        if "FOR USER REPLY — U.S. legal registrant" in line:
            in_us_reg_verbatim = True
            out[SECTION_REGISTRY_DATA].append(line)
            continue
        if in_us_reg_verbatim:
            if ls.startswith("[") and "U.S. legal registrant" not in line:
                in_us_reg_verbatim = False
                # fall through; classify this line normally
            else:
                out[SECTION_REGISTRY_DATA].append(line)
                continue
        if ls.startswith("[AUTHORITATIVE — FAA MASTER"):
            out[SECTION_AIRCRAFT_SPECS].append(line)
            continue
        if _is_registry_line(line):
            out[SECTION_REGISTRY_DATA].append(line)
            continue
        bucket = _classify_phly_line(line)
        out[bucket].append(line)
    return out


def _merge_lines(lines: List[str]) -> str:
    return "\n".join(lines).strip()


def filter_tavily_block(tavily: str, mask: Dict[str, bool]) -> str:
    """Drop lines ill-matched to intent (e.g. insurance, ownership in comparison)."""
    if not (tavily or "").strip():
        return ""
    if mask.get(SECTION_REGISTRY_DATA) and mask.get(SECTION_OPERATIONAL_DATA):
        return tavily.strip()
    lines_out: List[str] = []
    for line in tavily.splitlines():
        low = line.lower()
        if not mask.get(SECTION_REGISTRY_DATA, False):
            if re.search(
                r"\b(registrant|registered owner|legal owner|mailing address|faa master registrant)\b",
                low,
            ):
                continue
        if not mask.get(SECTION_MARKET_DATA, False):
            if re.search(r"\b(insurance|premium|hull value|liability coverage)\b", low):
                continue
            if re.search(r"\b(listing price|asking price|for sale|controller\.com)\b", low):
                continue
        lines_out.append(line)
    return "\n".join(lines_out).strip()


def rag_result_bucket(result: Dict[str, Any]) -> str:
    et = str(result.get("entity_type") or "").lower().replace(" ", "_")
    if et in ("faa_registration",):
        return SECTION_REGISTRY_DATA
    if et in ("aircraft_listing", "aircraft_sale", "listing"):
        return SECTION_MARKET_DATA
    if et in ("aircraft", "aviacost_aircraft_detail", "document"):
        return SECTION_AIRCRAFT_SPECS
    return SECTION_OPERATIONAL_DATA


def build_section_bodies(
    *,
    phly_authority: str,
    market_block: str,
    tavily_block: str,
    rag_results: List[Dict[str, Any]],
    mask: Dict[str, bool],
) -> Dict[str, str]:
    phly_parts = partition_phly_authority(phly_authority)
    bodies: Dict[str, List[str]] = {s: list(phly_parts.get(s) or []) for s in SECTION_ORDER}

    if (market_block or "").strip():
        bodies[SECTION_MARKET_DATA].extend(market_block.strip().splitlines())

    tav_f = filter_tavily_block(tavily_block, mask)
    if tav_f:
        bodies[SECTION_OPERATIONAL_DATA].extend(tav_f.splitlines())

    for r in rag_results:
        text = (r.get("full_context") or r.get("chunk_text") or "").strip()
        if not text:
            continue
        b = rag_result_bucket(r)
        bodies[b].extend(text.splitlines())

    return {s: _merge_lines(bodies[s]) for s in SECTION_ORDER}


def assemble_filtered_context(
    *,
    section_bodies: Dict[str, str],
    mask: Dict[str, bool],
    max_chars: int,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Apply mask, add section headers, trim to char budget (token proxy)."""
    chunks: List[str] = []
    included: List[str] = []
    per_section_parts: List[str] = []

    for sec in SECTION_ORDER:
        if not mask.get(sec, True):
            continue
        body = (section_bodies.get(sec) or "").strip()
        if not body:
            continue
        per_section_parts.append(f"{sec}\n{body}")
        included.append(sec)

    if not per_section_parts:
        return "", [], {"sections_included": [], "context_tokens_est": 0, "context_char_budget": max_chars}

    sep = "\n\n---\n\n"
    # Priority trim: drop from end of OPERATIONAL, then MARKET, REGISTRY, SPECS last
    full = sep.join(per_section_parts)

    def truncate(s: str, limit: int) -> str:
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 20)].rstrip() + "\n… [truncated]"

    if len(full) <= max_chars:
        final = full
    else:
        # Rebuild with greedy truncation (low-urgency sections shrink first)
        bodies = {s: (section_bodies.get(s) or "").strip() for s in SECTION_ORDER}
        if not included:
            final = ""
        else:
            remaining = max_chars - (len("---") + 48)  # slack for headers
            parts_rev: List[str] = []
            for sec in reversed(SECTION_ORDER):
                if sec not in included:
                    continue
                body = bodies.get(sec) or ""
                hdr = f"{sec}\n"
                need = len(hdr) + len(body) + len(sep)
                if remaining < len(hdr) + 40:
                    continue
                if need > remaining:
                    body = truncate(body, remaining - len(hdr))
                chunk = hdr + body
                parts_rev.append(chunk)
                remaining -= len(chunk) + len(sep)
            final = sep.join(reversed(parts_rev)) if parts_rev else truncate(full, max_chars)

    pieces = final.split(sep) if final else []
    meta = {
        "sections_included": included,
        "context_tokens_est": estimate_tokens(final),
        "context_char_budget": max_chars,
    }
    return final, pieces, meta
