"""
HyeAero.AI — Aircraft Research & Valuation Consultant API

Real-time AI assistant for aircraft research, market comparison, predictive pricing,
and resale advisory using Hye Aero's proprietary data (Controller, AircraftExchange, FAA, Internal DB).
"""

import os
import sys
from pathlib import Path
import csv
from functools import lru_cache

# Backend root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any

from config.config_loader import Config  # Config.from_env() without validate for flexible API
from database.postgres_client import PostgresClient
from rag.embedding_service import EmbeddingService
from rag.query_service import RAGQueryService
from vector_store.pinecone_client import PineconeClient
from services.market_comparison import run_comparison
from services.price_estimate import estimate_value, estimate_value_hybrid
from services.aviacost_lookup import lookup_aviacost
from services.aircraftpost_lookup import lookup_aircraftpost_fleet
from services.faa_master_lookup import fetch_faa_master_owner_rows
from services.tavily_owner_hint import enrich_faa_owners_with_tavily_hints
from services.tavily_llm_synthesis import enrich_faa_owners_with_tavily_llm_synthesis
from services.zoominfo_client import (
    search_companies as zoominfo_search_companies,
    enrich_company as zoominfo_enrich_company,
    enrich_contact as zoominfo_enrich_contact,
    normalize_phone as zoominfo_normalize_phone,
    phones_match as zoominfo_phones_match,
)

# Match method returned to frontend and logged
MATCH_METHOD_PHONE = "phone"
MATCH_METHOD_CONTENT_SCORE = "content_score"
MATCH_METHOD_LLM_FALLBACK = "llm_fallback"

import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# When content/word scoring is below this, try vector + LLM fallback to pick best match from ZoomInfo candidates.
ZOOMINFO_SCORE_THRESHOLD_FOR_LLM_FALLBACK = 1.0

# US state abbreviations (FAA state field) — used to infer US registrant vs foreign homonym companies in ZoomInfo.
_US_STATE_CODES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC AS GU MP PR VI".split()
)

# Legal/status suffixes and numbers to strip when ZoomInfo returns 0 (e.g. "JET ALLIANCE 84 LLC" -> "JET ALLIANCE")
_LEGAL_SUFFIXES = frozenset(
    "llc inc corp ltd co lp l.l.c l.l.p llp plc sa gmbh ag na ag".split()
)


def _clean_company_name(name: str) -> str:
    """Remove legal suffixes (LLC, Inc, etc.) and standalone numbers. Preserve core business name."""
    if not name or not isinstance(name, str):
        return ""
    tokens = name.split()
    kept = []
    for t in tokens:
        # Strip trailing punctuation (e.g. "Inc,")
        cleaned = re.sub(r"[^\w]", "", t)
        low = cleaned.lower()
        if not cleaned:
            continue
        if low in _LEGAL_SUFFIXES:
            continue
        if cleaned.isdigit():
            continue
        kept.append(t)
    return " ".join(kept).strip()


def _core_company_name(name: str, min_words: int = 2) -> str:
    """Cleaned name, then first 2–3 meaningful words. E.g. 'JET ALLIANCE 84 LLC' -> 'JET ALLIANCE'."""
    cleaned = _clean_company_name(name)
    tokens = cleaned.split()
    if len(tokens) <= min_words:
        return cleaned
    return " ".join(tokens[:max(min_words, 3)]).strip()


def _first_word_company_name(name: str) -> str:
    """First meaningful word only. E.g. 'SAXTON CRAIG J' -> 'SAXTON'. Use when 2-word core returns 0."""
    cleaned = _clean_company_name(name)
    tokens = cleaned.split()
    return tokens[0].strip() if tokens else ""


def _person_name_search_variants(name: str) -> List[str]:
    """
    Ordered, deduplicated query strings for person-led ZoomInfo *company* search (contact_full_name).

    Unlike company search variants, this permutes personal-name tokens: full string, without trailing
    initial, first+second in both orders, optional first+last when 3+ tokens, then single tokens (len>=2).

    Examples:
      - "Sarah Jason P" -> full, "Sarah Jason", "Jason Sarah", "Sarah", "Jason" (not lone "P")
      - "SMITH, JOHN A" -> structured LAST/FIRST variants plus "JOHN SMITH", etc.
    """
    if not name or not isinstance(name, str):
        return []
    raw = " ".join(name.split()).strip()
    if not raw:
        return []

    def _tok(w: str) -> str:
        return re.sub(r"[^\w]", "", w or "")

    variants: List[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = " ".join(s.split()).strip()
        if len(s) < 2:
            return
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        variants.append(s)

    add(raw)

    # FAA-style: LAST, FIRST [MIDDLE / INITIAL ...]
    if "," in raw:
        left, right = [p.strip() for p in raw.split(",", 1)]
        last = _tok(left)
        rtoks = [_tok(t) for t in right.split() if _tok(t)]
        if last and rtoks:
            given = " ".join(rtoks)
            add(f"{given} {last}")
            add(f"{last} {given}")
            if len(rtoks) >= 2 and len(rtoks[-1]) == 1:
                add(f"{rtoks[0]} {last}")
                add(f"{last} {rtoks[0]}")
            elif len(rtoks) >= 2:
                add(f"{rtoks[0]} {rtoks[-1]}")
                add(f"{rtoks[-1]} {rtoks[0]}")
            for t in rtoks:
                if len(t) >= 2:
                    add(t)
            if len(last) >= 2:
                add(last)
        return variants

    # Space-separated personal name
    toks = [_tok(t) for t in raw.split() if _tok(t)]
    if not toks:
        return variants

    significant = [t for t in toks if len(t) >= 2]

    if len(toks) >= 2 and len(toks[-1]) == 1:
        add(" ".join(toks[:-1]))

    if len(toks) >= 2:
        a, b = toks[0], toks[1]
        add(f"{a} {b}")
        if a.lower() != b.lower():
            add(f"{b} {a}")

    if len(toks) >= 3 and len(toks[-1]) >= 2:
        add(f"{toks[0]} {toks[-1]}")
        if toks[0].lower() != toks[-1].lower():
            add(f"{toks[-1]} {toks[0]}")

    for t in significant:
        add(t)

    return variants


def _normalize_name_tokens(s: str) -> set:
    """Order-independent name comparison: 'John Smith' and 'Smith John' -> same token set."""
    return set(_tokenize(s))


def _company_token_overlap(our_name: str, their_name: str) -> float:
    """Jaccard-like overlap for company names (handles ZoomInfo shortened names)."""
    our_t = _normalize_name_tokens(our_name)
    their_t = set(_tokenize(their_name or ""))
    if not our_t:
        return 0.0
    inter = len(our_t & their_t)
    return inter / len(our_t)  # fraction of our tokens found in theirs


def _tokenize(s: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, return non-empty tokens."""
    if not s or not isinstance(s, str):
        return []
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _normalize_website(url: Optional[str]) -> str:
    """Normalize URL for comparison: lowercase, strip protocol, www, trailing slash and path (optional)."""
    if not url or not isinstance(url, str):
        return ""
    s = url.strip().lower()
    for prefix in ("https://", "http://", "www."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("www."):
        s = s[4:]
    # Compare domain only (strip path) so https://acme.com/contact matches https://acme.com
    if "/" in s:
        s = s.split("/")[0]
    return s.rstrip("/") or ""


# Host substrings → treat as marketplace/listing; company site is not the registrant domain.
_OWNER_URL_LISTING_HOST_MARKERS = (
    "aircraftpost",
    "controller",
    "jetnet",
    "aminusa",
    "avbuyer",
    "aso.com",
    "globalair.com",
    "trade-a-plane",
    "controller.com",
)


def _owner_url_host_looks_like_listing_portal(host: str) -> bool:
    h = (host or "").lower()
    if h.startswith("www."):
        h = h[4:]
    return any(m in h for m in _OWNER_URL_LISTING_HOST_MARKERS)


def _zoominfo_domain_in_owner_url(their_domain: str, owner_url: str) -> bool:
    """True if normalized ZoomInfo company domain appears in owner_url (path, query, or host)."""
    if not their_domain or not owner_url:
        return False
    ou = owner_url.strip().lower()
    bare = ou.replace("https://", "").replace("http://", "")
    host_part = bare.split("/")[0].split("?")[0]
    if their_domain in host_part or their_domain in bare:
        return True
    return False


def _split_domain_label_to_company_words(label: str) -> str:
    """
    Turn a registrable domain label (e.g. ``fremontgroup``, ``acme-corp``) into a spaced company-style
    string for ZoomInfo company search (``Fremont Group``, ``Acme Corp``).
    """
    if not label or not isinstance(label, str):
        return ""
    raw = label.strip().replace("-", " ").replace("_", " ")
    low = raw.lower().replace(" ", "")
    if not low:
        return ""
    suffixes = (
        "international",
        "associates",
        "solutions",
        "services",
        "holdings",
        "ventures",
        "partners",
        "capital",
        "global",
        "systems",
        "group",
        "corp",
        "inc",
    )
    for suf in suffixes:
        if low.endswith(suf) and len(low) > len(suf) + 1:
            stem = low[: -len(suf)].rstrip("-")
            if stem:
                return f"{stem.title()} {suf.title()}"
    return raw.title()


def _llm_owner_url_aligns_with_zoominfo_company(
    owner_url: str,
    zoominfo_company_name: str,
    zoominfo_website: Optional[str],
    openai_key: str,
) -> str:
    """
    After matching AircraftPost → ZoomInfo by name, verify the owner listing URL plausibly refers to the same entity.
    Returns: 'yes' | 'no' | 'unknown'
    """
    if not owner_url or not openai_key:
        return "unknown"
    try:
        import openai

        client = openai.OpenAI(api_key=openai_key, timeout=25.0)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Does this aircraft owner/operator profile URL most likely refer to the SAME business entity as "
                        "the ZoomInfo company below (same operator or parent), not merely a similar name?\n"
                        "Reply with exactly one word: YES, NO, or UNKNOWN\n\n"
                        f"URL: {owner_url[:900]}\n"
                        f"ZoomInfo company name: {zoominfo_company_name[:300]}\n"
                        f"ZoomInfo website (may be empty): {zoominfo_website or '—'}"
                    ),
                }
            ],
            max_tokens=5,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        if text.startswith("Y"):
            return "yes"
        if text.startswith("N"):
            return "no"
        return "unknown"
    except Exception as e:
        logger.debug("llm_owner_url_aligns_with_zoominfo_company failed: %s", e)
        return "unknown"


def _faa_item_geo_bucket(item: dict) -> str:
    """Infer registrant geography from FAA-derived owner item: 'us' | 'intl' | 'unk'."""
    if (item.get("source_platform") or "").strip().lower() != "faa":
        return "unk"
    c = (item.get("country") or "").strip().upper().replace(".", "")
    st = (item.get("state") or "").strip().upper()
    if c in ("US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"):
        return "us"
    if len(st) == 2 and st in _US_STATE_CODES:
        return "us"
    if c in ("GB", "UK", "CANADA", "CA", "AUSTRALIA", "AU", "MEXICO", "MX", "FRANCE", "FR", "GERMANY", "DE"):
        return "intl"
    if c and c not in ("US", "USA"):
        # Any other explicit non-empty country → treat as non-US
        return "intl"
    return "unk"


def _owner_source_is_us_for_zoominfo(item: dict) -> bool:
    """True when owner context suggests US — narrow ZoomInfo company hits for FAA and AircraftPost."""
    sp = (item.get("source_platform") or "").strip().lower()
    if sp == "faa":
        return _faa_item_geo_bucket(item) == "us"
    if sp == "aircraftpost":
        cc = (item.get("country_code") or "").strip().upper()
        return cc in ("US", "USA")
    return False


def _zoominfo_company_geo_bucket(attrs: dict) -> str:
    """Infer ZoomInfo company geography: 'us' | 'intl' | 'unk'."""
    c_raw = str(attrs.get("country") or "").strip().lower()
    st = str(attrs.get("state") or "").strip().upper()
    st_lower = str(attrs.get("state") or "").strip().lower()
    city_lower = str(attrs.get("city") or "").strip().lower()
    if any(
        x in c_raw
        for x in (
            "united kingdom",
            "england",
            "scotland",
            "wales",
            "northern ireland",
            "great britain",
        )
    ):
        return "intl"
    if c_raw in ("gb", "uk"):
        return "intl"
    if "united states" in c_raw or c_raw in ("us", "usa"):
        return "us"
    if len(st) == 2 and st in _US_STATE_CODES:
        return "us"
    if "greater london" in st_lower or (city_lower == "london" and "united states" not in c_raw and c_raw not in ("us", "usa")):
        return "intl"
    return "unk"


def _zoominfo_foreign_signals_penalty(item: dict, attrs: dict) -> float:
    """
    When FAA context is US, penalize obvious non-US company signals (same-name homonyms).
    Same when AircraftPost row has US country_code.
    Returns negative delta to add to score (e.g. -3.0).
    """
    sp = (item.get("source_platform") or "").strip().lower()
    if sp == "faa":
        if _faa_item_geo_bucket(item) != "us":
            return 0.0
    elif sp == "aircraftpost":
        cc = (item.get("country_code") or "").strip().upper()
        if cc not in ("US", "USA"):
            return 0.0
    else:
        return 0.0
    penalty = 0.0
    tw = _normalize_website(attrs.get("website"))
    if tw.endswith(".co.uk") or tw.endswith(".uk"):
        penalty -= 3.0
    for p in (
        attrs.get("phone"),
        attrs.get("directPhone"),
        attrs.get("mainPhone"),
        attrs.get("mobilePhone"),
    ):
        ps = str(p or "").strip().replace(" ", "")
        if ps.startswith("+44") or ps.startswith("0044"):
            penalty -= 2.5
            break
    zb = _zoominfo_company_geo_bucket(attrs)
    if zb == "intl":
        penalty -= 4.0
    elif zb == "us":
        penalty += 1.0  # small bonus when both look US
    return penalty


def _location_match_score(our_tokens: List[str], their_str: str) -> float:
    """Score 0..1: fraction of our tokens that appear in their string."""
    if not our_tokens:
        return 0.0
    their_tokens = set(_tokenize(their_str or ""))
    if not their_tokens:
        return 0.0
    matches = sum(1 for t in our_tokens if t in their_tokens)
    return matches / len(our_tokens)


def _pick_best_zoominfo_company(
    companies: List[Any],
    source_platform: str,
    *,
    seller_location: Optional[str] = None,
    city: Optional[str] = None,
    street: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
) -> List[Any]:
    """
    When ZoomInfo returns multiple companies, pick the best match by location.
    Controller: use seller_location vs company address/city/state.
    FAA: use city and street vs company city and street.
    AircraftExchange: no location → return first.
    Returns list of one company (best match) or unchanged if single/empty.
    """
    if not companies or len(companies) <= 1:
        return companies
    platform = (source_platform or "").strip().lower()
    if platform == "aircraftexchange":
        return [companies[0]]
    # Build our location tokens for scoring
    our_tokens = []
    if platform == "controller" and seller_location:
        our_tokens = _tokenize(seller_location)
    elif platform == "faa":
        if city:
            our_tokens.extend(_tokenize(city))
        if street:
            our_tokens.extend(_tokenize(street))
        if state:
            our_tokens.extend(_tokenize(state))
        if country:
            our_tokens.extend(_tokenize(country))
    if not our_tokens:
        return [companies[0]]
    best_idx = 0
    best_score = 0.0
    for i, co in enumerate(companies):
        attrs = co.get("attributes") or {}
        their_parts = [
            attrs.get("address") or "",
            attrs.get("addressLine1") or "",
            attrs.get("city") or "",
            attrs.get("state") or "",
            attrs.get("country") or "",
        ]
        their_str = " ".join(their_parts)
        score = _location_match_score(our_tokens, their_str)
        if score > best_score:
            best_score = score
            best_idx = i
    return [companies[best_idx]]


def _score_zoominfo_company(co: Any, item: dict) -> tuple:
    """
    Score one ZoomInfo company against our item. Returns (score, matched).
    matched = { "company": bool, "person": bool, "phone": bool, "location": bool, "website": bool }.
    """
    attrs = co.get("attributes") or {}
    matched = {"company": False, "person": False, "phone": False, "location": False, "website": False}
    score = 0.0
    # Company name overlap
    our_company = (item.get("company_name") or "").strip()
    their_name = (attrs.get("name") or attrs.get("companyName") or "").strip()
    if our_company and their_name:
        # FAA registrant as person: name is not a legal entity; don't score vs company name (false overlaps).
        if (item.get("registrant_type") or "").lower() != "person":
            co_score = _company_token_overlap(our_company, their_name)
            if co_score > 0:
                matched["company"] = co_score >= 0.5
                score += co_score * 2.0
    # Website (strong signal when both sides have it)
    our_website = _normalize_website(item.get("website"))
    their_website = _normalize_website(attrs.get("website"))
    if our_website and their_website and our_website == their_website:
        matched["website"] = True
        score += 2.0
    elif their_website and item.get("owner_url"):
        # AircraftPost: listing URL may embed the ZoomInfo company's domain, or we matched via inferred name.
        if _zoominfo_domain_in_owner_url(their_website, str(item["owner_url"])):
            matched["website"] = True
            score += 2.0
    elif (
        (item.get("source_platform") or "").lower() == "aircraftpost"
        and item.get("owner_url")
        and their_name
        and not matched["website"]
    ):
        # Company **search** hits often omit ``website``; compare domain label to returned company name.
        dom = _normalize_website(str(item["owner_url"]))
        if dom and "." in dom:
            dlabel = dom.split(".")[0]
            guess = _split_domain_label_to_company_words(dlabel)
            if guess and _company_token_overlap(guess, their_name) >= 0.45:
                matched["website"] = True
                score += 1.75
    # Phone (company may have main phone)
    our_phone = zoominfo_normalize_phone(item.get("phone"))
    their_phones = [
        attrs.get("phone"),
        attrs.get("directPhone"),
        attrs.get("mainPhone"),
        attrs.get("mobilePhone"),
    ]
    for p in their_phones:
        if zoominfo_phones_match(item.get("phone"), p):
            matched["phone"] = True
            score += 1.5
            break
    # Location
    our_tokens = []
    if (item.get("source_platform") or "").lower() == "controller" and item.get("address"):
        our_tokens = _tokenize(item["address"])
    elif (item.get("source_platform") or "").lower() == "faa":
        if item.get("city"):
            our_tokens.extend(_tokenize(item["city"]))
        if item.get("street"):
            our_tokens.extend(_tokenize(item["street"]))
        if item.get("state"):
            our_tokens.extend(_tokenize(item["state"]))
        if item.get("country"):
            our_tokens.extend(_tokenize(item["country"]))
    if our_tokens:
        their_str = " ".join(
            [str(attrs.get(k) or "") for k in ("address", "addressLine1", "city", "state", "country", "zipCode")]
        )
        loc_score = _location_match_score(our_tokens, their_str)
        if loc_score > 0:
            matched["location"] = loc_score >= 0.5
            score += loc_score * 1.5
    elif (item.get("source_platform") or "").lower() == "aircraftpost":
        cc = (item.get("country_code") or "").strip().upper()
        zb = _zoominfo_company_geo_bucket(attrs)
        if cc in ("US", "USA") and zb == "us":
            matched["location"] = True
            score += 0.85
        elif len(cc) == 2 and cc not in ("US", "USA", "") and zb == "intl":
            matched["location"] = True
            score += 0.85
    score += _zoominfo_foreign_signals_penalty(item, attrs)
    return (score, matched)


def _score_zoominfo_contact(contact: Any, item: dict) -> tuple:
    """
    Score one ZoomInfo contact against our item. Returns (score, matched).
    matched includes website=False (contacts rarely have website; company scoring uses website).
    """
    attrs = contact.get("attributes") or {}
    matched = {"company": False, "person": False, "phone": False, "location": False, "website": False}
    score = 0.0
    # Person name (order-independent)
    our_names = []
    for k in ("contact_name", "broker_name", "company_name"):
        v = (item.get(k) or "").strip()
        if v and k == "company_name" and _clean_company_name(v) != v:
            continue  # skip company-only for person match when we have contact_name
        if v:
            our_names.append(v)
    their_full = (attrs.get("fullName") or attrs.get("firstName") or "") + " " + (attrs.get("lastName") or "")
    their_tokens = _normalize_name_tokens(their_full)
    for our_name in our_names:
        if not our_name:
            continue
        our_tokens = _normalize_name_tokens(our_name)
        if our_tokens and our_tokens <= their_tokens or (our_tokens & their_tokens):
            j = len(our_tokens & their_tokens) / len(our_tokens) if our_tokens else 0
            if j >= 0.5:
                matched["person"] = True
                score += 1.5
            score += j
            break
    # Company name (contact's company)
    our_company = (item.get("company_name") or "").strip()
    their_company = (attrs.get("companyName") or attrs.get("company") or "").strip()
    if our_company and their_company:
        c_score = _company_token_overlap(our_company, their_company)
        if c_score > 0:
            matched["company"] = c_score >= 0.5
            score += c_score
    # Phone
    their_phones = [attrs.get("phone"), attrs.get("directPhone"), attrs.get("mobilePhone"), attrs.get("workPhone")]
    for p in their_phones:
        if zoominfo_phones_match(item.get("phone"), p):
            matched["phone"] = True
            score += 1.5
            break
    # Location
    our_tokens = []
    if (item.get("source_platform") or "").lower() == "controller" and item.get("address"):
        our_tokens = _tokenize(item["address"])
    elif (item.get("source_platform") or "").lower() == "faa":
        if item.get("city"):
            our_tokens.extend(_tokenize(item["city"]))
        if item.get("street"):
            our_tokens.extend(_tokenize(item["street"]))
        if item.get("state"):
            our_tokens.extend(_tokenize(item["state"]))
        if item.get("country"):
            our_tokens.extend(_tokenize(item["country"]))
    if our_tokens:
        their_str = " ".join([str(attrs.get(k) or "") for k in ("city", "state", "address", "country")])
        loc_score = _location_match_score(our_tokens, their_str)
        if loc_score > 0:
            matched["location"] = loc_score >= 0.5
            score += loc_score * 1.5
    return (score, matched)


def _record_phone_matches(record: Any, our_phone: str, *, is_contact: bool) -> bool:
    """True if the ZoomInfo record has any phone field that matches our_phone (robust normalization)."""
    if not our_phone or not zoominfo_normalize_phone(our_phone):
        return False
    attrs = record.get("attributes") or {}
    if is_contact:
        phone_keys = ("phone", "directPhone", "mobilePhone", "workPhone", "phoneNumber")
    else:
        phone_keys = ("phone", "directPhone", "mainPhone", "mobilePhone")
    for k in phone_keys:
        val = attrs.get(k)
        if val and zoominfo_phones_match(our_phone, val):
            return True
    return False


def _pick_best_by_phone_first(
    companies: List[Any],
    contacts: List[Any],
    item: dict,
) -> tuple:
    """
    Phone-first: if we have a phone and any result matches it, return that immediately (one match = correct).
    Otherwise pick best by company/location/name score.
    Returns (best_record, result_type, matched).
    """
    our_phone = (item.get("phone") or "").strip()
    has_phone = bool(zoominfo_normalize_phone(our_phone))
    if has_phone:
        phone_matched_companies = [c for c in companies if _record_phone_matches(c, our_phone, is_contact=False)]
        phone_matched_contacts = [c for c in contacts if _record_phone_matches(c, our_phone, is_contact=True)]
        if phone_matched_contacts:
            # Prefer contact when phone matches (person is usually the one with that phone)
            best = phone_matched_contacts[0]
            used_location = False
            if len(phone_matched_contacts) > 1:
                our_tokens = []
                if (item.get("source_platform") or "").lower() == "controller" and item.get("address"):
                    our_tokens = _tokenize(item["address"])
                elif (item.get("source_platform") or "").lower() == "faa":
                    if item.get("city"):
                        our_tokens.extend(_tokenize(item["city"]))
                    if item.get("street"):
                        our_tokens.extend(_tokenize(item["street"]))
                if our_tokens:
                    best_score = -1.0
                    for c in phone_matched_contacts:
                        attrs = c.get("attributes") or {}
                        their_str = " ".join([str(attrs.get(k) or "") for k in ("city", "state", "address")])
                        s = _location_match_score(our_tokens, their_str)
                        if s > best_score:
                            best_score = s
                            best = c
                            used_location = s >= 0.5
            return (best, "contact", {"company": False, "person": False, "phone": True, "location": used_location, "website": False}, None)
        if phone_matched_companies:
            best = phone_matched_companies[0]
            used_location = False
            if len(phone_matched_companies) > 1:
                our_tokens = []
                if (item.get("source_platform") or "").lower() == "controller" and item.get("address"):
                    our_tokens = _tokenize(item["address"])
                elif (item.get("source_platform") or "").lower() == "faa":
                    if item.get("city"):
                        our_tokens.extend(_tokenize(item["city"]))
                    if item.get("street"):
                        our_tokens.extend(_tokenize(item["street"]))
                if our_tokens:
                    best_score = -1.0
                    for c in phone_matched_companies:
                        attrs = c.get("attributes") or {}
                        their_str = " ".join([str(attrs.get(k) or "") for k in ("address", "addressLine1", "city", "state")])
                        s = _location_match_score(our_tokens, their_str)
                        if s > best_score:
                            best_score = s
                            best = c
                            used_location = s >= 0.5
            return (best, "company", {"company": False, "person": False, "phone": True, "location": used_location, "website": False}, None)
    # No phone or no phone match: fall back to company/name/location scoring (return score for fallback trigger)
    best, btype, matched, score = _pick_best_zoominfo_result(companies, contacts, item)
    return (best, btype, matched, score)


def _pick_best_zoominfo_result(
    companies: List[Any],
    contacts: List[Any],
    item: dict,
) -> tuple:
    """
    From many companies and contacts, pick the single best match by score.
    Returns (best_record, result_type, matched) where result_type is "company" or "contact",
    and matched = { company, person, phone, location }.
    """
    best = None
    best_type = None
    best_score = -1.0
    best_matched = {"company": False, "person": False, "phone": False, "location": False, "website": False}
    for co in companies:
        score, matched = _score_zoominfo_company(co, item)
        if score > best_score:
            best_score = score
            best = co
            best_type = "company"
            best_matched = matched
    for c in contacts:
        score, matched = _score_zoominfo_contact(c, item)
        if score > best_score:
            best_score = score
            best = c
            best_type = "contact"
            best_matched = matched
    if best is None and companies:
        best = companies[0]
        best_type = "company"
        _, best_matched = _score_zoominfo_company(best, item)
        best_score = 0.0
    if best is None and contacts:
        best = contacts[0]
        best_type = "contact"
        _, best_matched = _score_zoominfo_contact(best, item)
        best_score = 0.0
    return (best, best_type, best_matched, best_score)


def _item_to_text(item: dict) -> str:
    """Build a single text block for our owner item (for embedding / LLM context)."""
    parts = []
    if item.get("company_name"):
        parts.append(f"Company: {item['company_name']}")
    if item.get("contact_name"):
        parts.append(f"Contact: {item['contact_name']}")
    if item.get("broker_name"):
        parts.append(f"Broker: {item['broker_name']}")
    if item.get("phone"):
        parts.append(f"Phone: {item['phone']}")
    if item.get("address"):
        parts.append(f"Address: {item['address']}")
    if item.get("city") or item.get("state") or item.get("zip_code") or item.get("country"):
        loc = ", ".join(filter(None, [item.get("city"), item.get("state"), item.get("zip_code"), item.get("country")]))
        if loc:
            parts.append(f"Location: {loc}")
    if item.get("street"):
        parts.append(f"Street: {item['street']}")
    if item.get("website"):
        parts.append(f"Website: {item['website']}")
    if item.get("owner_url"):
        parts.append(f"Owner URL: {item['owner_url']}")
    return " | ".join(parts) if parts else ""


def _zoominfo_record_to_text(record: Any, is_contact: bool) -> str:
    """
    Build a single text block for one ZoomInfo company or contact (for embedding / LLM context).
    Includes website when present so vector search and LLM can use it to verify company identity.
    """
    attrs = record.get("attributes") or {}
    parts = []
    if is_contact:
        name = (attrs.get("fullName") or "").strip() or f"{attrs.get('firstName') or ''} {attrs.get('lastName') or ''}".strip()
        if name:
            parts.append(f"Person: {name}")
        company = (attrs.get("companyName") or attrs.get("company") or "").strip()
        if company:
            parts.append(f"Company: {company}")
        if attrs.get("website"):
            parts.append(f"Website: {attrs.get('website')}")
        for k in ("phone", "directPhone", "mobilePhone", "workPhone"):
            if attrs.get(k):
                parts.append(f"Phone: {attrs[k]}")
                break
        for k in ("city", "state", "address"):
            if attrs.get(k):
                parts.append(f"Location: {attrs[k]}")
                break
    else:
        name = (attrs.get("name") or attrs.get("companyName") or "").strip()
        if name:
            parts.append(f"Company: {name}")
        if attrs.get("website"):
            parts.append(f"Website: {attrs.get('website')}")
        for k in ("phone", "directPhone", "mainPhone"):
            if attrs.get(k):
                parts.append(f"Phone: {attrs[k]}")
                break
        addr = [attrs.get("addressLine1"), attrs.get("city"), attrs.get("state"), attrs.get("zipCode")]
        loc = ", ".join(str(x) for x in addr if x)
        if loc:
            parts.append(f"Location: {loc}")
    return " | ".join(parts) if parts else str(attrs)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 if invalid."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pick_best_zoominfo_by_vector_and_llm(
    companies: List[Any],
    contacts: List[Any],
    item: dict,
    embedding_service: Any,
    openai_api_key: str,
) -> tuple:
    """
    Fallback when content/word matching is weak: rank candidates by vector similarity,
    then ask LLM to pick the best match. Returns (best_record, result_type, matched) or (None, None, {}).
    """
    candidates = []
    for c in companies:
        candidates.append(("company", c, _zoominfo_record_to_text(c, is_contact=False)))
    for c in contacts:
        candidates.append(("contact", c, _zoominfo_record_to_text(c, is_contact=True)))
    if not candidates:
        return (None, None, {})
    item_text = _item_to_text(item)
    if not item_text.strip():
        return (None, None, {})
    try:
        query_vec = embedding_service.embed_text(item_text)
        if not query_vec:
            return (None, None, {})
        texts = [t for _, _, t in candidates]
        cand_embeddings = embedding_service.embed_batch(texts, batch_size=50)
        scored = []
        for i, (rtype, record, _) in enumerate(candidates):
            if i < len(cand_embeddings) and cand_embeddings[i]:
                sim = _cosine_sim(query_vec, cand_embeddings[i])
                attrs = record.get("attributes") or {}
                if (item.get("source_platform") or "").strip().lower() == "faa":
                    # Down-rank same-name foreign companies (e.g. Valiair UK vs US registrant).
                    sim = sim + _zoominfo_foreign_signals_penalty(item, attrs) * 0.22
                scored.append((sim, rtype, record))
        if not scored:
            return (None, None, {})
        scored.sort(key=lambda x: -x[0])
        top_k = scored[:10]
        if len(top_k) == 1:
            _, rtype, record = top_k[0]
            return (record, rtype, {"company": False, "person": False, "phone": False, "location": False, "website": False, "llm_fallback": True})
        faa_geo_rule = ""
        if (item.get("source_platform") or "").strip().lower() == "faa":
            loc_hint = ", ".join(
                filter(
                    None,
                    [
                        item.get("city"),
                        item.get("state"),
                        item.get("country"),
                    ],
                )
            )
            faa_geo_rule = (
                "\nCRITICAL: Our registrant is from FAA records"
                + (f" (mailing / location hint: {loc_hint})." if loc_hint else ".")
                + " If several companies share the same name, choose the one in the SAME country/region as this registrant "
                "(typically United States for US city/state). Reject homonymous companies in other countries (e.g. UK .co.uk, +44 phone) "
                "unless you are certain it is the same entity.\n"
            )
        prompt = f"""We have owner/seller data from our database and {len(top_k)} candidate records from ZoomInfo. Pick the single best matching ZoomInfo candidate.
{faa_geo_rule}
Our data:
{item_text}

Candidates (reply with only the number 1-{len(top_k)} of the best match, or 0 if none match):
"""
        for idx, (_, rtype, record) in enumerate(top_k, 1):
            text = _zoominfo_record_to_text(record, is_contact=(rtype == "contact"))
            prompt += f"\n[{idx}] ({rtype}) {text}"
        prompt += "\nAnswer with only the number (e.g. 2):"

        import openai
        client = openai.OpenAI(api_key=openai_api_key, timeout=30.0)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        reply = (response.choices[0].message.content or "").strip()
        num = None
        for t in reply.split():
            t = "".join(c for c in t if c.isdigit())
            if t:
                num = int(t)
                break
        if num and 1 <= num <= len(top_k):
            _, rtype, record = top_k[num - 1]
            return (record, rtype, {"company": False, "person": False, "phone": False, "location": False, "website": False, "llm_fallback": True})
        if num == 0:
            return (None, None, {})
        return (top_k[0][2], top_k[0][1], {"company": False, "person": False, "phone": False, "location": False, "website": False, "llm_fallback": True})
    except Exception as e:
        logger.warning("ZoomInfo vector+LLM fallback failed: %s", e)
        return (None, None, {})


# --- Pydantic models ---
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=0)

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = None  # prior conversation for context

class ChatResponse(BaseModel):
    answer: str
    sources: List[Any] = []
    data_used: Optional[dict] = None  # e.g. {"aircraft listing": 3, "aircraft sale": 2}
    error: Optional[str] = None

class MarketComparisonRequest(BaseModel):
    models: List[str] = Field(..., min_length=1)
    region: Optional[str] = None
    max_hours: Optional[float] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    limit: int = Field(50, ge=1, le=200)

class PriceEstimateRequest(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    flight_hours: Optional[float] = None
    flight_cycles: Optional[int] = None
    region: Optional[str] = None

class ResaleAdvisoryRequest(BaseModel):
    query: Optional[str] = None
    listing_id: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None

# --- App ---
app = FastAPI(
    title="HyeAero.AI API",
    description="Aircraft Research & Valuation Consultant — market comparison, pricing, resale advisory, RAG chat",
    version="1.0.0",
)

# CORS: set CORS_ORIGINS on Render to your frontend URL (e.g. https://your-app.onrender.com)
_default_cors = "http://localhost:3000,http://127.0.0.1:3000,http://88.99.198.243:3000,http://88.99.198.243"
_cors_raw = os.getenv("CORS_ORIGINS", _default_cors).strip().split(",")
_allow_origins = [o.strip() for o in _cors_raw if o.strip()]
if not _allow_origins:
    _allow_origins = _default_cors.strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded dependencies
_config: Optional[Config] = None
_db: Optional[PostgresClient] = None
_rag: Optional[RAGQueryService] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()  # Do not call .validate() so API can run with partial config
    return _config

def get_db() -> PostgresClient:
    global _db
    if _db is None:
        c = get_config()
        if not c.postgres_connection_string:
            raise HTTPException(status_code=503, detail="PostgreSQL not configured")
        _db = PostgresClient(c.postgres_connection_string)
    return _db

def get_rag() -> RAGQueryService:
    global _rag
    if _rag is None:
        c = get_config()
        if not all([c.openai_api_key, c.pinecone_api_key, c.postgres_connection_string]):
            raise HTTPException(status_code=503, detail="RAG not configured (OpenAI, Pinecone, Postgres)")
        emb = EmbeddingService(
            api_key=c.openai_api_key,
            model=c.openai_embedding_model,
            dimension=c.openai_embedding_dimension,
        )
        pc = PineconeClient(
            api_key=c.pinecone_api_key,
            index_name=c.pinecone_index_name,
            dimension=c.pinecone_dimension,
            metric=c.pinecone_metric,
            host=c.pinecone_host,
        )
        if not pc.connect():
            raise HTTPException(status_code=503, detail="Pinecone connection failed")
        _rag = RAGQueryService(
            embedding_service=emb,
            pinecone_client=pc,
            postgres_client=PostgresClient(c.postgres_connection_string),
            openai_api_key=c.openai_api_key,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        )
    return _rag


def get_embedding_and_pinecone():
    """Return (embedding_service, pinecone_client) when configured and connected; else (None, None)."""
    c = get_config()
    if not c.openai_api_key or not c.pinecone_api_key:
        return None, None
    try:
        emb = EmbeddingService(
            api_key=c.openai_api_key,
            model=c.openai_embedding_model,
            dimension=c.openai_embedding_dimension,
        )
        pc = PineconeClient(
            api_key=c.pinecone_api_key,
            index_name=c.pinecone_index_name,
            dimension=c.pinecone_dimension,
            metric=c.pinecone_metric,
            host=c.pinecone_host,
        )
        if not pc.connect():
            return None, None
        return emb, pc
    except Exception:
        return None, None


def get_embedding_service_only():
    """Return (embedding_service, openai_api_key) when OpenAI is configured; else (None, None). Used for ZoomInfo vector+LLM fallback without Pinecone."""
    c = get_config()
    if not c.openai_api_key:
        return None, None
    try:
        emb = EmbeddingService(
            api_key=c.openai_api_key,
            model=c.openai_embedding_model,
            dimension=c.openai_embedding_dimension,
        )
        return emb, c.openai_api_key
    except Exception:
        return None, None

@app.get("/")
def root():
    return {"service": "HyeAero.AI", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/aircraft-models")
def aircraft_models():
    """Return distinct aircraft models (manufacturer + model) that have at least one listing.
    This ensures the Market Comparison dropdown only shows models that can return comparison results."""
    try:
        db = get_db()
        rows = db.execute_query(
            """
            SELECT DISTINCT a.manufacturer, a.model
            FROM aircraft_listings l
            INNER JOIN aircraft a ON l.aircraft_id = a.id
            WHERE ((a.manufacturer IS NOT NULL AND TRIM(a.manufacturer) != '')
               OR (a.model IS NOT NULL AND TRIM(a.model) != ''))
            ORDER BY a.manufacturer, a.model
            """
        )
        seen = set()
        models = []
        for r in rows:
            man = (r.get("manufacturer") or "").strip()
            mod = (r.get("model") or "").strip()
            label = f"{man} {mod}".strip()
            if label and label not in seen:
                seen.add(label)
                models.append(label)
        return {"models": models}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price-estimate-models")
def price_estimate_models():
    """Return distinct aircraft models from aircraft_sales that have at least one sale.
    Use this list for the Price Estimator dropdown so selected models always have comparable sales."""
    try:
        db = get_db()
        rows = db.execute_query(
            """
            SELECT DISTINCT manufacturer, model
            FROM aircraft_sales
            WHERE sold_price IS NOT NULL AND sold_price > 0
              AND (COALESCE(manufacturer,'') != '' OR COALESCE(model,'') != '')
            ORDER BY manufacturer, model
            """
        )
        seen = set()
        models = []
        for r in rows:
            man = (r.get("manufacturer") or "").strip()
            mod = (r.get("model") or "").strip()
            label = f"{man} {mod}".strip()
            if label and label not in seen:
                seen.add(label)
                models.append(label)
        sample = None
        test_payloads = []
        if models:
            sample = {"model": models[0], "region": "Global"}
            # First 5 models as test values (exact strings from DB — will return results)
            for m in models[:5]:
                test_payloads.append({"model": m, "region": "Global"})
        return {"models": models, "sample_request": sample, "test_payloads": test_payloads}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/answer", response_model=ChatResponse)
def rag_answer(req: ChatRequest):
    """Ask the Consultant: natural language over Hye Aero's sale history + market feed (RAG)."""
    try:
        rag = get_rag()
        history_dicts = [{"role": m.role, "content": m.content} for m in (req.history or [])]
        out = rag.answer(query=req.query.strip(), top_k=20, history=history_dicts if history_dicts else None)
        if out.get("error"):
            raise HTTPException(status_code=500, detail=out["error"])
        return ChatResponse(
            answer=out["answer"],
            sources=out.get("sources", []),
            data_used=out.get("data_used"),
            error=out.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/market-comparison")
def market_comparison(req: MarketComparisonRequest):
    """Dynamic market comparison: compare aircraft by model, age, hours, region (Controller, AircraftExchange, FAA, Internal DB)."""
    try:
        db = get_db()
        result = run_comparison(
            db=db,
            models=req.models,
            region=req.region,
            max_hours=req.max_hours,
            min_year=req.min_year,
            max_year=req.max_year,
            limit=req.limit,
        )
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/price-estimate")
def price_estimate(req: PriceEstimateRequest):
    """Predictive pricing: vector search first (Pinecone), then full details from PostgreSQL;
    if no vector results, fall back to PostgreSQL keyword search."""
    try:
        db = get_db()
        emb, pc = get_embedding_and_pinecone()
        result = estimate_value_hybrid(
            db=db,
            embedding_service=emb,
            pinecone_client=pc,
            manufacturer=req.manufacturer,
            model=req.model,
            year=req.year,
            flight_hours=req.flight_hours,
            flight_cycles=req.flight_cycles,
            region=req.region,
        )
        aviacost_ref = lookup_aviacost(db, manufacturer=req.manufacturer, model=req.model)
        if aviacost_ref:
            result["aviacost_reference"] = aviacost_ref
        ap_ref = lookup_aircraftpost_fleet(
            db,
            manufacturer=req.manufacturer,
            model=req.model,
            serial=None,
        )
        if ap_ref:
            result["aircraftpost_fleet_reference"] = ap_ref
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resale-advisory")
def resale_advisory(req: ResaleAdvisoryRequest):
    """Resale advisory: plain-English guidance (e.g. undervalued by X% given comparables). Uses RAG when query provided."""
    try:
        if req.query:
            rag = get_rag()
            out = rag.answer(
                query=f"Resale advisory: {req.query}. Provide plain-English guidance on whether this aircraft is over/undervalued and why, given comparable sales and market data. Use plain text and bullet points (-) only; do not use markdown headers (#) or double asterisks (**). You may use professional symbols (e.g. •, ✓, →) where appropriate.",
                top_k=8,
            )
            if out.get("error"):
                raise HTTPException(status_code=500, detail=out["error"])
            return {"insight": out["answer"], "sources": out.get("sources", []), "error": None}
        # Structured lookup by listing/model/year could be added here
        return {"insight": "Provide a natural-language query for resale guidance, or specify listing_id/model/year for structured lookup.", "sources": [], "error": None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_REGISTRATION_NOT_EMPTY = "registration_number IS NOT NULL AND TRIM(registration_number) <> ''"
_IDENTIFIERS_NOT_EMPTY = "(serial_number IS NOT NULL AND TRIM(serial_number) <> '') OR (registration_number IS NOT NULL AND TRIM(registration_number) <> '')"


@lru_cache(maxsize=1)
def _load_internaldb_aircraft_keys() -> tuple[list[str], list[str]]:
    """
    Load keys from the PhlyData (Internal DB) CSV:
      etl-pipeline/store/raw/internaldb/aircraft.csv

    Returns (serial_numbers, registration_numbers) as lists of trimmed strings.
    Used to filter the PhlyData aircraft list so the frontend shows only internal DB aircraft.
    """
    repo_root = Path(__file__).resolve().parents[2]
    internal_csv = repo_root / "etl-pipeline" / "store" / "raw" / "internaldb" / "aircraft.csv"
    if not internal_csv.exists():
        logger.warning("Phlydata filter: internal CSV not found at %s", internal_csv)
        return ([], [])

    serials: set[str] = set()
    regs: set[str] = set()
    with open(internal_csv, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return ([], [])
        for row in reader:
            s = (row.get("Serial Number") or "").strip()
            r = (row.get("Registration Number") or "").strip()
            if s:
                serials.add(s)
            if r:
                regs.add(r)
    return (list(serials), list(regs))


@lru_cache(maxsize=1)
def _load_internaldb_aircraft_rows() -> list[dict[str, object]]:
    """
    Load rows from the PhlyData (Internal DB) CSV:
      etl-pipeline/store/raw/internaldb/aircraft.csv

    Returns a list of objects with the exact fields the frontend needs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    internal_csv = repo_root / "etl-pipeline" / "store" / "raw" / "internaldb" / "aircraft.csv"
    if not internal_csv.exists():
        logger.warning("Phlydata aircraft rows: internal CSV not found at %s", internal_csv)
        return []

    def parse_int_or_none(v: object) -> Optional[int]:
        s = "" if v is None else str(v).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    import hashlib

    def stable_id(serial: str, reg: str, make: str, model: str, manufacturer_year: object, delivery_year: object, category: str) -> str:
        # Deterministic id from row content, so pagination/sorting is stable.
        key = "|".join(
            [
                serial.strip(),
                reg.strip(),
                make.strip(),
                model.strip(),
                str(manufacturer_year or "").strip(),
                str(delivery_year or "").strip(),
                category.strip(),
            ]
        )
        return hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()

    rows_out: list[dict[str, object]] = []
    with open(internal_csv, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        for row in reader:
            serial = (row.get("Serial Number") or "").strip()
            reg = (row.get("Registration Number") or "").strip()
            make = (row.get("Make") or "").strip()
            model = (row.get("Model") or "").strip()
            category = (row.get("Category") or "").strip()

            manufacturer_year = parse_int_or_none(row.get("Manufacturer Year"))
            delivery_year = parse_int_or_none(row.get("Delivery Year"))

            if not serial and not reg:
                continue

            rows_out.append(
                {
                    "id": stable_id(serial, reg, make, model, manufacturer_year, delivery_year, category),
                    "serial_number": serial or None,
                    "registration_number": reg or None,
                    "manufacturer": make or None,
                    "model": model or None,
                    "manufacturer_year": manufacturer_year,
                    "delivery_year": delivery_year,
                    "category": category or None,
                }
            )

    return rows_out


@app.get("/api/phlydata/aircraft")
def phlydata_aircraft_list(page: int = 1, page_size: int = 100, q: Optional[str] = None):
    """List aircraft from the PhlyData Internal DB table (`phlydata_aircraft`).

    Search query ``q`` matches **only** ``serial_number`` and ``registration_number`` (ILIKE substring),
    aligned with owner lookup fields (serial + tail / N-number).
    """
    page = max(1, page)
    # Allow the frontend to fetch the entire Internal DB dataset once.
    page_size = min(100000, max(1, page_size))
    offset = (page - 1) * page_size
    search = (q or "").strip()
    try:
        db = get_db()
        params: list[Any] = []
        where_sql = ""
        if search:
            like = f"%{search}%"
            where_sql = """
              WHERE (
                serial_number ILIKE %s OR registration_number ILIKE %s
              )
            """
            params = [like, like]

        total_sql = f"""
          SELECT COUNT(*) AS total
          FROM public.phlydata_aircraft
          {where_sql}
        """
        total_rows = db.execute_query(total_sql, tuple(params))
        total = int(total_rows[0]["total"]) if total_rows else 0

        rows_sql = f"""
          SELECT
            CAST(aircraft_id AS TEXT) AS id,
            serial_number,
            registration_number,
            manufacturer,
            model,
            manufacturer_year,
            delivery_year,
            category
          FROM public.phlydata_aircraft
          {where_sql}
          ORDER BY serial_number NULLS LAST, registration_number NULLS LAST
          LIMIT %s OFFSET %s
        """
        rows = db.execute_query(rows_sql, tuple(params + [page_size, offset]))
        return {"aircraft": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/phlydata/zoominfo/company")
def phlydata_zoominfo_company(company_name: Optional[str] = None, company_id: Optional[int] = None):
    """Get ZoomInfo company detail by company name or company ID (Enrich API). Uses ZoomInfo credits per successful match."""
    if not company_name and company_id is None:
        raise HTTPException(status_code=400, detail="Provide company_name or company_id")
    record, err = zoominfo_enrich_company(company_id=company_id, company_name=company_name or None)
    if err:
        raise HTTPException(status_code=400, detail=err)
    if not record:
        return {"company": None, "message": "No match"}
    return {"company": record, "message": "OK"}


@app.get("/api/phlydata/owners")
def phlydata_owners(
    serial: str,
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    registration: Optional[str] = None,
):
    """Get owner detail for a PhlyData aircraft.

    **Frontend should send** ``serial``, ``registration`` (tail), and ``model`` (plus optional ``manufacturer``).

    - **FAA MASTER** (``faa_master`` table): registrant/address from ingested FAA MASTER CSV. Match
      **N-number + serial** when ``registration`` is sent; else **serial + model** via ``mfr_mdl_code`` /
      ``faa_aircraft_reference``; else **serial-only** when ``model`` is omitted (same ambiguity rules as before).

    - **ZoomInfo** (PhlyData tab): driven from **FAA MASTER** registrant name + address only (up to 6 deduped
      company/person lookups). AircraftPost is **not** used for this tab.

    - **Tavily** (optional): if ``TAVILY_API_KEY`` is set and the registrant looks trustee-like (or corporate + address
      when ``TAVILY_WHEN_CORP_AND_ADDRESS=1``), ``tavily_web_hints`` lists web search results from registrant name +
      mailing address. Disable with ``TAVILY_DISABLED=1``.

    - **Tavily + LLM** (optional): when Tavily returns snippets and ``OPENAI_API_KEY`` is set, ``tavily_llm_synthesis``
      infers a likely operating company from snippets; **medium/high** confidence adds ZoomInfo lookup
      ``faa_tavily_llm_hint``. Disable with ``TAVILY_LLM_SYNTHESIS_DISABLED=1``.
    """
    if not serial or not serial.strip():
        raise HTTPException(status_code=400, detail="serial is required")
    serial = serial.strip()
    mfr = (manufacturer or "").strip() or None
    mdl = (model or "").strip() or None
    try:
        db = get_db()
        # Optional: use `phlydata_aircraft` for display fields (registration, years, category).
        aircraft_rows: list = []
        if mfr and mdl:
            aircraft_rows = db.execute_query(
                """
                SELECT
                  CAST(aircraft_id AS TEXT) AS id,
                  serial_number, registration_number, manufacturer, model,
                  manufacturer_year, delivery_year, category
                FROM public.phlydata_aircraft
                WHERE serial_number = %s
                  AND (manufacturer IS NULL OR manufacturer ILIKE %s)
                  AND (model IS NULL OR model ILIKE %s)
                LIMIT 1
                """,
                (serial, f"%{mfr}%", f"%{mdl}%"),
            )
        if not aircraft_rows and mdl:
            aircraft_rows = db.execute_query(
                """
                SELECT
                  CAST(aircraft_id AS TEXT) AS id,
                  serial_number, registration_number, manufacturer, model,
                  manufacturer_year, delivery_year, category
                FROM public.phlydata_aircraft
                WHERE serial_number = %s
                  AND (model IS NULL OR model ILIKE %s)
                LIMIT 1
                """,
                (serial, f"%{mdl}%"),
            )
        if not aircraft_rows:
            aircraft_rows = db.execute_query(
                """
                SELECT
                  CAST(aircraft_id AS TEXT) AS id,
                  serial_number, registration_number, manufacturer, model,
                  manufacturer_year, delivery_year, category
                FROM public.phlydata_aircraft
                WHERE serial_number = %s
                LIMIT 1
                """,
                (serial,),
            )

        aircraft = dict(aircraft_rows[0]) if aircraft_rows else {
            "id": None,
            "serial_number": serial,
            "registration_number": None,
            "manufacturer": mfr,
            "model": mdl,
            "manufacturer_year": None,
            "delivery_year": None,
            "category": None,
        }
        # Reflect explicit tail from the client (FAA MASTER / canonical N-number lookup).
        if (registration or "").strip():
            aircraft["registration_number"] = (registration or "").strip()

        # Listings (Controller / etc.) — optional; PhlyData aircraft tab uses FAA MASTER + ZoomInfo only.
        owners_from_listings = []

        # Owners from FAA MASTER table (ingested CSV): N-number + serial when tail known, else
        # serial+model via faa_aircraft_reference, else serial-only if model omitted.
        reg_for_faa = (registration or "").strip() or (aircraft.get("registration_number") or "").strip() or None
        faa_rows, faa_master_match_kind = fetch_faa_master_owner_rows(
            db,
            serial=serial,
            model=mdl,
            registration=reg_for_faa,
        )
        owners_from_faa = [dict(r) for r in faa_rows if r.get("registrant_name")]
        owners_from_faa = enrich_faa_owners_with_tavily_hints(owners_from_faa)
        owners_from_faa = enrich_faa_owners_with_tavily_llm_synthesis(owners_from_faa)

        # PhlyData tab: no AircraftPost owner / fleet / ZoomInfo-from-AircraftPost.
        aircraftpost_fleet: list = []
        owners_from_aircraftpost: list = []

        # Build ZoomInfo search payloads (FAA MASTER registrants only, deduped, max 6).
        def _listing_item(row: dict, platform: str, field: str) -> dict:
            company_name = (row.get("seller") or "").strip()
            return {
                "query_name": company_name,
                "source_platform": platform,
                "field_name": field,
                "company_name": company_name,
                "address": (row.get("seller_location") or "").strip() or None,
                "street": None,
                "city": None,
                "state": None,
                "zip_code": None,
                "country": None,
                "contact_name": (row.get("seller_contact_name") or "").strip() or None,
                "broker_name": (row.get("seller_broker") or "").strip() or None,
                "phone": (row.get("seller_phone") or "").strip() or None,
                "website": (row.get("seller_website") or "").strip() or None,  # add seller_website to SELECT when column exists
            }
        by_platform = {"controller": [], "aircraftexchange": [], "faa": []}
        seen_company = set()
        for row in owners_from_faa:
            company_name = (row.get("registrant_name") or "").strip()
            if not company_name or company_name.lower() in seen_company:
                continue
            seen_company.add(company_name.lower())
            street = (row.get("street") or "").strip()
            if (row.get("street2") or "").strip():
                street = f"{street} {(row.get('street2') or '').strip()}".strip() or street
            by_platform["faa"].append({
                "query_name": company_name,
                "source_platform": "faa",
                "field_name": "registrant_name",
                "company_name": company_name,
                # Used by contact scoring when ZoomInfo returns person records.
                "contact_name": company_name,
                "address": None,
                "street": street or None,
                "city": (row.get("city") or "").strip() or None,
                "state": (row.get("state") or "").strip() or None,
                "zip_code": (row.get("zip_code") or "").strip() or None,
                "country": (row.get("country") or "").strip() or None,
                "broker_name": None,
                "phone": None,
                "website": None,
                # Will be computed per-item in the ZoomInfo loop (company vs person).
                "registrant_type": None,
            })
        # Tavily snippets + LLM → operating company guess → extra ZoomInfo query (deduped).
        allow_low_zi = (os.getenv("TAVILY_LLM_ZOOMINFO_ON_LOW") or "").strip().lower() in ("1", "true", "yes")
        for row in owners_from_faa:
            syn = row.get("tavily_llm_synthesis") or {}
            if not isinstance(syn, dict) or syn.get("error"):
                continue
            conf = (syn.get("confidence") or "").lower()
            if conf not in ("high", "medium") and not (allow_low_zi and conf == "low"):
                continue
            qn = (syn.get("suggested_zoominfo_query") or syn.get("operating_company_name") or "").strip()
            if not qn:
                continue
            reg_nm = (row.get("registrant_name") or "").strip()
            if qn.lower() == reg_nm.lower():
                continue
            if qn.lower() in seen_company:
                continue
            seen_company.add(qn.lower())
            street = (row.get("street") or "").strip()
            if (row.get("street2") or "").strip():
                street = f"{street} {(row.get('street2') or '').strip()}".strip() or street
            w = (syn.get("website") or "").strip() or None
            ph = (syn.get("phone") or "").strip() or None
            by_platform["faa"].append(
                {
                    "query_name": qn,
                    "source_platform": "faa_tavily_llm_hint",
                    "field_name": "tavily_llm_synthesis",
                    "company_name": qn,
                    "contact_name": qn,
                    "address": None,
                    "street": street or None,
                    "city": (row.get("city") or "").strip() or None,
                    "state": (row.get("state") or "").strip() or None,
                    "zip_code": (row.get("zip_code") or "").strip() or None,
                    "country": (row.get("country") or "").strip() or None,
                    "broker_name": None,
                    "phone": ph,
                    "website": w,
                    "registrant_type": "company",
                }
            )
        items_to_lookup: list = []
        seen_lookup_key: set = set()

        def _append_lookup_item(it: dict) -> None:
            cn = (it.get("company_name") or "").strip().lower()
            if not cn or cn in seen_lookup_key or len(items_to_lookup) >= 6:
                return
            seen_lookup_key.add(cn)
            items_to_lookup.append(it)

        for it in by_platform["faa"]:
            _append_lookup_item(it)
            if len(items_to_lookup) >= 6:
                break

        logger.info(
            "phlydata/owners: items_to_lookup count=%s (faa=%s)",
            len(items_to_lookup),
            len(by_platform["faa"]),
        )

        zoominfo_enrichment = []
        zoominfo_contacts_access_denied = False
        zoominfo_registrant_type_hint: Optional[str] = None
        for idx, item in enumerate(items_to_lookup):
            url_alignment_source = None
            owner_url_llm_alignment = None
            all_companies = []
            all_contacts = []
            zoominfo_error = None
            logger.info("phlydata/owners: item[%s] query_name=%r source_platform=%s has_phone=%s has_location=%s",
                        idx, item.get("query_name"), item.get("source_platform"),
                        bool((item.get("phone") or "").strip()),
                        bool((item.get("address") or item.get("city") or item.get("street"))))

            # Decide whether registrant is likely a company vs a person.
            registrant_name = (item.get("company_name") or "").strip()
            registrant_type = (item.get("registrant_type") or "").strip().lower() or None

            legal_markers = any(tok in registrant_name.lower() for tok in list(_LEGAL_SUFFIXES) + [" inc", " llc", " ltd", " corp", " co."])
            has_middle_initial = bool(re.search(r"\b[A-Z]\.\b", registrant_name))
            has_comma = "," in registrant_name
            has_person_hint = has_comma or has_middle_initial

            if registrant_type not in {"company", "person"}:
                if legal_markers:
                    registrant_type = "company"
                elif has_person_hint:
                    registrant_type = "person"
                else:
                    # Use LLM only when ambiguous.
                    llm_type = "company"
                    emb_svc, openai_key = get_embedding_service_only()
                    if openai_key:
                        try:
                            import openai

                            client = openai.OpenAI(api_key=openai_key, timeout=30.0)
                            prompt = (
                                "Classify this aircraft owner/registrant name as either an individual person or an organization/company. "
                                "Return only one word: person OR company.\n\n"
                                f"Name: {registrant_name}\n"
                                f"Source: {item.get('source_platform') or 'unknown'}\n"
                                f"Location hint (optional): state={item.get('state')}, country={item.get('country')}, country_code={item.get('country_code')}\n"
                            )
                            resp = client.chat.completions.create(
                                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=5,
                                temperature=0,
                            )
                            llm_type = (resp.choices[0].message.content or "").strip().lower()
                        except Exception:
                            llm_type = "company"
                    registrant_type = "person" if llm_type == "person" else "company"

            item["registrant_type"] = registrant_type

            # Build search variants: company names use cleanup/core/first-word; persons use name permutations.
            original = registrant_name
            if registrant_type == "person":
                queries = _person_name_search_variants(original)
                if not queries:
                    queries = [original]
            else:
                queries = [original]
                cleaned = _clean_company_name(original)
                if cleaned and cleaned.lower() != original.lower():
                    queries.append(cleaned)
                core = _core_company_name(original)
                if core and core.lower() not in {q.lower() for q in queries}:
                    queries.append(core)
                first_word = _first_word_company_name(original)
                if first_word and first_word.lower() not in {q.lower() for q in queries}:
                    queries.append(first_word)

            if registrant_type == "person":
                # Company search by person name (fullName / firstName+lastName + geo). Avoids Contact Search API
                # which often returns 403 without api:data:contact scope.
                all_companies = []
                all_contacts = []
                st = (item.get("state") or "").strip() or None
                ctry = (item.get("country") or "").strip() or None
                z = (item.get("zip_code") or "").strip() or None
                for q in queries:
                    companies, zoominfo_error = zoominfo_search_companies(
                        None,
                        page_size=25,
                        contact_full_name=q,
                        state=st,
                        country=ctry,
                        zip_code=z,
                    )
                    if zoominfo_error:
                        logger.info(
                            "phlydata/owners: item[%s] company_search_by_person query=%r -> error=%s",
                            idx,
                            q,
                            zoominfo_error,
                        )
                        break
                    if companies:
                        if _owner_source_is_us_for_zoominfo(item):
                            us_only = [
                                c
                                for c in companies
                                if _zoominfo_company_geo_bucket(c.get("attributes") or {}) == "us"
                            ]
                            if us_only:
                                logger.info(
                                    "phlydata/owners: item[%s] ZoomInfo person-led query=%r: narrowed %s -> %s US-region companies",
                                    idx,
                                    q,
                                    len(companies),
                                    len(us_only),
                                )
                                companies = us_only
                        logger.info(
                            "phlydata/owners: item[%s] company_search_by_person query=%r -> %s companies",
                            idx,
                            q,
                            len(companies),
                        )
                        all_companies.extend(companies)
                        break
                    all_companies.extend(companies)
                if zoominfo_error and not all_companies:
                    logger.info(
                        "phlydata/owners: item[%s] company_search_by_person all queries failed, total companies=0",
                        idx,
                    )
            else:
                # Company search (progressive fallback)
                for q in queries:
                    companies, zoominfo_error = zoominfo_search_companies(q, page_size=25)
                    if zoominfo_error:
                        logger.info("phlydata/owners: item[%s] company_search query=%r -> error=%s", idx, q, zoominfo_error)
                        break
                    if companies:
                        # When FAA registrant is US, drop obvious non-US same-name hits (e.g. Valiair UK vs Carlsbad US).
                        if _owner_source_is_us_for_zoominfo(item):
                            us_only = [
                                c
                                for c in companies
                                if _zoominfo_company_geo_bucket(c.get("attributes") or {}) == "us"
                            ]
                            if us_only:
                                logger.info(
                                    "phlydata/owners: item[%s] ZoomInfo query=%r: narrowed %s -> %s US-region companies",
                                    idx,
                                    q,
                                    len(companies),
                                    len(us_only),
                                )
                                companies = us_only
                        logger.info("phlydata/owners: item[%s] company_search query=%r -> %s companies", idx, q, len(companies))
                        all_companies.extend(companies)
                        break
                    all_companies.extend(companies)
                if zoominfo_error and not all_companies:
                    logger.info("phlydata/owners: item[%s] company_search all queries failed, total companies=0", idx)

            # Phone-first: if we have phone and any result matches it, return that immediately; else best by company/location/name
            best_record, result_type, matched, content_score = _pick_best_by_phone_first(all_companies, all_contacts, item)
            if content_score is None and matched.get("phone"):
                match_method = MATCH_METHOD_PHONE
                logger.info("phlydata/owners: item[%s] match_method=phone result_type=%s", idx, result_type)
            else:
                match_method = MATCH_METHOD_CONTENT_SCORE
                logger.info("phlydata/owners: item[%s] match_method=content_score result_type=%s content_score=%s matched=%s",
                            idx, result_type, content_score, matched)

            # Fallback: when content/word scoring is weak, use vector similarity + LLM to pick best from ZoomInfo candidates
            if (
                content_score is not None
                and content_score < ZOOMINFO_SCORE_THRESHOLD_FOR_LLM_FALLBACK
                and (all_companies or all_contacts)
            ):
                emb_svc, openai_key = get_embedding_service_only()
                if emb_svc and openai_key:
                    llm_best, llm_type, llm_matched = _pick_best_zoominfo_by_vector_and_llm(
                        all_companies, all_contacts, item, emb_svc, openai_key
                    )
                    if llm_best is not None:
                        best_record, result_type, matched = llm_best, llm_type, llm_matched
                        match_method = MATCH_METHOD_LLM_FALLBACK
                        logger.info("phlydata/owners: item[%s] match_method=llm_fallback result_type=%s", idx, result_type)
                else:
                    logger.info("phlydata/owners: item[%s] llm_fallback skipped (no embedding/openai)", idx)

            # Return only correct results: do not return name-only matches (no phone, location, or website) unless LLM confirms.
            # Many similar names exist (e.g. FAA "EVANS ROBERT W SR" vs ZoomInfo "Robert W Evans"); without phone,
            # location, or website we cannot be sure. So: if current match is content_score with only name overlap,
            # require LLM confirmation; if LLM says "none" or we have no LLM, return no ZoomInfo result for this item.
            name_only = (
                match_method == MATCH_METHOD_CONTENT_SCORE
                and best_record
                and not matched.get("phone")
                and not matched.get("location")
                and not matched.get("website")
            )
            if name_only and (all_companies or all_contacts):
                emb_svc, openai_key = get_embedding_service_only()
                if emb_svc and openai_key:
                    llm_best, llm_type, llm_matched = _pick_best_zoominfo_by_vector_and_llm(
                        all_companies, all_contacts, item, emb_svc, openai_key
                    )
                    if llm_best is not None:
                        best_record, result_type, matched = llm_best, llm_type, llm_matched
                        match_method = MATCH_METHOD_LLM_FALLBACK
                        logger.info("phlydata/owners: item[%s] name_only -> llm_fallback confirmed result_type=%s", idx, result_type)
                    else:
                        best_record = None
                        result_type = None
                        logger.info("phlydata/owners: item[%s] name_only -> llm said no match, returning no ZoomInfo result", idx)
                else:
                    best_record = None
                    result_type = None
                    logger.info("phlydata/owners: item[%s] name_only -> no LLM available, returning no ZoomInfo result (avoid wrong match)", idx)

            # Enrich: fetch full company/contact detail after we picked the best match.
            if best_record and result_type == "company" and best_record.get("id"):
                enriched_company = None
                try:
                    cid = best_record.get("id")
                    company_id_int = None
                    if cid is not None:
                        try:
                            company_id_int = int(cid)
                        except (TypeError, ValueError):
                            pass
                    if company_id_int is not None:
                        enriched_company, enrich_err = zoominfo_enrich_company(company_id=company_id_int, company_name=None)
                        if enriched_company:
                            best_record = enriched_company
                            logger.info("phlydata/owners: item[%s] company_enrich company_id=%s -> success (full detail)", idx, cid)
                        else:
                            logger.warning("phlydata/owners: item[%s] company_enrich company_id=%s -> failed (no detail) err=%s, trying by name", idx, cid, enrich_err or "no match")
                    if not enriched_company:
                        # Fallback: enrich by company name (sometimes works when enrich by ID returns NO_MATCH)
                        company_name_for_enrich = (best_record.get("attributes") or {}).get("name") or (best_record.get("attributes") or {}).get("companyName") or item.get("query_name") or item.get("company_name")
                        if company_name_for_enrich and isinstance(company_name_for_enrich, str) and company_name_for_enrich.strip():
                            enriched_company, enrich_err = zoominfo_enrich_company(company_id=None, company_name=company_name_for_enrich.strip())
                            if enriched_company:
                                best_record = enriched_company
                                logger.info("phlydata/owners: item[%s] company_enrich company_name=%r -> success (full detail)", idx, company_name_for_enrich.strip())
                            else:
                                logger.info("phlydata/owners: item[%s] company_enrich by name -> no_detail (using search result) err=%s", idx, enrich_err or "no match")
                    if not enriched_company and company_id_int is not None:
                        logger.info("phlydata/owners: item[%s] company_enrich company_id=%s -> no_detail (using search result)", idx, cid)
                except Exception as e:
                    logger.warning("phlydata/owners: item[%s] company_enrich failed: %s", idx, e)
            elif best_record and result_type == "contact":
                try:
                    pid = best_record.get("id")
                    person_id_int = None
                    if pid is not None:
                        try:
                            person_id_int = int(pid)
                        except (TypeError, ValueError):
                            person_id_int = None
                    enriched_contact, enrich_err = zoominfo_enrich_contact(
                        person_id=person_id_int,
                        full_name=item.get("company_name") if not person_id_int else None,
                    )
                    if enriched_contact:
                        best_record = enriched_contact
                        logger.info("phlydata/owners: item[%s] contact_enrich -> success (email)", idx)
                    else:
                        logger.warning("phlydata/owners: item[%s] contact_enrich failed: %s", idx, enrich_err or "no match")
                except Exception as e:
                    logger.warning("phlydata/owners: item[%s] contact_enrich failed: %s", idx, e)

            companies_out = [best_record] if (best_record and result_type == "company") else []
            contacts_out = [best_record] if (best_record and result_type == "contact") else []
            payload = {
                "query_name": item["query_name"],
                "source_platform": item["source_platform"],
                "field_name": item["field_name"],
                "companies": companies_out,
                "contacts": contacts_out,
                "best_result_type": result_type,
                "match_method": match_method,
                "matched": matched,
                "registrant_type": registrant_type,
                "context_sent": {
                    "company_name": item["company_name"],
                    "street": item.get("street"),
                    "city": item.get("city"),
                    "state": item.get("state"),
                    "zip_code": item.get("zip_code"),
                    "country": item.get("country"),
                    "address": item.get("address"),
                    "contact_name": item.get("contact_name"),
                    "broker_name": item.get("broker_name"),
                    "phone": item.get("phone"),
                    "website": item.get("website"),
                    "owner_url": item.get("owner_url"),
                    "inferred_company_from_url": item.get("inferred_company_from_url"),
                    "owner_url_alignment_source": url_alignment_source,
                    "owner_url_llm_alignment": owner_url_llm_alignment,
                },
            }
            if zoominfo_error:
                payload["zoominfo_error"] = zoominfo_error
            # Include when we have a match, or when we have an error that needs to be shown.
            if zoominfo_error or companies_out or contacts_out:
                zel = str(zoominfo_error or "").lower()
                # Only when the failure is clearly the Contacts Search endpoint (we use company search for persons now).
                payload["needs_contacts_admin_permission"] = bool(
                    zoominfo_error
                    and (
                        "contacts/search" in zel
                        or "gtm/data/v1/contacts/search" in zel
                    )
                )
                zoominfo_enrichment.append(payload)
            logger.info("phlydata/owners: item[%s] done -> match_method=%s best_result_type=%s companies=%s contacts=%s",
                        idx, match_method, result_type, len(companies_out), len(contacts_out))

        owner_lookup_sources: List[str] = []
        if owners_from_faa:
            owner_lookup_sources.append("faa")

        return {
            "aircraft": aircraft,
            "owners_from_listings": owners_from_listings,
            "owners_from_faa": owners_from_faa,
            "owners_from_aircraftpost": owners_from_aircraftpost,
            "owner_lookup_sources": owner_lookup_sources,
            # How ``faa_master`` was matched: n_number_serial (tail+serial), serial_model, serial_only, or null.
            "faa_master_match_kind": faa_master_match_kind,
            "zoominfo_enrichment": zoominfo_enrichment,
            "aircraftpost_fleet": aircraftpost_fleet,
            "zoominfo_contacts_access_denied": zoominfo_contacts_access_denied,
            "zoominfo_registrant_type_hint": zoominfo_registrant_type_hint,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/pdf")
def export_pdf():
    """Placeholder: export results to PDF for clients or pitch decks. Frontend can generate client-side or call this with payload."""
    return {"message": "Use frontend Download PDF for chat/comparison/valuation reports. Server-side PDF generation can be added here."}
