"""
ZoomInfo company and contact search for PhlyData owner enrichment.

- Company Search: companyName and/or person-style filters (fullName, firstName/lastName) plus optional geo.
  ZoomInfo's published schema emphasizes companyName; person fields are used for "companies tied to this person"
  when contact search is unavailable (403). Disambiguation is done in the owners endpoint.
- Contact Search: fullName; optional when api:data:contact scope is enabled.
- On 401 Unauthorized, the client refreshes the access token using ZOOMINFO_REFRESH_TOKEN and retries.
- For deployment: do NOT put ZOOMINFO_ACCESS_TOKEN in .env (it changes on refresh). Set only
  ZOOMINFO_CLIENT_ID, ZOOMINFO_CLIENT_SECRET, ZOOMINFO_REFRESH_TOKEN. Optionally set ZOOMINFO_TOKEN_FILE
  to a writable path (e.g. /data/zoominfo_token) so the refreshed token is persisted across restarts.
"""

import base64
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JSON_API = "application/vnd.api+json"
ZOOMINFO_TOKEN_URL = "https://okta-login.zoominfo.com/oauth2/default/v1/token"


def _token_file_path() -> Optional[Path]:
    p = (os.getenv("ZOOMINFO_TOKEN_FILE") or "").strip()
    return Path(p).resolve() if p else None


def _read_token_from_file() -> bool:
    """Load ZOOMINFO_ACCESS_TOKEN (and optionally REFRESH_TOKEN) from ZOOMINFO_TOKEN_FILE. Sets os.environ. Returns True if access token was set."""
    path = _token_file_path()
    if not path or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("ZOOMINFO_ACCESS_TOKEN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["ZOOMINFO_ACCESS_TOKEN"] = val
            elif line.startswith("ZOOMINFO_REFRESH_TOKEN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["ZOOMINFO_REFRESH_TOKEN"] = val
        return bool((os.getenv("ZOOMINFO_ACCESS_TOKEN") or "").strip())
    except Exception as e:
        logger.debug("ZoomInfo token file read failed: %s", e)
        return False


def _write_token_to_file(access_token: str, refresh_token: Optional[str] = None) -> bool:
    """Persist access token (and optional refresh token) to ZOOMINFO_TOKEN_FILE. Returns True if written."""
    path = _token_file_path()
    if not path:
        return False
    try:
        lines = [f"ZOOMINFO_ACCESS_TOKEN={access_token}\n"]
        if refresh_token:
            lines.append(f"ZOOMINFO_REFRESH_TOKEN={refresh_token}\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")
        logger.debug("ZoomInfo token written to %s", path)
        return True
    except Exception as e:
        logger.warning("ZoomInfo token file write failed: %s", e)
        return False


def _get_config() -> Tuple[str, str]:
    token = (os.getenv("ZOOMINFO_ACCESS_TOKEN") or "").strip()
    base = (os.getenv("ZOOMINFO_BASE_URL") or "https://api.zoominfo.com/gtm").rstrip("/")
    # If no token in env, try token file (deployment: token not in .env)
    if not token and _token_file_path():
        _read_token_from_file()
        token = (os.getenv("ZOOMINFO_ACCESS_TOKEN") or "").strip()
    # If still no token, try refresh once (lazy refresh on first use)
    if not token and all(
        (os.getenv(k) or "").strip()
        for k in ("ZOOMINFO_CLIENT_ID", "ZOOMINFO_CLIENT_SECRET", "ZOOMINFO_REFRESH_TOKEN")
    ):
        if _refresh_access_token():
            token = (os.getenv("ZOOMINFO_ACCESS_TOKEN") or "").strip()
    return token, base


def _refresh_access_token() -> bool:
    """
    Exchange ZOOMINFO_REFRESH_TOKEN for a new access token. Sets os.environ and optionally updates backend/.env.
    Returns True if a new token was set, False otherwise (missing credentials or refresh failed).
    """
    client_id = (os.getenv("ZOOMINFO_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("ZOOMINFO_CLIENT_SECRET") or "").strip()
    refresh_token = (os.getenv("ZOOMINFO_REFRESH_TOKEN") or "").strip()
    if not client_id or not client_secret or not refresh_token:
        logger.warning("ZoomInfo refresh skipped: ZOOMINFO_CLIENT_ID, ZOOMINFO_CLIENT_SECRET, ZOOMINFO_REFRESH_TOKEN required in .env")
        return False
    try:
        import requests
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        r = requests.post(
            ZOOMINFO_TOKEN_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=30,
        )
        if r.status_code != 200:
            logger.warning("ZoomInfo token refresh failed: HTTP %s %s", r.status_code, r.text[:200])
            return False
        body = r.json()
        access_token = body.get("access_token")
        new_refresh = body.get("refresh_token")
        if not access_token:
            logger.warning("ZoomInfo token refresh: no access_token in response")
            return False
        os.environ["ZOOMINFO_ACCESS_TOKEN"] = access_token
        if new_refresh:
            os.environ["ZOOMINFO_REFRESH_TOKEN"] = new_refresh
        logger.info("ZoomInfo access token refreshed successfully (expires_in=%s)", body.get("expires_in"))
        # Persist: prefer ZOOMINFO_TOKEN_FILE (deployment); else backend/.env (local dev)
        if _write_token_to_file(access_token, new_refresh):
            pass  # token file used
        else:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                try:
                    lines = []
                    for line in open(env_path, "r", encoding="utf-8"):
                        if line.strip().startswith("ZOOMINFO_ACCESS_TOKEN="):
                            lines.append(f"ZOOMINFO_ACCESS_TOKEN={access_token}\n")
                        elif line.strip().startswith("ZOOMINFO_REFRESH_TOKEN=") and new_refresh:
                            lines.append(f"ZOOMINFO_REFRESH_TOKEN={new_refresh}\n")
                        else:
                            lines.append(line if line.endswith("\n") else line + "\n")
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    logger.debug("Could not write refreshed token to .env: %s", e)
        return True
    except Exception as e:
        logger.warning("ZoomInfo token refresh failed: %s", e)
        return False


def _strip(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    return t if t else None


def normalize_phone(phone: Optional[str]) -> str:
    """
    Normalize phone for comparison: digits only. No region or length assumptions.
    E.g. +44 (0) 20 7123 4567, 1-911-112-1312 -> 4402071234567, 19111121312.
    """
    if not phone or not isinstance(phone, str):
        return ""
    return re.sub(r"\D", "", phone)


def _normalized_digit_variants(digits: str) -> list:
    """Return digit string and variant without leading zeros (national format)."""
    if not digits:
        return []
    out = [digits]
    stripped = digits.lstrip("0") or "0"
    if stripped != digits:
        out.append(stripped)
    return out


def phones_match(our_phone: Optional[str], their_phone: Optional[str]) -> bool:
    """
    Format-agnostic phone match so we don't miss: works for any length (9, 10, 11, 12+) and any country.
    - Exact match after normalizing to digits.
    - Suffix match: one number is the other with country/area prefix (e.g. 9111121312 vs 19111121312, or 1234567890 vs 441234567890).
    - Leading-zero variant: 0123456789 vs 123456789 (national format).
    - Containment: shorter number (min 6 digits) appears inside the longer (avoids 12 matching 123456789012).
    """
    our = normalize_phone(our_phone)
    their = normalize_phone(their_phone)
    if not our or not their:
        return False
    # Exact match
    if our == their:
        return True
    # Suffix match (one is the other with prefix: country code, area, etc.)
    min_meaningful = 6  # avoid "1" or "12" matching end of a long number
    if len(our) >= min_meaningful and len(their) >= len(our) and their.endswith(our):
        return True
    if len(their) >= min_meaningful and len(our) >= len(their) and our.endswith(their):
        return True
    # Leading-zero variant (e.g. 0123456789 vs 123456789)
    for o in _normalized_digit_variants(our):
        for t in _normalized_digit_variants(their):
            if o == t:
                return True
            if len(o) >= min_meaningful and len(t) >= len(o) and t.endswith(o):
                return True
            if len(t) >= min_meaningful and len(o) >= len(t) and o.endswith(t):
                return True
    # Containment when both are long enough (same number with extra digits e.g. extension)
    if len(our) >= min_meaningful and len(their) >= min_meaningful:
        if our in their or their in our:
            return True
    return False


def _parse_person_name_for_company_search(full_name: str) -> Dict[str, str]:
    """
    Build person-name fragments for CompanySearch (same camelCase as ContactSearch: fullName, firstName, lastName).
    Handles FAA-style 'LAST, FIRST M' and space-separated names.
    """
    full_name = _strip(full_name)
    out: Dict[str, str] = {}
    if not full_name:
        return out
    out["fullName"] = full_name
    if "," in full_name:
        left, right = [p.strip() for p in full_name.split(",", 1)]
        if left and right:
            out["lastName"] = left
            out["firstName"] = right
    else:
        parts = full_name.split()
        if len(parts) >= 2:
            out["firstName"] = parts[0]
            out["lastName"] = parts[-1]
    return out


def _merge_company_search_geo(
    attrs: Dict[str, Any],
    *,
    state: Optional[str],
    country: Optional[str],
    zip_code: Optional[str],
    person_led: bool,
) -> Dict[str, Any]:
    merged = dict(attrs)
    st = _strip(state)
    ctry = _strip(country)
    z = _strip(zip_code)
    if st:
        merged["state"] = st
    if ctry:
        merged["country"] = ctry
    if z:
        merged["zipCode"] = z
    # Prefer person-or-HQ location when we have a person-led query and any geo (ZoomInfo CompanySearch).
    if person_led and (st or ctry or z):
        merged["locationSearchType"] = "PersonOrHQ"
    return merged


def search_companies(
    company_name: Optional[str] = None,
    page_size: int = 10,
    *,
    contact_full_name: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> Tuple[List[Any], Optional[str]]:
    """
    Search ZoomInfo for companies (CompanySearch).

    - Legacy: pass ``company_name`` only (uses attributes.companyName).
    - Person-led (no contact API): pass ``contact_full_name`` with optional ``state``/``country``/``zip_code``.
      Sends fullName and/or firstName+lastName parsed from the registrant string, plus geo and
      locationSearchType=PersonOrHQ when geo is present.

    Tries attribute variants in order until results are non-empty or a non-retryable error occurs.
    On 401, refreshes token and retries once per request.
    """
    company_name = _strip(company_name)
    contact_full_name = _strip(contact_full_name)

    if not company_name and not contact_full_name:
        return [], None

    token, base = _get_config()
    if not token:
        logger.warning("ZOOMINFO_ACCESS_TOKEN not set in backend .env; skipping ZoomInfo.")
        return [], "ZoomInfo token not configured. Set ZOOMINFO_ACCESS_TOKEN in backend/.env (same as phlydata-zoominfo/.env)."

    import requests

    # Ordered attribute bases: try structured name first, then fullName only, then companyName-as-string fallback.
    attr_variants: List[Dict[str, Any]] = []
    person_led = bool(contact_full_name)
    if contact_full_name:
        parsed = _parse_person_name_for_company_search(contact_full_name)
        fn = parsed.get("fullName")
        fn_part = parsed.get("firstName")
        ln_part = parsed.get("lastName")
        if fn_part and ln_part:
            attr_variants.append({"firstName": fn_part, "lastName": ln_part})
        if fn:
            attr_variants.append({"fullName": fn})
        # Last resort: some datasets match better as a free-text companyName query.
        attr_variants.append({"companyName": contact_full_name})
    elif company_name:
        attr_variants.append({"companyName": company_name})

    url = f"{base}/data/v1/companies/search"
    params = {"page[number]": 1, "page[size]": min(100, max(1, page_size))}

    last_error: Optional[str] = None
    for base_attrs in attr_variants:
        attributes = _merge_company_search_geo(
            base_attrs,
            state=state,
            country=country,
            zip_code=zip_code,
            person_led=person_led,
        )
        body = {"data": {"type": "CompanySearch", "attributes": attributes}}
        for attempt in range(2):
            try:
                token, base = _get_config()
                headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}
                r = requests.post(url, json=body, params=params, headers=headers, timeout=15)
                if r.status_code == 401 and attempt == 0 and _refresh_access_token():
                    logger.info("ZoomInfo company search 401 -> token refreshed, retrying")
                    continue
                if r.status_code == 400:
                    try:
                        err_body = r.json()
                    except Exception:
                        err_body = r.text
                    logger.info(
                        "ZoomInfo company search 400 for attributes keys=%s -> trying next variant: %s",
                        list(attributes.keys()),
                        err_body,
                    )
                    last_error = f"ZoomInfo error: 400 Client Error {err_body}"
                    break  # next variant
                r.raise_for_status()
                data = r.json()
                rows = data.get("data") or []
                if rows:
                    return rows, None
                # Empty OK: try next variant for person-led search
                last_error = None
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 401 and attempt == 0 and _refresh_access_token():
                    logger.info("ZoomInfo company search 401 -> token refreshed, retrying")
                    continue
                logger.warning("ZoomInfo company search failed attributes=%s: %s", list(attributes.keys()), e)
                return [], f"ZoomInfo error: {str(e)}"
            except Exception as e:
                logger.warning("ZoomInfo company search failed attributes=%s: %s", list(attributes.keys()), e)
                return [], f"ZoomInfo error: {str(e)}"

    if last_error:
        return [], last_error
    return [], None


# Output fields for company enrich. Only request fields allowed by your ZoomInfo plan.
# Do NOT include addressLine1 - many plans return 400 "Invalid field 'addressline1'". Use "street" instead.
# Align with phlydata-zoominfo/zoominfo_enrich_company.py DEFAULT_OUTPUT_FIELDS (which works).
DEFAULT_COMPANY_ENRICH_FIELDS = [
    "id", "ticker", "name", "website", "socialMediaUrls",
    "phone", "street", "city", "state", "zipCode", "country",
    "revenue", "employeeRange", "employeeCount", "industries", "foundedYear", "companyStatus",
    "certified", "continent", "locationCount", "numberOfContactsInZoomInfo", "parentId", "parentName",
]


def enrich_company(
    company_id: Optional[int] = None,
    company_name: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get detailed company data from ZoomInfo by company ID or company name.
    Returns (company_detail, error_reason). On 401, refreshes token and retries once.
    """
    if company_id is None and not _strip(company_name):
        return None, "Provide company_id or company_name."
    token, base = _get_config()
    if not token:
        return None, "ZoomInfo token not configured. Set ZOOMINFO_ACCESS_TOKEN in backend/.env."
    fields = output_fields or DEFAULT_COMPANY_ENRICH_FIELDS
    match_input = {}
    if company_id is not None:
        match_input["companyId"] = int(company_id)
    if company_name:
        match_input["companyName"] = _strip(company_name)
    if not match_input:
        return None, "Provide company_id or company_name."
    import requests
    url = f"{base}/data/v1/companies/enrich"
    body = {
        "data": {
            "type": "CompanyEnrich",
            "attributes": {
                "matchCompanyInput": [match_input],
                "outputFields": fields,
            },
        },
    }
    for attempt in range(2):
        try:
            token, base = _get_config()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}
            r = requests.post(url, json=body, headers=headers, timeout=15)
            if r.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo enrich 401 -> token refreshed, retrying")
                continue
            if r.status_code == 403:
                try:
                    err = r.json()
                except Exception:
                    err = r.text
                logger.warning("ZoomInfo company enrich 403 Forbidden (detail API not in plan or no access): %s", err)
                return None, f"ZoomInfo enrich not allowed or not in plan: {err}"
            if r.status_code == 400:
                try:
                    err = r.json()
                except Exception:
                    err = r.text
                return None, f"ZoomInfo enrich 400 Bad Request (check outputFields or request format): {err}"
            r.raise_for_status()
            data = r.json()
            records = data.get("data") or []
            if not records:
                return None, None
            first = records[0]
            if first.get("meta", {}).get("matchStatus") == "NO_MATCH":
                return None, None
            return first, None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo enrich 401 -> token refreshed, retrying")
                continue
            logger.warning("ZoomInfo company enrich failed: %s", e)
            return None, f"ZoomInfo error: {str(e)}"
        except Exception as e:
            logger.warning("ZoomInfo company enrich failed: %s", e)
            return None, f"ZoomInfo error: {str(e)}"
    return None, "ZoomInfo error: retry after refresh failed"


def search_contacts(full_name: Optional[str], page_size: int = 25) -> Tuple[List[Any], Optional[str]]:
    """
    Search ZoomInfo for contacts (people) by full name.

    Note:
    - ZoomInfo's contacts search endpoint does not directly return emails/phone numbers.
      It returns contact ids and hints; use `enrich_contact()` to fetch email/phone/address.
    - If your ZoomInfo plan does not include contacts/search, ZoomInfo returns 403 and we
      return ([], <error>).
    """
    full_name = _strip(full_name)
    if not full_name:
        return [], None
    token, base = _get_config()
    if not token:
        logger.warning("ZOOMINFO_ACCESS_TOKEN not set in backend .env; skipping ZoomInfo contacts search.")
        return [], "ZoomInfo token not configured."

    import requests

    url = f"{base}/data/v1/contacts/search"
    params = {"page[number]": 1, "page[size]": min(100, max(1, page_size))}
    body = {"data": {"type": "ContactSearch", "attributes": {"fullName": full_name}}}

    for attempt in range(2):
        token, base = _get_config()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}
        try:
            r = requests.post(url, json=body, params=params, headers=headers, timeout=15)
            if r.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo contacts search 401 -> token refreshed, retrying")
                continue
            if r.status_code == 403:
                try:
                    err = r.json()
                except Exception:
                    err = r.text
                return [], f"ZoomInfo contacts search forbidden (plan/access missing): {err}"
            r.raise_for_status()
            data = r.json()
            return (data.get("data") or []), None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo contacts search 401 -> token refreshed, retrying")
                continue
            logger.warning("ZoomInfo contacts search failed for %r: %s", full_name, e)
            return [], f"ZoomInfo error: {str(e)}"
        except Exception as e:
            logger.warning("ZoomInfo contacts search failed for %r: %s", full_name, e)
            return [], f"ZoomInfo error: {str(e)}"
    return [], "ZoomInfo error: retry after refresh failed"


def enrich_contact(person_id: Optional[int] = None, full_name: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Enrich one ZoomInfo contact (person) to fetch details like email.

    Uses `/data/v1/contacts/enrich`.

    Provide either `person_id` (preferred) or `full_name` (fallback matching input).
    """
    if person_id is None and not _strip(full_name):
        return None, "Provide person_id or full_name."

    token, base = _get_config()
    if not token:
        return None, "ZoomInfo token not configured."

    import requests

    # Select fields that the frontend can render (email + basic identity/location).
    output_fields = [
        "id",
        "fullName",
        "firstName",
        "lastName",
        "email",
        "companyName",
        "phone",
        "directPhone",
        "mobilePhone",
        "workPhone",
        "street",
        "city",
        "state",
        "zipCode",
        "country",
    ]
    required_fields = ["id", "companyName"]

    match_input: dict = {}
    if person_id is not None:
        match_input["personId"] = int(person_id)
    else:
        # Best-effort: split full name into first/last when possible.
        parts = str(full_name).strip().split()
        if len(parts) >= 2:
            match_input["firstName"] = parts[0]
            match_input["lastName"] = " ".join(parts[1:])
        else:
            # If only one token, use it as firstName and leave lastName empty.
            match_input["firstName"] = parts[0]

    url = f"{base}/data/v1/contacts/enrich"
    body = {
        "data": {
            "type": "ContactEnrich",
            "attributes": {
                "matchPersonInput": [match_input],
                "outputFields": output_fields,
                "requiredFields": required_fields,
            },
        }
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}

    for attempt in range(2):
        token, base = _get_config()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}
        try:
            r = requests.post(url, json=body, headers=headers, timeout=15)
            if r.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo contacts enrich 401 -> token refreshed, retrying")
                continue
            if r.status_code == 403:
                try:
                    err = r.json()
                except Exception:
                    err = r.text
                return None, f"ZoomInfo contacts enrich forbidden (plan/access missing): {err}"
            if r.status_code == 400:
                try:
                    err = r.json()
                except Exception:
                    err = r.text
                return None, f"ZoomInfo contacts enrich 400 Bad Request: {err}"
            r.raise_for_status()
            data = r.json()
            records = data.get("data") or []
            if not records:
                return None, None
            first = records[0]
            meta = first.get("meta") or {}
            if meta.get("matchStatus") == "NO_MATCH":
                return None, None
            attrs = first.get("attributes") or {}
            # Normalize street field into `address` so frontend can display it consistently.
            if not attrs.get("address") and attrs.get("street"):
                attrs["address"] = attrs.get("street")
            return first, None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo contacts enrich 401 -> token refreshed, retrying")
                continue
            return None, f"ZoomInfo error: {str(e)}"
        except Exception as e:
            return None, f"ZoomInfo error: {str(e)}"
    return None, "ZoomInfo error: retry after refresh failed"
