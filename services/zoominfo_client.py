"""
ZoomInfo company and contact search for PhlyData owner enrichment.

- Company Search: companyName only; disambiguation by location/phone/name is done in the owners endpoint.
- Contact Search: fullName; used for person names (seller_contact_name, broker, registrant when person-like).
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
from typing import List, Any, Optional, Tuple

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


def search_companies(company_name: Optional[str], page_size: int = 10) -> Tuple[List[Any], Optional[str]]:
    """
    Search ZoomInfo for companies. Payload contains only companyName.
    Returns (companies, error_reason). On 401, refreshes token and retries once.
    """
    company_name = _strip(company_name)
    if not company_name:
        return [], None
    token, base = _get_config()
    if not token:
        logger.warning("ZOOMINFO_ACCESS_TOKEN not set in backend .env; skipping ZoomInfo.")
        return [], "ZoomInfo token not configured. Set ZOOMINFO_ACCESS_TOKEN in backend/.env (same as phlydata-zoominfo/.env)."
    import requests
    url = f"{base}/data/v1/companies/search"
    params = {"page[number]": 1, "page[size]": min(100, max(1, page_size))}
    body = {"data": {"type": "CompanySearch", "attributes": {"companyName": company_name}}}
    for attempt in range(2):
        try:
            token, base = _get_config()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}
            r = requests.post(url, json=body, params=params, headers=headers, timeout=15)
            if r.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo company search 401 -> token refreshed, retrying")
                continue
            r.raise_for_status()
            data = r.json()
            return (data.get("data") or []), None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401 and attempt == 0 and _refresh_access_token():
                logger.info("ZoomInfo company search 401 -> token refreshed, retrying")
                continue
            logger.warning("ZoomInfo company search failed for %r: %s", company_name, e)
            return [], f"ZoomInfo error: {str(e)}"
        except Exception as e:
            logger.warning("ZoomInfo company search failed for %r: %s", company_name, e)
            return [], f"ZoomInfo error: {str(e)}"
    return [], "ZoomInfo error: retry after refresh failed"


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
    Contact Search disabled: ZoomInfo plan returns 403 for contacts/search.
    Returns empty list so matching uses company data only.
    """
    return [], None
