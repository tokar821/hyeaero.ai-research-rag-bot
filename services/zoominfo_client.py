"""
ZoomInfo company search for PhlyData owner enrichment.

Uses the same API as phlydata-zoominfo (GTM Data API company search).
Expects ZOOMINFO_ACCESS_TOKEN (and optionally ZOOMINFO_BASE_URL) in environment
(e.g. backend/.env). If token is missing, search returns empty results.
"""

import os
import logging
from typing import List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

JSON_API = "application/vnd.api+json"


def _get_config() -> Tuple[str, str]:
    token = (os.getenv("ZOOMINFO_ACCESS_TOKEN") or "").strip()
    base = (os.getenv("ZOOMINFO_BASE_URL") or "https://api.zoominfo.com/gtm").rstrip("/")
    return token, base


def search_companies(company_name: Optional[str], page_size: int = 10) -> List[Any]:
    """
    Search ZoomInfo for companies by name (partial match).
    Returns list of company items: { "id", "type", "attributes": { "name", ... } }.
    Returns [] if token is missing or request fails.
    """
    if not company_name or not company_name.strip():
        return []
    company_name = company_name.strip()
    token, base = _get_config()
    if not token:
        logger.debug("ZOOMINFO_ACCESS_TOKEN not set; skipping ZoomInfo company search")
        return []
    try:
        import requests
        url = f"{base}/data/v1/companies/search"
        params = {"page[number]": 1, "page[size]": min(100, max(1, page_size))}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": JSON_API, "Accept": JSON_API}
        body = {"data": {"type": "CompanySearch", "attributes": {"companyName": company_name}}}
        r = requests.post(url, json=body, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("data") or []
    except Exception as e:
        logger.warning("ZoomInfo company search failed for %r: %s", company_name, e)
        return []
