"""
Tests: ZoomInfo company search using contact_full_name (person-led CompanySearch).

Mocks HTTP — no token or live ZoomInfo required. Run from repo backend/:

    python -m unittest tests.test_zoominfo_company_search_by_person -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Backend root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.zoominfo_client import search_companies  # noqa: E402


class TestZoominfoCompanySearchByPerson(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["ZOOMINFO_ACCESS_TOKEN"] = "test-access-token"
        os.environ["ZOOMINFO_BASE_URL"] = "https://api.zoominfo.com/gtm"

    def tearDown(self) -> None:
        for k in ("ZOOMINFO_ACCESS_TOKEN", "ZOOMINFO_BASE_URL"):
            os.environ.pop(k, None)

    def _mock_response(self, status: int, payload: dict | None = None, text: str = "") -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.text = text
        if payload is not None:
            r.json.return_value = payload
        r.raise_for_status.side_effect = lambda: None
        return r

    def test_person_name_first_request_uses_company_search_with_first_last(self) -> None:
        """Two-word name -> first variant is firstName + lastName on CompanySearch."""
        bodies: list[dict] = []

        def fake_post(url, json=None, **kwargs):
            bodies.append(json)
            return self._mock_response(200, {"data": [{"type": "companies", "id": "1"}]})

        with patch("requests.post", side_effect=fake_post):
            rows, err = search_companies(
                None, page_size=10, contact_full_name="Sarah Jason"
            )

        self.assertIsNone(err)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0]["data"]["type"], "CompanySearch")
        attrs = bodies[0]["data"]["attributes"]
        self.assertEqual(attrs.get("firstName"), "Sarah")
        self.assertEqual(attrs.get("lastName"), "Jason")

    def test_400_on_first_variant_retries_full_name(self) -> None:
        """400 on structured name -> client tries next variant (fullName)."""
        bodies: list[dict] = []
        calls = {"n": 0}

        def fake_post(url, json=None, **kwargs):
            bodies.append(json)
            calls["n"] += 1
            if calls["n"] == 1:
                r = self._mock_response(400, {"errors": [{"title": "bad"}]})
                return r
            return self._mock_response(200, {"data": [{"type": "companies", "id": "2"}]})

        with patch("requests.post", side_effect=fake_post):
            rows, err = search_companies(
                None, page_size=10, contact_full_name="Sarah Jason"
            )

        self.assertIsNone(err)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[1]["data"]["attributes"].get("fullName"), "Sarah Jason")

    def test_single_token_uses_full_name_then_company_name_variants(self) -> None:
        """One-word name -> no firstName/lastName pair; tries fullName then companyName."""
        bodies: list[dict] = []
        calls = {"n": 0}

        def fake_post(url, json=None, **kwargs):
            bodies.append(json)
            calls["n"] += 1
            if calls["n"] == 1:
                return self._mock_response(200, {"data": []})
            return self._mock_response(200, {"data": [{"type": "companies", "id": "3"}]})

        with patch("requests.post", side_effect=fake_post):
            rows, err = search_companies(
                None, page_size=10, contact_full_name="Jason"
            )

        self.assertIsNone(err)
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(len(bodies), 2)
        self.assertEqual(bodies[0]["data"]["attributes"].get("fullName"), "Jason")
        self.assertEqual(bodies[1]["data"]["attributes"].get("companyName"), "Jason")

    def test_person_plus_geo_sets_location_search_type(self) -> None:
        bodies: list[dict] = []

        def fake_post(url, json=None, **kwargs):
            bodies.append(json)
            return self._mock_response(200, {"data": [{"type": "companies", "id": "4"}]})

        with patch("requests.post", side_effect=fake_post):
            rows, err = search_companies(
                None,
                page_size=10,
                contact_full_name="Jane Doe",
                state="CA",
                country="United States",
                zip_code="92008",
            )

        self.assertIsNone(err)
        attrs = bodies[0]["data"]["attributes"]
        self.assertEqual(attrs.get("state"), "CA")
        self.assertEqual(attrs.get("country"), "United States")
        self.assertEqual(attrs.get("zipCode"), "92008")
        self.assertEqual(attrs.get("locationSearchType"), "PersonOrHQ")


if __name__ == "__main__":
    unittest.main()
