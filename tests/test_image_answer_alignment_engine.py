"""Tests for image + answer alignment plan (grouping, confidence, premium facet rate)."""

from __future__ import annotations

import os

import pytest

from services import image_answer_alignment_engine as iae


def test_groups_by_aircraft_and_caps_three(monkeypatch):
    monkeypatch.delenv("CONSULTANT_IMAGE_ANSWER_ALIGNMENT", raising=False)
    phly = [
        {"manufacturer": "Bombardier", "model": "Challenger 350"},
        {"manufacturer": "Dassault", "model": "Falcon 2000"},
    ]
    imgs = [
        {
            "url": "https://example.com/a.jpg",
            "description": "Challenger 350 cabin interior seating",
            "page_url": "",
        },
        {
            "url": "https://example.com/b.jpg",
            "description": "Challenger 350 galley",
            "page_url": "",
        },
        {
            "url": "https://example.com/c.jpg",
            "description": "Falcon 2000 cabin layout",
            "page_url": "",
        },
        {
            "url": "https://example.com/d.jpg",
            "description": "Falcon 2000 windows",
            "page_url": "",
        },
    ]
    plan = iae.build_image_answer_alignment_plan(
        user_query="compare cabins",
        aircraft_images=imgs,
        phly_rows=phly,
        gallery_meta={"image_query_engine": {"confidence": 0.9}},
        marketing_type_hint=None,
    )
    assert len(plan["image_groups"]) <= 3
    models = {g["model"] for g in plan["image_groups"]}
    assert "Challenger 350" in models or any("Challenger" in m for m in models)
    assert "Falcon 2000" in models or any("Falcon" in m for m in models)
    assert plan["alignment_ok"] is True
    assert "1:1" in plan["llm_directives"] or "1:1" in plan["llm_directives"].replace(" ", "")


def test_low_confidence_marks_not_aligned():
    phly = [{"manufacturer": "Gulfstream", "model": "G650"}]
    imgs = [
        {
            "url": "https://x.com/1.jpg",
            "description": "Gulfstream G650 interior",
            "page_url": "",
        }
    ]
    plan = iae.build_image_answer_alignment_plan(
        user_query="show cabin",
        aircraft_images=imgs,
        phly_rows=phly,
        gallery_meta={"image_query_engine": {"confidence": 0.55}},
        marketing_type_hint=None,
    )
    assert plan["alignment_ok"] is False
    assert plan["weak_images_message"]
    assert "reliable interior" in plan["weak_images_message"]


def test_section_mismatch_when_cockpit_requested():
    phly = [{"manufacturer": "Cessna", "model": "Citation CJ3"}]
    imgs = [
        {
            "url": "https://x.com/cabin.jpg",
            "description": "Citation CJ3 cabin seating layout",
            "page_url": "",
        },
        {
            "url": "https://x.com/cabin2.jpg",
            "description": "CJ3 interior salon",
            "page_url": "",
        },
    ]
    premium = {
        "type": "GENERAL",
        "tail_number": "",
        "image_type": "cockpit",
        "image_facets": [],
        "validate_images": False,
    }
    plan = iae.build_image_answer_alignment_plan(
        user_query="cockpit photos",
        aircraft_images=imgs,
        phly_rows=phly,
        gallery_meta={
            "consultant_premium_intent": premium,
            "image_query_engine": {"confidence": 0.95},
        },
        marketing_type_hint=None,
    )
    assert plan["section_alignment_rate"] < 0.5
    assert plan["alignment_ok"] is False


def test_consultant_toggle_off(monkeypatch):
    monkeypatch.setenv("CONSULTANT_IMAGE_ANSWER_ALIGNMENT", "0")
    assert iae.consultant_image_answer_alignment_enabled() is False


def test_marketing_hint_leads_candidates_over_phly():
    phly = [{"manufacturer": "Bombardier", "model": "Challenger 300"}]
    plan = iae.build_image_answer_alignment_plan(
        user_query="G650 interior",
        aircraft_images=[
            {"url": "https://x.com/1.jpg", "description": "Gulfstream G650 cabin", "page_url": ""},
        ],
        phly_rows=phly,
        gallery_meta={},
        marketing_type_hint="Gulfstream G650",
    )
    cand0 = (plan.get("aircraft_candidates") or [{}])[0].get("model", "")
    assert "650" in cand0 or "Gulfstream" in cand0


def test_format_layered_block_truncates():
    plan = {
        "llm_directives": "x" * 100,
        "image_groups": [
            {
                "model": "Test",
                "images": [{"url": "u", "description": "d"}],
            }
        ],
    }
    s = iae.format_alignment_block_for_layered_context(plan, max_chars=50)
    assert len(s) <= 50
    assert s.endswith("...")
