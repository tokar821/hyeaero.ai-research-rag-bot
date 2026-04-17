from __future__ import annotations

import asyncio

import pytest

from services import premium_image_pipeline as pip
from services.premium_image_pipeline import ImageHit, ImageIntent


def test_premium_pipeline_tail_accepts_high_confidence(monkeypatch):
    async def fake_intent(_: str) -> ImageIntent:
        return ImageIntent(tail="N807JS", model=None, view="exterior")

    monkeypatch.setattr(pip, "extract_image_intent", fake_intent)

    def fake_search(_q: str):
        return [
            ImageHit(
                title="N807JS ramp — JetPhotos",
                imageUrl="https://cdn.jetphotos.com/photo/807js.jpg",
                source="JetPhotos",
            )
        ]

    monkeypatch.setattr(pip, "search_images", fake_search)

    out = asyncio.run(pip.get_premium_aircraft_images("show N807JS exterior"))
    assert out["success"] is True
    assert len(out["images"]) == 1
    assert out["images"][0]["score"] >= 0.7


def test_premium_pipeline_rejects_tail_mismatch(monkeypatch):
    async def fake_intent(_: str) -> ImageIntent:
        return ImageIntent(tail="N807JS", model=None, view=None)

    monkeypatch.setattr(pip, "extract_image_intent", fake_intent)

    def fake_search(_q: str):
        return [
            ImageHit(
                title="N807JT parked",
                imageUrl="https://cdn.jetphotos.com/photo/807jt.jpg",
                source="JetPhotos",
            )
        ]

    monkeypatch.setattr(pip, "search_images", fake_search)

    out = asyncio.run(pip.get_premium_aircraft_images("show N807JS"))
    assert out["success"] is False
