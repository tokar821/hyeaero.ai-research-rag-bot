"""ChunkingService: entity-single vs document multi-chunk."""

from rag.chunking_service import (
    ChunkingService,
    STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES,
)


def test_structured_types_include_checklist_entities():
    assert "aircraft" in STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES
    assert "aircraft_listing" in STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES
    assert "faa_registration" in STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES
    assert "document" not in STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES


def test_aircraft_single_chunk_even_when_long():
    svc = ChunkingService(chunk_size=100, chunk_overlap=20)
    text = "Aircraft: Gulfstream G650\n" + ("x" * 500)
    chunks = svc.chunk_for_entity("aircraft", text, {"entity_type": "aircraft"})
    assert len(chunks) == 1
    assert chunks[0]["text"] == text.strip()
    assert chunks[0]["metadata"]["total_chunks"] == 1
    assert chunks[0]["metadata"]["chunking_strategy"] == "entity_single"


def test_listing_single_chunk():
    svc = ChunkingService(chunk_size=50, chunk_overlap=10)
    text = "Listing\n" + ("y" * 200)
    chunks = svc.chunk_for_entity("aircraft_listing", text, {})
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunking_strategy"] == "entity_single"


def test_faa_single_chunk():
    svc = ChunkingService(chunk_size=40, chunk_overlap=5)
    text = "FAA Registration: N12345\n" + ("z" * 300)
    chunks = svc.chunk_for_entity("faa_registration", text, {})
    assert len(chunks) == 1


def test_document_multi_chunk_when_long():
    svc = ChunkingService(chunk_size=100, chunk_overlap=20)
    text = "para. " * 80
    chunks = svc.chunk_for_entity("document", text, {})
    assert len(chunks) > 1
    for c in chunks:
        assert c["metadata"]["chunking_strategy"] == "character_window"


def test_chunk_text_backward_compat_short_is_single_with_strategy():
    svc = ChunkingService(chunk_size=1000, chunk_overlap=200)
    chunks = svc.chunk_text("short", {})
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunking_strategy"] == "character_window"


def test_chunk_text_backward_compat_long_splits():
    svc = ChunkingService(chunk_size=80, chunk_overlap=10)
    text = "Hello world. " * 30
    chunks = svc.chunk_text(text, {})
    assert len(chunks) >= 2
