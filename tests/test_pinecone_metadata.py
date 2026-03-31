from rag.pinecone_metadata import (
    build_vector_metadata,
    infer_pinecone_entity_filter,
    normalize_serial_for_metadata,
    normalize_tail_for_metadata,
    sanitize_pinecone_metadata_dict,
    normalize_aircraft_model_metadata,
)


def test_normalize_serial_strips_prefixes():
    assert normalize_serial_for_metadata("SN-6201") == "6201"
    assert normalize_serial_for_metadata("sn6201") == "6201"
    assert normalize_serial_for_metadata("Serial 6201") == "6201"
    assert normalize_serial_for_metadata("S/N: 12345") == "12345"


def test_normalize_tail_uppercase():
    assert normalize_tail_for_metadata("n123ab") == "N123AB"
    assert normalize_tail_for_metadata("N 123 AB") == "N123AB"


def test_aircraft_model_uppercase():
    assert normalize_aircraft_model_metadata("Citation X") == "CITATION X"


def test_sanitize_small_payload():
    m = sanitize_pinecone_metadata_dict(
        {
            "entity_type": "aircraft_listing",
            "entity_id": "550e8400-e29b-41d4-a716-446655440000",
            "aircraft_model": "G650",
            "manufacturer": "Gulfstream",
            "serial_number": "6201",
            "tail_number": "N1HQ",
            "year": 2016,
            "source_table": "aircraft_listings",
            "chunk_index": 0,
            "total_chunks": 1,
            "chunking_strategy": "entity_single",
        }
    )
    raw = str(m)
    assert len(raw) < 1024
    assert m["tail_number"] == "N1HQ"
    assert m["year"] == 2016


def test_build_vector_metadata_aircraft():
    r = {
        "id": "u1",
        "manufacturer": "Gulfstream",
        "model": "G-650",
        "serial_number": "SN-6201",
        "registration_number": "n999ab",
        "manufacturer_year": 2014,
    }
    meta = build_vector_metadata("aircraft", r)
    assert meta["entity_type"] == "aircraft"
    assert meta["entity_id"] == "u1"
    assert meta["serial_number"] == "6201"
    assert meta["tail_number"] == "N999AB"
    assert meta["aircraft_model"] == "G-650"
    assert meta["source_table"] == "aircraft"


def test_build_vector_metadata_faa():
    r = {
        "id": "f1",
        "n_number": "n123ab",
        "serial_number": "ABC-99",
        "year_mfr": 2010,
    }
    meta = build_vector_metadata("faa_registration", r)
    assert meta["tail_number"] == "N123AB"
    assert meta["year"] == 2010


def test_infer_filter_listing():
    f = infer_pinecone_entity_filter("Citation X listing for sale in Florida")
    assert f is not None
    assert "aircraft_listing" in str(f)


def test_infer_filter_specs():
    f = infer_pinecone_entity_filter("G650 specs range cruise")
    assert f is not None
    assert "aviacost_aircraft_detail" in str(f)


def test_infer_filter_faa():
    f = infer_pinecone_entity_filter("FAA registration N123AB registrant")
    assert f is not None
    assert "faa_registration" in str(f)
