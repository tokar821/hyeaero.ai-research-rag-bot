from unittest.mock import MagicMock

from vector_store.pinecone_client import PineconeClient


def test_delete_by_metadata_filter_calls_index():
    pc = PineconeClient(api_key="k", index_name="idx", dimension=1024)
    pc.index = MagicMock()
    ok = pc.delete_by_metadata_filter({"entity_type": {"$in": ["aircraft"]}}, namespace=None)
    assert ok is True
    pc.index.delete.assert_called_once()
    call_kw = pc.index.delete.call_args.kwargs
    assert call_kw["filter"] == {"entity_type": {"$in": ["aircraft"]}}
    assert call_kw.get("namespace") is None


def test_delete_all_in_namespace():
    pc = PineconeClient(api_key="k", index_name="idx", dimension=1024)
    pc.index = MagicMock()
    ok = pc.delete_all_in_namespace("phlydata_aircraft")
    assert ok is True
    pc.index.delete.assert_called_once_with(
        delete_all=True, namespace="phlydata_aircraft"
    )


def test_delete_all_namespace_not_found_is_ok():
    """Pinecone returns 404 when namespace was never created — treat as success."""
    pc = PineconeClient(api_key="k", index_name="idx", dimension=1024)
    pc.index = MagicMock()

    def boom(**_kwargs):
        raise Exception(
            '(404) Reason: Not Found\nHTTP response body: {"code":5,"message":"Namespace not found"}'
        )

    pc.index.delete = boom
    assert pc.delete_all_in_namespace("phlydata_aircraft") is True


def test_structured_constants_exclude_documents_include_aircraftpost():
    from rag.structured_reembed_constants import (
        DOCUMENT_ENTITY_TYPE,
        STRUCTURED_AVIATION_ALL_ENTITY_TYPES,
        STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES,
    )

    assert DOCUMENT_ENTITY_TYPE not in STRUCTURED_AVIATION_ALL_ENTITY_TYPES
    assert "aircraftpost_fleet_aircraft" in STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES
