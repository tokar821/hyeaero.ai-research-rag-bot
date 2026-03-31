from unittest.mock import MagicMock

from rag.embeddings_metadata_cleanup import delete_embeddings_metadata_for_entity_types


def test_delete_builds_any_query():
    db = MagicMock()
    db.execute_update.return_value = 3
    n = delete_embeddings_metadata_for_entity_types(
        db, ["aircraft", "aircraft_listing"], embedding_model="text-embedding-3-large"
    )
    assert n == 3
    assert db.execute_update.called
    sql = db.execute_update.call_args[0][0]
    assert "ANY(%s::text[])" in sql
    assert "embedding_model" in sql
