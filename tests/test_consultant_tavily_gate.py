import pytest

from rag.consultant_tavily_gate import (
    should_run_consultant_tavily_after_internal,
)


@pytest.mark.parametrize(
    "vec,sql,force,min_v,expect_run,reason_substr",
    [
        (0, False, False, 3, True, "sql"),
        (2, True, False, 3, True, "vector"),
        (3, True, False, 3, False, "sufficient"),
        (10, False, False, 3, True, "sql"),
        (2, True, False, 2, False, "sufficient"),
        (0, True, True, 3, True, "always"),
    ],
)
def test_tavily_after_internal(vec, sql, force, min_v, expect_run, reason_substr):
    run, reason = should_run_consultant_tavily_after_internal(
        vector_result_count=vec,
        sql_context_nonempty=sql,
        force_always=force,
        min_vector_results=min_v,
    )
    assert run is expect_run
    assert reason_substr in reason
