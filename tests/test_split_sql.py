from database.split_sql import sql_statements


def test_split_respects_semicolon_inside_single_quoted_string():
    sql = """
    SELECT 1;
    COMMENT ON TABLE t IS 'a; b';
    SELECT 2;
    """
    parts = sql_statements(sql)
    assert len(parts) == 3
    assert "COMMENT ON TABLE t IS 'a; b'" in parts[1]


def test_doubled_quote_inside_string():
    sql = "SELECT 'isn''t; split' AS x; SELECT 2;"
    parts = sql_statements(sql)
    assert len(parts) == 2
