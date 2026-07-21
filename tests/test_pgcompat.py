from dashboard.pgcompat import translate_sql, HybridRow

def test_basic_placeholder():
    assert translate_sql("SELECT * FROM t WHERE id=?") == "SELECT * FROM t WHERE id=%s"

def test_multiple_placeholders():
    assert translate_sql("INSERT INTO t (a,b) VALUES (?,?)") == "INSERT INTO t (a,b) VALUES (%s,%s)"

def test_question_mark_in_string_literal_is_left_alone():
    assert translate_sql("SELECT '?' , x FROM t WHERE y=?") == "SELECT '?' , x FROM t WHERE y=%s"

def test_literal_percent_is_escaped():
    assert translate_sql("SELECT * FROM t WHERE name LIKE '%foo%'") == "SELECT * FROM t WHERE name LIKE '%%foo%%'"

def test_hybrid_row_index_and_key():
    r = HybridRow(["id", "v"], (7, "hi"))
    assert r[0] == 7
    assert r["v"] == "hi"
    assert r["id"] == 7

def test_apostrophe_in_line_comment_does_not_break_later_placeholder():
    sql = "SELECT x -- user's note\nFROM t WHERE id=?"
    assert translate_sql(sql) == "SELECT x -- user's note\nFROM t WHERE id=%s"

def test_apostrophe_in_block_comment_ignored():
    sql = "SELECT x /* it's fine */ FROM t WHERE id=?"
    assert translate_sql(sql) == "SELECT x /* it's fine */ FROM t WHERE id=%s"

def test_percent_in_line_comment_still_escaped_but_harmless():
    # % is escaped globally; a placeholder after a comment must still convert
    sql = "UPDATE t SET a=? -- 50% off\nWHERE id=?"
    assert translate_sql(sql) == "UPDATE t SET a=%s -- 50%% off\nWHERE id=%s"

def test_hybrid_row_duplicate_columns_first_wins():
    r = HybridRow(["id", "v", "id"], (1, "x", 2))
    assert r["id"] == 1
    assert r[2] == 2
