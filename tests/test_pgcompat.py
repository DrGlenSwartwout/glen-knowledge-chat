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
