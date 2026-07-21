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

def test_hybrid_row_case_insensitive_name_lookup():
    r = HybridRow(["Total", "V"], (5, "x"))
    assert r["total"] == 5
    assert r["Total"] == 5
    assert r["v"] == "x"
    assert r["V"] == "x"
    assert r[0] == 5

def test_hybrid_row_first_wins_still_holds_case_insensitively():
    r = HybridRow(["id", "V", "ID"], (1, "x", 2))
    assert r["id"] == 1
    assert r["ID"] == 1


# ---------------------------------------------------------------------------
# DDL-idiom auto-translation: AUTOINCREMENT -> IDENTITY, datetime('now') -> now()::text
# ---------------------------------------------------------------------------

def test_autoincrement_translated_in_create_table():
    sql = "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)"
    assert translate_sql(sql) == "CREATE TABLE t (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, v TEXT)"

def test_autoincrement_translated_lowercase():
    sql = "create table t (id integer primary key autoincrement, v text)"
    assert translate_sql(sql) == "create table t (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, v text)"

def test_autoincrement_translated_with_extra_whitespace():
    sql = "CREATE TABLE t (id INTEGER   PRIMARY KEY    AUTOINCREMENT, v TEXT)"
    assert translate_sql(sql) == "CREATE TABLE t (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, v TEXT)"

def test_datetime_now_translated_in_default_clause():
    sql = "CREATE TABLE t (id INTEGER, created_at TEXT DEFAULT (datetime('now')))"
    assert translate_sql(sql) == "CREATE TABLE t (id INTEGER, created_at TEXT DEFAULT (now()::text))"

def test_datetime_now_translated_in_where_clause():
    sql = "SELECT * FROM t WHERE created_at > datetime('now')"
    assert translate_sql(sql) == "SELECT * FROM t WHERE created_at > now()::text"

def test_plain_select_only_placeholder_converted():
    assert translate_sql("SELECT a FROM t WHERE id=?") == "SELECT a FROM t WHERE id=%s"

def test_placeholder_autoincrement_and_datetime_now_together():
    sql = ("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, "
           "created_at TEXT DEFAULT (datetime('now')), v TEXT); "
           "INSERT INTO t (v) VALUES (?)")
    expected = ("CREATE TABLE t (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "created_at TEXT DEFAULT (now()::text), v TEXT); "
                "INSERT INTO t (v) VALUES (%s)")
    assert translate_sql(sql) == expected

def test_ddl_idioms_idempotent_on_already_postgres_sql():
    sql = ("CREATE TABLE t (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
           "created_at TEXT DEFAULT (now()::text))")
    assert translate_sql(sql) == sql

def test_ddl_idioms_idempotent_when_translated_twice():
    sql = "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT (datetime('now')))"
    once = translate_sql(sql)
    twice = translate_sql(once)
    assert once == twice

def test_autoincrement_as_string_literal_data_is_also_translated_known_risk():
    # KNOWN, ACCEPTED RISK (documented in report): the DDL-idiom regexes run on
    # the raw SQL before the quote-aware placeholder pass, so a string literal
    # that happens to contain the exact phrase "INTEGER PRIMARY KEY AUTOINCREMENT"
    # as DATA (not schema) will also be rewritten. This is a mechanical, low-blast
    # -radius idiom (DDL-only in practice) and the residual risk of a literal data
    # value colliding with this exact DDL phrase is judged acceptable for v1.
    sql = "INSERT INTO t (note) VALUES ('id INTEGER PRIMARY KEY AUTOINCREMENT')"
    expected = "INSERT INTO t (note) VALUES ('id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY')"
    assert translate_sql(sql) == expected


# ---------------------------------------------------------------------------
# DDL-idiom auto-translation v2: INSERT OR IGNORE -> ON CONFLICT DO NOTHING
# ---------------------------------------------------------------------------

def test_insert_or_ignore_translated_basic():
    sql = "INSERT OR IGNORE INTO t (a,b) VALUES (?,?)"
    assert translate_sql(sql) == "INSERT INTO t (a,b) VALUES (%s,%s) ON CONFLICT DO NOTHING"

def test_insert_or_ignore_case_insensitive():
    sql = "insert or ignore into t (a,b) values (?,?)"
    assert translate_sql(sql) == "INSERT INTO t (a,b) values (%s,%s) ON CONFLICT DO NOTHING"

def test_insert_or_ignore_mixed_case_and_whitespace():
    sql = "Insert  Or   Ignore Into t (a) VALUES (?)"
    assert translate_sql(sql) == "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING"

def test_insert_or_ignore_with_returning():
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?) RETURNING id"
    assert translate_sql(sql) == "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id"

def test_insert_or_ignore_with_returning_lowercase():
    sql = "insert or ignore into t (a) values (?) returning id"
    assert translate_sql(sql) == "INSERT INTO t (a) values (%s) ON CONFLICT DO NOTHING returning id"

def test_insert_or_ignore_with_trailing_semicolon():
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?);"
    assert translate_sql(sql) == "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING"

def test_insert_or_ignore_with_trailing_whitespace():
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?)   \n"
    assert translate_sql(sql) == "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING"

def test_normal_insert_unaffected_except_placeholder():
    sql = "INSERT INTO t (a,b) VALUES (?,?)"
    assert translate_sql(sql) == "INSERT INTO t (a,b) VALUES (%s,%s)"

def test_insert_or_ignore_combined_with_autoincrement_and_datetime_now():
    sql = ("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, "
           "created_at TEXT DEFAULT (datetime('now')), v TEXT); "
           "INSERT OR IGNORE INTO t (v) VALUES (?)")
    expected = ("CREATE TABLE t (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "created_at TEXT DEFAULT (now()::text), v TEXT); "
                "INSERT INTO t (v) VALUES (%s) ON CONFLICT DO NOTHING")
    assert translate_sql(sql) == expected

def test_insert_or_ignore_idiom_idempotent_when_applied_twice():
    # Exercise the DDL-idiom layer directly (not the full translate_sql, which
    # unconditionally escapes '%' on every call -- applying that full pipeline
    # twice to its own output is never idempotent once a placeholder exists,
    # independent of this feature). The idiom transform itself must be stable.
    from dashboard.pgcompat import _translate_ddl_idioms
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?)"
    once = _translate_ddl_idioms(sql)
    twice = _translate_ddl_idioms(once)
    assert once == twice

def test_insert_or_ignore_idempotent_on_already_postgres_sql():
    # No placeholder -> no '%' escaping concern; a statement that already
    # carries ON CONFLICT DO NOTHING (and no "INSERT OR IGNORE") is untouched.
    sql = "INSERT INTO t (a) VALUES ('x') ON CONFLICT DO NOTHING"
    assert translate_sql(sql) == sql

def test_insert_or_replace_not_translated():
    # Out of scope: INSERT OR REPLACE must be left untouched by this translator.
    sql = "INSERT OR REPLACE INTO t (a) VALUES (?)"
    assert translate_sql(sql) == "INSERT OR REPLACE INTO t (a) VALUES (%s)"


# ---------------------------------------------------------------------------
# DDL-idiom v2 FIX: quote/comment-aware ON CONFLICT placement
# ---------------------------------------------------------------------------

def test_insert_or_ignore_returning_word_inside_string_literal_is_not_a_clause():
    # BUG 1: "returning" as string DATA must not be mistaken for a trailing
    # RETURNING clause -- the literal must stay intact and the clause must
    # land at the true end of the statement, after the literal.
    sql = "INSERT OR IGNORE INTO t (a, s) VALUES (?, 'returning')"
    expected = "INSERT INTO t (a, s) VALUES (%s, 'returning') ON CONFLICT DO NOTHING"
    assert translate_sql(sql) == expected

def test_insert_or_ignore_trailing_line_comment_clause_lands_before_comment():
    # BUG 2: a trailing "-- comment" must not swallow the appended clause --
    # it has to be spliced in before the comment, not after it.
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?) -- hi"
    expected = "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING -- hi"
    assert translate_sql(sql) == expected

def test_insert_or_ignore_trailing_block_comment_clause_lands_before_comment():
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?) /* note */"
    expected = "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING /* note */"
    assert translate_sql(sql) == expected

def test_insert_or_ignore_genuine_returning_still_works():
    sql = "INSERT OR IGNORE INTO t (a) VALUES (?) RETURNING id"
    expected = "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id"
    assert translate_sql(sql) == expected

def test_insert_or_ignore_string_literal_returning_and_real_returning_both_present():
    # A string literal containing "returning" AND a genuine trailing RETURNING
    # clause in the same statement must resolve to the real clause -- the
    # literal must not be touched, and the fake match must not win.
    sql = "INSERT OR IGNORE INTO t (a, s) VALUES (?, 'returning') RETURNING id"
    expected = "INSERT INTO t (a, s) VALUES (%s, 'returning') ON CONFLICT DO NOTHING RETURNING id"
    assert translate_sql(sql) == expected
