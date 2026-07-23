import re
import sqlite3
from dashboard import portal_library as lib
from dashboard import pgcompat

def _db():
    cx = sqlite3.connect(":memory:")
    lib.init_table(cx)
    return cx

def test_grant_ddl_is_postgres_portable():
    # Prod runs Postgres; SQLite-only tests miss dialect bugs. pgcompat translates
    # `INTEGER PRIMARY KEY AUTOINCREMENT` to a PG identity column, but a bare
    # `INTEGER PRIMARY KEY` is NOT translated -> INSERTs without an explicit id
    # hit a null-PK violation on Postgres (the prod 500). Guard: no such pattern
    # may survive translation of this module's DDL.
    captured = []
    class _Rec:
        def execute(self, sql, *a):
            captured.append(sql)
            return self
    lib.init_table(_Rec())
    assert captured, "init_table issued no SQL"
    for sql in captured:
        pg = pgcompat.translate_sql(sql)
        assert not re.search(r"(?i)\bINTEGER\s+PRIMARY\s+KEY\b(?!\s+AUTOINCREMENT)", pg), \
            f"non-identity INTEGER PRIMARY KEY survives PG translation: {pg}"

def test_grant_is_idempotent_per_email_slug():
    cx = _db()
    assert lib.grant(cx, "A@x.com", "healing-glaucoma-starter", "healingglaucoma.com") is True
    # second grant of same (email, slug) is a no-op, email normalized
    assert lib.grant(cx, "a@x.com", "healing-glaucoma-starter") is False
    assert cx.execute("SELECT COUNT(*) FROM ebook_grants WHERE email='a@x.com'").fetchone()[0] == 1

def test_list_and_has():
    cx = _db()
    lib.grant(cx, "b@x.com", "healing-glaucoma-starter", "healingglaucoma.com")
    lib.grant(cx, "b@x.com", "refreshing-vision-starter", "refreshingvision.com")
    slugs = {r["slug"] for r in lib.list_for_email(cx, "b@x.com")}
    assert slugs == {"healing-glaucoma-starter", "refreshing-vision-starter"}
    row = next(r for r in lib.list_for_email(cx, "b@x.com") if r["slug"] == "healing-glaucoma-starter")
    assert row["source_site"] == "healingglaucoma.com"
    assert lib.has(cx, "b@x.com", "healing-glaucoma-starter") is True
    assert lib.has(cx, "b@x.com", "nope") is False
