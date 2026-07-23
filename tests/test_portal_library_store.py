import sqlite3
from dashboard import portal_library as lib

def _db():
    cx = sqlite3.connect(":memory:")
    lib.init_table(cx)
    return cx

def test_grant_is_idempotent_per_email_slug():
    cx = _db()
    assert lib.grant(cx, "A@x.com", "healing-glaucoma-starter", "healingglaucoma.com") is True
    # second grant of same (email, slug) is a no-op, email normalized
    assert lib.grant(cx, "a@x.com", "healing-glaucoma-starter") is False
    assert cx.execute("SELECT COUNT(*) FROM portal_library WHERE email='a@x.com'").fetchone()[0] == 1

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
