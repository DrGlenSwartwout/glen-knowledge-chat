import sqlite3
from dashboard import biofield_reveals as br

def _db():
    cx = sqlite3.connect(":memory:"); br.init_table(cx); return cx

def test_list_for_email_returns_rows_newest_first():
    cx = _db()
    br.upsert(cx, "a@x.com", "2026-07-10", {"greeting": "hi"}, [], "t")
    br.upsert(cx, "a@x.com", "2026-07-18", {"greeting": "yo"}, [], "t")
    br.upsert(cx, "b@x.com", "2026-07-18", {}, [], "t")
    rows = br.list_for_email(cx, "A@x.com")            # case-insensitive
    assert [r["scan_date"] for r in rows] == ["2026-07-18", "2026-07-10"]
    assert rows[0]["interpretation"] == {"greeting": "yo"}

def test_list_for_email_empty_when_none():
    assert br.list_for_email(_db(), "none@x.com") == []

def test_list_for_email_applies_remedy_substitution():
    # Funnel parity: the portal read path must swap a curated non-purchasable
    # matched remedy ("Relax") for its sellable equivalent ("Stress Release")
    # exactly like the funnel client read path (get_by_token_hash) does.
    cx = _db()
    br.upsert(
        cx, "c@x.com", "2026-07-18", {},
        [{"name": "Relax", "slug": "relax", "meaning": "old"}], "t",
        layers=[{"n": 1, "title": "T", "meaning": "m",
                 "remedy": {"name": "Relax", "slug": "relax"}}],
    )
    rows = br.list_for_email(cx, "c@x.com")
    assert rows[0]["remedies"][0]["name"] == "Stress Release"
    assert rows[0]["remedies"][0]["slug"] == "stress-release"
    assert rows[0]["layers"][0]["remedy"]["name"] == "Stress Release"
    assert rows[0]["layers"][0]["remedy"]["slug"] == "stress-release"
