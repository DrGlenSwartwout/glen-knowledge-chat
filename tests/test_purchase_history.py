import sqlite3
from datetime import datetime, timedelta, timezone
from dashboard import purchase_history as ph

def _cx():
    cx = sqlite3.connect(":memory:"); ph.init_purchase_history_table(cx); return cx

def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()

def test_replace_source_is_idempotent_and_scoped():
    cx = _cx()
    n1 = ph.replace_source(cx, "fmp", [("A@x.com","neuro-mag",_iso(10),"inv1"),
                                        ("a@x.com","neuro-mag",_iso(10),"inv1b")])
    assert n1 == 2
    # re-run replaces, does not accumulate
    n2 = ph.replace_source(cx, "fmp", [("a@x.com","terrain-restore",_iso(5),"inv2")])
    assert n2 == 1
    assert ph.slugs_since(cx, "a@x.com", 365) == {"terrain-restore"}
    # a different source is untouched by replacing 'fmp'
    ph.replace_source(cx, "groovekart", [("a@x.com","wholomega",_iso(3),"gk9")])
    ph.replace_source(cx, "fmp", [("a@x.com","neuro-mag",_iso(2),"inv3")])
    assert ph.slugs_since(cx, "a@x.com", 365) == {"neuro-mag","wholomega"}

def test_window_excludes_old():
    cx = _cx()
    ph.replace_source(cx, "fmp", [("a@x.com","old-sku",_iso(400),"i1"),
                                   ("a@x.com","new-sku",_iso(100),"i2")])
    assert ph.slugs_since(cx, "a@x.com", 365) == {"new-sku"}
