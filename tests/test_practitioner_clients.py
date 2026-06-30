import dashboard.practitioner_portal as pp


def _seed(db):
    pp.record_dispensary_order("prac-1", invoice_id="i1", customer_email="Karin@X.com", db_path=db)
    pp.record_dispensary_order("prac-2", invoice_id="i2", customer_email="bob@x.com", db_path=db)


def test_belongs_true_for_own_client_case_insensitive(tmp_path):
    db = str(tmp_path / "t.db"); _seed(db)
    assert pp.client_belongs_to_practitioner("prac-1", "karin@x.com", db_path=db) is True
    assert pp.client_belongs_to_practitioner("prac-1", "  KARIN@X.COM ", db_path=db) is True


def test_belongs_false_for_other_practitioners_client(tmp_path):
    db = str(tmp_path / "t.db"); _seed(db)
    # bob belongs to prac-2 — prac-1 must NOT be able to claim him
    assert pp.client_belongs_to_practitioner("prac-1", "bob@x.com", db_path=db) is False


def test_belongs_false_for_unknown_or_empty(tmp_path):
    db = str(tmp_path / "t.db"); _seed(db)
    assert pp.client_belongs_to_practitioner("prac-1", "nobody@x.com", db_path=db) is False
    assert pp.client_belongs_to_practitioner("prac-1", "", db_path=db) is False
    assert pp.client_belongs_to_practitioner("prac-1", None, db_path=db) is False
    assert pp.client_belongs_to_practitioner("", "karin@x.com", db_path=db) is False
    assert pp.client_belongs_to_practitioner(None, "karin@x.com", db_path=db) is False


import sqlite3


def _seed_full(db):
    pp.record_dispensary_order("prac-1", invoice_id="i1", customer_email="karin@x.com", db_path=db)
    pp.record_dispensary_order("prac-1", invoice_id="i1b", customer_email="karin@x.com", db_path=db)  # repeat
    pp.record_dispensary_order("prac-1", invoice_id="i3", customer_email="larry@x.com", db_path=db)
    pp.record_dispensary_order("prac-2", invoice_id="i2", customer_email="bob@x.com", db_path=db)
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE IF NOT EXISTS people (email TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, name) VALUES (?, ?)", ("karin@x.com", "Karin Doe"))
    cx.commit(); cx.close()


def test_search_empty_q_returns_empty(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    assert pp.search_clients("prac-1", "", db_path=db) == []
    assert pp.search_clients("prac-1", "   ", db_path=db) == []


def test_search_by_email_and_by_joined_name(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    by_email = pp.search_clients("prac-1", "karin", db_path=db)
    assert {"email": "karin@x.com", "name": "Karin Doe"} in by_email
    by_name = pp.search_clients("prac-1", "doe", db_path=db)
    assert any(c["email"] == "karin@x.com" for c in by_name)


def test_search_dedupes_repeat_orders(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    res = pp.search_clients("prac-1", "karin", db_path=db)
    assert sum(1 for c in res if c["email"] == "karin@x.com") == 1


def test_search_never_returns_other_practitioners_client(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    assert pp.search_clients("prac-1", "bob", db_path=db) == []          # bob is prac-2's
    broad = pp.search_clients("prac-1", "x.com", db_path=db)              # matches everyone's email
    assert all(c["email"] != "bob@x.com" for c in broad)                 # but bob never leaks to prac-1


def test_search_respects_limit(tmp_path):
    db = str(tmp_path / "t.db"); _seed_full(db)
    assert len(pp.search_clients("prac-1", "x.com", limit=1, db_path=db)) == 1
