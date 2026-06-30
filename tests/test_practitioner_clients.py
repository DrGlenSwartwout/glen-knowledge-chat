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
