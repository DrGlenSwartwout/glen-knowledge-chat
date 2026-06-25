# tests/test_sourcing.py
import sqlite3
import pytest
from dashboard.ingredient_catalog import init_ingredients_schema, list_sources_for_ingredient
from dashboard import sourcing as sc


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        sc.init_sourcing_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'Curcumin')")
        cx.execute("INSERT INTO suppliers (id,company) VALUES (7,'Pharmako')")
        cx.commit()
    return db


_Q = {"gmail_msg_id": "m1", "from_email": "sales@pharmako.com", "subject": "HydroCurc quote",
      "supplier_name": "Pharmako", "ingredient_name": "Curcumin", "price": 334.0, "price_unit": "kg",
      "currency": "USD", "moq": 25.0, "moq_unit": "kg", "lead_time_days": 9, "confidence": 0.9}


def test_stage_idempotent(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        assert sc.stage_quotes(cx, [_Q]) == 1
        assert sc.stage_quotes(cx, [_Q]) == 0          # same gmail_msg_id → no dup
        cx.commit()
    assert len(sc.list_quotes(db_path=db)) == 1


def test_match_and_approve_creates_source(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        sc.stage_quotes(cx, [_Q]); cx.commit()
    qid = sc.list_quotes(db_path=db)[0]["id"]
    with sqlite3.connect(db) as cx:
        sc.match_quote(cx, qid); cx.commit()           # fuzzy: name → ids
    q = sc.get_quote(qid, db_path=db)
    assert q["ingredient_id"] == 1 and q["supplier_id"] == 7
    sid = sc.approve_quote(qid, db_path=db)             # → create_source on ingredient 1
    srcs = list_sources_for_ingredient(1, db_path=db)
    assert len(srcs) == 1 and srcs[0]["id"] == sid and srcs[0]["price_per_unit"] == 334.0
    assert srcs[0]["minimum_order"] == 25.0 and srcs[0]["lead_time_days"] == 9
    assert sc.get_quote(qid, db_path=db)["status"] == "applied"
    with pytest.raises(ValueError):
        sc.approve_quote(qid, db_path=db)              # already applied


def test_approve_requires_ingredient(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        sc.stage_quotes(cx, [{**_Q, "ingredient_name": "Unknownium"}]); cx.commit()
    qid = sc.list_quotes(db_path=db)[0]["id"]
    with sqlite3.connect(db) as cx:
        sc.match_quote(cx, qid); cx.commit()           # no ingredient match → ingredient_id stays NULL
    with pytest.raises(ValueError):
        sc.approve_quote(qid, db_path=db)              # can't apply without a matched ingredient
    sc.dismiss_quote(qid, db_path=db)
    assert sc.get_quote(qid, db_path=db)["status"] == "dismissed"


def test_dismiss_guard_on_applied(tmp_path):
    """An applied quote (which created a source) cannot be dismissed — would leave it inconsistent."""
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        sc.stage_quotes(cx, [_Q]); cx.commit()
        sc.match_quote(cx, sc.list_quotes(db_path=db)[0]["id"]); cx.commit()
    qid = sc.list_quotes(db_path=db)[0]["id"]
    sc.approve_quote(qid, db_path=db)               # status → applied
    with pytest.raises(ValueError):
        sc.dismiss_quote(qid, db_path=db)
    assert sc.get_quote(qid, db_path=db)["status"] == "applied"
