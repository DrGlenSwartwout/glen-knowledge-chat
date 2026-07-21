"""Biofield pipeline: the per-client sequential checklist (/api/console/biofield-pipeline).

Imports app (needs real secrets + writable DATA_DIR); skipped under plain pytest,
runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch \
    python3 -m pytest tests/test_biofield_pipeline.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import app
    import dashboard
    from dashboard import orders as _orders, client_portal as _cp
except Exception as _e:  # pragma: no cover
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)


def _auth(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "sek", raising=False)
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sek", raising=False)
    cx = sqlite3.connect(db)
    _cp.init_client_portal_table(cx)
    cx.close()
    return db


def _seed_portal(db, email, name, status):
    cx = sqlite3.connect(db)
    _cp.init_client_portal_table(cx)
    _cp.upsert_portal(cx, email, name, {"biofield_status": status,
                                        "layers": [{"n": 1, "title": "L", "remedy": "r"}]})
    cx.close()


def _seed_biofield_order(db, email, name, status, pay_status):
    cx = sqlite3.connect(db)
    _orders.init_orders_table(cx)
    for col, ddl in (("portal_published", "INTEGER NOT NULL DEFAULT 0"), ("invoice_token", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
        except Exception:
            pass
    items = json.dumps([{"slug": "biofield-analysis", "name": "Biofield Analysis", "qty": 1, "line_cents": 30000}])
    cx.execute("INSERT INTO orders (source,external_ref,name,email,status,pay_status,total_cents,"
               "items_json,address_json,created_at) VALUES (?,?,?,?,?,?,?,?,'{}',?)",
               ("test", "R-" + email, name, email, status, pay_status, 30000, items,
                "2026-07-08T00:00:00+00:00"))
    cx.commit(); cx.close()


def _clients(db, all_=False):
    c = app.app.test_client()
    q = "/api/console/biofield-pipeline" + ("?all=1&key=sek" if all_ else "?key=sek")
    return {x["email"]: x for x in c.get(q, headers={"X-Console-Key": "sek"}).get_json()["clients"]}


def test_pipeline_handed_off_client(monkeypatch, tmp_path):
    db = _auth(monkeypatch, tmp_path)
    _seed_portal(db, "ho@x.com", "Handed Off", "ai_draft")
    _seed_biofield_order(db, "ho@x.com", "Handed Off", "done", "paid")
    c = _clients(db)["ho@x.com"]
    s = c["steps"]
    assert s["paid"]["done"] is True
    assert s["handed_off"]["done"] is True and s["handed_off"]["awaiting_publish"] is True
    assert s["analysis_published"]["done"] is False   # ai_draft, not yet published
    assert s["invoice_paid"]["done"] is True and s["fulfilled"]["done"] is True
    assert c["complete"] is False                     # analysis not published -> in flight


def test_pipeline_completed_hidden_unless_all(monkeypatch, tmp_path):
    db = _auth(monkeypatch, tmp_path)
    _seed_portal(db, "done@x.com", "All Done", "confirmed")
    _seed_biofield_order(db, "done@x.com", "All Done", "done", "paid")
    assert "done@x.com" not in _clients(db)           # complete -> hidden by default
    shown = _clients(db, all_=True)["done@x.com"]
    assert shown["complete"] is True
    assert shown["steps"]["analysis_published"]["done"] is True


def test_pipeline_paid_not_handed_off(monkeypatch, tmp_path):
    # Steve Fox shape: paid the analysis, intake not handed off yet -> appears, awaiting.
    db = _auth(monkeypatch, tmp_path)
    _seed_biofield_order(db, "steve@x.com", "Steve Fox", "new", "paid")
    c = _clients(db)["steve@x.com"]
    assert c["steps"]["paid"]["done"] is True
    assert c["steps"]["handed_off"]["done"] is False


def _ensure_orders_table(db):
    # The pipeline endpoint reads the orders table (always present in prod). Portal-only
    # tests must create it so the orders pass doesn't hit "no such table".
    cx = sqlite3.connect(db)
    _orders.init_orders_table(cx)
    for col, ddl in (("portal_published", "INTEGER NOT NULL DEFAULT 0"), ("invoice_token", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
        except Exception:
            pass
    cx.commit(); cx.close()


def test_pipeline_excludes_none_status_buyer_portals(monkeypatch, tmp_path):
    # Every buyer is auto-provisioned a portal with biofield_status='none'. Those are
    # not biofield clients and must NOT pollute the pipeline (the '(unnamed)' flood).
    db = _auth(monkeypatch, tmp_path)
    _ensure_orders_table(db)
    _seed_portal(db, "buyer@x.com", "", "none")
    _seed_portal(db, "real@x.com", "Real Client", "requested")
    shown = _clients(db, all_=True)
    assert "buyer@x.com" not in shown        # 'none' buyer portal excluded
    assert "real@x.com" in shown             # a requested biofield client stays


def test_pipeline_none_portal_with_biofield_order_still_shown(monkeypatch, tmp_path):
    # A 'none' buyer portal is excluded by the portal pass, but if that same email has
    # a live biofield-analysis order, the orders pass must still surface them.
    db = _auth(monkeypatch, tmp_path)
    _seed_portal(db, "both@x.com", "Both", "none")
    _seed_biofield_order(db, "both@x.com", "Both", "new", "paid")
    assert "both@x.com" in _clients(db, all_=True)


def test_pipeline_requires_key(monkeypatch, tmp_path):
    _auth(monkeypatch, tmp_path)
    r = app.app.test_client().get("/api/console/biofield-pipeline")
    assert r.status_code == 401
