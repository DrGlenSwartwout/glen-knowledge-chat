"""Invoice publish + email, surfaced in the composer (automated workflow for Rae).

Covers GET /api/console/client-invoice (the composer's Invoice panel) and the new
`email` flag on POST /api/console/order/<id>/publish-to-portal.

Imports app (needs real secrets + writable DATA_DIR), so it's skipped under plain
pytest and runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch \
    python3 -m pytest tests/test_invoice_email_composer.py
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
    from dashboard import orders as _orders
except Exception as _e:  # pragma: no cover - exercised only under plain pytest
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)


def _seed_order(db, email="pt@x.com", name="Pt Name"):
    cx = sqlite3.connect(db)
    _orders.init_orders_table(cx)
    for col, ddl in (("portal_published", "INTEGER NOT NULL DEFAULT 0"),
                     ("invoice_token", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
        except Exception:
            pass  # already present
    items = json.dumps([
        {"slug": "biofield-analysis", "name": "Biofield Analysis", "qty": 1, "line_cents": 10000},
        {"slug": "liver-support", "name": "Liver Support", "qty": 2, "line_cents": 3000}])
    cx.execute(
        "INSERT INTO orders (source,external_ref,name,email,status,pay_status,total_cents,"
        "items_json,address_json,created_at,invoice_token,portal_published) "
        "VALUES (?,?,?,?,?,?,?,?,'{}',?,?,0)",
        ("test", "INV-1", name, email, "proposed", "unpaid", 13000, items,
         "2026-07-08T00:00:00+00:00", "tok-preset"))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders WHERE external_ref='INV-1'").fetchone()[0]
    cx.close()
    return oid


def _auth(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sek", raising=False)
    return db


def test_client_invoice_endpoint(monkeypatch, tmp_path):
    db = _auth(monkeypatch, tmp_path)
    oid = _seed_order(db)
    c = app.app.test_client()
    r = c.get("/api/console/client-invoice?email=pt@x.com&key=sek").get_json()
    assert r["ok"] and r["order"]["id"] == oid
    assert r["order"]["status"] == "proposed"
    assert r["order"]["portal_published"] is False
    assert r["order"]["total_dollars"] == "130.00"
    lines = {l["name"]: l for l in r["order"]["lines"]}
    assert lines["Liver Support"]["qty"] == 2
    assert lines["Liver Support"]["amount_dollars"] == "30.00"
    assert f"edit_order={oid}" in r["order"]["edit_url"]
    # unknown email -> no order
    r2 = c.get("/api/console/client-invoice?email=nobody@x.com&key=sek").get_json()
    assert r2["ok"] and r2["order"] is None


def test_publish_invoice_with_email(monkeypatch, tmp_path):
    db = _auth(monkeypatch, tmp_path)
    oid = _seed_order(db)
    sent = {}
    monkeypatch.setattr(app, "_send_full_report_email",
                        lambda to, name, subj, body: sent.update(to=to, subj=subj, body=body))
    c = app.app.test_client()
    r = c.post(f"/api/console/order/{oid}/publish-to-portal",
               headers={"X-Console-Key": "sek"}, json={"email": True}).get_json()
    assert r["ok"] and r["emailed"] is True
    assert sent["to"] == "pt@x.com" and "invoice is ready" in sent["subj"].lower()
    cx = sqlite3.connect(db)
    pub = cx.execute("SELECT portal_published FROM orders WHERE id=?", (oid,)).fetchone()[0]
    cx.close()
    assert pub == 1


def test_publish_invoice_without_email(monkeypatch, tmp_path):
    db = _auth(monkeypatch, tmp_path)
    oid = _seed_order(db)
    sent = {}
    monkeypatch.setattr(app, "_send_full_report_email",
                        lambda *a, **k: sent.update(called=True))
    c = app.app.test_client()
    r = c.post(f"/api/console/order/{oid}/publish-to-portal",
               headers={"X-Console-Key": "sek"}, json={"email": False}).get_json()
    assert r["ok"] and r["emailed"] is False
    assert sent == {}   # no email sent


def test_publish_invoice_email_failure_still_publishes(monkeypatch, tmp_path):
    db = _auth(monkeypatch, tmp_path)
    oid = _seed_order(db)
    def boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(app, "_send_full_report_email", boom)
    c = app.app.test_client()
    r = c.post(f"/api/console/order/{oid}/publish-to-portal",
               headers={"X-Console-Key": "sek"}, json={"email": True}).get_json()
    assert r["ok"] is True and r["emailed"] is False   # publish succeeded despite email failure
    cx = sqlite3.connect(db)
    pub = cx.execute("SELECT portal_published FROM orders WHERE id=?", (oid,)).fetchone()[0]
    cx.close()
    assert pub == 1
