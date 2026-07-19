import sqlite3
import pytest
from biofield_local_app import create_app
from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


@pytest.fixture
def client(tmp_path):
    db = str(tmp_path / "t.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx, name="Donna Banks", email="d@x.com", date="2026-07-06")
        add_chain_row(cx, tid, layer=1, head="", most_affected="", remedy="Liver Support")
        add_chain_row(cx, tid, layer=2, head="", most_affected="", remedy="Green Jasper Gem Elixir")
    calls = {}

    def fake_catalog():
        return [{"slug": "liver-support", "name": "Liver Support"}]

    def fake_create(customer, lines, replace_open=False, invoice_note=None):
        calls["lines"] = lines
        return {"ok": True, "order_id": 7, "external_ref": "INH-Z", "total_cents": 10000, "error": None,
                "accepted_slugs": ["biofield-analysis", "liver-support"]}

    def fake_link(oid):
        return {"ok": True, "print_url": "https://x/invoice/tok?print=1", "error": None}

    app = create_app(db_path=db, invoice_fetch_catalog=fake_catalog,
                     invoice_create=fake_create, invoice_link=fake_link)
    app.testing = True
    c = app.test_client()
    c._calls = calls
    c._tid = tid
    return c


def test_invoice_happy_path(client):
    r = client.post(f"/author/{client._tid}/invoice")
    j = r.get_json()
    assert j["ok"] and j["print_url"].endswith("print=1")
    assert j["skipped"] == ["Green Jasper Gem Elixir"]
    # Biofield is the top line; Liver Support resolved; elixir skipped
    assert client._calls["lines"][0]["slug"] == "biofield-analysis"
    assert {"slug": "liver-support", "qty": 1} in client._calls["lines"]


def test_invoice_requires_email(tmp_path):
    db = str(tmp_path / "e.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx, name="", email="", date="")  # no header -> no email
    app = create_app(db_path=db, invoice_fetch_catalog=lambda: [],
                     invoice_create=lambda *a, **k: {"ok": True},
                     invoice_link=lambda *a: {"ok": True, "print_url": ""})
    app.testing = True
    r = app.test_client().post(f"/author/{tid}/invoice")
    assert r.status_code == 400 and "email" in r.get_json()["error"].lower()


def test_invoice_create_failure_is_502(tmp_path):
    db = str(tmp_path / "f.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx, name="D", email="d@x.com", date="2026-07-06")
        add_chain_row(cx, tid, layer=1, head="", most_affected="", remedy="Liver Support")
    app = create_app(db_path=db, invoice_fetch_catalog=lambda: [],
                     invoice_create=lambda *a, **k: {"ok": False, "error": "Couldn't reach the console."},
                     invoice_link=lambda *a: {"ok": False})
    app.testing = True
    r = app.test_client().post(f"/author/{tid}/invoice")
    assert r.status_code == 502 and "console" in r.get_json()["error"].lower()


def test_invoice_warns_when_biofield_line_dropped(tmp_path):
    db = str(tmp_path / "w.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx, name="D", email="d@x.com", date="2026-07-06")
        add_chain_row(cx, tid, layer=1, head="", most_affected="", remedy="Liver Support")
    app = create_app(db_path=db, invoice_fetch_catalog=lambda: [{"slug": "liver-support", "name": "Liver Support"}],
                     invoice_create=lambda c, l, **k: {"ok": True, "order_id": 9, "external_ref": "INH-W",
                                                  "total_cents": 5000, "accepted_slugs": ["liver-support"]},
                     invoice_link=lambda oid: {"ok": True, "print_url": "https://x/invoice/t?print=1"})
    app.testing = True
    r = app.test_client().post(f"/author/{tid}/invoice")
    j = r.get_json()
    assert j["ok"] and "Biofield Analysis line was not accepted" in j["warning"]
