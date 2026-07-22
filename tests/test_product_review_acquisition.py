"""First-touch acquisition source for product-review clients. Data-layer tests are
pure sqlite; the route test needs the app (skips if secrets missing) and stubs the
portal-link email so nothing is sent."""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import customers as cu


def _people_cx():
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, "
               "name TEXT, phone TEXT, source TEXT, created_at TEXT, updated_at TEXT)")
    return cx


def _source(cx, email):
    r = cx.execute("SELECT source FROM people WHERE lower(email)=?", (email.lower(),)).fetchone()
    return r[0] if r else None


def test_new_person_gets_given_source():
    cx = _people_cx()
    cu.find_or_create_by_email(cx, email="a@x.com", name="A", source="product-review")
    assert _source(cx, "a@x.com") == "product-review"


def test_default_source_is_order_entry():
    cx = _people_cx()
    cu.find_or_create_by_email(cx, email="b@x.com")
    assert _source(cx, "b@x.com") == "order-entry"


def test_first_touch_existing_person_keeps_original_source():
    cx = _people_cx()
    cu.find_or_create_by_email(cx, email="c@x.com", source="order-entry")     # already a customer
    cu.find_or_create_by_email(cx, email="c@x.com", source="product-review")  # later submits a review
    assert _source(cx, "c@x.com") == "order-entry"                            # NOT overwritten


# ---- route test ----
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")
os.environ.setdefault("SUPPLEMENT_REVIEW_ENABLED", "on")

import pytest

try:
    import app
    import dashboard
except Exception as e:  # pragma: no cover
    pytest.skip(f"app import needs secrets: {e}", allow_module_level=True)


def test_public_start_tags_source_and_console_lists(monkeypatch, tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SUPPLEMENT_REVIEW_ENABLED", "on")
    monkeypatch.setattr(app, "LOG_DB", dbp, raising=False)
    for o in (app, dashboard):
        monkeypatch.setattr(o, "CONSOLE_SECRET", "sek", raising=False)
    monkeypatch.setattr(app, "send_evox_setup_link", lambda *a, **k: None, raising=False)
    app._init_people_table()
    c = app.app.test_client()

    r = c.post("/api/product-review/public/start",
               json={"email": "new@x.com", "name": "New Client",
                     "product_name": "Fish Oil", "tos_agreed": True})
    assert r.status_code == 200 and r.get_json()["ok"] is True

    cx = sqlite3.connect(dbp)
    assert _source(cx, "new@x.com") == "product-review"
    cx.close()

    j = c.get("/api/console/product-review/new-clients", headers={"X-Console-Key": "sek"}).get_json()
    assert j["total"] >= 1 and any(cl["email"] == "new@x.com" for cl in j["clients"])
    # gated without the key
    assert c.get("/api/console/product-review/new-clients").status_code == 401
