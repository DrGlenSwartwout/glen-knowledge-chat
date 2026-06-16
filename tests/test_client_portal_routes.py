# tests/test_client_portal_routes.py
import sqlite3
import pytest


# ── Data layer (dashboard/client_portal.py) ─────────────────────────────────

def test_upsert_and_get_roundtrip(tmp_path):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cp.init_client_portal_table(cx)
    content = {
        "greeting": "Aloha Brooke.",
        "video": {"url": "https://app.heygen.com/share/abc", "label": "Watch"},
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "r", "dosing": "d"}],
        "reorder_items": [{"slug": "nous-energy", "qty": 1}],
    }
    token, pid = cp.upsert_portal(cx, "brooke@example.com", "Brooke Webb", content)
    assert token and isinstance(token, str)
    got = cp.get_portal_by_token(cx, token)
    assert got["name"] == "Brooke Webb"
    assert got["email"] == "brooke@example.com"
    assert got["content"]["reorder_items"][0]["slug"] == "nous-energy"


def test_get_unknown_token_returns_none(tmp_path):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cp.init_client_portal_table(cx)
    assert cp.get_portal_by_token(cx, "not-a-real-token") is None


def test_upsert_same_email_keeps_link_and_updates_content(tmp_path):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cp.init_client_portal_table(cx)
    t1, _ = cp.upsert_portal(cx, "b@x.com", "Brooke", {"greeting": "one"})
    t2, _ = cp.upsert_portal(cx, "b@x.com", "Brooke", {"greeting": "two"})
    assert t2 is None  # update does not re-mint a token
    # the originally-shared link still works and now shows the updated content
    assert cp.get_portal_by_token(cx, t1)["content"]["greeting"] == "two"


# ── Routes (app.py) ─────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email="brooke@example.com", name="Brooke Webb", content=None):
    from dashboard import client_portal as cp
    content = content or {
        "greeting": "Aloha Brooke.",
        "video": {"url": "https://app.heygen.com/share/x", "label": "Watch"},
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "r", "dosing": "d"}],
        "reorder_items": [{"slug": "nous-energy", "qty": 1}],
    }
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


def test_portal_page_served(client):
    c, appmod = client
    tok = _seed_portal(appmod)
    r = c.get(f"/portal/{tok}")
    assert r.status_code == 200


def test_api_portal_returns_enriched_content(client):
    c, appmod = client
    tok = _seed_portal(appmod)
    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    j = r.get_json()
    assert j["name"] == "Brooke Webb"
    assert j["reorder_items"][0]["slug"] == "nous-energy"
    assert j["reorder_items"][0].get("name")  # enriched from the catalog


def test_api_portal_bad_token_404(client):
    c, _ = client
    r = c.get("/api/portal/not-a-real-token")
    assert r.status_code == 404


def test_admin_upsert_requires_secret(client):
    c, _ = client
    r = c.post("/admin/portal/upsert", json={"email": "x@y.com", "name": "X", "content": {}})
    assert r.status_code in (401, 403)


def test_admin_upsert_creates_and_returns_token(client):
    c, _ = client
    r = c.post("/admin/portal/upsert?key=test-secret",
               json={"email": "x@y.com", "name": "X", "content": {"greeting": "hi"}})
    assert r.status_code == 200
    j = r.get_json()
    assert j["token"] and j["url"].endswith(j["token"])
    r2 = c.get(f"/api/portal/{j['token']}")
    assert r2.status_code == 200


# ── Practitioner-special pricing ────────────────────────────────────────────

def test_priced_lines_honor_per_item_override():
    import app as appmod
    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": "nous-energy", "qty": 1, "price_cents": 2500}])
    assert subtotal == 2500
    assert lines[0]["amount"] == 25.0
    assert items_rec[0]["qty"] == 1


def test_priced_lines_fall_back_to_catalog():
    import app as appmod
    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": "nous-energy", "qty": 1}])
    # catalog price for nous-energy is $69.97
    assert subtotal == 6997
    assert lines[0]["amount"] == 69.97


def test_api_portal_shows_regular_and_special_price(client):
    c, appmod = client
    tok = _seed_portal(appmod, content={
        "greeting": "hi", "video": {}, "layers": [],
        "pricing_note": "Your certified-practitioner price.",
        "reorder_items": [{"slug": "nous-energy", "qty": 1, "price_cents": 2500}],
    })
    r = c.get(f"/api/portal/{tok}")
    j = r.get_json()
    it = j["reorder_items"][0]
    assert it["price_cents"] == 2500            # the special price (what he pays)
    assert it["regular_price_cents"] == 6997    # the struck-through catalog price
    assert it["is_special"] is True
    assert j["pricing_note"] == "Your certified-practitioner price."


def test_portal_checkout_charges_special_price(client, monkeypatch):
    c, appmod = client
    tok = _seed_portal(appmod, content={
        "greeting": "hi", "video": {}, "layers": [],
        "reorder_items": [{"slug": "nous-energy", "qty": 1, "price_cents": 2500}],
    })
    captured = {}
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})

    def _fake_invoice(cust, lines, **kw):
        captured["lines"] = lines
        total = sum(l["amount"] * l["qty"] for l in lines)
        return {"Id": "INV1", "DocNumber": "1001", "TotalAmt": total}
    monkeypatch.setattr(qbo_billing, "create_invoice", _fake_invoice)
    monkeypatch.setattr(appmod, "_ingest_order", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: "https://checkout.stripe/x")

    r = c.post(f"/api/portal/{tok}/checkout")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/x"
    assert captured["lines"][0]["amount"] == 25.0  # charged the special price, not catalog


def test_portal_checkout_bad_token_404(client):
    c, _ = client
    r = c.post("/api/portal/not-a-real-token/checkout")
    assert r.status_code == 404
