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


def test_admin_upsert_emails_link_when_send_true(client, monkeypatch):
    c, appmod = client
    sent = {}
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subj, body, **k: sent.update(to=to, body=body))
    r = c.post("/admin/portal/upsert?key=test-secret",
               json={"email": "send@y.com", "name": "S", "content": {"greeting": "hi"},
                     "send": True})
    assert r.status_code == 200
    tok = r.get_json()["token"]
    assert sent["to"] == "send@y.com"
    assert tok in sent["body"]  # the emailed link contains the freshly-minted token


def test_admin_upsert_does_not_email_by_default(client, monkeypatch):
    c, appmod = client
    sent = {}
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda *a, **k: sent.update(called=True))
    c.post("/admin/portal/upsert?key=test-secret",
           json={"email": "nosend@y.com", "name": "N", "content": {}})
    assert "called" not in sent


# ── Role-aware view endpoint (identity seam + view assembler) ────────────────

def _seed_person(appmod, email, name="C", roles='["client"]'):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(appmod.LOG_DB)
    pi._ensure_people_table(cx)
    cx.execute(
        "INSERT OR IGNORE INTO people (email, name, roles, created_at, updated_at) "
        "VALUES (?,?,?,?,?)", (email, name, roles, "t", "t"))
    cx.commit()
    cx.close()


def test_api_portal_view_returns_role_aware_blocks(client):
    c, appmod = client
    _seed_person(appmod, "view@example.com", "View Client")
    tok = _seed_portal(appmod, email="view@example.com", name="View Client")
    r = c.get(f"/api/portal/{tok}/view")
    assert r.status_code == 200
    j = r.get_json()
    assert j["account"]["email"] == "view@example.com"
    assert "Client" in j["account"]["role_badges"]
    assert j["orders"]["visible"] is True
    assert j["biofield"]["visible"] is True       # seeded portal has layers/video
    assert j["upgrade"] == {"enabled": False}  # offers dark by default
    assert j["auth_method"] == "token"            # session login is dark


def test_api_portal_view_bad_token_404(client):
    c, _ = client
    r = c.get("/api/portal/not-a-real-token/view")
    assert r.status_code == 404


# ── Tokenless /portal/me (logged-in home) resolves content via session ───────

def _login_cookie(appmod, email, name="Me"):
    """Enable login, seed a person, and return a valid rm_portal_session value."""
    from dashboard import portal_identity as pi
    _seed_person(appmod, email, name)
    cx = sqlite3.connect(appmod.LOG_DB)
    pid = cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]
    sess = pi.create_client_session(cx, pid, email)
    cx.commit()
    cx.close()
    return sess


def test_api_portal_content_resolves_via_session(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    sess = _login_cookie(appmod, "me@example.com", "Me Client")
    # the same person also has biofield/reorder portal content
    _seed_portal(appmod, email="me@example.com", name="Me Client")
    c.set_cookie("rm_portal_session", sess)

    r = c.get("/api/portal/me")
    assert r.status_code == 200
    j = r.get_json()
    assert j["name"] == "Me Client"
    assert j["reorder_items"][0]["slug"] == "nous-energy"


def test_api_portal_me_without_session_is_404(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    r = c.get("/api/portal/me")
    assert r.status_code == 404


def test_portal_me_page_served_when_enabled(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    assert c.get("/portal/me").status_code == 200


def test_portal_checkout_resolves_via_session(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    sess = _login_cookie(appmod, "co@example.com", "CO")
    _seed_portal(appmod, email="co@example.com", name="CO", content={
        "greeting": "hi", "video": {}, "layers": [],
        "reorder_items": [{"slug": "nous-energy", "qty": 1, "price_cents": 2500}]})
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_invoice",
                        lambda cust, lines, **kw: {"Id": "INV1", "DocNumber": "1", "TotalAmt": 25.0})
    monkeypatch.setattr(appmod, "_ingest_order", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: "https://checkout.stripe/me")
    c.set_cookie("rm_portal_session", sess)

    r = c.post("/api/portal/me/checkout")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/me"


# ── Scaffolded client login (dark behind CLIENT_LOGIN_ENABLED) ───────────────

def test_client_login_routes_dark_by_default(client):
    c, _ = client
    assert c.get("/portal/login").status_code == 404
    assert c.get("/portal/login-verify?token=x").status_code == 404
    assert c.post("/portal/login-request", json={"email": "a@b.com"}).status_code == 404
    assert c.get("/portal/me").status_code == 404


def test_client_login_verify_sets_session_when_enabled(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    _seed_person(appmod, "li@example.com", "LI")
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(appmod.LOG_DB)
    pid = cx.execute("SELECT id FROM people WHERE email=?", ("li@example.com",)).fetchone()[0]
    magic = pi.create_client_magic_link(cx, pid, "li@example.com")
    cx.commit()
    cx.close()

    r = c.get(f"/portal/login-verify?token={magic}", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert "rm_portal_session=" in r.headers.get("Set-Cookie", "")


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
