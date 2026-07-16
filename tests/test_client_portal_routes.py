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


def test_portal_reorder_uses_client_ff_price(client, monkeypatch):
    # Unified FF pricing: the client's FF flat (client_prices.__all_ff__) drives the
    # portal reorder prices, with the same precedence as the invoice — baked per-item
    # override > per-SKU client special > FF flat (FF-eligible only) > regular.
    c, appmod = client
    prods = {
        "ff-prod": {"name": "FF Prod", "price_cents": 6997, "qty_pricing": True},
        "non-ff":  {"name": "Non FF",  "price_cents": 7000, "qty_pricing": False},
        "ff-ov":   {"name": "FF Ov",   "price_cents": 6000, "qty_pricing": True},
        "ff-sku":  {"name": "FF Sku",  "price_cents": 8000, "qty_pricing": True},
    }
    monkeypatch.setattr(appmod, "_get_product", lambda s: prods.get(s))
    email = "ffclient@example.com"
    from dashboard import client_prices as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_table(cx)
    cp.set_ff_flat(cx, email, 5000)          # $50 flat for all FFs
    cp.set_price(cx, email, "ff-sku", 4000)  # per-SKU special beats the flat
    cx.commit(); cx.close()
    tok = _seed_portal(appmod, email=email, name="FF Client", content={
        "greeting": "hi", "layers": [],
        "reorder_items": [
            {"slug": "ff-prod", "qty": 1},                      # FF, no override -> flat 5000
            {"slug": "non-ff", "qty": 1},                       # not FF -> regular 7000
            {"slug": "ff-ov", "qty": 1, "price_cents": 3000},   # baked override wins -> 3000
            {"slug": "ff-sku", "qty": 1},                       # per-SKU special -> 4000
        ]})
    j = c.get(f"/api/portal/{tok}").get_json()
    items = {it["slug"]: it for it in j["reorder_items"]}
    assert items["ff-prod"]["price_cents"] == 5000 and items["ff-prod"]["is_special"] is True
    assert items["non-ff"]["price_cents"] == 7000 and items["non-ff"]["is_special"] is False
    assert items["ff-ov"]["price_cents"] == 3000    # per-item baked override wins
    assert items["ff-sku"]["price_cents"] == 4000   # per-SKU client special wins over the flat


def test_portal_reorder_no_client_price_uses_regular(client, monkeypatch):
    # No client FF price set -> FF item shows the regular catalog price (not special).
    c, appmod = client
    monkeypatch.setattr(appmod, "_get_product",
                        lambda s: {"name": "FF", "price_cents": 6997, "qty_pricing": True} if s == "ff-prod" else None)
    tok = _seed_portal(appmod, email="plain@example.com", name="Plain", content={
        "greeting": "hi", "layers": [], "reorder_items": [{"slug": "ff-prod", "qty": 1}]})
    it = c.get(f"/api/portal/{tok}").get_json()["reorder_items"][0]
    assert it["price_cents"] == 6997 and it["is_special"] is False


def test_portal_current_scan_date_wins_over_later(client):
    # A manual hand-off stamps content.current_scan_date. The display must show THAT
    # report even when a reveal owns a LATER-dated per-scan report (which would win
    # under plain latest-by-date). This is the stale-reveal fix.
    c, appmod = client
    from dashboard import portal_biofield_reports as pbr
    email = "cur@example.com"
    cx = sqlite3.connect(appmod.LOG_DB)
    pbr.init_table(cx)
    pbr.upsert_report(cx, email, "2026-07-09", "",
                      {"layers": [{"n": 1, "title": "RevealLayer", "remedy": "x"}], "reorder_items": []}, "confirmed")
    pbr.upsert_report(cx, email, "2026-07-07", "",
                      {"layers": [{"n": 1, "title": "ManualLayer", "remedy": "y"}], "reorder_items": []}, "confirmed")
    cx.commit(); cx.close()
    tok = _seed_portal(appmod, email=email, name="Cur", content={
        "biofield_status": "confirmed", "current_scan_date": "2026-07-07", "layers": []})
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["scan_date"] == "2026-07-07"                       # current pointer wins over 07-09
    assert "ManualLayer" in [L.get("title") for L in j["layers"]]


def test_portal_past_invoices_in_history(client):
    # History tab: portal-published PAID/done invoices surface under past_invoices;
    # an unpaid published one stays under the live `invoices` card, not history.
    c, appmod = client
    from dashboard import orders as _orders
    email = "hist@example.com"
    cx = sqlite3.connect(appmod.LOG_DB)
    _orders.init_orders_table(cx)
    for col, ddl in (("portal_published", "INTEGER NOT NULL DEFAULT 0"), ("invoice_token", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
        except Exception:
            pass
    cx.execute("INSERT INTO orders (source,external_ref,name,email,status,pay_status,total_cents,"
               "items_json,address_json,created_at,portal_published,invoice_token,paid_at) "
               "VALUES ('t','PAID','H',?,'done','paid',9900,'[]','{}','2026-07-01',1,'tok-paid','2026-07-02')",
               (email,))
    cx.execute("INSERT INTO orders (source,external_ref,name,email,status,pay_status,total_cents,"
               "items_json,address_json,created_at,portal_published,invoice_token) "
               "VALUES ('t','OPEN','H',?,'proposed','unpaid',5000,'[]','{}','2026-07-05',1,'tok-open')",
               (email,))
    # Paid OUTSIDE the portal: not portal-published, no token — must STILL show in history.
    cx.execute("INSERT INTO orders (source,external_ref,name,email,status,pay_status,total_cents,"
               "items_json,address_json,created_at,portal_published) "
               "VALUES ('t','EXT','H',?,'done','paid',30000,'[]','{}','2026-06-20',0)",
               (email,))
    cx.commit(); cx.close()
    tok = _seed_portal(appmod, email=email, name="H", content={"biofield_status": "confirmed", "layers": []})
    j = c.get(f"/api/portal/{tok}").get_json()
    past = {i["token"]: i for i in (j.get("past_invoices") or [])}
    live = {i["token"] for i in (j.get("invoices") or [])}
    amounts = {i["amount_dollars"] for i in (j.get("past_invoices") or [])}
    assert "tok-paid" in past and past["tok-paid"]["amount_dollars"] == "99.00" and past["tok-paid"]["link"]
    assert "300.00" in amounts                                 # paid-outside-portal order (no token) is listed
    ext = [i for i in j["past_invoices"] if i["amount_dollars"] == "300.00"][0]
    assert ext["link"] == "" and ext["paid"] is True           # tokenless -> no link, still a receipt
    assert "tok-open" in live and "tok-open" not in past       # unpaid stays a live pay card


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


def test_view_endpoint_includes_eligible_offer(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_enabled_offer_keys", lambda: {"live_group", "biofield"})
    _seed_person(appmod, "vo@example.com", "VO")
    tok = _seed_portal(appmod, email="vo@example.com", name="VO")
    j = c.get(f"/api/portal/{tok}/view").get_json()
    assert j["upgrade"]["enabled"] is True
    assert j["upgrade"]["offer"]["key"] == "live_group"


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


# ── Group-join offer checkout (mirrors the studio card-vault flow) ───────────

def test_group_join_checkout_dark_by_default(client):
    c, _ = client
    assert c.post("/portal/offer/live-group/checkout").status_code == 404


def test_group_join_checkout_returns_stripe_url(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_portal_offers_enabled", lambda: True)
    _seed_person(appmod, "gj@example.com", "GJ")
    tok = _seed_portal(appmod, email="gj@example.com", name="GJ")
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "create_setup_session",
                        lambda **k: {"url": "https://checkout.stripe/grp"})
    r = c.post(f"/portal/offer/live-group/checkout?token={tok}")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/grp"


def test_group_join_return_creates_membership(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_portal_offers_enabled", lambda: True)
    from dashboard import stripe_pay, subscriptions as subs
    monkeypatch.setattr(stripe_pay, "get_session",
                        lambda sid: {"metadata": {"email": "gj2@example.com"},
                                     "setup_intent": "si_1"})
    monkeypatch.setattr(stripe_pay, "get_setup_intent",
                        lambda si: {"customer": "cus_1", "payment_method": "pm_1"})
    r = c.get("/portal/offer/live-group/return?session_id=sess_1",
              follow_redirects=False)
    assert r.status_code in (302, 303)
    import sqlite3 as _sq
    cx = _sq.connect(appmod.LOG_DB)
    cx.row_factory = _sq.Row  # active_memberships_by_email does dict(row)
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    assert subs.active_memberships_by_email(cx, "gj2@example.com")
    cx.close()


# ── Scaffolded client login (dark behind CLIENT_LOGIN_ENABLED) ───────────────

def test_client_login_routes_dark_by_default(client, monkeypatch):
    c, appmod = client
    # Force the flag OFF so this asserts the dark behavior deterministically — the
    # ambient env (e.g. CLIENT_LOGIN_ENABLED=1 in the prod Doppler config used to
    # run the suite) must not leak in and flip these routes live.
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: False)
    assert c.get("/portal/login").status_code == 404
    assert c.get("/portal/login-verify?token=x").status_code == 404
    assert c.post("/portal/login-request", json={"email": "a@b.com"}).status_code == 404
    assert c.get("/portal/me").status_code == 404


def test_client_login_page_carries_join_wiring(client, monkeypatch):
    # When the page is live it must ship the new-visitor "join" path: the reveal
    # status probe, the provision-anyone endpoint, and the join button. The block
    # itself stays hidden until the JS sees HEALING_OASIS_ENABLED at runtime.
    c, appmod = client
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    r = c.get("/portal/login")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "/api/healing-oasis/status" in body
    assert "/api/healing-oasis/request" in body
    assert 'id="joinGo"' in body


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

    # GET only confirms (mail scanners prefetch it); the POST signs in.
    assert c.get(f"/portal/login-verify?token={magic}").status_code == 200
    r = c.post(f"/portal/login-verify?token={magic}", follow_redirects=False)
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
    # Paid-only (QBO Stage 3): api_client_portal_checkout no longer calls
    # create_invoice -- the exact QBO line payload is persisted via
    # set_order_qbo_lines instead, so capture it the same way.
    c, appmod = client
    tok = _seed_portal(appmod, content={
        "greeting": "hi", "video": {}, "layers": [],
        "reorder_items": [{"slug": "nous-energy", "qty": 1, "price_cents": 2500}],
    })
    captured = {}
    from dashboard import qbo_billing

    def boom(*a, **k):
        raise AssertionError("api_client_portal_checkout must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda *a, **k: None)
    monkeypatch.setattr(appmod._bos_orders, "set_order_qbo_lines",
                        lambda cx, ref, payload: captured.setdefault("payload", payload))
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder",
                        lambda out, email: "https://checkout.stripe/x")

    r = c.post(f"/api/portal/{tok}/checkout")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/x"
    # charged the special price, not catalog
    assert captured["payload"]["lines"][0]["amount"] == 25.0


def test_portal_checkout_bad_token_404(client):
    c, _ = client
    r = c.post("/api/portal/not-a-real-token/checkout")
    assert r.status_code == 404


# ── Biofield interest/request transitions (+ GHL tags) ──────────────────────

def test_biofield_interest_and_request_transitions(client):
    c, appmod = client
    from dashboard import client_portal as cp
    email = "transition@example.com"
    tok = _seed_portal(appmod, email=email, name="T Client", content={
        "biofield_status": "ai_draft", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m",
                    "remedy": "R", "dosing": "d"}]})

    r1 = c.post(f"/api/portal/{tok}/biofield/interest")
    assert r1.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cp.get_biofield_status(cx, email) == "interested"
    cx.close()

    r2 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r2.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cp.get_biofield_status(cx, email) == "requested"

    rows = cx.execute("SELECT payload_json FROM ghl_write_queue").fetchall()
    cx.close()
    blob = "".join(r[0] for r in rows)
    assert "e4l:interested" in blob
    assert "e4l:requested" in blob


def test_biofield_transition_bad_token_404(client):
    c, _ = client
    r = c.post("/api/portal/bogustoken/biofield/interest")
    assert r.status_code == 404


def test_content_endpoint_blurs_remedies_until_confirmed(client):
    c, appmod = client
    tok = _seed_portal(appmod, "blur@y.com", "Blur", {
        "biofield_status": "interested",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["blurred"] is True and j["biofield_status"] == "interested"
    L = j["layers"][0]
    assert L["title"] == "Calm" and L["meaning"] == "m"
    assert "remedy" not in L and "dosing" not in L


def test_content_endpoint_reveals_remedies_when_confirmed(client):
    c, appmod = client
    tok = _seed_portal(appmod, "rev@y.com", "Rev", {
        "biofield_status": "confirmed",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["blurred"] is False and j["layers"][0]["remedy"] == "Nous Energy"


def test_content_endpoint_reports_newest_and_scan_date_param(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    tok = _seed_portal(appmod, "ms@y.com", "MS", {"layers": []})  # ensures token row
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    old = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    R.upsert_report(cx, "ms@y.com", today, "s1",
                    {"layers": [{"n": 1, "title": "New", "remedy": "Y", "dosing": "2"}]}, "interested")
    R.upsert_report(cx, "ms@y.com", old, "s0",
                    {"layers": [{"n": 1, "title": "Old", "remedy": "X", "dosing": "1"}]}, "confirmed")
    cx.close()
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["scan_date"] == today and j["scan_dates"] == [today, old]
    assert j["blurred"] is True and "remedy" not in j["layers"][0]
    j2 = c.get(f"/api/portal/{tok}?scan_date={old}").get_json()
    assert j2["scan_date"] == old and j2["blurred"] is False and j2["layers"][0]["remedy"] == "X"


def test_transition_targets_scan_date_and_rejects_out_of_window(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    tok = _seed_portal(appmod, "tw@y.com", "TW", {"layers": []})
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    old = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    R.upsert_report(cx, "tw@y.com", today, "s1", {"layers": []}, "ai_draft")
    R.upsert_report(cx, "tw@y.com", old, "s0", {"layers": []}, "ai_draft")
    cx.close()
    r = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": today})
    assert r.status_code == 200 and r.get_json()["status"] == "requested"
    cx = sqlite3.connect(appmod.LOG_DB)
    assert R.get_report(cx, "tw@y.com", today)["status"] == "requested"
    r2 = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": old})
    assert r2.status_code == 409
    assert R.get_report(cx, "tw@y.com", old)["status"] == "ai_draft"


def test_admin_upsert_with_scan_date_writes_report(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3
    r = c.post("/admin/portal/upsert?key=test-secret", json={
        "email": "ad@y.com", "name": "Ad", "scan_date": "2026-06-05", "scan_id": "s9",
        "content": {"biofield_status": "ai_draft", "layers": [{"n": 1, "title": "T", "remedy": "R"}]}})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    rep = R.get_report(cx, "ad@y.com", "2026-06-05")
    assert rep is not None and rep["status"] == "ai_draft" and rep["scan_id"] == "s9"


def test_admin_upsert_never_downgrades_confirmed(client):
    # A re-hand-off pushes biofield_status='ai_draft'. If the report at this scan_date
    # is already confirmed (published), it must STAY confirmed — never re-blurred.
    c, appmod = client
    from dashboard import portal_biofield_reports as R, client_portal as cp
    email, sd = "keep@y.com", "2026-07-07"
    # 1) publish/confirm at this scan_date
    c.post("/admin/portal/upsert?key=test-secret", json={
        "email": email, "name": "K", "scan_date": sd,
        "content": {"biofield_status": "confirmed", "layers": [{"n": 1, "title": "T", "remedy": "R"}]}})
    # 2) a later re-hand-off with ai_draft must NOT downgrade it
    c.post("/admin/portal/upsert?key=test-secret", json={
        "email": email, "name": "K", "scan_date": sd,
        "content": {"biofield_status": "ai_draft", "layers": [{"n": 1, "title": "T2", "remedy": "R2"}]}})
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx); cp.init_client_portal_table(cx)
    rep = R.get_report(cx, email, sd)
    status = cp.get_biofield_status(cx, email)
    cx.close()
    assert rep["status"] == "confirmed"          # per-scan report stays published
    assert status == "confirmed"                 # content_json stays confirmed
    # and the newer content DID land (re-sync still updates the report body)
    assert rep["content"]["layers"][0]["title"] == "T2"


def test_admin_upsert_new_client_still_ai_draft(client):
    # The guard must not over-preserve: a brand-new client's first hand-off stays draft.
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    c.post("/admin/portal/upsert?key=test-secret", json={
        "email": "fresh@y.com", "name": "F", "scan_date": "2026-07-07",
        "content": {"biofield_status": "ai_draft", "layers": [{"n": 1, "title": "T", "remedy": "R"}]}})
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    rep = R.get_report(cx, "fresh@y.com", "2026-07-07"); cx.close()
    assert rep["status"] == "ai_draft"


def test_admin_delete_portal_removes_all_traces(client):
    c, appmod = client
    from dashboard import client_portal as cp, portal_biofield_reports as R
    import sqlite3
    cx = sqlite3.connect(appmod.LOG_DB); cp.init_client_portal_table(cx); R.init_table(cx)
    cp.upsert_portal(cx, "del@y.com", "Del", {"layers": []})
    R.upsert_report(cx, "del@y.com", "2026-06-05", "s", {"layers": []}, "ai_draft")
    appmod._log_biofield_correction(cx, "del@y.com", "2026-06-05", {"layers": []})
    cx.close()
    r = c.post("/admin/portal/delete?key=test-secret", json={"email": "del@y.com"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cp.get_portal_content_by_email(cx, "del@y.com") is None
    assert R.list_report_dates(cx, "del@y.com") == []


def test_admin_delete_requires_key(client):
    c, _ = client
    assert c.post("/admin/portal/delete", json={"email": "x@y.com"}).status_code == 401


def test_reissue_link_rotates_token(client):
    c, appmod = client
    tok = _seed_portal(appmod, "ri@y.com", "RI", {"layers": []})
    r = c.post("/admin/portal/reissue-link?key=test-secret", json={"email": "ri@y.com"})
    assert r.status_code == 200
    newtok = (r.get_json()["url"]).rstrip("/").split("/")[-1]
    assert newtok and newtok != tok
    assert c.get(f"/api/portal/{tok}").status_code == 404          # old link dead
    assert c.get(f"/api/portal/{newtok}").status_code == 200       # new link works


def test_reissue_link_404_when_no_portal(client):
    c, _ = client
    assert c.post("/admin/portal/reissue-link?key=test-secret",
                  json={"email": "nobody@y.com"}).status_code == 404


def test_content_endpoint_returns_findings_without_clinical_notes(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    tok = _seed_portal(appmod, "fd@y.com", "FD", {"layers": []})
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    R.upsert_report(cx, "fd@y.com", datetime.date.today().isoformat(), "s1",
        {"layers": [{"n": 1, "title": "T", "remedy": "R", "patterns": ["ED4"]}],
         "findings": [{"code": "ED4", "name": "Nerve Driver",
                       "description": "supports nerve impulses", "rank": 1,
                       "clinical_notes": "SECRET clinician note"}]}, "interested")
    cx.close()
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["findings"][0]["name"] == "Nerve Driver"
    assert j["findings"][0]["description"] == "supports nerve impulses"
    assert j["findings"][0]["code"] == "ED4" and j["findings"][0]["rank"] == 1
    assert "clinical_notes" not in j["findings"][0]


def test_content_endpoint_findings_empty_when_none(client):
    c, appmod = client
    tok = _seed_portal(appmod, "nf@y.com", "NF", {"layers": [{"n": 1, "title": "X"}]})
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["findings"] == []


def test_content_endpoint_returns_notify_on(client):
    c, appmod = client
    import sqlite3
    from dashboard import notify_state as N
    tok = _seed_portal(appmod, "non@y.com", "Non", {"layers": []})
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["notify_on"] is True                      # default = subscribed
    cx = sqlite3.connect(appmod.LOG_DB); N.set_opt(cx, "non@y.com", "out"); cx.commit()
    j2 = c.get(f"/api/portal/{tok}").get_json()
    assert j2["notify_on"] is False                    # opted out


def test_admin_delete_clears_notify_and_process_rows(client):
    c, appmod = client
    from dashboard import notify_state as N, process_queue as Q
    import sqlite3
    _seed_portal(appmod, "wipe@y.com", "Wipe", {"layers": []})
    cx = sqlite3.connect(appmod.LOG_DB)
    N.set_opt(cx, "wipe@y.com", "in"); Q.enqueue(cx, "wipe@y.com", "2026-06-05"); cx.close()
    r = c.post("/admin/portal/delete?key=test-secret", json={"email": "wipe@y.com"})
    assert r.status_code == 200
    d = r.get_json()["deleted"]
    assert d["notify_state"] == 1 and d["process_requests"] == 1
    cx = sqlite3.connect(appmod.LOG_DB)
    assert N.get_state(cx, "wipe@y.com")["opt_status"] == "default"   # row gone -> default
    assert all(p["email"] != "wipe@y.com" for p in Q.list_pending(cx))
