# tests/test_founding_reserve_route.py
import app as appmod
import dashboard.founding as founding


def _setup(monkeypatch, is_open=True):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "f@x.com")
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: True)
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug": s, "name": "Neuro Magnesium", "price_cents": 8000, "qbo_item_id": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "is_open", lambda cx, slug, now_iso=None: is_open)
    inv = {"n": 0}
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: inv.update(n=inv["n"] + 1) or {"Id": "X"})
    cap = {}
    monkeypatch.setattr(appmod.stripe_pay, "create_setup_session",
        lambda **k: cap.update(k) or {"id": "cs", "url": "https://stripe/setup"})
    monkeypatch.setenv("FOUNDING_LAUNCH_ENABLED", "true")
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    return cap, inv


def test_reserve_creates_zero_charge_vault_session(monkeypatch):
    cap, inv = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/begin/founding/reserve", json={"slug": "neuro-magnesium",
               "items": [{"slug": "neuro-magnesium", "qty": 1}], "address": {"state": "HI", "country": "US", "name": "F"}})
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://stripe/setup"     # setup-mode session, $0
    assert cap["metadata"]["kind"] == "founding_reserve"
    assert cap["metadata"]["slug"] == "neuro-magnesium"
    assert inv["n"] == 0                 # no QBO invoice at reservation (vault only)


def test_reserve_closed_returns_409(monkeypatch):
    cap, inv = _setup(monkeypatch, is_open=False)
    c = appmod.app.test_client()
    r = c.post("/begin/founding/reserve", json={"slug": "neuro-magnesium",
               "items": [{"slug": "neuro-magnesium", "qty": 1}], "address": {"state": "HI"}})
    assert r.status_code == 409
    assert r.get_json()["error"] == "founding_closed"


def test_reserve_disabled_when_flag_off(monkeypatch):
    cap, inv = _setup(monkeypatch); monkeypatch.setenv("FOUNDING_LAUNCH_ENABLED", "false")
    c = appmod.app.test_client()
    r = c.post("/begin/founding/reserve", json={"slug": "neuro-magnesium", "items": [], "address": {}})
    assert r.status_code == 400
