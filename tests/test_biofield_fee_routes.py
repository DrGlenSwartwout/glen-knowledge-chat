# tests/test_biofield_fee_routes.py
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


def _app(db, store):
    """Inject fakes: fee_get reads `store`, fee_set/clear mutate it."""
    def fee_get(email):
        return {"available": True, "courtesy_cents": store.get(email), "note": "special" if store.get(email) else ""}
    def fee_set(email, cents, note):
        store[email] = int(cents); return {"ok": True}
    def fee_clear(email):
        store.pop(email, None); return {"ok": True}
    return create_app(db, fee_get=fee_get, fee_set=fee_set, fee_clear=fee_clear)


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_author_page_shows_fee_panel(tmp_path):
    client = _app(str(tmp_path / "c.db"), {}).test_client()
    tid = _new(client, "j@x.com")
    html = client.get(f"/author/{tid}").get_data(as_text=True)
    assert "feepanel" in html and "$300" in html and "$997" in html


def test_set_fee_route(tmp_path):
    store = {}
    client = _app(str(tmp_path / "c.db"), store).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/fee", json={"dollars": "100", "note": "special"}).get_json()
    assert j["ok"] and store["j@x.com"] == 10000
    assert "Courtesy" in j["html"] and "$100" in j["html"]


def test_clear_fee_route(tmp_path):
    store = {"j@x.com": 10000}
    client = _app(str(tmp_path / "c.db"), store).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/fee/clear", json={}).get_json()
    assert j["ok"] and "j@x.com" not in store and "Standard" in j["html"]


def test_set_fee_no_email_is_400(tmp_path):
    client = _app(str(tmp_path / "c.db"), {}).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    r = client.post(f"/author/{tid}/fee", json={"dollars": "100"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_set_fee_bad_amount_is_400(tmp_path):
    client = _app(str(tmp_path / "c.db"), {}).test_client()
    tid = _new(client, "j@x.com")
    r = client.post(f"/author/{tid}/fee", json={"dollars": "-5"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_set_fee_zero_is_comp(tmp_path):
    store = {}
    client = _app(str(tmp_path / "c.db"), store).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/fee", json={"dollars": "0", "note": "comp"}).get_json()
    assert j["ok"] and store["j@x.com"] == 0
    assert "Courtesy: $0" in j["html"]
