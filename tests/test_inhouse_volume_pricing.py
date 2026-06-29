"""In-house order-entry volume pricing: $69.97 functional-formulation capsules
get the order-wide (total-FF-quantity) linear volume rate per line; everything
else stays at list price; server is the sole price authority."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import pricing as _pricing


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:  # missing env in some CI
        pytest.skip(f"app not importable: {e}")


FF = {"slug": "brain", "qty_pricing": True, "price_cents": 6997, "name": "Brain Boost"}
FF2 = {"slug": "bone", "qty_pricing": True, "price_cents": 6997, "name": "Bone Builder"}
NONFF = {"slug": "mix", "price_cents": 7000, "name": "Drink Mix"}
_CAT = {"brain": FF, "bone": FF2, "mix": NONFF}


def test_ff_unit_cents_by_total_qty():
    appmod = _app()
    s = _pricing.load_settings(None)
    f = appmod._inhouse_ff_unit_cents
    # PAID MEMBER (#394: member=True) — effective = round(6997*(1-volume_pct(total)/100)),
    # floored at list*0.57
    assert f(FF, 1, s, member=True) == 6997     # vp(1)=0%
    assert f(FF, 3, s, member=True) == 6450     # vp(3)≈7.82%
    assert f(FF, 6, s, member=True) == 5629     # vp(6)≈19.55%
    assert f(FF, 12, s, member=True) == 3988    # vp(12)=43% (== floor)
    assert f(FF, 99, s, member=True) == 3988    # capped at the 12-unit max
    assert f(NONFF, 12, s, member=True) == 7000  # non-FF: list price, no volume
    # NON-MEMBER / trial (#394: member=False default) — always regular list price
    assert f(FF, 6, s, member=False) == 6997
    assert f(FF, 12, s) == 6997                 # member defaults False → no volume


def test_total_ff_qty_sums_only_ff(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod, "_get_product", _CAT.get)
    lines = [{"slug": "brain", "qty": 4}, {"slug": "bone", "qty": 2}, {"slug": "mix", "qty": 5}]
    assert appmod._inhouse_total_ff_qty(lines) == 6  # 4+2; mix excluded


def test_multi_ff_lines_share_total_rate():
    appmod = _app()
    s = _pricing.load_settings(None)
    # total FF qty 6 → BOTH FF lines priced at the vp(6) rate, even a qty-2 line (members only)
    assert appmod._inhouse_ff_unit_cents(FF, 6, s, member=True) == 5629
    assert appmod._inhouse_ff_unit_cents(FF2, 6, s, member=True) == 5629


def test_line_unit_override_wins():
    appmod = _app()
    s = _pricing.load_settings(None)
    assert appmod._inhouse_line_unit_cents(FF, 5000, 12, s) == 5000   # explicit override (member-independent)
    assert appmod._inhouse_line_unit_cents(FF, None, 12, s, member=True) == 3988    # FF volume (paid member)
    assert appmod._inhouse_line_unit_cents(FF, None, 12, s, member=False) == 6997   # non-member regular
    assert appmod._inhouse_line_unit_cents(NONFF, None, 12, s, member=True) == 7000


def test_price_preview_route(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod, "_get_product", _CAT.get)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: {"id": "m"})  # active grant
    monkeypatch.setattr(appmod, "membership_category", lambda e: "full")  # paid member → volume pricing
    client = appmod.app.test_client()
    key = appmod.dashboard.CONSOLE_SECRET or ""
    r = client.post("/api/orders/price-preview",
                    json={"email": "full@x.com",
                          "lines": [{"slug": "brain", "qty": 4},
                                    {"slug": "bone", "qty": 2},
                                    {"slug": "mix", "qty": 1}]},
                    headers={"X-Console-Key": key})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] and j["total_ff_qty"] == 6
    by = {l["slug"]: l for l in j["lines"]}
    assert by["brain"]["is_ff"] and by["brain"]["effective_unit_cents"] == 5629
    assert by["bone"]["effective_unit_cents"] == 5629          # qty-2 FF line, total rate
    assert (not by["mix"]["is_ff"]) and by["mix"]["effective_unit_cents"] == 7000
    assert j["subtotal_cents"] == 5629 * 4 + 5629 * 2 + 7000 * 1


def test_price_preview_owner_only():
    appmod = _app()
    client = appmod.app.test_client()
    r = client.post("/api/orders/price-preview", json={"lines": []},
                    headers={"X-Console-Key": "wrong-key"})
    assert r.status_code == 401


def test_manual_charges_ff_effective_no_double_discount(monkeypatch, tmp_path):
    appmod = _app()
    db = str(tmp_path / "m.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "_get_product", _CAT.get)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)  # paid member → volume pricing
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    O.init_orders_table(cx)
    cx.close()
    client = appmod.app.test_client()
    key = appmod.dashboard.CONSOLE_SECRET or ""
    # No email → customer find/create + address save are skipped (only orders table needed).
    r = client.post("/api/orders/manual", json={
        "customer": {"name": "T", "address": {"address1": "1", "city": "Hilo",
                                              "state": "HI", "zip": "96720", "country": "US"}},
        "lines": [{"slug": "brain", "qty": 6}], "method": "Zelle"},
        headers={"X-Console-Key": key})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"]
    # FF qty6 charged at the effective 5629/unit; NO separate volume discount applied.
    assert j["lines"][0]["unit_cents"] == 5629
    assert j["totals"]["subtotal_cents"] == 5629 * 6
    assert j["totals"]["discount_cents"] == 0
