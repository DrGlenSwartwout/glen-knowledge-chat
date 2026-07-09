"""'Special price for the FFs on THIS invoice' must not silently drop lines.

The `these_ff_cents` branch saves a slug only when it is FF-eligible
(`qty_pricing` and not `info_only`) and present+active in the catalog. Everything
else was skipped with no signal: the response was `ok: true` and the UI toast only
reported a count, so pricing an infoceutical or a service looked like it worked and
persisted nothing. 766 of 978 active catalog products fall in that skipped set.

The endpoint now returns `skipped: [{slug, reason}]` so the UI can say so.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import app          # noqa: E402
import dashboard    # noqa: E402


def _auth(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "chat_log.db"), raising=False)
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sek", raising=False)


def _catalog(monkeypatch, products):
    monkeypatch.setattr(app, "_PRODUCTS", {"products": products, "default_price_cents": 6997},
                        raising=False)


FF = {"name": "Neuro Magnesium", "price_cents": 6997, "qty_pricing": True}
INFOCEUTICAL = {"name": "EI8 Microbes-Liver Integrator", "price_cents": 6997}   # no qty_pricing
SERVICE = {"name": "Biofield Analysis", "price_cents": 30000, "info_only": True}
RETIRED = {"name": "Old Drops", "price_cents": 6997, "qty_pricing": True, "inactive": True}


def test_these_ff_reports_slugs_it_skipped(monkeypatch, tmp_path):
    _auth(monkeypatch, tmp_path)
    _catalog(monkeypatch, {"neuro-magnesium": FF, "ei8": INFOCEUTICAL,
                           "biofield-analysis": SERVICE, "old-drops": RETIRED})
    c = app.app.test_client()
    r = c.post("/api/console/client-prices?key=sek",
               headers={"X-Console-Key": "sek"},
               json={"email": "deb@x.com", "these_ff_cents": 5000,
                     "slugs": ["neuro-magnesium", "ei8", "biofield-analysis",
                               "old-drops", "not-in-catalog"]}).get_json()
    assert r["ok"] is True
    assert r["saved"] == 1
    assert r["applied"] == ["neuro-magnesium"]

    skipped = {s["slug"]: s["reason"] for s in r["skipped"]}
    assert set(skipped) == {"ei8", "biofield-analysis", "old-drops", "not-in-catalog"}
    assert skipped["ei8"] == "not a Functional Formulation"
    assert skipped["biofield-analysis"] == "not a Functional Formulation"
    assert skipped["old-drops"] == "not in the catalog"       # inactive -> _get_product None
    assert skipped["not-in-catalog"] == "not in the catalog"


def test_these_ff_all_eligible_reports_nothing_skipped(monkeypatch, tmp_path):
    _auth(monkeypatch, tmp_path)
    _catalog(monkeypatch, {"neuro-magnesium": FF})
    c = app.app.test_client()
    r = c.post("/api/console/client-prices?key=sek",
               headers={"X-Console-Key": "sek"},
               json={"email": "deb@x.com", "these_ff_cents": 5000,
                     "slugs": ["neuro-magnesium"]}).get_json()
    assert r["saved"] == 1 and r["skipped"] == []


def test_zero_dollar_comp_is_saved_not_treated_as_unset(monkeypatch, tmp_path):
    """$0 is a real price (a comp), not a missing value."""
    _auth(monkeypatch, tmp_path)
    _catalog(monkeypatch, {"neuro-magnesium": FF})
    c = app.app.test_client()
    r = c.post("/api/console/client-prices?key=sek", headers={"X-Console-Key": "sek"},
               json={"email": "deb@x.com", "ff_flat_cents": 0}).get_json()
    assert r["ok"] and r["ff_flat_cents"] == 0
    g = c.get("/api/console/client-prices?key=sek&email=deb@x.com",
              headers={"X-Console-Key": "sek"}).get_json()
    assert g["ff_flat_cents"] == 0          # not None


def test_get_returns_ff_flat_and_per_sku_for_readback(monkeypatch, tmp_path):
    """order-new.html prefills from this payload; both fields must be readable."""
    _auth(monkeypatch, tmp_path)
    _catalog(monkeypatch, {"neuro-magnesium": FF})
    c = app.app.test_client()
    c.post("/api/console/client-prices?key=sek", headers={"X-Console-Key": "sek"},
           json={"email": "deb@x.com", "ff_flat_cents": 4500})
    c.post("/api/console/client-prices?key=sek", headers={"X-Console-Key": "sek"},
           json={"email": "deb@x.com", "slug": "neuro-magnesium", "price_cents": 4200})
    g = c.get("/api/console/client-prices?key=sek&email=deb@x.com",
              headers={"X-Console-Key": "sek"}).get_json()
    assert g["ff_flat_cents"] == 4500
    assert [(p["slug"], p["price_cents"]) for p in g["prices"]] == [("neuro-magnesium", 4200)]
