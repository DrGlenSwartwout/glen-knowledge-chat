import json, app as appmod


def test_preview_prices_a_known_product(monkeypatch):
    client = appmod.app.test_client()
    # stub product lookup + tax so the test is deterministic
    monkeypatch.setattr(appmod, "_get_product",
                        lambda slug: {"slug": slug, "price_cents": 7000} if slug == "neuro-mag" else None)
    r = client.post("/api/pricing/preview", json={
        "items": [{"slug": "neuro-mag", "qty": 1}],
        "subscriber_tier_pct": 15, "ship_to_state": "CA"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["lines"][0]["line_total_cents"] == 5950   # 15% off 7000
    assert body["discount_cents"] == 1050


def test_preview_skips_unknown_product(monkeypatch):
    client = appmod.app.test_client()
    monkeypatch.setattr(appmod, "_get_product", lambda slug: None)
    r = client.post("/api/pricing/preview", json={"items": [{"slug": "nope", "qty": 1}]})
    assert r.status_code == 200
    assert r.get_json()["lines"] == []


def test_preview_tolerates_junk_qty(monkeypatch):
    # a non-numeric qty from a public body must NOT 500 — it defaults to 1
    client = appmod.app.test_client()
    monkeypatch.setattr(appmod, "_get_product",
                        lambda slug: {"slug": slug, "price_cents": 7000})
    r = client.post("/api/pricing/preview", json={"items": [{"slug": "neuro-mag", "qty": "abc"}]})
    assert r.status_code == 200
    assert r.get_json()["lines"][0]["qty"] == 1
