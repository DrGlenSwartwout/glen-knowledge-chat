from dashboard import pricing

SETTINGS = pricing.load_settings({})

def _item(slug, product):
    return {"slug": slug, "name": slug, "qty": 1, "product": product,
            "unit_cents": 10000, "months": 1, "volume_eligible": True}

def test_scalar_tier_uniform_backcompat():
    items = [_item("a", {}), _item("b", {})]
    res = pricing.compute(items, settings=SETTINGS, subscriber_tier_pct=20)
    assert [ln["pct_applied"] for ln in res["lines"]] == [20, 20]

def test_callable_tier_per_line():
    items = [_item("bundle", {"bundle": True}), _item("single", {})]
    def resolver(it):
        return 29 if it["product"].get("bundle") else 3
    res = pricing.compute(items, settings=SETTINGS, subscriber_tier_pct=resolver)
    by_slug = {ln["slug"]: ln["pct_applied"] for ln in res["lines"]}
    assert by_slug["bundle"] == 29
    assert by_slug["single"] == 3

def test_callable_none_falls_back_to_coupon():
    items = [_item("a", {})]
    res = pricing.compute(items, settings=SETTINGS,
                          subscriber_tier_pct=lambda it: None, coupon_pct=10)
    assert res["lines"][0]["pct_applied"] == 10
