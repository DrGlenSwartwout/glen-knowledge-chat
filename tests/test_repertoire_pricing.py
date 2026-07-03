from dashboard import pricing


def _item(slug, qty=1, list_cents=6997):
    """Aligned to the real compute() item contract (see test_pricing_engine.py):
    'product'/'unit_cents' carry the list price, 'months' drives volume ramps."""
    return {
        "slug": slug, "name": slug, "qty": qty,
        "product": {"slug": slug, "price_cents": list_cents},
        "unit_cents": list_cents, "months": qty, "volume_eligible": True,
    }


def test_member_repertoire_sku_gets_reorder_rate():
    s = pricing.load_settings({})
    # qty 1 => same_sku ramp gives 0%; regular price = list
    out_reg = pricing.compute([_item("neuro-mag", 1)], settings=s)
    assert out_reg["lines"][0]["line_total_cents"] == 6997

    # member with slug in repertoire => ~29% off (repertoire_reorder_pct default)
    out_mem = pricing.compute([_item("neuro-mag", 1)], settings=s,
                               repertoire_slugs={"neuro-mag"})
    assert out_mem["lines"][0]["line_total_cents"] < 5100  # ~$50
    assert out_mem["lines"][0]["line_total_cents"] == 4968  # round(6997*0.71)


def test_non_repertoire_sku_unchanged_for_member():
    s = pricing.load_settings({})
    out = pricing.compute([_item("brand-new", 1)], settings=s,
                           repertoire_slugs={"neuro-mag"})
    assert out["lines"][0]["line_total_cents"] == 6997  # first buy = regular


def test_repertoire_ignored_when_line_not_volume_eligible():
    s = pricing.load_settings({})
    item = _item("neuro-mag", 1)
    item["volume_eligible"] = False
    out = pricing.compute([item], settings=s, repertoire_slugs={"neuro-mag"})
    assert out["lines"][0]["line_total_cents"] == 6997


def test_non_member_default_none_unaffected():
    s = pricing.load_settings({})
    out = pricing.compute([_item("neuro-mag", 1)], settings=s)  # repertoire_slugs=None
    assert out["lines"][0]["line_total_cents"] == 6997


def test_repertoire_best_of_never_stacks_with_higher_offer():
    s = pricing.load_settings({})
    # qty 12 same-SKU ramp already gives 29% (matches repertoire rate) -> best-of, not additive
    item = _item("neuro-mag", qty=12, list_cents=7000)
    out = pricing.compute([item], settings=s, repertoire_slugs={"neuro-mag"})
    # same as non-repertoire qty-12 pricing: 29% off 84000
    out_no_rep = pricing.compute([item], settings=s)
    assert out["lines"][0]["line_total_cents"] == out_no_rep["lines"][0]["line_total_cents"]
