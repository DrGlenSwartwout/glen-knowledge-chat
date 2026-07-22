from dashboard import remedy_upgrades as ru

_CAT = {"neuro-mag": {"name": "Neuro-Mag", "url": "/p/neuro-mag"}}

def test_mapped_ingredient_returns_our_equivalent():
    up = ru.suggest_upgrade("Magnesium Glycinate", "Acme", catalog=_CAT)
    assert up and up["slug"] == "neuro-mag" and up["url"] == "/p/neuro-mag"
    assert up["reason"]                    # a non-empty clinical reason

def test_unmapped_product_returns_none():
    assert ru.suggest_upgrade("Organic Kale Powder", "Acme", catalog=_CAT) is None

def test_well_chosen_product_not_swapped():
    # a product we already consider optimal maps to None on purpose (clinical integrity)
    assert ru.suggest_upgrade("Neuro-Mag", "Remedy Match", catalog=_CAT) is None
