from dashboard import remedy_upgrades as ru

_CAT = {"focus-neuro-magnesium-powder": {"name": "Focus Neuro-Magnesium", "url": "/p/focus-neuro-magnesium-powder"}}

def test_mapped_ingredient_returns_our_equivalent():
    up = ru.suggest_upgrade("Magnesium Glycinate", "Acme", catalog=_CAT)
    assert up and up["slug"] == "focus-neuro-magnesium-powder"
    assert up["url"] == "/p/focus-neuro-magnesium-powder"
    assert up["reason"]                    # a non-empty clinical reason

def test_unmapped_product_returns_none():
    assert ru.suggest_upgrade("Organic Kale Powder", "Acme", catalog=_CAT) is None

def test_well_chosen_product_not_swapped():
    # a product we already consider optimal maps to None on purpose (clinical integrity)
    assert ru.suggest_upgrade("Focus Neuro-Magnesium", "Remedy Match", catalog=_CAT) is None

def test_our_own_product_by_name_not_swapped():
    cat = {"mitochondrial-biogenesis": {"name": "Mitochondrial Biogenesis", "url": "/p/mitochondrial-biogenesis"}}
    # client is already on OUR Mitochondrial Biogenesis (mapped slug's catalog
    # name matches the input product name exactly) -> no self-referential upgrade
    assert ru.suggest_upgrade("Mitochondrial Biogenesis", "Acme", catalog=cat) is None

def test_coq10_now_maps_to_mitochondrial_biogenesis():
    # coq10 -> mitochondrial-biogenesis (was previously self-mapped to coq10,
    # which the name-equality guard suppressed); now a genuinely different
    # slug/name, so the guard no longer suppresses it and it fires.
    cat = {"mitochondrial-biogenesis": {"name": "Mitochondrial Biogenesis", "url": "/p/mitochondrial-biogenesis"}}
    up = ru.suggest_upgrade("CoQ10", "Acme", catalog=cat)
    assert up and up["slug"] == "mitochondrial-biogenesis"
    assert up["reason"]

def test_probiotics_maps_to_microbiome():
    cat = {"microbiome": {"name": "Microbiome", "url": "/p/microbiome"}}
    up = ru.suggest_upgrade("Probiotics", "Acme", catalog=cat)
    assert up and up["slug"] == "microbiome"
    assert up["reason"]

def test_vitamin_c_maps_to_synergy_c():
    cat = {"vitamin-c-syntropy": {"name": "Synergy C", "url": "/p/vitamin-c-syntropy"}}
    up = ru.suggest_upgrade("Vitamin C", "Acme", catalog=cat)
    assert up and up["slug"] == "vitamin-c-syntropy"
    assert up["reason"]

def test_our_own_brand_never_swapped():
    # brand guard alone suppresses, regardless of product-name mapping
    assert ru.suggest_upgrade("Fish Oil", "Healing Oasis", catalog=_CAT) is None
    assert ru.suggest_upgrade("Fish Oil", "Remedy Match", catalog=_CAT) is None
