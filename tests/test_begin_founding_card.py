import begin_funnel


def test_founding_card_in_catalog():
    c = begin_funnel.CARD_CATALOG["founding_offer"]
    assert c["base_url"] == "/begin/product/neuro-magnesium"
    assert c["internal"] is True


def test_surface_with_founding_prepends_when_open():
    state = {"awareness_stage": "product", "current_rung": "assess", "unlocked_gates": ["quiz"]}
    cards = begin_funnel.surface_with_founding(state, ["neuro magnesium"], ref="", founding_open=True)
    assert cards[0]["key"] == "founding_offer"
    assert len(cards) <= 3


def test_surface_with_founding_absent_when_closed():
    state = {"awareness_stage": "product", "current_rung": "assess", "unlocked_gates": ["quiz"]}
    cards = begin_funnel.surface_with_founding(state, ["neuro magnesium"], ref="", founding_open=False)
    assert all(c["key"] != "founding_offer" for c in cards)
