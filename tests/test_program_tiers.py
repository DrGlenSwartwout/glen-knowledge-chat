from dashboard import program_tiers as pt
from dashboard import family_plan as fp
from dashboard import portal_offers as po


def _by_key(tiers):
    return {t["key"]: t for t in tiers}


def test_free_is_always_owned():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["free"]["state"] == "owned"
    assert t["free"]["checkout_path"] is None


def test_paid_available_when_enabled_and_not_owned():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["paid"]["state"] == "available"
    assert t["paid"]["price_cents"] == po.MEMBERSHIP_PRICE_CENTS
    assert t["paid"]["checkout_path"] == "/portal/offer/live-group/checkout"


def test_paid_coming_soon_when_flag_off():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=False, family_enabled=True))
    assert t["paid"]["state"] == "coming_soon"


def test_paid_owned_wins_over_enabled():
    t = _by_key(pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["paid"]["state"] == "owned"


def test_family_prices_are_data_sourced():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["family"]["price_cents"] == fp.PLAN["amount_cents"]
    assert t["family"]["value_cents"] == fp.PLAN["value_cents"]
    assert t["family"]["name"] == fp.PLAN["label"]


def test_current_tier_key_prefers_family():
    tiers = pt.program_blocks(
        paid_owned=True, family_owned=True,
        paid_enabled=True, family_enabled=True)
    assert pt.current_tier_key(tiers) == "family"
    tiers2 = pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_enabled=True, family_enabled=True)
    assert pt.current_tier_key(tiers2) == "paid"


def test_grow_paths_shape():
    keys = {g["key"] for g in pt.GROW_PATHS}
    assert keys == {"practitioner", "coach", "cert"}
    assert all(g["url"] for g in pt.GROW_PATHS)
