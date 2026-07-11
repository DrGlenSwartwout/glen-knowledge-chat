from dashboard import program_tiers as pt
from dashboard import family_plan as fp
from dashboard import portal_offers as po


def _by_key(tiers):
    return {t["key"]: t for t in tiers}


def test_free_is_always_owned():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["free"]["state"] == "owned"
    assert t["free"]["checkout_path"] is None


def test_paid_available_points_at_continuous_care():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["paid"]["state"] == "available"
    assert t["paid"]["checkout_path"] == "/portal/offer/continuous-care/checkout"
    assert t["paid"]["cta_kind"] == "checkout_post"
    assert any("12" in b for b in t["paid"]["benefits"])  # term note present


def test_cta_kinds_and_family_has_no_checkout_route():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["free"]["cta_kind"] == "none"
    assert t["family"]["cta_kind"] == "arrange"
    assert t["family"]["checkout_path"] is None


def test_paid_coming_soon_when_not_live():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=False, family_enabled=True))
    assert t["paid"]["state"] == "coming_soon"


def test_paid_owned_wins():
    t = _by_key(pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["paid"]["state"] == "owned"


def test_family_prices_are_data_sourced():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_live=True, family_enabled=True))
    assert t["family"]["price_cents"] == fp.PLAN["amount_cents"]
    assert t["family"]["value_cents"] == fp.PLAN["value_cents"]
    assert t["family"]["name"] == fp.PLAN["label"]


def test_current_tier_key_prefers_family():
    tiers = pt.program_blocks(
        paid_owned=True, family_owned=True,
        paid_live=True, family_enabled=True)
    assert pt.current_tier_key(tiers) == "family"
    tiers2 = pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_live=True, family_enabled=True)
    assert pt.current_tier_key(tiers2) == "paid"


def test_grow_paths_shape():
    keys = {g["key"] for g in pt.GROW_PATHS}
    assert keys == {"practitioner", "coach", "cert"}
    assert all(g["url"] for g in pt.GROW_PATHS)
