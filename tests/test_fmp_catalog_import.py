import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fmp_catalog_import import build_entry, select_and_build, slugify, _cents


def test_cents_parses_text_price():
    assert _cents("70") == 7000
    assert _cents("70.00") == 7000
    assert _cents("") is None and _cents("n/a") is None


def test_build_entry_ff_gets_volume_pricing_and_regular_anchor():
    row = {"product_name": "Digestzymes HomeoEnergetic Drops", "type": "Functional Formulation",
           "sold_price": "70", "retail_sug_price": "80", "id_pk": "75"}
    slug, e = build_entry(row)
    assert slug == "digestzymes-homeoenergetic-drops"
    # FMP's round '70' is shorthand for the $69.97 FF base; app derives the $80 Value
    # anchor from price == 6997 exactly, so importing at 7000 would kill the anchor.
    assert e["price_cents"] == 6997 and e["regular_cents"] == 8000
    assert e["qty_pricing"] is True          # FF -> volume rate
    assert e["no_groovekart"] is True and e["fmp_id"] == "75"


def test_build_entry_ff_off_base_price_passes_through():
    """Only the round-$70 shorthand is normalized. Genuinely different FF price points
    (CDS $35, WholOmega 120-count $190) import at their real price."""
    _slug, cds = build_entry({"product_name": "CDS", "type": "Functional Formulation",
                              "sold_price": "35", "retail_sug_price": "40", "id_pk": "887"})
    assert cds["price_cents"] == 3500
    _slug, wo = build_entry({"product_name": "WholOmega 120 gelcaps", "type": "Functional Formulation",
                             "sold_price": "$190", "retail_sug_price": "230", "id_pk": "448"})
    assert wo["price_cents"] == 19000


def test_build_entry_non_ff_at_70_is_not_normalized():
    """The 6997 normalization is FF-only — an Essence at $70 stays $70."""
    _slug, e = build_entry({"product_name": "Green Jasper Gem Elixir", "type": "Essence",
                            "sold_price": "70", "retail_sug_price": "80", "id_pk": "626"})
    assert e["price_cents"] == 7000


def test_build_entry_drops_srp_at_or_below_price():
    """FMP has rows where retail_sug_price <= sold_price. An SRP anchor below the charge
    price is incoherent — omit the field rather than emit a nonsense Value."""
    _slug, below = build_entry({"product_name": "Protogen", "type": "Homeopathic",
                                "sold_price": "70", "retail_sug_price": "40", "id_pk": "976"})
    assert "regular_cents" not in below
    _slug, equal = build_entry({"product_name": "Equal", "type": "Essence",
                                "sold_price": "70", "retail_sug_price": "70", "id_pk": "9"})
    assert "regular_cents" not in equal


def test_build_entry_essence_no_volume_pricing():
    row = {"product_name": "Green Jasper Gem Elixir in Terrain Restore", "type": "Essence",
           "sold_price": "70", "retail_sug_price": "80", "id_pk": "626"}
    _slug, e = build_entry(row)
    assert e["qty_pricing"] is False         # non-FF -> list price only


def test_build_entry_skips_no_price():
    assert build_entry({"product_name": "X", "type": "Essence", "sold_price": ""}) is None


def test_select_skips_inactive_wrong_type_and_existing():
    existing = {"vitality": {"name": "Vitality"}}
    rows = [
        {"product_name": "New Essence", "type": "Essence", "active": "Yes", "sold_price": "70", "id_pk": "1"},
        {"product_name": "Old Book", "type": "Book", "active": "Yes", "sold_price": "20", "id_pk": "2"},
        {"product_name": "Dead Essence", "type": "Essence", "active": "No", "sold_price": "70", "id_pk": "3"},
        {"product_name": "Vitality", "type": "Functional Formulation", "active": "Yes", "sold_price": "70", "id_pk": "4"},
    ]
    additions, skipped, collisions, by_type = select_and_build(rows, existing)
    assert set(additions) == {"new-essence"}          # book(type), dead(inactive), vitality(exists) all excluded
    assert by_type == {"Essence": 1}


def test_select_de_collides_slugs():
    existing = {"terrain-restore": {"name": "Terrain Restore"}}
    rows = [
        {"product_name": "Terrain Restore", "type": "Tincture", "active": "Yes", "sold_price": "70", "id_pk": "1"},
        {"product_name": "Rescue!", "type": "Essence", "active": "Yes", "sold_price": "70", "id_pk": "2"},
        {"product_name": "Rescue?", "type": "Essence", "active": "Yes", "sold_price": "70", "id_pk": "3"},
    ]
    additions, skipped, collisions, by_type = select_and_build(rows, existing)
    # "Terrain Restore" name already exists -> skipped entirely (not a collision)
    assert "terrain-restore" not in additions or additions["terrain-restore"]["name"] != "Terrain Restore"
    # two "rescue" slugs collide -> second gets -2
    assert "rescue" in additions and "rescue-2" in additions
    assert any(c[1] == "rescue-2" for c in collisions)


def test_cents_strips_dollar_sign_and_commas():
    assert _cents("$40") == 4000
    assert _cents("1,200") == 120000


def test_infoceutical_flat_price_overrides_fmp():
    slug, e = build_entry({"product_name": "ED9 Muscle Energetic Driver Infoceutical",
                           "type": "Infoceutical", "sold_price": "$40", "id_pk": "209"})
    assert e["price_cents"] == 3997          # flat $39.97, not FMP's $40
    assert e["qty_pricing"] is False         # infoceuticals are list price (not FF volume)
