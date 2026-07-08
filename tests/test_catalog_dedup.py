"""Duplicate FMP records for the same eye-drop product are deprecated, not deleted.

FMP carried two near-identical records for each of two eye drops. Invoice/prescription
history (00 System/fmp-extracts) identifies the live one in each pair:

    372  Clear Lens Eye Drops ACES+CAT      16 invoices, 3 prescriptions  <- live
    440  Clear Lens+ Eye Drops ACES+CAT      1 invoice,  0 prescriptions
    390  Neuro Eye Drops ACES+GL Lite        1 invoice,  0 prescriptions  <- live
    369  Neuro+ Eye Drops                    0 invoices, 0 prescriptions

The loser is marked `inactive` rather than removed: prod order history lives in
chat_log.db on the Render disk, and a deleted slug would orphan any line item that
references it. `inactive` makes _get_product return None (unsellable) while leaving
the record readable.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RETIRED = {
    "clear-lens-eye-drops-aces-cat-eye-drops-2": "440",
    "neuro-eye-drops": "369",
}
LIVE = {
    "clear-lens-eye-drops-aces-cat-eye-drops": ("372", "Clear Lens Eye Drops"),
    "neuro-eye-drops-aces-gl-lite-eye-drops": ("390", "Neuro Eye Drops"),
}


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def test_duplicate_records_are_deprecated_not_deleted():
    prods = _products()
    for slug, fmp_id in RETIRED.items():
        assert slug in prods, f"{slug} was deleted; it must stay for order history"
        assert prods[slug].get("inactive") is True, f"{slug} should be inactive"
        assert prods[slug]["fmp_id"] == fmp_id


def test_surviving_record_is_live_and_renamed():
    prods = _products()
    for slug, (fmp_id, name) in LIVE.items():
        e = prods[slug]
        assert e["fmp_id"] == fmp_id
        assert e["name"] == name
        assert not e.get("inactive")


def test_rename_does_not_touch_pinecone_title():
    """pinecone_title is the retrieval key against the vector store — renaming the
    display name must not move it (see reference_product_catalog_pinecone_coupling)."""
    prods = _products()
    assert prods["clear-lens-eye-drops-aces-cat-eye-drops"]["pinecone_title"] == \
        "Clear Lens Eye Drops ACES+CAT Eye Drops"
    assert prods["neuro-eye-drops-aces-gl-lite-eye-drops"]["pinecone_title"].startswith(
        "Neuro Eye Drops")


def test_serenity_capsule_and_drink_mix_both_stay_sellable():
    """Not duplicates: 305 is a capsule ('1 capsule daily'), 1081 a drink mix
    ('1 scoop 2 times a day'). 305 carries FMP's trailing '*' = discontinuing but
    STILL sellable, and it is the only one of the two with any sales history."""
    prods = _products()
    for slug in ("serenity-blue-green-balance", "serenity-bluegreen-balance-drink-mix"):
        assert not prods[slug].get("inactive"), f"{slug} must remain sellable"
