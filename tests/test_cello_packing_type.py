"""Tests for the cello-refill packing bottle_type + packing_bottle_type helper.

Task 1 of the cello-pack packing feature: prices "Cellophane refill packs"
(cello) as a tighter-packing shipping unit than a standard bottle."""

import sqlite3
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from dashboard import shipping as S


def test_packing_bottle_type_maps_cello_and_passes_through(monkeypatch):
    prod = {"slug": "mag", "bottle_type": "default"}
    # A cello/refill line short-circuits to the cello unit WITHOUT touching the DB
    # (case/space-insensitive, matching packing_bottle_type's normalization).
    assert S.packing_bottle_type(prod, "refill") == S.CELLO_BOTTLE_TYPE
    assert S.packing_bottle_type(prod, "  REFILL ") == S.CELLO_BOTTLE_TYPE
    # Non-cello formats delegate to resolve_bottle_type and return its result. Isolate
    # that call with a sentinel so the test is independent of the shared
    # product_bottle_types DB — in a full-suite run another test may set a "mag"
    # override there, which previously made this test order-dependent. resolve_bottle_type
    # has its own tests for the default/override fallback.
    monkeypatch.setattr(S, "resolve_bottle_type", lambda slug, product, **kw: ("passthrough", slug))
    assert S.packing_bottle_type(prod, "bottle") == ("passthrough", "mag")
    assert S.packing_bottle_type(prod, None) == ("passthrough", "mag")


def test_cello_type_is_registered_and_packs_smaller_than_a_standard_bottle(tmp_path):
    db_path = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db_path) as cx:
        S.init_shipping_schema(cx)  # seeds bottle_types (incl. cello-refill) on a fresh db

    dims = {r["name"]: (r["diameter_mm"], r["height_mm"])
            for r in S.list_bottle_types(db_path=db_path)}

    assert S.CELLO_BOTTLE_TYPE in dims

    # cello pack occupies less volume than a standard 30-cap bottle -> tighter
    # packing, more per box.
    dia_c, h_c = dims[S.CELLO_BOTTLE_TYPE]
    dia_b, h_b = dims["30 Caps"]
    assert dia_c * dia_c * h_c < dia_b * dia_b * h_b
