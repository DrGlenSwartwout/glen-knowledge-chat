"""Tests for dashboard.wholesale_pricing — practitioner progressive wholesale
pricing engine (Phase 1, pure functions).

Locks the agreed blended-price model:
  F = 4000 - clamp(modules,0,12)*125   (cents)
  blended unit price piecewise-linear in total bottles q, knots at
  (1, 5000), (B, (5000+F)//2), (2B, F), flat F beyond 2B.
"""

import sqlite3

import pytest


# ── certification_floor_cents ─────────────────────────────────────────────────

def test_floor_uncertified_is_4000():
    from dashboard.wholesale_pricing import certification_floor_cents
    assert certification_floor_cents(0) == 4000


def test_floor_fully_certified_is_2500():
    from dashboard.wholesale_pricing import certification_floor_cents
    assert certification_floor_cents(12) == 2500


def test_floor_midway_six_modules_is_3250():
    from dashboard.wholesale_pricing import certification_floor_cents
    assert certification_floor_cents(6) == 3250


def test_floor_clamps_below_zero_and_above_twelve():
    from dashboard.wholesale_pricing import certification_floor_cents
    assert certification_floor_cents(-3) == 4000
    assert certification_floor_cents(99) == 2500


# ── blended_unit_price_cents: the five locked points ──────────────────────────

def test_locked_uncertified_full_box_is_4500():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    assert blended_unit_price_cents(20, 0, 20) == 4500   # $45.00


def test_locked_uncertified_two_boxes_is_4000():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    assert blended_unit_price_cents(40, 0, 20) == 4000   # $40.00 floor


def test_locked_certified_full_box_is_3750():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    assert blended_unit_price_cents(20, 12, 20) == 3750  # $37.50


def test_locked_certified_thirty_is_3125():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    assert blended_unit_price_cents(30, 12, 20) == 3125  # $31.25


def test_locked_certified_two_boxes_is_2500():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    assert blended_unit_price_cents(40, 12, 20) == 2500  # $25.00 floor


# ── blended_unit_price_cents: shape ───────────────────────────────────────────

def test_single_bottle_is_5000_regardless_of_cert():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    assert blended_unit_price_cents(1, 0, 20) == 5000
    assert blended_unit_price_cents(1, 12, 20) == 5000


def test_floor_holds_beyond_two_boxes():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    # certified floor 2500 holds for 40, 60, 200
    assert blended_unit_price_cents(60, 12, 20) == 2500
    assert blended_unit_price_cents(200, 12, 20) == 2500


def test_unit_price_monotonic_non_increasing():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    prev = blended_unit_price_cents(1, 12, 20)
    for q in range(2, 61):
        cur = blended_unit_price_cents(q, 12, 20)
        assert cur <= prev, f"unit went up at q={q}: {cur} > {prev}"
        prev = cur


def test_order_total_monotonic_non_decreasing():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    prev = 1 * blended_unit_price_cents(1, 12, 20)
    for q in range(2, 61):
        cur = q * blended_unit_price_cents(q, 12, 20)
        assert cur >= prev, f"total dropped at q={q}: {cur} < {prev}"
        prev = cur


def test_non_twenty_box_size_b12_uncertified():
    from dashboard.wholesale_pricing import blended_unit_price_cents
    # B=12, uncertified F=4000: midpoint at q=12 = (5000+4000)//2 = 4500;
    # floor at q=24 = 4000.
    assert blended_unit_price_cents(12, 0, 12) == 4500
    assert blended_unit_price_cents(24, 0, 12) == 4000


# ── _product_pricing ──────────────────────────────────────────────────────────

def test_product_pricing_from_injected_catalog():
    from dashboard.wholesale_pricing import _product_pricing
    catalog = {"x": {"name": "X Formula", "qbo_item_id": "9",
                     "price_cents": 7000, "bottle_type": "dropper 1oz",
                     "cogs_cents": 1500, "fulfillment_cents": 800}}
    p = _product_pricing("x", catalog=catalog)
    assert p["name"] == "X Formula"
    assert p["retail_cents"] == 7000
    assert p["bottle_type"] == "dropper 1oz"
    assert p["cogs_cents"] == 1500
    assert p["fulfillment_cents"] == 800


def test_product_pricing_unknown_slug_falls_back_to_defaults():
    from dashboard.wholesale_pricing import (
        _product_pricing, DEFAULT_BOTTLE_TYPE,
    )
    p = _product_pricing("nope", catalog={})
    assert p["bottle_type"] == DEFAULT_BOTTLE_TYPE
    assert p["cogs_cents"] is None
    assert p["fulfillment_cents"] is None
    assert p["retail_cents"] > 0  # default retail present


# ── _resolve_B (box-fit matrix) ───────────────────────────────────────────────

def _seed_matrix(tmp_path, rows):
    """rows: [(bottle_name, L_capacity)]. Returns db_path with matrix seeded."""
    from dashboard.shipping import (
        init_shipping_schema, add_bottle_type, set_box_capacity,
    )
    db_path = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db_path) as cx:
        init_shipping_schema(cx)
    for name, L in rows:
        bid = add_bottle_type(name, db_path=db_path)
        if L is not None:
            set_box_capacity(bid, "L", L, db_path=db_path)
    return db_path


def test_resolve_b_single_bottle_type(tmp_path):
    from dashboard.wholesale_pricing import _resolve_B
    db = _seed_matrix(tmp_path, [("dropper 1oz", 24)])
    catalog = {"x": {"name": "X", "bottle_type": "dropper 1oz"}}
    B, warnings = _resolve_B([{"slug": "x", "qty": 5}], db_path=db, catalog=catalog)
    assert B == 24
    assert warnings == []


def test_resolve_b_mixed_types_takes_min_L(tmp_path):
    from dashboard.wholesale_pricing import _resolve_B
    db = _seed_matrix(tmp_path, [("dropper 1oz", 24), ("capsule 90ct", 16)])
    catalog = {"a": {"bottle_type": "dropper 1oz"},
               "b": {"bottle_type": "capsule 90ct"}}
    B, warnings = _resolve_B(
        [{"slug": "a", "qty": 5}, {"slug": "b", "qty": 5}],
        db_path=db, catalog=catalog,
    )
    assert B == 16  # min of 24, 16


def test_resolve_b_empty_matrix_falls_back_with_warning(tmp_path):
    from dashboard.wholesale_pricing import _resolve_B, DEFAULT_B
    db = _seed_matrix(tmp_path, [])  # no bottle types at all
    catalog = {"x": {"bottle_type": "dropper 1oz"}}
    B, warnings = _resolve_B([{"slug": "x", "qty": 5}], db_path=db, catalog=catalog)
    assert B == DEFAULT_B
    assert any("fallback" in w.lower() or "empty" in w.lower() for w in warnings)


def test_resolve_b_unknown_bottle_type_warns(tmp_path):
    from dashboard.wholesale_pricing import _resolve_B, DEFAULT_B
    db = _seed_matrix(tmp_path, [("dropper 1oz", 24)])
    catalog = {"x": {"bottle_type": "mystery jar"}}  # not in matrix
    B, warnings = _resolve_B([{"slug": "x", "qty": 5}], db_path=db, catalog=catalog)
    assert B == DEFAULT_B
    assert warnings  # at least one warning about the missing bottle type


# ── order_quote ───────────────────────────────────────────────────────────────

def test_order_quote_single_line_certified_two_boxes(tmp_path):
    from dashboard.wholesale_pricing import order_quote
    db = _seed_matrix(tmp_path, [("dropper 1oz", 20)])
    catalog = {"x": {"name": "X", "bottle_type": "dropper 1oz",
                     "price_cents": 7000}}
    q = order_quote([{"slug": "x", "qty": 40}], {"modules_completed": 12},
                    db_path=db, catalog=catalog)
    assert q["blended_unit_price_cents"] == 2500
    assert q["total_bottles"] == 40
    assert q["B_effective"] == 20
    assert q["floor_cents"] == 2500
    assert len(q["lines"]) == 1
    assert q["lines"][0]["line_total_cents"] == 100000  # 2500 * 40
    assert q["subtotal_cents"] == 100000


def test_order_quote_multi_line_prices_whole_order_at_one_blended_unit(tmp_path):
    from dashboard.wholesale_pricing import order_quote
    db = _seed_matrix(tmp_path, [("dropper 1oz", 20)])
    catalog = {"a": {"name": "A", "bottle_type": "dropper 1oz", "price_cents": 7000},
               "b": {"name": "B", "bottle_type": "dropper 1oz", "price_cents": 7000}}
    # total q = 20+20 = 40 -> certified floor 2500 for the whole order
    q = order_quote([{"slug": "a", "qty": 20}, {"slug": "b", "qty": 20}],
                    {"modules_completed": 12}, db_path=db, catalog=catalog)
    assert q["total_bottles"] == 40
    assert q["blended_unit_price_cents"] == 2500
    units = {ln["unit_price_cents"] for ln in q["lines"]}
    assert units == {2500}  # same blended unit on every line
    assert q["subtotal_cents"] == 100000


def test_order_quote_margin_ok_true_with_warning_when_cogs_unset(tmp_path):
    from dashboard.wholesale_pricing import order_quote
    db = _seed_matrix(tmp_path, [("dropper 1oz", 20)])
    catalog = {"x": {"name": "X", "bottle_type": "dropper 1oz", "price_cents": 7000}}
    q = order_quote([{"slug": "x", "qty": 40}], {"modules_completed": 12},
                    db_path=db, catalog=catalog)
    assert q["margin_ok"] is True
    assert any("cogs" in w.lower() for w in q["margin_warnings"])


def test_order_quote_margin_not_ok_when_cost_exceeds_blended(tmp_path):
    from dashboard.wholesale_pricing import order_quote
    db = _seed_matrix(tmp_path, [("dropper 1oz", 20)])
    # blended at 40 certified = 2500; cost 2000+800 = 2800 > 2500
    catalog = {"x": {"name": "X", "bottle_type": "dropper 1oz",
                     "price_cents": 7000, "cogs_cents": 2000,
                     "fulfillment_cents": 800}}
    q = order_quote([{"slug": "x", "qty": 40}], {"modules_completed": 12},
                    db_path=db, catalog=catalog)
    assert q["margin_ok"] is False
    assert q["margin_warnings"]
