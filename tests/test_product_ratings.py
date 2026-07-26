"""product_ratings: one row per product, never-downgrade state machine,
unrated can't advance to a color."""
import sqlite3
import pytest
from dashboard import product_ratings as pr

GREEN = {"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"}
RED = {"color": "red", "red_hits": ["Magnesium Stearate"], "yellow_hits": [], "avoidlist_version": "v1"}
UNRATED = {"color": "unrated", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    pr.init_tables(cx)
    return cx


def _rec(cx, key, screen):
    pr.record_screen(cx, key, brand="B", product_name="N",
                     other_ingredients_raw="...", other_ingredients_parsed=["..."],
                     screen=screen)


def test_one_row_per_product():
    cx = _cx()
    _rec(cx, "brand-x", GREEN)
    _rec(cx, "brand-x", GREEN)  # same key again
    assert cx.execute("SELECT COUNT(*) FROM product_ratings").fetchone()[0] == 1


def test_screen_sets_color_and_hits():
    cx = _cx()
    _rec(cx, "k", RED)
    row = pr.get(cx, "k")
    assert row["color"] == "red" and row["status"] == "screened"
    assert row["red_hits"] == ["Magnesium Stearate"]
    assert row["avoidlist_version"] == "v1"


def test_unrated_screen_lands_unrated_not_a_color():
    cx = _cx()
    _rec(cx, "k", UNRATED)
    row = pr.get(cx, "k")
    assert row["status"] == "unrated"
    assert row["color"] is None, "unrated must not be stored as a color"


def test_green_advances_screened_to_ai_draft_to_confirmed():
    cx = _cx()
    _rec(cx, "k", GREEN)
    pr.set_tier2(cx, "k", 8.5, '{"note":"good"}')
    assert pr.get(cx, "k")["status"] == "ai_draft"
    pr.confirm(cx, "k")
    assert pr.get(cx, "k")["status"] == "confirmed"


def test_red_confirms_without_tier2():
    cx = _cx()
    _rec(cx, "k", RED)
    pr.confirm(cx, "k")  # red skips ai_draft
    assert pr.get(cx, "k")["status"] == "confirmed"


def test_never_downgrades_a_confirmed_row():
    cx = _cx()
    _rec(cx, "k", GREEN)
    pr.set_tier2(cx, "k", 8.5, "{}")
    pr.confirm(cx, "k")
    _rec(cx, "k", RED)  # a later re-screen must not walk it back
    row = pr.get(cx, "k")
    assert row["status"] == "confirmed"


def test_unrated_cannot_advance_to_tier2():
    cx = _cx()
    _rec(cx, "k", UNRATED)
    with pytest.raises(ValueError):
        pr.set_tier2(cx, "k", 8.5, "{}")
