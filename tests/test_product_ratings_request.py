import sqlite3
from dashboard import product_ratings as pr

GREEN = {"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"}


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); return cx


def test_request_creates_a_requested_row():
    cx = _cx()
    out = pr.request(cx, "brand-x", brand="Brand", product_name="X", requested_by="a@b.com")
    assert out["created"] is True and out["status"] == "requested"
    assert pr.get(cx, "brand-x")["status"] == "requested"


def test_request_is_idempotent_and_never_downgrades():
    cx = _cx()
    pr.request(cx, "k", brand="B", product_name="N", requested_by="a@b.com")
    pr.record_screen(cx, "k", brand="B", product_name="N",
                     other_ingredients_raw="Magnesium Stearate",
                     other_ingredients_parsed=["Magnesium Stearate"],
                     screen={"color": "red", "red_hits": ["Magnesium Stearate"],
                             "yellow_hits": [], "avoidlist_version": "v1"})
    out = pr.request(cx, "k", brand="B", product_name="N", requested_by="c@d.com")
    assert out["created"] is False and out["status"] == "screened"
    assert pr.get(cx, "k")["status"] == "screened", "request must not walk a screened row back"
