"""Tests for dashboard.practitioner_settings — pure sqlite (:memory:), no app import."""
import sqlite3
from dashboard import practitioner_settings as ps


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    ps.init_settings_table(cx)
    return cx


def test_defaults_when_unset():
    cx = _cx()
    s = ps.get_settings(cx, "p1")
    assert s["branding"] == {} and s["pricing"] == {"default_markup_pct": 0, "overrides": {}}


def test_set_and_get_roundtrip():
    cx = _cx()
    ps.set_branding(cx, "p1", {"practice_name": "Acme", "brand_color_1": "#0a0",
                               "logo_url": "https://x/l.png"})
    ps.set_pricing(cx, "p1", {"default_markup_pct": 15, "overrides": {"brain-boost": 8500}})
    s = ps.get_settings(cx, "p1")
    assert s["branding"]["practice_name"] == "Acme"
    assert s["pricing"]["default_markup_pct"] == 15
    assert s["pricing"]["overrides"]["brain-boost"] == 8500


def test_price_for_uses_override_then_markup_then_retail():
    cx = _cx()
    ps.set_pricing(cx, "p1", {"default_markup_pct": 20, "overrides": {"a": 9000}})
    # override wins
    assert ps.price_cents_for(cx, "p1", "a", retail_cents=7000, map_cents=6700) == 9000
    # else markup over retail: 7000*1.20 = 8400
    assert ps.price_cents_for(cx, "p1", "b", retail_cents=7000, map_cents=6700) == 8400
    # markup that lands below MAP clamps up to MAP
    ps.set_pricing(cx, "p1", {"default_markup_pct": -20, "overrides": {}})
    assert ps.price_cents_for(cx, "p1", "b", retail_cents=7000, map_cents=6700) == 6700


def test_set_branding_does_not_overwrite_pricing():
    """set_branding should leave existing pricing intact (and vice versa)."""
    cx = _cx()
    ps.set_pricing(cx, "p1", {"default_markup_pct": 10, "overrides": {}})
    ps.set_branding(cx, "p1", {"practice_name": "Clinic A"})
    s = ps.get_settings(cx, "p1")
    assert s["pricing"]["default_markup_pct"] == 10
    assert s["branding"]["practice_name"] == "Clinic A"
