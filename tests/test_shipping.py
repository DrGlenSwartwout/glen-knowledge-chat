"""Tests for dashboard.shipping — USPS Flat Rate auto-update + box-fit matrix."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


# ── Schema ────────────────────────────────────────────────────────────────────

def test_init_shipping_schema_seeds_default_rates(tmp_path):
    """First init seeds the April 26 2026 USPS Flat Rate prices so the order
    tool works on fresh deploy. Second init must NOT duplicate them."""
    from dashboard.shipping import init_shipping_schema, get_current_rates
    db_path = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db_path) as cx:
        init_shipping_schema(cx)
        init_shipping_schema(cx)  # idempotent
        count = cx.execute("SELECT COUNT(*) FROM usps_rates").fetchone()[0]
    assert count == 3  # one row per box size, no duplicates

    rates = get_current_rates(db_path=db_path)
    assert rates["S"]["charged_cents"] == 1300
    assert rates["M"]["charged_cents"] == 2300
    assert rates["L"]["charged_cents"] == 3200
    assert rates["S"]["effective_date"] == "2026-04-26"


def test_init_shipping_schema_creates_tables(tmp_path):
    """init_shipping_schema must create bottle_types, box_capacity, usps_rates."""
    db_path = tmp_path / "chat_log.db"
    repo_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys, os, sqlite3\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"db_path = {str(db_path)!r}\n"
        "from dashboard.shipping import init_shipping_schema\n"
        "with sqlite3.connect(db_path) as cx:\n"
        "    init_shipping_schema(cx)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"

    with sqlite3.connect(str(db_path)) as cx:
        tables = {
            r[0] for r in
            cx.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "bottle_types" in tables
    assert "box_capacity" in tables
    assert "usps_rates" in tables


# ── Rounding rule ─────────────────────────────────────────────────────────────

def test_round_to_dollar_rounds_50_up():
    from dashboard.shipping import round_to_dollar
    assert round_to_dollar(1250) == 1300  # $12.50 → $13.00

def test_round_to_dollar_rounds_49_down():
    from dashboard.shipping import round_to_dollar
    assert round_to_dollar(1249) == 1200  # $12.49 → $12.00

def test_round_to_dollar_rounds_99_up():
    from dashboard.shipping import round_to_dollar
    assert round_to_dollar(1299) == 1300

def test_round_to_dollar_keeps_whole_dollars():
    from dashboard.shipping import round_to_dollar
    assert round_to_dollar(1200) == 1200

def test_round_to_dollar_actual_usps_rates():
    """The April 26 2026 retail rates → Glen's charged prices."""
    from dashboard.shipping import round_to_dollar
    assert round_to_dollar(1265) == 1300  # Small  $12.65 → $13
    assert round_to_dollar(2295) == 2300  # Medium $22.95 → $23
    assert round_to_dollar(3150) == 3200  # Large  $31.50 → $32


# ── Box-fit matrix ────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_db(tmp_path):
    """Sqlite db with shipping schema + Rae's box-fit matrix seeded.

    Matrix (per-box max qty): dropper 1oz: S=6, M=12, L=24
                              cap 60ct: S=4, M=8, L=16
    """
    from dashboard.shipping import init_shipping_schema
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(str(db_path)) as cx:
        init_shipping_schema(cx)
        cx.execute("INSERT INTO bottle_types (name) VALUES ('dropper 1oz')")
        cx.execute("INSERT INTO bottle_types (name) VALUES ('cap 60ct')")
        rows = cx.execute("SELECT id, name FROM bottle_types").fetchall()
        ids = {name: i for i, name in rows}
        for name, caps in {
            "dropper 1oz": {"S": 6, "M": 12, "L": 24},
            "cap 60ct":    {"S": 4, "M": 8,  "L": 16},
        }.items():
            for size, qty in caps.items():
                cx.execute(
                    "INSERT INTO box_capacity (bottle_type_id, box_size, qty) "
                    "VALUES (?, ?, ?)",
                    (ids[name], size, qty),
                )
        # Rates: the April 26 2026 Glen charged prices
        for size, retail, charged in [("S", 1265, 1300), ("M", 2295, 2300), ("L", 3150, 3200)]:
            cx.execute(
                "INSERT INTO usps_rates (box_size, usps_retail_cents, charged_cents, "
                "effective_date, source_url, confirmed_by) VALUES (?, ?, ?, ?, ?, ?)",
                (size, retail, charged, "2026-04-26", "https://www.usps.com/", "seed"),
            )
        cx.commit()
    return str(db_path)


def test_pick_box_smallest_fit(seeded_db):
    from dashboard.shipping import pick_box
    # 5 droppers — fits in S (capacity 6)
    assert pick_box({"dropper 1oz": 5}, db_path=seeded_db) == "S"

def test_pick_box_escalates_to_medium(seeded_db):
    from dashboard.shipping import pick_box
    # 8 droppers — exceeds S (6), fits in M (12)
    assert pick_box({"dropper 1oz": 8}, db_path=seeded_db) == "M"

def test_pick_box_escalates_to_large(seeded_db):
    from dashboard.shipping import pick_box
    # 20 droppers — exceeds M (12), fits in L (24)
    assert pick_box({"dropper 1oz": 20}, db_path=seeded_db) == "L"

def test_pick_box_returns_none_when_exceeds_large(seeded_db):
    from dashboard.shipping import pick_box
    # 30 droppers — exceeds L (24)
    assert pick_box({"dropper 1oz": 30}, db_path=seeded_db) is None

def test_pick_box_combines_bottle_types_fractional(seeded_db):
    """Mixed order: 3 droppers (3/6 of S) + 2 caps (2/4 of S) = 1.0 → Small fits exactly."""
    from dashboard.shipping import pick_box
    assert pick_box({"dropper 1oz": 3, "cap 60ct": 2}, db_path=seeded_db) == "S"

def test_pick_box_combines_bottle_types_overflow(seeded_db):
    """3 droppers (3/6=0.5 of S) + 3 caps (3/4=0.75 of S) = 1.25 → escalate to M.

    In M: 3/12 + 3/8 = 0.25 + 0.375 = 0.625 → fits."""
    from dashboard.shipping import pick_box
    assert pick_box({"dropper 1oz": 3, "cap 60ct": 3}, db_path=seeded_db) == "M"

def test_pick_box_unknown_bottle_raises(seeded_db):
    from dashboard.shipping import pick_box, UnknownBottleType
    with pytest.raises(UnknownBottleType):
        pick_box({"nonexistent": 1}, db_path=seeded_db)

def test_pick_box_empty_order_returns_none(seeded_db):
    from dashboard.shipping import pick_box
    assert pick_box({}, db_path=seeded_db) is None


# ── Rate retrieval ────────────────────────────────────────────────────────────

def test_get_current_rates_returns_all_three_sizes(seeded_db):
    from dashboard.shipping import get_current_rates
    rates = get_current_rates(db_path=seeded_db)
    assert set(rates.keys()) == {"S", "M", "L"}
    assert rates["S"]["charged_cents"] == 1300
    assert rates["M"]["charged_cents"] == 2300
    assert rates["L"]["charged_cents"] == 3200

def test_get_current_rates_picks_most_recent(tmp_path):
    """When two effective_date rows exist for the same size, the newer one wins."""
    from dashboard.shipping import init_shipping_schema, get_current_rates
    db_path = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db_path) as cx:
        init_shipping_schema(cx)
        for date, retail, charged in [
            ("2026-01-18", 1170, 1200),  # old Small price
            ("2026-04-26", 1265, 1300),  # new Small price
        ]:
            cx.execute(
                "INSERT INTO usps_rates (box_size, usps_retail_cents, charged_cents, "
                "effective_date, source_url, confirmed_by) VALUES (?, ?, ?, ?, ?, ?)",
                ("S", retail, charged, date, "https://www.usps.com/", "seed"),
            )
        cx.commit()
    rates = get_current_rates(db_path=db_path)
    assert rates["S"]["charged_cents"] == 1300
    assert rates["S"]["effective_date"] == "2026-04-26"


# ── Quote (end-to-end pricing) ────────────────────────────────────────────────

def test_quote_returns_box_and_cost(seeded_db):
    """quote() ties pick_box + get_current_rates together for the order page."""
    from dashboard.shipping import quote
    result = quote({"dropper 1oz": 5}, db_path=seeded_db)
    assert result["box_size"] == "S"
    assert result["shipping_cents"] == 1300

def test_quote_returns_none_when_too_big(seeded_db):
    from dashboard.shipping import quote
    result = quote({"dropper 1oz": 30}, db_path=seeded_db)
    assert result["box_size"] is None
    assert result["shipping_cents"] is None
    assert "exceeds" in result["error"].lower()


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def test_add_bottle_type(seeded_db):
    from dashboard.shipping import add_bottle_type, list_bottle_types
    new_id = add_bottle_type("jar 8oz", notes="heavy", db_path=seeded_db)
    types = list_bottle_types(db_path=seeded_db)
    assert any(t["id"] == new_id and t["name"] == "jar 8oz" for t in types)

def test_set_box_capacity(seeded_db):
    """set_box_capacity is upsert: can call twice on same bottle/size."""
    from dashboard.shipping import (
        add_bottle_type, set_box_capacity, get_capacity_matrix,
    )
    new_id = add_bottle_type("jar 8oz", db_path=seeded_db)
    set_box_capacity(new_id, "S", 2, db_path=seeded_db)
    set_box_capacity(new_id, "M", 4, db_path=seeded_db)
    set_box_capacity(new_id, "L", 8, db_path=seeded_db)
    # Re-set S to confirm upsert
    set_box_capacity(new_id, "S", 3, db_path=seeded_db)
    matrix = get_capacity_matrix(db_path=seeded_db)
    jar_row = next(r for r in matrix if r["name"] == "jar 8oz")
    assert jar_row["S"] == 3
    assert jar_row["M"] == 4
    assert jar_row["L"] == 8

def test_set_box_capacity_rejects_invalid_size(seeded_db):
    from dashboard.shipping import add_bottle_type, set_box_capacity
    new_id = add_bottle_type("jar 8oz", db_path=seeded_db)
    with pytest.raises(ValueError):
        set_box_capacity(new_id, "XL", 2, db_path=seeded_db)

def test_delete_bottle_type_cascades(seeded_db):
    """Deleting a bottle type also removes its capacity rows."""
    from dashboard.shipping import (
        add_bottle_type, set_box_capacity, delete_bottle_type, get_capacity_matrix,
    )
    new_id = add_bottle_type("temp", db_path=seeded_db)
    set_box_capacity(new_id, "S", 1, db_path=seeded_db)
    delete_bottle_type(new_id, db_path=seeded_db)
    matrix = get_capacity_matrix(db_path=seeded_db)
    assert not any(r["name"] == "temp" for r in matrix)


# ── Manual rate update (Glen's approval flow) ────────────────────────────────

def test_propose_rate_update_creates_pending_row(seeded_db):
    """propose_rate_update inserts a usps_rates row with confirmed_by='pending'."""
    from dashboard.shipping import propose_rate_update, list_pending_rate_updates
    propose_rate_update(
        box_size="S", usps_retail_cents=1300, source_url="https://usps.com/test",
        effective_date="2027-01-18", db_path=seeded_db,
    )
    pending = list_pending_rate_updates(db_path=seeded_db)
    assert len(pending) == 1
    assert pending[0]["box_size"] == "S"
    assert pending[0]["usps_retail_cents"] == 1300
    assert pending[0]["charged_cents"] == 1300  # rounding rule applied

def test_confirm_rate_update_marks_confirmed(seeded_db):
    """confirm_rate_update sets confirmed_by + confirmed_at; row becomes the active rate."""
    from dashboard.shipping import (
        propose_rate_update, confirm_rate_update, get_current_rates,
        list_pending_rate_updates,
    )
    propose_rate_update(
        box_size="S", usps_retail_cents=1400, source_url="https://usps.com/test",
        effective_date="2027-01-18", db_path=seeded_db,
    )
    pending_id = list_pending_rate_updates(db_path=seeded_db)[0]["id"]
    confirm_rate_update(pending_id, confirmed_by="glen", db_path=seeded_db)
    assert list_pending_rate_updates(db_path=seeded_db) == []
    rates = get_current_rates(db_path=seeded_db)
    assert rates["S"]["charged_cents"] == 1400


# ── USPS rate watcher ─────────────────────────────────────────────────────────

SAMPLE_USPS_HTML = """
<html><body>
<h2>Priority Mail Flat Rate</h2>
<ul>
  <li>Flat Rate Envelope: $11.95</li>
  <li>Small Flat Rate Box: $13.50</li>
  <li>Medium Flat Rate Box: $24.00</li>
  <li>Large Flat Rate Box: $33.10</li>
</ul>
</body></html>
"""

def test_parse_usps_html_extracts_three_sizes():
    from dashboard.shipping import _parse_usps_html
    out = _parse_usps_html(SAMPLE_USPS_HTML)
    assert out == {"S": 1350, "M": 2400, "L": 3310}

def test_parse_usps_html_raises_on_missing_size():
    from dashboard.shipping import _parse_usps_html
    bad = "<html><body><p>Small Flat Rate Box: $13.50 (medium price moved)</p></body></html>"
    with pytest.raises(ValueError, match="missing sizes"):
        _parse_usps_html(bad)


def test_check_usps_rates_proposes_when_retail_differs(seeded_db, monkeypatch):
    """If scraped retail differs from current, propose an update."""
    from dashboard import shipping
    # Seeded db has S=$12.65 retail. Pretend USPS now reports $13.50.
    monkeypatch.setattr(shipping, "fetch_usps_retail_prices",
                        lambda timeout=30: {"S": 1350, "M": 2295, "L": 3150})
    summary = shipping.check_usps_rates(today="2027-01-18", db_path=seeded_db)
    assert summary["scraped"] == {"S": 1350, "M": 2295, "L": 3150}
    proposed_sizes = [p["box_size"] for p in summary["proposed"]]
    assert proposed_sizes == ["S"]   # only S differed
    assert "M" in summary["unchanged"]
    assert "L" in summary["unchanged"]

def test_check_usps_rates_skips_existing_pending(seeded_db, monkeypatch):
    """A second run shouldn't pile up duplicate pending rows for the same retail."""
    from dashboard import shipping
    monkeypatch.setattr(shipping, "fetch_usps_retail_prices",
                        lambda timeout=30: {"S": 1350, "M": 2295, "L": 3150})
    shipping.check_usps_rates(today="2027-01-18", db_path=seeded_db)
    second = shipping.check_usps_rates(today="2027-01-18", db_path=seeded_db)
    assert second["proposed"] == []
    assert "S (already pending)" in second["unchanged"]
    pending = shipping.list_pending_rate_updates(db_path=seeded_db)
    assert len(pending) == 1   # still just the one from the first run

def test_check_usps_rates_handles_fetch_failure(seeded_db, monkeypatch):
    """A network error should be captured in errors[], not raise."""
    from dashboard import shipping
    def boom(timeout=30):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(shipping, "fetch_usps_retail_prices", boom)
    summary = shipping.check_usps_rates(today="2027-01-18", db_path=seeded_db)
    assert summary["scraped"] is None
    assert summary["proposed"] == []
    assert any("fetch failed" in e for e in summary["errors"])
