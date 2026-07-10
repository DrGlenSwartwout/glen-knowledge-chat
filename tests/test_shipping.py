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


# ── Schema extensions: dimensions + packing settings ──────────────────────────

def test_schema_adds_dims_and_seeds_standard_bottles(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, get_bottle_dims, get_packing_settings
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
        init_shipping_schema(cx)  # idempotent
        cols = {r[1] for r in cx.execute("PRAGMA table_info(bottle_types)")}
    assert {"diameter_mm", "height_mm"} <= cols
    dims = get_bottle_dims(db_path=db)
    assert dims["15ml"] == (30, 100)
    assert dims["120 caps"] == (72, 100)
    assert len(dims) == 14   # 10 prod names + 4 PENDING_BOTTLE_NAMES
    assert get_packing_settings(db_path=db) == {"wrap_mm": 6, "box_margin_mm": 5}   # prod's live values

def test_set_packing_setting_updates_value(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, set_packing_setting, get_packing_settings
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    set_packing_setting("wrap_mm", 9, db_path=db)
    assert get_packing_settings(db_path=db)["wrap_mm"] == 9


# ── geometric path ────────────────────────────────────────────────────────────

@pytest.fixture
def geo_db(tmp_path):
    """Schema with the 8 seeded standard bottle types (dims) + rates."""
    import sqlite3
    from dashboard.shipping import init_shipping_schema
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    return db

def test_quote_geometric_single_box(geo_db):
    from dashboard.shipping import quote
    q = quote({"15ml": 5}, db_path=geo_db)
    assert q["box_sizes"] == ["S"]
    assert q["shipping_cents"] == 1300  # S charged rate

def test_quote_geometric_multibox_sums_rates(geo_db):
    from dashboard.shipping import quote
    q = quote({"15ml": 57}, db_path=geo_db)  # needs 2 boxes (57 > 56 that fit in L)
    assert len(q["box_sizes"]) == 2
    assert q["box_sizes"][0] == "L"
    assert q["shipping_cents"] == sum(b["charged_cents"] for b in q["box_breakdown"])

def test_pick_box_geometric_with_padding(geo_db):
    from dashboard.shipping import pick_box
    # 5ml in S with default padding still fits at least 1 -> S
    assert pick_box({"Dropper 5 mL": 4}, db_path=geo_db) == "S"

def test_override_cap_forces_larger_box(geo_db):
    import sqlite3
    from dashboard.shipping import pick_box, set_box_capacity
    with sqlite3.connect(geo_db) as cx:
        bid = cx.execute("SELECT id FROM bottle_types WHERE name='15ml'").fetchone()[0]
    set_box_capacity(bid, "S", 2, db_path=geo_db)  # cap S at 2 of 15ml
    # 4 of 15ml geometrically fit S, but cap=2 forces escalation to M
    assert pick_box({"15ml": 4}, db_path=geo_db) == "M"


# ── Rename/add migration tests ────────────────────────────────────────────────

def test_migration_renames_100cos_and_adds_30ml(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, get_bottle_dims
    db = str(tmp_path / "chat_log.db")
    # Simulate an already-seeded older DB: insert a 100cos row, no 30ml
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE bottle_types (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "name TEXT NOT NULL UNIQUE, notes TEXT, created_at TEXT NOT NULL "
                   "DEFAULT (datetime('now')))")
        cx.execute("ALTER TABLE bottle_types ADD COLUMN diameter_mm INTEGER")
        cx.execute("ALTER TABLE bottle_types ADD COLUMN height_mm INTEGER")
        cx.execute("INSERT INTO bottle_types (name, diameter_mm, height_mm) "
                   "VALUES ('100cos', 70, 70)")
        cx.commit()
        init_shipping_schema(cx)
    dims = get_bottle_dims(db_path=db)
    assert "100cos" not in dims
    assert dims["30 g"] == (70, 70)   # renamed row KEEPS its own measured dims
    assert dims["30ml"] == (40, 110)

def test_fresh_seed_has_30g_and_30ml_not_100cos(tmp_path):
    import sqlite3
    from dashboard.shipping import init_shipping_schema, get_bottle_dims
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    dims = get_bottle_dims(db_path=db)
    assert dims["30 g"] == (65, 75)   # prod's measured dims
    assert dims["30ml"] == (40, 110)
    assert "100cos" not in dims
    assert len(dims) == 14   # 10 prod names + 4 PENDING_BOTTLE_NAMES


# ── Dimension-aware bottle CRUD ───────────────────────────────────────────────

def test_add_and_update_bottle_with_dims(tmp_path):
    import sqlite3
    from dashboard.shipping import (init_shipping_schema, add_bottle_type,
                                    update_bottle_type, get_bottle_dims)
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    bid = add_bottle_type("250ml-spray", diameter_mm=55, height_mm=180, db_path=db)
    assert get_bottle_dims(db_path=db)["250ml-spray"] == (55, 180)
    update_bottle_type(bid, "250ml-spray", diameter_mm=60, height_mm=185, db_path=db)
    assert get_bottle_dims(db_path=db)["250ml-spray"] == (60, 185)


# ── Product override resolver ────────────────────────────────────────────────────

def test_product_override_crud_and_resolution(tmp_path):
    import sqlite3
    from dashboard.shipping import (init_shipping_schema, set_product_bottle_override,
        clear_product_bottle_override, list_product_bottle_overrides, resolve_bottle_type)
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    # resolution with no override falls to products.json value, then default
    assert resolve_bottle_type("x", {"bottle_type": "15ml"}, db_path=db) == "15ml"
    assert resolve_bottle_type("y", {}, db_path=db) == "default"
    # override wins
    set_product_bottle_override("x", "30ml", db_path=db)
    assert resolve_bottle_type("x", {"bottle_type": "15ml"}, db_path=db) == "30ml"
    assert list_product_bottle_overrides(db_path=db)["x"] == "30ml"
    clear_product_bottle_override("x", db_path=db)
    assert resolve_bottle_type("x", {"bottle_type": "15ml"}, db_path=db) == "15ml"


def test_delete_rate_reverts_to_previous_confirmed(tmp_path):
    from dashboard.shipping import (init_shipping_schema, propose_rate_update,
                                    confirm_rate_update, delete_rate, get_current_rates)
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    rid = propose_rate_update("S", 3375, "u", "2026-07-06", db_path=db)   # wrong rate on top of the seed
    confirm_rate_update(rid, "glen", db_path=db)
    assert get_current_rates(db_path=db)["S"]["usps_retail_cents"] == 3375
    assert delete_rate(rid, db_path=db) == "S"
    assert get_current_rates(db_path=db)["S"]["usps_retail_cents"] == 1265  # reverted to the seed


def test_delete_rate_refuses_last_confirmed(tmp_path):
    import pytest as _pt
    from dashboard.shipping import init_shipping_schema, delete_rate
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
        seed_id = cx.execute("SELECT id FROM usps_rates WHERE box_size='S'").fetchone()[0]
    with _pt.raises(ValueError):
        delete_rate(seed_id, db_path=db)   # only confirmed S rate -> no fallback, refuse


def test_check_usps_rates_flags_implausible_jump(tmp_path, monkeypatch):
    from dashboard import shipping as sh
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        sh.init_shipping_schema(cx)
    # scraper returns a >40% jump for S (the mis-scrape); M/L unchanged
    monkeypatch.setattr(sh, "fetch_usps_retail_prices",
                        lambda *a, **k: {"S": 3375, "M": 2295, "L": 3150})
    summary = sh.check_usps_rates(today="2026-07-07", db_path=db)
    assert "S" in [f["box_size"] for f in summary["flagged"]]     # flagged, not proposed
    assert not any(p["box_size"] == "S" for p in summary["proposed"])
    assert "M" in summary["unchanged"] and "L" in summary["unchanged"]
