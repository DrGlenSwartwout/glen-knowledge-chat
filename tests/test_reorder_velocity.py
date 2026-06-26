import sqlite3
import pytest
from dashboard import reorder as ro


@pytest.fixture
def db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(p)
    cx.execute("CREATE TABLE product_sales (product_fmp_id TEXT, period TEXT, units REAL, revenue_cents INTEGER, source TEXT)")
    # product 425: last-3 = 2026-06(12)+2026-05(6) = 18 → vel_3mo 6.0 ; +2025-09(24) within last 12 → 12mo sum 42 → 3.5
    cx.executemany("INSERT INTO product_sales(product_fmp_id,period,units,revenue_cents,source) VALUES (?,?,?,?, 'fmp')", [
        ("425", "2026-06", 12, 0), ("425", "2026-05", 6, 0), ("425", "2025-09", 24, 0),
        ("999", "2026-06", 9, 0),  # has velocity but no formulation → dropped from plan
    ])
    cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
    cx.execute("INSERT INTO formulations(id,fmp_id,name) VALUES (1,'425','Microbiome')")
    cx.execute("CREATE TABLE formulation_items (formulation_id INTEGER, ingredient_id INTEGER, dose REAL, dose_unit TEXT)")
    cx.execute("INSERT INTO formulation_items(formulation_id,ingredient_id,dose,dose_unit) VALUES (1, 99, 2.0, 'g')")
    cx.commit(); cx.close()
    return p


def test_product_velocity_3_and_12_month(db):
    v = ro.product_velocity(db)
    assert v["425"]["vel_3mo"] == pytest.approx(6.0)    # (12+6)/3
    assert v["425"]["vel_12mo"] == pytest.approx(3.5)   # (12+6+24)/12


def test_velocity_plan_basis_horizon_and_formulation_map(db):
    p3 = ro.velocity_plan(basis="3mo", horizon_months=3, db_path=db)
    assert p3 == [{"formulation_id": 1, "qty": pytest.approx(18.0)}]      # 6 * 3 ; 999 dropped (no formulation)
    p12 = ro.velocity_plan(basis="12mo", horizon_months=2, db_path=db)
    assert p12 == [{"formulation_id": 1, "qty": pytest.approx(7.0)}]      # 3.5 * 2
    pmax = ro.velocity_plan(basis="max", horizon_months=1, db_path=db)
    assert pmax == [{"formulation_id": 1, "qty": pytest.approx(6.0)}]     # max(6, 3.5) * 1


def test_velocity_table_shape(db):
    t = ro.velocity_table(basis="3mo", horizon_months=3, db_path=db)
    assert len(t) == 1 and t[0]["fmp_id"] == "425" and t[0]["name"] == "Microbiome"
    assert t[0]["vel_3mo"] == pytest.approx(6.0) and t[0]["vel_12mo"] == pytest.approx(3.5)
    assert t[0]["projected_qty"] == pytest.approx(18.0)


def test_velocity_plan_feeds_bom_demand(db):
    plan = ro.velocity_plan(basis="3mo", horizon_months=3, db_path=db)
    dem = ro.bom_demand(plan, db_path=db)
    assert dem[99]["demand"] == pytest.approx(36.0)  # dose 2 * qty 18


def test_product_velocity_absent_table_returns_empty(tmp_path):
    """product_sales table doesn't exist yet → no error, just empty results."""
    p = str(tmp_path / "absent.db")
    cx = sqlite3.connect(p)
    # No product_sales table — only formulations (always present in prod)
    cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
    cx.commit(); cx.close()
    assert ro.product_velocity(db_path=p) == {}
    assert ro.velocity_plan(db_path=p) == []
    assert ro.velocity_table(db_path=p) == []


def test_empty_product_sales_returns_empty(tmp_path):
    p = str(tmp_path / "empty.db")
    cx = sqlite3.connect(p)
    cx.execute("CREATE TABLE product_sales (product_fmp_id TEXT, period TEXT, units REAL, revenue_cents INTEGER, source TEXT)")
    cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
    cx.commit(); cx.close()
    assert ro.velocity_plan(db_path=p) == []
