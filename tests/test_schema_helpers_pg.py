"""Schema-port validation for the 4 dashboard `init_*_schema(cx)` helpers
(formulations, inventory, ingredients, materials) — batch 3a.

Each helper takes a bare `cx`, so it can be validated against a live Postgres
DB in isolation (no app.py-import coupling). Postgres tests are skip-guarded
on PG_DSN so they're a no-op in secretless CI; the SQLite tests always run.
"""
import os
import pytest

from dashboard import db

pg = bool(os.environ.get("PG_DSN"))

MODULES = {
    "formulations": {
        "import_path": "dashboard.formulations",
        "fn": "init_formulations_schema",
        "tables": ["formulations", "formulation_items"],
    },
    "inventory": {
        "import_path": "dashboard.inventory",
        "fn": "init_inventory_schema",
        "tables": ["inventory_txns"],
    },
    "ingredients": {
        "import_path": "dashboard.ingredient_catalog",
        "fn": "init_ingredients_schema",
        "tables": ["suppliers", "ingredients", "ingredient_sources"],
    },
    "materials": {
        "import_path": "dashboard.materials_catalog",
        "fn": "init_materials_schema",
        "tables": ["materials", "material_suppliers", "product_suppliers"],
    },
}


def _load(spec):
    import importlib
    mod = importlib.import_module(spec["import_path"])
    return getattr(mod, spec["fn"])


# ---------------------------------------------------------------------------
# Postgres smoke tests — each schema must create cleanly on live Postgres.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
@pytest.mark.parametrize("name", sorted(MODULES))
def test_schema_creates_on_postgres(name, monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    spec = MODULES[name]
    init_fn = _load(spec)
    cx = db.connect("/data/chat_log.db")
    try:
        # formulations/inventory/materials all carry FK REFERENCES into
        # ingredients/suppliers (created by ingredient_catalog); Postgres
        # validates FK targets at CREATE TABLE time (unlike SQLite), and
        # app.py's real init order always brings up ingredients first —
        # mirror that dependency here so each module can still be smoke
        # tested in isolation.
        from dashboard.ingredient_catalog import init_ingredients_schema
        init_ingredients_schema(cx)
        cx.commit()
        for tbl in spec["tables"]:
            cx.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cx.commit()
        init_fn(cx)
        cx.commit()
        for tbl in spec["tables"]:
            row = cx.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema=current_schema() AND table_name=?",
                (tbl,),
            ).fetchone()
            assert row[0] == 1, f"{name}: table {tbl} was not created on postgres"
    finally:
        cx.close()


# ---------------------------------------------------------------------------
# SQLite tests — default backend, must stay byte-identical / unaffected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODULES))
def test_schema_creates_on_sqlite(name, tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    spec = MODULES[name]
    init_fn = _load(spec)
    db_path = str(tmp_path / f"{name}.db")
    cx = db.connect(db_path)
    try:
        init_fn(cx)
        for tbl in spec["tables"]:
            # trivial SELECT ... LIMIT 0 must not raise if the table exists
            cx.execute(f"SELECT * FROM {tbl} LIMIT 0")
    finally:
        cx.close()


# ---------------------------------------------------------------------------
# Batch 3b — 5 more dashboard `init_*_schema(cx)` helpers finishing the
# product/inventory subsystem: purchase_orders, production, shipping,
# sourcing, tracking. Same mechanic as 3a, but each module's cross-module
# FK dependency chain differs, so `deps` lists the OTHER modules' init
# functions (in app.py's real init order) that must run first on Postgres
# (which validates FK targets at CREATE TABLE time, unlike SQLite).
# ---------------------------------------------------------------------------

# name -> (import_path, fn) for every dependency `init_*_schema` a batch-3b
# module's tables carry a FOREIGN KEY into.
DEP_FNS = {
    "ingredients": ("dashboard.ingredient_catalog", "init_ingredients_schema"),
    "materials": ("dashboard.materials_catalog", "init_materials_schema"),
    "formulations": ("dashboard.formulations", "init_formulations_schema"),
}

MODULES_3B = {
    "purchase_orders": {
        "import_path": "dashboard.purchase_orders",
        "fn": "init_purchase_orders_schema",
        "tables": ["purchase_orders", "po_items", "po_receiving"],
        # purchase_orders.supplier_id -> suppliers; po_items.ingredient_id -> ingredients,
        # po_items.material_id -> materials (materials_catalog itself needs suppliers,
        # i.e. ingredients, first) — mirrors app.py's real init order.
        "deps": ["ingredients", "materials"],
    },
    "production": {
        "import_path": "dashboard.production",
        "fn": "init_production_schema",
        "tables": ["production_runs", "production_run_items"],
        # production_runs.formulation_id -> formulations (needs ingredients);
        # production_run_items.ingredient_id -> ingredients, .material_id -> materials.
        "deps": ["ingredients", "formulations", "materials"],
    },
    "shipping": {
        "import_path": "dashboard.shipping",
        "fn": "init_shipping_schema",
        "tables": [
            "bottle_types", "box_capacity", "usps_rates",
            "product_bottle_types", "packing_settings",
        ],
        "deps": [],  # box_capacity FKs bottle_types, both owned by this module
    },
    "sourcing": {
        "import_path": "dashboard.sourcing",
        "fn": "init_sourcing_schema",
        "tables": ["supplier_quotes"],
        # supplier_quotes.supplier_id -> suppliers, .ingredient_id -> ingredients,
        # .applied_source_id -> ingredient_sources — all owned by ingredient_catalog.
        "deps": ["ingredients"],
    },
    "tracking": {
        "import_path": "dashboard.tracking",
        "fn": "init_tracking_schema",
        "tables": ["shipments"],
        "deps": [],  # no FKs at all
    },
}


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
@pytest.mark.parametrize("name", sorted(MODULES_3B))
def test_schema_3b_creates_on_postgres(name, monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    spec = MODULES_3B[name]
    init_fn = _load(spec)
    cx = db.connect("/data/chat_log.db")
    try:
        import importlib
        for dep in spec["deps"]:
            dep_path, dep_fn_name = DEP_FNS[dep]
            dep_fn = getattr(importlib.import_module(dep_path), dep_fn_name)
            dep_fn(cx)
            cx.commit()
        for tbl in spec["tables"]:
            cx.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cx.commit()
        init_fn(cx)
        cx.commit()
        for tbl in spec["tables"]:
            row = cx.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema=current_schema() AND table_name=?",
                (tbl,),
            ).fetchone()
            assert row[0] == 1, f"{name}: table {tbl} was not created on postgres"
    finally:
        cx.close()


@pytest.mark.parametrize("name", sorted(MODULES_3B))
def test_schema_3b_creates_on_sqlite(name, tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    spec = MODULES_3B[name]
    init_fn = _load(spec)
    db_path = str(tmp_path / f"{name}.db")
    cx = db.connect(db_path)
    try:
        import importlib
        for dep in spec["deps"]:
            dep_path, dep_fn_name = DEP_FNS[dep]
            dep_fn = getattr(importlib.import_module(dep_path), dep_fn_name)
            dep_fn(cx)
        init_fn(cx)
        for tbl in spec["tables"]:
            # trivial SELECT ... LIMIT 0 must not raise if the table exists
            cx.execute(f"SELECT * FROM {tbl} LIMIT 0")
    finally:
        cx.close()
