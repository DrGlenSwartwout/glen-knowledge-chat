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
