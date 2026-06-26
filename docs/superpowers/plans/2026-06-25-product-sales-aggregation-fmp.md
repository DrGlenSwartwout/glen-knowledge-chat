# Product Sales Aggregation (FMP invoices) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `product_sales` aggregate table in `chat_log.db` (units + revenue per product per month) from FMP `invoice_items`, plus a console-gated `GET /api/console/top-products` endpoint and a server-side import, so top-sellers is one query and the data can feed reorder demand.

**Architecture:** A pure-testable module `dashboard/product_sales.py` (table init + aggregation + ranking + idempotent write) consumed by a local import script (`scripts/import_invoices_from_fmp.py`) and a server-side console import endpoint. Product name comes from the line `description` (always present); slug from `products.json` fmp_id→slug when matched. Mirrors the prior FMP-phase module + import-endpoint + console pattern.

**Tech Stack:** Python 3 / Flask, sqlite, pytest. Reuses: the console-key auth pattern (`X-Console-Key`/`CONSOLE_SECRET`), `data/products.json` (carries `fmp_id`), the multipart-import endpoint pattern (`/api/ingredients/import`, `/api/formulations/import`).

## Global Constraints

- Source = FMP `invoice_items` (extracted to `/tmp/fmp-export/newapp/invoice_items.csv`, 3047 rows). Real columns: `id_fk_product` (= products' `fmp_id`), `qty`, `zc_ext_price` (line revenue, **dollars**), `zc_year`, `zc_month`, `invoice_date`, `description`, `fee_name`.
- **Skip lines with a blank `id_fk_product`** (fees / non-product lines).
- `revenue_cents` = round(dollars × 100), stripping `$`/commas; `units` = `qty`.
- **Monthly grain:** `period` = `'YYYY-MM'`; unique on `(product_fmp_id, period, source)`; `source` default `'fmp'`.
- Product **name** = the line `description` (most common per product); **slug** = `products.json` fmp_id→slug, else NULL.
- **Idempotent import:** `DELETE FROM product_sales WHERE source='fmp'` then bulk insert (a re-import after a fresh extract refreshes cleanly).
- Endpoint + import are **console-key gated** (`if CONSOLE_SECRET: key = X-Console-Key or ?key; if key != CONSOLE_SECRET → 401`). No public flag. Additive table.
- Out of scope: app `orders.items_json` fold-in (`source='app'`), reorder-demand wiring, service filtering.
- Local test command (app-importing tests): `mkdir -p /tmp/jshell-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest …` (work in the worktree). Pure-module tests run with plain `python3 -m pytest`.

---

### Task 1: `dashboard/product_sales.py` — table, aggregation, ranking

**Files:**
- Create: `dashboard/product_sales.py`
- Test: `tests/test_product_sales.py`

**Interfaces:**
- Produces:
  - `init_product_sales_table(cx)`
  - `slug_map_from_products_json(path) -> dict`  (fmp_id → slug)
  - `aggregate_rows(rows, slug_for) -> list[dict]`  (`rows` = invoice_items dicts; `slug_for` = `{fmp_id: slug}`; returns rows with keys `product_fmp_id, product_slug, product_name, period, units, revenue_cents, source`)
  - `write_fmp_sales(cx, agg_rows) -> int`  (idempotent: clears `source='fmp'`, inserts; returns count)
  - `top_products(cx, *, year=None, by='revenue', limit=20) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_product_sales.py
import json, sqlite3
import pytest
from dashboard import product_sales as ps


def _rows():
    return [
        {"id_fk_product": "425", "qty": "2", "zc_ext_price": "$138.00", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Microbiome", "fee_name": ""},
        {"id_fk_product": "425", "qty": "1", "zc_ext_price": "69", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/9/2026", "description": "Microbiome", "fee_name": ""},
        {"id_fk_product": "425", "qty": "3", "zc_ext_price": "207", "zc_year": "2025", "zc_month": "12", "invoice_date": "12/1/2025", "description": "Microbiome", "fee_name": ""},
        {"id_fk_product": "", "qty": "1", "zc_ext_price": "10", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Shipping", "fee_name": "Shipping"},  # fee → skip
        {"id_fk_product": "73", "qty": "1", "zc_ext_price": "90", "zc_year": "", "zc_month": "", "invoice_date": "2026-06-15", "description": "Nous Energy", "fee_name": ""},  # period from invoice_date
    ]


def test_aggregate_groups_skips_fees_and_converts():
    agg = ps.aggregate_rows(_rows(), {"425": "microbiome"})
    by = {(r["product_fmp_id"], r["period"]): r for r in agg}
    assert ("", "2026-06") not in by  # fee line skipped
    m = by[("425", "2026-06")]
    assert m["units"] == 3 and m["revenue_cents"] == 20700 and m["product_slug"] == "microbiome" and m["product_name"] == "Microbiome"
    assert by[("425", "2025-12")]["revenue_cents"] == 20700
    assert by[("73", "2026-06")]["units"] == 1 and by[("73", "2026-06")]["product_slug"] is None  # date fallback + unmatched slug


def test_write_idempotent_and_top_products():
    cx = sqlite3.connect(":memory:")
    ps.init_product_sales_table(cx)
    agg = ps.aggregate_rows(_rows(), {"425": "microbiome"})
    n1 = ps.write_fmp_sales(cx, agg)
    n2 = ps.write_fmp_sales(cx, agg)  # re-import
    assert n1 == n2  # idempotent: same row count, no duplicates
    assert cx.execute("SELECT COUNT(*) FROM product_sales").fetchone()[0] == n1
    top = ps.top_products(cx, year=2026, by="revenue", limit=10)
    assert top[0]["product_fmp_id"] == "425" and top[0]["revenue_cents"] == 20700  # 2026 only
    tu = ps.top_products(cx, year=None, by="units", limit=10)
    assert tu[0]["product_fmp_id"] == "425" and tu[0]["units"] == 6  # all-time units
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_product_sales.py -q`
Expected: FAIL — `No module named 'dashboard.product_sales'`.

- [ ] **Step 3: Write `dashboard/product_sales.py`**

```python
# dashboard/product_sales.py
"""Aggregate FMP invoice line items into per-product, per-month sales.
Pure helpers (no Flask). Reads build dicts by hand (row_factory-independent)."""
import json
import re
from collections import defaultdict, Counter

_COLS = ["product_fmp_id", "product_slug", "product_name", "period",
         "units", "revenue_cents", "source"]


def init_product_sales_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS product_sales (
            product_fmp_id TEXT NOT NULL,
            product_slug   TEXT,
            product_name   TEXT,
            period         TEXT NOT NULL,
            units          REAL NOT NULL DEFAULT 0,
            revenue_cents  INTEGER NOT NULL DEFAULT 0,
            source         TEXT NOT NULL DEFAULT 'fmp'
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_product_sales "
               "ON product_sales(product_fmp_id, period, source)")
    cx.commit()


def slug_map_from_products_json(path):
    out = {}
    try:
        prods = (json.load(open(path)) or {}).get("products", {})
        for slug, p in prods.items():
            fid = str((p or {}).get("fmp_id") or "").strip()
            if fid:
                out[fid] = slug
    except Exception:
        pass
    return out


def _money_cents(x):
    s = re.sub(r"[^0-9.\-]", "", str(x or ""))
    try:
        return int(round(float(s) * 100)) if s not in ("", "-", ".") else 0
    except ValueError:
        return 0


def _num(x):
    try:
        return float(str(x or "0").strip() or 0)
    except ValueError:
        return 0.0


def _period(row):
    y = str(row.get("zc_year") or "").strip()
    m = str(row.get("zc_month") or "").strip()
    if y[:4].isdigit() and m.isdigit():
        return f"{y[:4]}-{int(m):02d}"
    d = str(row.get("invoice_date") or "").strip()
    mo = re.match(r"(\d{4})-(\d{1,2})", d)            # YYYY-MM(-DD)
    if mo:
        return f"{mo.group(1)}-{int(mo.group(2)):02d}"
    mo = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", d)  # M/D/YYYY
    if mo:
        return f"{mo.group(3)}-{int(mo.group(1)):02d}"
    return ""


def aggregate_rows(rows, slug_for):
    units = defaultdict(float)
    cents = defaultdict(int)
    names = defaultdict(Counter)
    for r in rows:
        pid = str(r.get("id_fk_product") or "").strip()
        if not pid:               # fee / non-product line
            continue
        period = _period(r)
        if not period:
            continue
        key = (pid, period)
        units[key] += _num(r.get("qty"))
        cents[key] += _money_cents(r.get("zc_ext_price"))
        desc = str(r.get("description") or "").strip().split("\n")[0]
        if desc:
            names[key][desc] += 1
    out = []
    for (pid, period) in units:
        name = names[(pid, period)].most_common(1)[0][0] if names[(pid, period)] else ""
        out.append({"product_fmp_id": pid, "product_slug": slug_for.get(pid),
                    "product_name": name, "period": period,
                    "units": units[(pid, period)], "revenue_cents": cents[(pid, period)],
                    "source": "fmp"})
    return out


def write_fmp_sales(cx, agg_rows):
    cx.execute("DELETE FROM product_sales WHERE source='fmp'")
    cx.executemany(
        "INSERT INTO product_sales(product_fmp_id,product_slug,product_name,period,units,revenue_cents,source) "
        "VALUES (?,?,?,?,?,?,?)",
        [(r["product_fmp_id"], r["product_slug"], r["product_name"], r["period"],
          r["units"], r["revenue_cents"], r.get("source", "fmp")) for r in agg_rows])
    cx.commit()
    return cx.execute("SELECT COUNT(*) FROM product_sales WHERE source='fmp'").fetchone()[0]


def top_products(cx, *, year=None, by="revenue", limit=20):
    order = "rev DESC" if by == "revenue" else "units DESC"
    where, params = "", []
    if year:
        where = "WHERE period LIKE ?"
        params.append(f"{int(year)}-%")
    rows = cx.execute(
        f"SELECT product_fmp_id, MAX(product_name) name, MAX(product_slug) slug, "
        f"SUM(units) units, SUM(revenue_cents) rev FROM product_sales {where} "
        f"GROUP BY product_fmp_id ORDER BY {order} LIMIT ?", params + [int(limit)]).fetchall()
    return [{"product_fmp_id": r[0], "product_name": r[1], "product_slug": r[2],
             "units": r[3], "revenue_cents": r[4]} for r in rows]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_product_sales.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/product_sales.py tests/test_product_sales.py
git commit -m "feat(sales): product_sales module — aggregate/rank/idempotent-write FMP invoice items"
```

---

### Task 2: Import script `scripts/import_invoices_from_fmp.py`

**Files:**
- Create: `scripts/import_invoices_from_fmp.py`
- Test: `tests/test_import_invoices.py`

**Interfaces:**
- Consumes: `product_sales.aggregate_rows/write_fmp_sales/slug_map_from_products_json/init_product_sales_table`.
- Produces: `run_import(items_csv, products_json, db_path, write=False) -> dict` (counts).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_invoices.py
import csv, json, sqlite3
from pathlib import Path
import importlib.util


def _load():
    p = Path(__file__).resolve().parent.parent / "scripts" / "import_invoices_from_fmp.py"
    spec = importlib.util.spec_from_file_location("imp_inv", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_run_import_dry_then_write(tmp_path):
    items = tmp_path / "invoice_items.csv"
    with open(items, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id_fk_product", "qty", "zc_ext_price", "zc_year", "zc_month", "invoice_date", "description", "fee_name"])
        w.writeheader()
        w.writerow({"id_fk_product": "425", "qty": "2", "zc_ext_price": "138", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Microbiome", "fee_name": ""})
        w.writerow({"id_fk_product": "", "qty": "1", "zc_ext_price": "10", "zc_year": "2026", "zc_month": "6", "invoice_date": "6/3/2026", "description": "Shipping", "fee_name": "Shipping"})
    pj = tmp_path / "products.json"
    pj.write_text(json.dumps({"products": {"microbiome": {"fmp_id": "425"}}}))
    db = tmp_path / "chat_log.db"
    mod = _load()
    dry = mod.run_import(str(items), str(pj), str(db), write=False)
    assert dry["product_rows"] == 1 and dry["written"] == 0
    res = mod.run_import(str(items), str(pj), str(db), write=True)
    assert res["written"] == 1
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT product_slug, units, revenue_cents FROM product_sales").fetchone()
    assert row == ("microbiome", 2.0, 13800)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_import_invoices.py -q`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Write `scripts/import_invoices_from_fmp.py`**

```python
#!/usr/bin/env python3
"""Import FMP invoice_items.csv into the product_sales table (idempotent).
Usage: python3 scripts/import_invoices_from_fmp.py --items /tmp/fmp-export/newapp/invoice_items.csv \
         --products data/products.json --db chat_log.db [--write]"""
import argparse, csv, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
csv.field_size_limit(sys.maxsize)
from dashboard import product_sales as ps


def run_import(items_csv, products_json, db_path, write=False):
    rows = list(csv.DictReader(open(items_csv)))
    slug_for = ps.slug_map_from_products_json(products_json)
    agg = ps.aggregate_rows(rows, slug_for)
    out = {"line_items": len(rows), "product_rows": len(agg), "written": 0}
    if write:
        with sqlite3.connect(db_path) as cx:
            ps.init_product_sales_table(cx)
            out["written"] = ps.write_fmp_sales(cx, agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="/tmp/fmp-export/newapp/invoice_items.csv")
    ap.add_argument("--products", default="data/products.json")
    ap.add_argument("--db", default="chat_log.db")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    res = run_import(a.items, a.products, a.db, write=a.write)
    print(res)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_import_invoices.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/import_invoices_from_fmp.py tests/test_import_invoices.py
git commit -m "feat(sales): import_invoices_from_fmp script (dry-run/write, idempotent)"
```

---

### Task 3: Endpoints + console view

**Files:**
- Modify: `app.py` (`GET /api/console/top-products`, `POST /api/console/sales/import`, a Top-Products console section)
- Test: `tests/test_top_products_api.py`

**Interfaces:**
- Consumes: `product_sales.top_products/init_product_sales_table/aggregate_rows/write_fmp_sales/slug_map_from_products_json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_top_products_api.py
import sqlite3, pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    from dashboard import product_sales as ps
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ps.init_product_sales_table(cx)
        ps.write_fmp_sales(cx, [
            {"product_fmp_id": "425", "product_slug": "microbiome", "product_name": "Microbiome",
             "period": "2026-06", "units": 63, "revenue_cents": 434000, "source": "fmp"},
            {"product_fmp_id": "73", "product_slug": None, "product_name": "Nous Energy",
             "period": "2025-06", "units": 10, "revenue_cents": 52000, "source": "fmp"},
        ])
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_top_products_authed_year_filter(client):
    r = client.get("/api/console/top-products?year=2026&limit=5", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    items = r.get_json()["products"]
    assert items and items[0]["product_fmp_id"] == "425" and len(items) == 1  # 2026 only


def test_top_products_requires_auth(client):
    assert client.get("/api/console/top-products").status_code in (401, 403)
```

- [ ] **Step 2: Run to verify it fails**

Run: `mkdir -p /tmp/jshell-test && cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_top_products_api.py -q`
Expected: FAIL — routes 404.

- [ ] **Step 3: Add the routes in `app.py`** (place near other `/api/console/*` routes; copy the console-auth guard verbatim from a neighbor — grep `X-Console-Key` for the exact pattern):

```python
@app.route("/api/console/top-products", methods=["GET"])
def api_console_top_products():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    year = request.args.get("year")
    by = request.args.get("by", "revenue")
    limit = int(request.args.get("limit", 20))
    from dashboard import product_sales as _ps
    with sqlite3.connect(LOG_DB) as cx:
        _ps.init_product_sales_table(cx)
        items = _ps.top_products(cx, year=year, by=by, limit=limit)
    return jsonify({"products": items})


@app.route("/api/console/sales/import", methods=["POST"])
def api_console_sales_import():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    f = request.files.get("invoice_items")
    if not f:
        return jsonify({"ok": False, "error": "invoice_items CSV required"}), 400
    import csv as _csv, io as _io
    from dashboard import product_sales as _ps
    _csv.field_size_limit(2**31 - 1)
    rows = list(_csv.DictReader(_io.StringIO(f.read().decode("utf-8", "replace"))))
    slug_for = _ps.slug_map_from_products_json(str(STATIC.parent / "data" / "products.json"))
    agg = _ps.aggregate_rows(rows, slug_for)
    write = (request.form.get("write", "") or "").strip().lower() in ("1", "true", "yes", "on")
    written = 0
    if write:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _ps.init_product_sales_table(cx)
            written = _ps.write_fmp_sales(cx, agg)
    return jsonify({"ok": True, "line_items": len(rows), "product_rows": len(agg), "written": written})
```

> `STATIC.parent / "data" / "products.json"` — confirm the repo's products.json path (it's `data/products.json` at repo root). Adjust the path expression so it resolves to that file (e.g. `Path(__file__).parent / "data" / "products.json"`).

- [ ] **Step 4: Add a minimal Top-Products console view** — add a read-only section to the existing business-ops console (grep `@app.route("/console/finance"` for the pattern; add a sibling `/console/top-products` route serving a small static page, OR add a section to an existing console page that fetches `/api/console/top-products`). A static `static/console-top-products.html` that fetches the endpoint and renders a ranked table + a year selector + the import file control is sufficient. (Manual/visual QA — no automated test for the HTML.)

- [ ] **Step 5: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_top_products_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-top-products.html tests/test_top_products_api.py
git commit -m "feat(sales): /api/console/top-products + sales/import endpoints + console view"
```

---

### Task 4: Integration smoke + real-data dry-run + PR

**Files:** none (verification only).

- [ ] **Step 1: Full new-suite green**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_product_sales.py tests/test_import_invoices.py tests/test_top_products_api.py -q`
Expected: all pass.

- [ ] **Step 2: Real-data dry-run** (validates against the actual extract)

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 scripts/import_invoices_from_fmp.py --items /tmp/fmp-export/newapp/invoice_items.csv --products data/products.json --db /tmp/jshell-test/sales.db --write`
Expected: prints a dict with `line_items: 3047`, a `product_rows` count (a few hundred), and `written` == `product_rows`. Then `sqlite3 /tmp/jshell-test/sales.db "SELECT product_name, SUM(revenue_cents)/100 FROM product_sales WHERE period LIKE '2026-%' GROUP BY product_fmp_id ORDER BY 2 DESC LIMIT 5"` should show Biofield Analysis / WholOmega / Microbiome at the top (matches the controller's earlier preview).

- [ ] **Step 3: Open the PR** (do not push/PR from a task during subagent execution — the controller opens it after the final review)

---

## Self-Review

**Spec coverage:**
- `product_sales` table (monthly grain, unique key) → Task 1 (`init_product_sales_table`). ✓
- Aggregation: group by (product, period), sum units + revenue_cents, skip fees, period from `zc_year`+`zc_month` w/ `invoice_date` fallback, cents conversion → Task 1 (`aggregate_rows`). ✓
- Name from `description`, slug from products.json fmp_id → Task 1 (`aggregate_rows` + `slug_map_from_products_json`). ✓
- Idempotent FMP-slice rebuild → Task 1 (`write_fmp_sales`). ✓
- `top_products` ranking (year filter, revenue/units) → Task 1. ✓
- Import script (dry-run/write) → Task 2. ✓
- `GET /api/console/top-products` + `POST /api/console/sales/import` (console-gated) + console view → Task 3. ✓
- Real-data validation → Task 4. ✓
- Out-of-scope (app fold-in, reorder wiring, service filtering) → not built. ✓

**Placeholder scan:** none — concrete code throughout. Two implementer-resolved spots: the products.json path expression in Task 3 Step 3 (named, with the exact target), and the console view in Task 3 Step 4 (the pattern + the contract; HTML is manual-QA).

**Type consistency:** `aggregate_rows`/`write_fmp_sales`/`top_products`/`slug_map_from_products_json` signatures match across Tasks 1–3; `run_import(items_csv, products_json, db_path, write)` consistent Task 2; the endpoint returns `{products:[…]}` consumed by the test + console view.
