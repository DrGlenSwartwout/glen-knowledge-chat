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
