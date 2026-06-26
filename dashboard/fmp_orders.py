"""FMP client order history — read-only reference import.

One projection schema (4 TEXT tables) built from the already-exported FileMaker
CSVs, used identically for the local lookup and the prod push, so the lookup
logic is the same in both places.

PRIVACY: the projection carries ONLY names, company, email, phone, addresses,
and order data (dates / amounts / line items / status). It deliberately excludes
every clinical field in the FMP client master (diagnose1..3, dob, gender,
doctor, notes, biofield/causal-chain). The fixed column lists below ARE the
privacy boundary.

Ship-to is client-level, not per-order: FMP invoices carry no ship-to FK and
clients_address links to the client only. Addresses are surfaced as
"on file for this client", not per-order.

stdlib-only (csv, json, sqlite3, os) so it is offline-testable.
"""

import csv
import json
import os

_CLIENT_COLS = ["id_pk", "name_first", "name_last", "company", "email",
                "phone_res", "phone_cell", "phone_business"]
_INV_COLS    = ["id_pk", "id_fk_client", "invoice_date", "status",
                "subtotal", "total", "shipping", "outstanding"]
_ITEM_COLS   = ["id_pk", "id_fk_invoice", "id_fk_product", "description",
                "qty", "price", "ext_price"]
_ADDR_COLS   = ["id_pk", "id_fk_client", "type", "street", "city",
                "province", "postal_code", "country"]

# FMP CSV column -> projection column (only where the names differ).
# NOTE: status comes from `closed` (the real status flag, e.g. "Active") — NOT
# `zcRecordStatus` (a FileMaker record-position calc). outstanding comes from
# `zc_overdue_balance` (per-invoice) — NOT `zs_ar_os_total` (a client-level AR
# summary repeated on every row).
_INV_MAP  = {"closed": "status", "zc_invoice_subtotal": "subtotal",
             "zc_invoice_total": "total", "shipping_fee": "shipping",
             "zc_overdue_balance": "outstanding"}
_ITEM_MAP = {"zc_ext_price": "ext_price"}
_ADDR_MAP = {"address_street": "street", "address_city": "city",
             "address_province": "province", "address_postal_code": "postal_code",
             "address_country": "country"}

_TABLES = {
    "fmp_clients": _CLIENT_COLS,
    "fmp_invoices": _INV_COLS,
    "fmp_invoice_items": _ITEM_COLS,
    "fmp_client_addresses": _ADDR_COLS,
}


def ensure_tables(cx):
    for t, cols in _TABLES.items():
        cx.execute(f"CREATE TABLE IF NOT EXISTS {t} ("
                   + ", ".join(f"{c} TEXT" for c in cols) + ")")


def _replace(cx, table, cols, rows):
    cx.execute(f"DROP TABLE IF EXISTS {table}")
    cx.execute(f"CREATE TABLE {table} (" + ", ".join(f"{c} TEXT" for c in cols) + ")")
    ph = ",".join("?" * len(cols))
    cx.executemany(f"INSERT INTO {table} VALUES ({ph})", rows)
    return len(rows)


def _iso_date(s):
    """Normalize an FMP date to sortable ISO 'YYYY-MM-DD'. FMP exports
    'M/D/YYYY'; pass through values already ISO; leave anything unparseable
    as-is (it just sorts to one end)."""
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s  # already ISO
    parts = s.split("/")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        m, d, y = parts
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return s


def _csv_rows(path, cols, colmap):
    """Read `path`, returning row tuples in `cols` order. Each projection col is
    read from its CSV source name (colmap maps the differing names; same-named
    cols are read directly)."""
    src = {v: k for k, v in colmap.items()}  # projcol -> csvcol
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(tuple((r.get(src.get(c, c), "") or "") for c in cols))
    return out


def build_projection_from_csv(cx, export_dir):
    """(Re)build the four projection tables from the FMP CSV export dir.
    Idempotent; returns row counts."""
    export_dir = str(export_dir)
    counts = {}
    counts["clients"]   = _replace(cx, "fmp_clients", _CLIENT_COLS,
                                   _csv_rows(os.path.join(export_dir, "clients.csv"), _CLIENT_COLS, {}))
    inv_rows = _csv_rows(os.path.join(export_dir, "invoices.csv"), _INV_COLS, _INV_MAP)
    _di = _INV_COLS.index("invoice_date")
    inv_rows = [tuple(_iso_date(v) if i == _di else v for i, v in enumerate(r)) for r in inv_rows]
    counts["invoices"]  = _replace(cx, "fmp_invoices", _INV_COLS, inv_rows)
    counts["items"]     = _replace(cx, "fmp_invoice_items", _ITEM_COLS,
                                   _csv_rows(os.path.join(export_dir, "invoice_items.csv"), _ITEM_COLS, _ITEM_MAP))
    counts["addresses"] = _replace(cx, "fmp_client_addresses", _ADDR_COLS,
                                   _csv_rows(os.path.join(export_dir, "clients_address.csv"), _ADDR_COLS, _ADDR_MAP))
    cx.commit()
    return counts


def _clients_where(client_id, email, name):
    if client_id:
        return "id_pk=?", [client_id]
    if email:
        return "lower(email)=lower(?)", [email]
    if name:
        return "(name_first LIKE ? OR name_last LIKE ? OR company LIKE ?)", [f"%{name}%"] * 3
    return "0", []


def client_order_history(cx, *, client_id=None, email=None, name=None):
    """Resolve matching client(s) and return their order history newest-first,
    each order with its line items, plus the client's address(es) on file.
    None-raising; empty list if nothing matches."""
    cx.row_factory = None
    where, args = _clients_where(client_id, email, name)
    clients = cx.execute(
        f"SELECT {','.join(_CLIENT_COLS)} FROM fmp_clients WHERE {where} "
        f"ORDER BY name_last, name_first LIMIT 50", args).fetchall()
    out = []
    for row in clients:
        c = dict(zip(_CLIENT_COLS, row))
        cid = c["id_pk"]
        addrs = [dict(zip(_ADDR_COLS, a)) for a in cx.execute(
            f"SELECT {','.join(_ADDR_COLS)} FROM fmp_client_addresses WHERE id_fk_client=?",
            [cid]).fetchall()]
        invs = cx.execute(
            f"SELECT {','.join(_INV_COLS)} FROM fmp_invoices WHERE id_fk_client=? "
            f"ORDER BY invoice_date DESC", [cid]).fetchall()
        orders = []
        for iv in invs:
            ivd = dict(zip(_INV_COLS, iv))
            items = [dict(zip(_ITEM_COLS, it)) for it in cx.execute(
                f"SELECT {','.join(_ITEM_COLS)} FROM fmp_invoice_items WHERE id_fk_invoice=?",
                [ivd["id_pk"]]).fetchall()]
            orders.append({
                "id": ivd["id_pk"], "date": ivd["invoice_date"], "status": ivd["status"],
                "subtotal": ivd["subtotal"], "total": ivd["total"],
                "shipping": ivd["shipping"], "outstanding": ivd["outstanding"],
                "items": [{"description": i["description"], "qty": i["qty"],
                           "price": i["price"], "ext_price": i["ext_price"],
                           "product_id": i["id_fk_product"]} for i in items],
            })
        out.append({
            "client": {"id": cid, "name": (c["name_first"] + " " + c["name_last"]).strip(),
                       "company": c["company"], "email": c["email"],
                       "phones": [p for p in (c["phone_cell"], c["phone_res"], c["phone_business"]) if p]},
            "addresses": [{k: a[k] for k in ("type", "street", "city", "province", "postal_code", "country")}
                          for a in addrs],
            "orders": orders,
        })
    return out


def to_payload(cx):
    """Dump the four projection tables as a JSON-able dict for the prod push."""
    spec = {"clients": ("fmp_clients", _CLIENT_COLS), "invoices": ("fmp_invoices", _INV_COLS),
            "items": ("fmp_invoice_items", _ITEM_COLS), "addresses": ("fmp_client_addresses", _ADDR_COLS)}
    return {key: [list(r) for r in cx.execute(f"SELECT {','.join(cols)} FROM {tbl}").fetchall()]
            for key, (tbl, cols) in spec.items()}


def ingest_payload(cx, payload):
    """Replace the four projection tables from a to_payload() dict. Idempotent."""
    counts = {}
    counts["clients"]   = _replace(cx, "fmp_clients", _CLIENT_COLS, [tuple(r) for r in payload.get("clients", [])])
    counts["invoices"]  = _replace(cx, "fmp_invoices", _INV_COLS, [tuple(r) for r in payload.get("invoices", [])])
    counts["items"]     = _replace(cx, "fmp_invoice_items", _ITEM_COLS, [tuple(r) for r in payload.get("items", [])])
    counts["addresses"] = _replace(cx, "fmp_client_addresses", _ADDR_COLS, [tuple(r) for r in payload.get("addresses", [])])
    cx.commit()
    return counts


if __name__ == "__main__":
    import sqlite3
    import sys
    db = os.environ.get("LOCAL_DB", "chat_log.db")
    export = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fmp-export/newapp"
    cx = sqlite3.connect(db)
    ensure_tables(cx)
    print(build_projection_from_csv(cx, export))
