# FMP Client Order History — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Bring FMP past client orders into the app as read-only reference, queryable in the web console (orders + line items + client-level addresses on file).

**Architecture:** One projection schema (4 TEXT tables) built from the already-exported CSVs, used identically for the local lookup and the prod push. `dashboard/fmp_orders.py` owns the schema, the CSV→projection builder, and `client_order_history`. A console-gated ingest endpoint replicates the projection into the prod LOG_DB. A console page + read API surface the lookup.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest; vanilla JS console page.

## Global Constraints

- **Projection columns are the privacy boundary.** Carry ONLY: client `id_pk, name_first, name_last, company, email, phone_res, phone_cell, phone_business`; invoice `id_pk, id_fk_client, invoice_date, status, subtotal, total, shipping, outstanding`; item `id_pk, id_fk_invoice, id_fk_product, description, qty, price, ext_price`; address `id_pk, id_fk_client, type, street, city, province, postal_code, country`. NEVER include `diagnose1..3, dob, gender, doctor, notes` or any biofield/clinical field.
- FMP→projection column mapping (invoices): `closed`→status (real status flag e.g. "Active"; NOT `zcRecordStatus`, a record-position calc), `zc_invoice_subtotal`→subtotal, `zc_invoice_total`→total, `shipping_fee`→shipping, `zc_overdue_balance`→outstanding (per-invoice; NOT `zs_ar_os_total`, a client-level AR summary). (items): `zc_ext_price`→ext_price. (addresses): `address_street`→street, `address_city`→city, `address_province`→province, `address_postal_code`→postal_code, `address_country`→country.
- Ship-to is **client-level, not per-order** — surface addresses labeled as such; do not imply a per-order link.
- All projection tables TEXT columns. Builder + ingest are idempotent (DROP/recreate then bulk insert).
- Export dir: `/tmp/fmp-export/newapp/`. Local DB: `chat_log.db`. Prod DB: `LOG_DB`.
- Endpoints `_bos_actor()`-gated (401 otherwise); `?dry_run=1` writes nothing.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`. `dashboard/fmp_orders.py` is offline-importable (stdlib only: `csv`, `sqlite3`, `json`, `os`). `app.py` is NOT → Tasks 2 & 3 verified live.

---

### Task 1: `dashboard/fmp_orders.py` — schema, builder, lookup

**Files:**
- Create: `dashboard/fmp_orders.py`
- Test: `tests/test_fmp_orders.py`

**Interfaces:**
- Produces: `ensure_tables(cx)`, `build_projection_from_csv(cx, export_dir) -> dict`, `client_order_history(cx, *, client_id=None, email=None, name=None) -> list[dict]`, `to_payload(cx) -> dict`.

- [ ] **Step 1: Write failing tests** (`tests/test_fmp_orders.py`):

```python
import csv, sqlite3, pathlib
from dashboard import fmp_orders as fo

def _cx():
    cx = sqlite3.connect(":memory:")
    fo.ensure_tables(cx)
    return cx

def _seed(cx):
    cx.executemany("INSERT INTO fmp_clients (id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business) VALUES (?,?,?,?,?,?,?,?)", [
        ("c1","JoAnn","Cuddigan","Sun Star Organics","joann@sunstarorganics.com","","808-555","" ),
        ("c2","Pam","Schreur","","iamsure@att.net","","",""),
    ])
    cx.executemany("INSERT INTO fmp_invoices (id_pk,id_fk_client,invoice_date,status,subtotal,total,shipping,outstanding) VALUES (?,?,?,?,?,?,?,?)", [
        ("i1","c1","2026-04-01","Closed","100.00","113.00","13.00","0.00"),
        ("i2","c1","2026-01-15","Closed","50.00","50.00","0.00","0.00"),
    ])
    cx.executemany("INSERT INTO fmp_invoice_items (id_pk,id_fk_invoice,id_fk_product,description,qty,price,ext_price) VALUES (?,?,?,?,?,?,?)", [
        ("it1","i1","p1","Lens-Zyme","2","35.00","70.00"),
        ("it2","i1","p2","Lipid Cleanse","1","30.00","30.00"),
        ("it3","i2","p1","Lens-Zyme","1","50.00","50.00"),
    ])
    cx.executemany("INSERT INTO fmp_client_addresses (id_pk,id_fk_client,type,street,city,province,postal_code,country) VALUES (?,?,?,?,?,?,?,?)", [
        ("a1","c1","","123 Farm Rd","Asheville","NC","28801","USA"),
        ("a2","c1","","old 9 Other St","Portland","OR","97214","USA"),
    ])
    cx.commit()

def test_lookup_by_client_id_orders_newest_first_items_grouped():
    cx=_cx(); _seed(cx)
    res = fo.client_order_history(cx, client_id="c1")
    assert len(res)==1
    c=res[0]
    assert c["client"]["email"]=="joann@sunstarorganics.com"
    assert [o["id"] for o in c["orders"]]==["i1","i2"]      # newest first
    assert len(c["orders"][0]["items"])==2                   # i1 has 2 lines
    assert {i["description"] for i in c["orders"][0]["items"]}=={"Lens-Zyme","Lipid Cleanse"}
    assert len(c["addresses"])==2                            # both on file

def test_lookup_by_email_case_insensitive():
    cx=_cx(); _seed(cx)
    res = fo.client_order_history(cx, email="JoAnn@SunStarOrganics.com")
    assert len(res)==1 and res[0]["client"]["id"]=="c1"

def test_lookup_by_name_like_matches_company_and_person():
    cx=_cx(); _seed(cx)
    assert {r["client"]["id"] for r in fo.client_order_history(cx, name="cuddigan")}=={"c1"}
    assert {r["client"]["id"] for r in fo.client_order_history(cx, name="sun star")}=={"c1"}
    assert fo.client_order_history(cx, name="zzz")==[]

def test_build_projection_from_csv(tmp_path):
    d=tmp_path
    (d/"clients.csv").write_text("id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business\nc1,Jo,Cud,SSO,jo@x.com,,,\n")
    (d/"invoices.csv").write_text("id_pk,id_fk_client,invoice_date,zcRecordStatus,zc_invoice_subtotal,zc_invoice_total,shipping_fee,zs_ar_os_total\ni1,c1,2026-04-01,Closed,100,113,13,0\n")
    (d/"invoice_items.csv").write_text("id_pk,id_fk_invoice,id_fk_product,description,qty,price,zc_ext_price\nit1,i1,p1,Lens-Zyme,2,35,70\n")
    (d/"clients_address.csv").write_text("id_pk,id_fk_client,type,address_street,address_city,address_province,address_postal_code,address_country\na1,c1,,123 Farm Rd,Asheville,NC,28801,USA\n")
    cx=sqlite3.connect(":memory:")
    counts = fo.build_projection_from_csv(cx, d)
    assert counts=={"clients":1,"invoices":1,"items":1,"addresses":1}
    res = fo.client_order_history(cx, client_id="c1")
    assert res[0]["orders"][0]["total"]=="113"
    assert res[0]["orders"][0]["items"][0]["ext_price"]=="70"
    assert res[0]["addresses"][0]["city"]=="Asheville"

def test_to_payload_roundtrips():
    cx=_cx(); _seed(cx)
    pay = fo.to_payload(cx)
    assert set(pay)=={"clients","invoices","items","addresses"}
    cx2=sqlite3.connect(":memory:"); fo.ensure_tables(cx2)
    fo.ingest_payload(cx2, pay)
    assert fo.client_order_history(cx2, client_id="c1")[0]["client"]["id"]=="c1"

def test_none_raising_on_empty():
    cx=_cx()
    assert fo.client_order_history(cx, name="anything")==[]
```

- [ ] **Step 2: Run, verify fail** — `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_fmp_orders.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `dashboard/fmp_orders.py`.** Define the four tables + column lists as constants. Key code:

```python
import csv, json, os

_CLIENT_COLS  = ["id_pk","name_first","name_last","company","email","phone_res","phone_cell","phone_business"]
_INV_COLS     = ["id_pk","id_fk_client","invoice_date","status","subtotal","total","shipping","outstanding"]
_ITEM_COLS    = ["id_pk","id_fk_invoice","id_fk_product","description","qty","price","ext_price"]
_ADDR_COLS    = ["id_pk","id_fk_client","type","street","city","province","postal_code","country"]

# FMP CSV column -> projection column (only where they differ)
_INV_MAP  = {"zcRecordStatus":"status","zc_invoice_subtotal":"subtotal","zc_invoice_total":"total","shipping_fee":"shipping","zs_ar_os_total":"outstanding"}
_ITEM_MAP = {"zc_ext_price":"ext_price"}
_ADDR_MAP = {"address_street":"street","address_city":"city","address_province":"province","address_postal_code":"postal_code","address_country":"country"}

_TABLES = {
    "fmp_clients": _CLIENT_COLS, "fmp_invoices": _INV_COLS,
    "fmp_invoice_items": _ITEM_COLS, "fmp_client_addresses": _ADDR_COLS,
}

def ensure_tables(cx):
    for t, cols in _TABLES.items():
        cx.execute(f"CREATE TABLE IF NOT EXISTS {t} (" + ", ".join(f"{c} TEXT" for c in cols) + ")")

def _replace(cx, table, cols, rows):
    cx.execute(f"DROP TABLE IF EXISTS {table}")
    cx.execute(f"CREATE TABLE {table} (" + ", ".join(f"{c} TEXT" for c in cols) + ")")
    ph = ",".join("?"*len(cols))
    cx.executemany(f"INSERT INTO {table} VALUES ({ph})", rows)
    return len(rows)

def _csv_rows(path, cols, colmap):
    """Yield row tuples in `cols` order, reading each projection col from its CSV
    source name (colmap maps differing names; same-name cols read directly)."""
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            src = {v: k for k, v in colmap.items()}  # projcol -> csvcol
            out.append(tuple((r.get(src.get(c, c), "") or "") for c in cols))
    return out

def build_projection_from_csv(cx, export_dir):
    export_dir = str(export_dir)
    counts = {}
    counts["clients"]  = _replace(cx, "fmp_clients", _CLIENT_COLS, _csv_rows(os.path.join(export_dir,"clients.csv"), _CLIENT_COLS, {}))
    counts["invoices"] = _replace(cx, "fmp_invoices", _INV_COLS, _csv_rows(os.path.join(export_dir,"invoices.csv"), _INV_COLS, _INV_MAP))
    counts["items"]    = _replace(cx, "fmp_invoice_items", _ITEM_COLS, _csv_rows(os.path.join(export_dir,"invoice_items.csv"), _ITEM_COLS, _ITEM_MAP))
    counts["addresses"]= _replace(cx, "fmp_client_addresses", _ADDR_COLS, _csv_rows(os.path.join(export_dir,"clients_address.csv"), _ADDR_COLS, _ADDR_MAP))
    cx.commit()
    return counts

def _clients_where(client_id, email, name):
    if client_id: return "id_pk=?", [client_id]
    if email:     return "lower(email)=lower(?)", [email]
    if name:      return "(name_first LIKE ? OR name_last LIKE ? OR company LIKE ?)", [f"%{name}%"]*3
    return "0", []

def client_order_history(cx, *, client_id=None, email=None, name=None):
    cx.row_factory = None
    where, args = _clients_where(client_id, email, name)
    clients = cx.execute(f"SELECT {','.join(_CLIENT_COLS)} FROM fmp_clients WHERE {where} ORDER BY name_last, name_first LIMIT 50", args).fetchall()
    out = []
    for row in clients:
        c = dict(zip(_CLIENT_COLS, row))
        cid = c["id_pk"]
        addrs = [dict(zip(_ADDR_COLS, a)) for a in cx.execute(
            f"SELECT {','.join(_ADDR_COLS)} FROM fmp_client_addresses WHERE id_fk_client=?", [cid]).fetchall()]
        invs = cx.execute(
            f"SELECT {','.join(_INV_COLS)} FROM fmp_invoices WHERE id_fk_client=? ORDER BY invoice_date DESC", [cid]).fetchall()
        orders = []
        for iv in invs:
            ivd = dict(zip(_INV_COLS, iv))
            items = [dict(zip(_ITEM_COLS, it)) for it in cx.execute(
                f"SELECT {','.join(_ITEM_COLS)} FROM fmp_invoice_items WHERE id_fk_invoice=?", [ivd["id_pk"]]).fetchall()]
            orders.append({"id":ivd["id_pk"],"date":ivd["invoice_date"],"status":ivd["status"],
                           "subtotal":ivd["subtotal"],"total":ivd["total"],"shipping":ivd["shipping"],
                           "outstanding":ivd["outstanding"],
                           "items":[{"description":i["description"],"qty":i["qty"],"price":i["price"],
                                     "ext_price":i["ext_price"],"product_id":i["id_fk_product"]} for i in items]})
        out.append({"client":{"id":cid,"name":(c["name_first"]+" "+c["name_last"]).strip(),
                              "company":c["company"],"email":c["email"],
                              "phones":[p for p in (c["phone_cell"],c["phone_res"],c["phone_business"]) if p]},
                    "addresses":[{k:a[k] for k in ("type","street","city","province","postal_code","country")} for a in addrs],
                    "orders":orders})
    return out

def to_payload(cx):
    return {key: [list(r) for r in cx.execute(f"SELECT {','.join(cols)} FROM {tbl}").fetchall()]
            for key, (tbl, cols) in {"clients":("fmp_clients",_CLIENT_COLS),"invoices":("fmp_invoices",_INV_COLS),
                                     "items":("fmp_invoice_items",_ITEM_COLS),"addresses":("fmp_client_addresses",_ADDR_COLS)}.items()}

def ingest_payload(cx, payload):
    counts = {}
    counts["clients"]  = _replace(cx, "fmp_clients", _CLIENT_COLS, [tuple(r) for r in payload.get("clients",[])])
    counts["invoices"] = _replace(cx, "fmp_invoices", _INV_COLS, [tuple(r) for r in payload.get("invoices",[])])
    counts["items"]    = _replace(cx, "fmp_invoice_items", _ITEM_COLS, [tuple(r) for r in payload.get("items",[])])
    counts["addresses"]= _replace(cx, "fmp_client_addresses", _ADDR_COLS, [tuple(r) for r in payload.get("addresses",[])])
    cx.commit()
    return counts

if __name__ == "__main__":
    import sqlite3, sys
    db = os.environ.get("LOCAL_DB", "chat_log.db")
    export = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fmp-export/newapp"
    cx = sqlite3.connect(db); ensure_tables(cx)
    print(build_projection_from_csv(cx, export))
```

- [ ] **Step 4: Run, verify pass** — `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_fmp_orders.py -v` → 6 pass.

- [ ] **Step 5: Build the local projection** — `cd ~/deploy-chat && ~/.venvs/deploy-chat311/bin/python -m dashboard.fmp_orders /tmp/fmp-export/newapp` (writes into local `chat_log.db`). Expect `{'clients':7846,'invoices':425,'items':3048,'addresses':5754}`. Smoke: `client_order_history(cx, name="Cuddigan")` returns Sun Star Organics with orders + addresses.

- [ ] **Step 6: Commit** — `git add dashboard/fmp_orders.py tests/test_fmp_orders.py && git commit -m "feat(fmp-orders): projection schema + CSV builder + client_order_history"`

---

### Task 2: Prod ingest endpoint

**Files:**
- Modify: `app.py` (add near the other `/api/console/*` backfill endpoints, e.g. after `/api/console/backfill-member-people`).

**Interfaces:**
- Consumes: `fmp_orders.ensure_tables`, `fmp_orders.ingest_payload`.

- [ ] **Step 1: Add the route:**

```python
@app.route("/api/console/fmp-orders-ingest", methods=["POST"])
def api_console_fmp_orders_ingest():
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import fmp_orders as _fo
    payload = request.get_json(silent=True) or {}
    if not any(k in payload for k in ("clients","invoices","items","addresses")):
        return jsonify({"ok": False, "error": "empty payload"}), 400
    dry = request.args.get("dry_run", "0") == "1"
    if dry:
        return jsonify({"ok": True, "dry_run": True, "counts": {k: len(payload.get(k, [])) for k in ("clients","invoices","items","addresses")}})
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _fo.ensure_tables(cx)
        counts = _fo.ingest_payload(cx, payload)
    return jsonify({"ok": True, "counts": counts})
```

- [ ] **Step 2: Parse-check + commit** — `~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"` then `git add app.py && git commit -m "feat(fmp-orders): POST /api/console/fmp-orders-ingest (dry_run aware)"`

- [ ] **Step 3: Live verification (post-deploy — record in report).** Build payload locally + POST:
  - dry: `python -c "import sqlite3,json; from dashboard import fmp_orders as fo; cx=sqlite3.connect('chat_log.db'); print(json.dumps(fo.to_payload(cx)))" > /tmp/fmp_payload.json` then `curl -s -X POST "$HOST/api/console/fmp-orders-ingest?dry_run=1" -H "X-Console-Key: $CONSOLE_SECRET" -H "Content-Type: application/json" --data @/tmp/fmp_payload.json` → counts {7846,425,3048,5754}.
  - real (drop `?dry_run=1`) → same counts.

---

### Task 3: Console lookup API + page + nav

**Files:**
- Modify: `app.py` (read endpoint).
- Create: `static/console-client-orders.html`.
- Modify: nav source (`/api/me` nav list or `static/op-nav.js`) — add the "Client Orders" entry (follow the existing console nav pattern).

**Interfaces:**
- Consumes: `fmp_orders.client_order_history`.

- [ ] **Step 1: Add the read endpoint:**

```python
@app.route("/api/console/fmp-orders", methods=["GET"])
def api_console_fmp_orders():
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import fmp_orders as _fo
    client_id = (request.args.get("client_id") or "").strip() or None
    email = (request.args.get("email") or "").strip() or None
    name = (request.args.get("name") or "").strip() or None
    if not (client_id or email or (name and len(name) >= 2)):
        return jsonify({"ok": True, "results": []})
    with sqlite3.connect(LOG_DB) as cx:
        results = _fo.client_order_history(cx, client_id=client_id, email=email, name=name)
    return jsonify({"ok": True, "results": results})
```

- [ ] **Step 2: Create `static/console-client-orders.html`** — console-styled page: a search box (name/email), calls `/api/console/fmp-orders?name=...` (send the console key the same way the other console pages do — match an existing page like `console-payments.html` for auth + look), renders each matched client as a card: header (name · company · email · phones), an "Addresses on file (client-level — not per-order)" block, then orders newest-first (date · status · total · outstanding) each listing line items (qty × description — ext_price). Escape all values. Empty state + "no orders" handling.

- [ ] **Step 3: Wire nav** — add a "Client Orders" entry to the console nav (owner + Rae visible), following the existing `op-nav.js` / `/api/me` nav structure. (Inspect a sibling entry like Payments/Orders and mirror it.)

- [ ] **Step 4: Parse-check + commit** — `~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"` then commit all three files.

- [ ] **Step 5: Live verification (post-deploy — record in report).** After Task 2 ingest: `GET /api/console/fmp-orders?name=Cuddigan` → Sun Star Organics with orders + address. Headless-render `console-client-orders.html` (authed), search "Cuddigan", assert orders + addresses render + zero console errors.

---

## Self-Review

**1. Spec coverage:** projection schema + builder + lookup → Task 1; prod push → Task 2; console API + page + nav → Task 3. Privacy boundary enforced by the fixed `_*_COLS` lists (Task 1) reused by ingest (Task 2) and read (Task 3). ✅
**2. Placeholder scan:** Task 1 is complete code; Tasks 2–3 give exact routes + concrete code + the one inspection step each (match a sibling console page / nav entry) which is the right call in an unfamiliar file. No TBDs. ✅
**3. Type consistency:** `client_order_history`/`to_payload`/`ingest_payload`/`ensure_tables` signatures identical across tasks; payload keys `clients|invoices|items|addresses` consistent in `to_payload` (T1), the ingest endpoint (T2), and `ingest_payload` (T1); projection column lists single-sourced. ✅
