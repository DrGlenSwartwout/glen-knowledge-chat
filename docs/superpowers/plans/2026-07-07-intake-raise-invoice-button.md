# Raise Invoice Button (Biofield Intake) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Raise invoice →" button to the local Biofield Intake page (`/author/<id>`) that creates the client's pickup invoice (Biofield Analysis top line + authored remedies) on the Orders board and returns a print link.

**Architecture:** A new pure module `dashboard/biofield_invoice.py` assembles order lines (name→slug via the prod catalog, Biofield always first) and makes three injected prod calls (fetch catalog, create order, mint print link). A new local route `POST /author/<id>/invoice` wires them, mirroring the existing fee routes. The Fee card gains the button + a result area. Prod does all pricing (courtesy + volume); the local app never computes a price.

**Tech Stack:** Python 3.11 (venv `~/.venvs/deploy-chat311`), Flask (local app), `urllib` for prod calls, `difflib` for fuzzy match, pytest.

## Global Constraints

- **Prod is the single pricing authority** — the local app POSTs `[{slug, qty}]` lines to `/api/orders/manual`; it never sets prices. The `biofield-analysis` line carries no `unit_cents` so the client's `client_prices` courtesy applies server-side.
- **Errors are explicit, never silent** — every failed prod call returns a human-readable `error` string surfaced in the UI (no blank degrade).
- **Network is injected** — `default_fetch_catalog` / `default_create_order` / `default_invoice_link` are the real defaults; `create_app` accepts overrides so tests never hit prod.
- **Copy rules (Glen):** no em dashes, no ALL CAPS, no "Hook:" labels in any user-facing string.
- **Local-app only** — `render_fee_panel` and `biofield_local_app.py` are not imported by prod `app.py`; do not add a feature flag.
- Run tests with `~/.venvs/deploy-chat311/bin/python -m pytest`. Commit after each task.

---

### Task 1: Pure line assembly — `biofield_invoice.py`

**Files:**
- Create: `dashboard/biofield_invoice.py`
- Test: `tests/test_biofield_invoice.py`

**Interfaces:**
- Produces: `BIOFIELD_SLUG = "biofield-analysis"`; `resolve_line_slug(name, catalog) -> str | None`; `build_invoice_lines(client, remedies, catalog) -> {"lines": list, "skipped": list}` where `catalog` is `[{"slug","name"}]`, `lines` is `[{"slug","qty"}]` with `lines[0]` always the Biofield line.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_biofield_invoice.py
from dashboard.biofield_invoice import BIOFIELD_SLUG, resolve_line_slug, build_invoice_lines

CATALOG = [{"slug": "liver-support", "name": "Liver Support"},
           {"slug": "vitality", "name": "Vitality"},
           {"slug": "gastrozyme", "name": "GastroZyme"}]


def test_resolve_exact_case_insensitive():
    assert resolve_line_slug("liver support", CATALOG) == "liver-support"
    assert resolve_line_slug("GASTROZYME", CATALOG) == "gastrozyme"


def test_resolve_fuzzy_close_match():
    # minor spelling drift still resolves (difflib cutoff 0.82)
    assert resolve_line_slug("Gastrozime", CATALOG) == "gastrozyme"


def test_resolve_none_when_no_match():
    assert resolve_line_slug("Green Jasper Gem Elixir", CATALOG) is None
    assert resolve_line_slug("", CATALOG) is None


def test_biofield_is_always_top_line():
    out = build_invoice_lines({"email": "d@x.com"}, ["Liver Support"], CATALOG)
    assert out["lines"][0] == {"slug": BIOFIELD_SLUG, "qty": 1}


def test_resolvable_remedies_become_lines_unresolvable_skipped():
    out = build_invoice_lines({"email": "d@x.com"},
                              ["Liver Support", "Vitality", "Green Jasper Gem Elixir"], CATALOG)
    slugs = [l["slug"] for l in out["lines"]]
    assert slugs == [BIOFIELD_SLUG, "liver-support", "vitality"]   # order preserved, biofield first
    assert out["skipped"] == ["Green Jasper Gem Elixir"]
    assert all(l["qty"] == 1 for l in out["lines"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice.py -q`
Expected: FAIL — `ModuleNotFoundError: dashboard.biofield_invoice`.

- [ ] **Step 3: Write the module**

```python
# dashboard/biofield_invoice.py
"""Assemble a client's pickup invoice from the authored Biofield intake and raise it
on prod (Orders board). Pure line-assembly + injected prod calls; mirrors
biofield_fee.py. Prod is the pricing authority — this module sends [{slug,qty}] only.
"""
import difflib
import json as _json
import os
import urllib.parse
import urllib.request

BIOFIELD_SLUG = "biofield-analysis"


def resolve_line_slug(name, catalog):
    """A remedy NAME -> a sellable catalog slug. Exact (case-insensitive) first,
    then a difflib close match (cutoff 0.82). None when nothing matches."""
    name = (name or "").strip().lower()
    if not name:
        return None
    by_name = {}
    for it in catalog or []:
        n = (it.get("name") or "").strip().lower()
        if n and n not in by_name:
            by_name[n] = it.get("slug")
    if name in by_name:
        return by_name[name] or None
    match = difflib.get_close_matches(name, list(by_name.keys()), n=1, cutoff=0.82)
    return (by_name[match[0]] or None) if match else None


def build_invoice_lines(client, remedies, catalog):
    """Biofield Analysis is always lines[0]; then one qty-1 line per resolvable
    remedy (order preserved). Unresolvable names go to 'skipped', never mispriced."""
    lines = [{"slug": BIOFIELD_SLUG, "qty": 1}]
    skipped = []
    for rname in remedies or []:
        rname = (rname or "").strip()
        if not rname:
            continue
        slug = resolve_line_slug(rname, catalog)
        if slug:
            lines.append({"slug": slug, "qty": 1})
        else:
            skipped.append(rname)
    return {"lines": lines, "skipped": skipped}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_invoice.py tests/test_biofield_invoice.py
git commit -m "feat(intake): biofield_invoice line assembly (biofield-first, name->slug)"
```

---

### Task 2: Injected prod calls — catalog / create / print link

**Files:**
- Modify: `dashboard/biofield_invoice.py` (append)
- Test: `tests/test_biofield_invoice_net.py`

**Interfaces:**
- Consumes: `BIOFIELD_SLUG` (Task 1).
- Produces: `default_fetch_catalog() -> list`; `default_create_order(customer, lines) -> {"ok","order_id","external_ref","total_cents","error"}`; `default_invoice_link(order_id) -> {"ok","print_url","error"}`. All read `CONSOLE_SECRET` + `PUBLIC_BASE_URL` via a private `_console()`.

- [ ] **Step 1: Write the failing tests** (monkeypatch `urlopen`, no network)

```python
# tests/test_biofield_invoice_net.py
import io, json
import dashboard.biofield_invoice as bi


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _fake_urlopen(payload):
    def _open(req, timeout=0):
        return _Resp(json.dumps(payload).encode())
    return _open


def _env(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")


def test_fetch_catalog_parses_products(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"products": [{"slug": "vitality", "name": "Vitality"}]}))
    assert bi.default_fetch_catalog() == [{"slug": "vitality", "name": "Vitality"}]


def test_create_order_ok(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"ok": True, "order_id": 42, "external_ref": "INH-AB",
                                       "totals": {"total_cents": 12345}}))
    out = bi.default_create_order({"name": "D", "email": "d@x.com"}, [{"slug": "biofield-analysis", "qty": 1}])
    assert out["ok"] and out["order_id"] == 42 and out["external_ref"] == "INH-AB"
    assert out["total_cents"] == 12345


def test_create_order_server_not_ok_is_explicit(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"ok": False, "error": "no valid products"}))
    out = bi.default_create_order({"email": "d@x.com"}, [])
    assert out["ok"] is False and out["error"] == "no valid products"


def test_create_order_no_console_is_explicit(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    out = bi.default_create_order({"email": "d@x.com"}, [{"slug": "biofield-analysis", "qty": 1}])
    assert out["ok"] is False and out["error"]


def test_invoice_link_ok(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"ok": True, "link": "https://illtowell.com/invoice/tok?print=1"}))
    out = bi.default_invoice_link(42)
    assert out["ok"] and out["print_url"].endswith("print=1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice_net.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'default_fetch_catalog'`.

- [ ] **Step 3: Append the network defaults**

```python
# dashboard/biofield_invoice.py  (append)

def _console():
    key = os.environ.get("CONSOLE_SECRET")
    if not key:
        return None, None
    base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
    return base, key


def default_fetch_catalog():
    base, key = _console()
    if not base:
        return []
    try:
        url = f"{base}/api/console/biofield-portal/catalog?key=" + urllib.parse.quote(key)
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = _json.loads(r.read().decode() or "{}")
        return resp.get("products") or []
    except Exception:
        return []


def default_create_order(customer, lines):
    base, key = _console()
    if not base:
        return {"ok": False, "error": "No console configured (CONSOLE_SECRET missing)."}
    try:
        body = {"customer": {"name": customer.get("name") or "", "email": customer.get("email") or ""},
                "lines": lines, "pickup": True,
                "invoice_note": "Biofield Analysis and remedies. Payable by check."}
        url = f"{base}/api/orders/manual?key=" + urllib.parse.quote(key)
        req = urllib.request.Request(url, data=_json.dumps(body).encode(), method="POST",
                                     headers={"X-Console-Key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = _json.loads(r.read().decode() or "{}")
        if not resp.get("ok"):
            return {"ok": False, "error": resp.get("error") or "Order creation failed."}
        totals = resp.get("totals") or {}
        return {"ok": True, "order_id": resp.get("order_id"),
                "external_ref": resp.get("external_ref"),
                "total_cents": totals.get("total_cents"), "error": None}
    except Exception:
        return {"ok": False, "error": "Couldn't reach the console to create the order."}


def default_invoice_link(order_id):
    base, key = _console()
    if not base or not order_id:
        return {"ok": False, "error": "link unavailable"}
    try:
        url = (f"{base}/api/console/order/{int(order_id)}/invoice-link?key="
               + urllib.parse.quote(key))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = _json.loads(r.read().decode() or "{}")
        if resp.get("ok") and resp.get("link"):
            return {"ok": True, "print_url": resp["link"], "error": None}
        return {"ok": False, "error": "link unavailable"}
    except Exception:
        return {"ok": False, "error": "link unavailable"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice_net.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_invoice.py tests/test_biofield_invoice_net.py
git commit -m "feat(intake): biofield_invoice prod calls (catalog/create/link), explicit errors"
```

---

### Task 3: Local route `POST /author/<id>/invoice`

**Files:**
- Modify: `biofield_local_app.py` — `create_app(...)` signature + new route after `author_fee_clear` (near line 473)
- Test: `tests/test_biofield_invoice_route.py`

**Interfaces:**
- Consumes: `build_invoice_lines` (Task 1); `default_fetch_catalog` / `default_create_order` / `default_invoice_link` (Task 2).
- Produces: route returning `{"ok": True, "print_url", "external_ref", "added": int, "skipped": list, "total_dollars": str}` on success; `{"ok": False, "error"}` with 400 (no email) or 502 (create failed). `create_app` gains kwargs `invoice_fetch_catalog=None, invoice_create=None, invoice_link=None`.

- [ ] **Step 1: Write the failing tests** (inject fakes; no prod)

```python
# tests/test_biofield_invoice_route.py
import sqlite3
import pytest
from biofield_local_app import create_app
from dashboard.biofield_authoring import init_auth_tables, create_test, update_header, add_chain_row


@pytest.fixture
def client(tmp_path):
    db = str(tmp_path / "t.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx)                        # returns a numeric test id
        update_header(cx, tid, name="Donna Banks", email="d@x.com", date="2026-07-06")
        add_chain_row(cx, tid, layer=1, remedy="Liver Support")
        add_chain_row(cx, tid, layer=2, remedy="Green Jasper Gem Elixir")
    calls = {}

    def fake_catalog():
        return [{"slug": "liver-support", "name": "Liver Support"}]

    def fake_create(customer, lines):
        calls["lines"] = lines
        return {"ok": True, "order_id": 7, "external_ref": "INH-Z", "total_cents": 10000, "error": None}

    def fake_link(oid):
        return {"ok": True, "print_url": "https://x/invoice/tok?print=1", "error": None}

    app = create_app(db_path=db, invoice_fetch_catalog=fake_catalog,
                     invoice_create=fake_create, invoice_link=fake_link)
    app.testing = True
    c = app.test_client()
    c._calls = calls
    c._tid = tid
    return c


def test_invoice_happy_path(client):
    r = client.post(f"/author/{client._tid}/invoice")
    j = r.get_json()
    assert j["ok"] and j["print_url"].endswith("print=1")
    assert j["skipped"] == ["Green Jasper Gem Elixir"]
    # Biofield is the top line; Liver Support resolved; elixir skipped
    assert client._calls["lines"][0]["slug"] == "biofield-analysis"
    assert {"slug": "liver-support", "qty": 1} in client._calls["lines"]


def test_invoice_requires_email(tmp_path):
    db = str(tmp_path / "e.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx)                        # no header -> no email
    app = create_app(db_path=db, invoice_fetch_catalog=lambda: [],
                     invoice_create=lambda *a: {"ok": True},
                     invoice_link=lambda *a: {"ok": True, "print_url": ""})
    app.testing = True
    r = app.test_client().post(f"/author/{tid}/invoice")
    assert r.status_code == 400 and "email" in r.get_json()["error"].lower()


def test_invoice_create_failure_is_502(tmp_path):
    db = str(tmp_path / "f.db")
    with sqlite3.connect(db) as cx:
        init_auth_tables(cx)
        tid = create_test(cx)
        update_header(cx, tid, name="D", email="d@x.com", date="2026-07-06")
        add_chain_row(cx, tid, layer=1, remedy="Liver Support")
    app = create_app(db_path=db, invoice_fetch_catalog=lambda: [],
                     invoice_create=lambda *a: {"ok": False, "error": "Couldn't reach the console."},
                     invoice_link=lambda *a: {"ok": False})
    app.testing = True
    r = app.test_client().post(f"/author/{tid}/invoice")
    assert r.status_code == 502 and "console" in r.get_json()["error"].lower()
```

> Before writing code, confirm the real signatures of `create_test`, `update_header`, and `add_chain_row` in `dashboard/biofield_authoring.py` and adjust the test setup calls to match (keyword names may differ). The behavior asserted stays the same.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice_route.py -q`
Expected: FAIL — route 404 / unexpected kwargs to `create_app`.

- [ ] **Step 3: Wire deps + add the route**

In `biofield_local_app.py`, import the module near the other `dashboard` imports:
```python
from dashboard import biofield_invoice
```
Extend the `create_app` signature (add to the existing kwargs list, ~line 175):
```python
               e4l_db=None, fee_get=None, fee_set=None, fee_clear=None,
               invoice_fetch_catalog=None, invoice_create=None, invoice_link=None):
```
Set defaults near the `fee_get = fee_get or ...` block (~line 194):
```python
    invoice_fetch_catalog = invoice_fetch_catalog or biofield_invoice.default_fetch_catalog
    invoice_create = invoice_create or biofield_invoice.default_create_order
    invoice_link = invoice_link or biofield_invoice.default_invoice_link
```
Add the route immediately after `author_fee_clear` (~line 473):
```python
    @app.route("/author/<test_id>/invoice", methods=["POST"])
    def author_invoice(test_id):
        from dashboard.biofield_fee import cents_to_dollars
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
        client = rep.get("client") or {}
        email = (client.get("email") or "").strip()
        if not email:
            return {"ok": False, "error": "Add a client email in the header first."}, 400
        remedies = [(l.get("remedy") or "").strip()
                    for l in (rep.get("layers") or []) if (l.get("remedy") or "").strip()]
        catalog = invoice_fetch_catalog()
        built = biofield_invoice.build_invoice_lines(client, remedies, catalog)
        created = invoice_create({"name": client.get("name"), "email": email}, built["lines"])
        if not created.get("ok"):
            return {"ok": False, "error": created.get("error") or "Order creation failed."}, 502
        link = invoice_link(created.get("order_id"))
        total = created.get("total_cents")
        return {"ok": True,
                "print_url": link.get("print_url") if link.get("ok") else "",
                "external_ref": created.get("external_ref"),
                "added": len(built["lines"]),
                "skipped": built["skipped"],
                "total_dollars": cents_to_dollars(total) if total is not None else ""}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice_route.py -q`
Expected: PASS (3 passed). If a setup helper signature differs, fix the test's setup calls (not the assertions) and re-run.

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py tests/test_biofield_invoice_route.py
git commit -m "feat(intake): POST /author/<id>/invoice raises the pickup order + print link"
```

---

### Task 4: UI — button + result in the Fee card

**Files:**
- Modify: `dashboard/biofield_report_html.py` — `render_fee_panel` (the `controls` block + `js`, ~lines 843-864)
- Test: `tests/test_biofield_fee_panel.py` (add cases)

**Interfaces:**
- Consumes: the `POST /author/<id>/invoice` route (Task 3).
- Produces: rendered HTML containing a `Raise invoice` button, an `#invresult` container, and a `raiseInvoice()` JS function, present only when `state["has_email"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_biofield_fee_panel.py  (add)
def test_panel_has_raise_invoice_button_when_email():
    html = render_fee_panel(_state(courtesy_cents=10000))
    assert "Raise invoice" in html
    assert "id=invresult" in html
    assert "function raiseInvoice()" in html


def test_panel_no_invoice_button_without_email():
    html = render_fee_panel(_state(email="", has_email=False, available=False))
    assert "Raise invoice" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_fee_panel.py -q`
Expected: FAIL — "Raise invoice" not found.

- [ ] **Step 3: Add the button, result div, and JS**

In `render_fee_panel`, append the button + result container to the `controls` string (after the preset-buttons row, before the closing of `controls`):
```python
        "<div class=btnrow style='margin-top:10px'>"
        "<button class=btn onclick=raiseInvoice()>Raise invoice &rarr;</button>"
        "<span id=invstat class=food></span></div>"
        "<div id=invresult class=food style='margin-top:6px'></div>")
```
(Adjust: the current `controls` ends with `"<span id=feestat class=food></span></div>")` — change that trailing `)` so the new lines are concatenated inside the same string, then close.)

Extend the `js` string (before its closing `"</script>"`) with:
```python
        "function raiseInvoice(){var s=document.getElementById('invstat');"
        "var out=document.getElementById('invresult');s.textContent=' working...';out.textContent='';"
        "fetch(location.pathname.replace(/\\/$/,'')+'/invoice',{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:'{}'})"
        ".then(r=>r.json()).then(j=>{s.textContent='';"
        "if(!j.ok){out.textContent=j.error||'Could not raise the invoice.';return;}"
        "var parts=[];"
        "if(j.print_url){parts.push('<a href=\"'+j.print_url+'\" target=_blank>Print invoice</a>');}"
        "else if(j.external_ref){parts.push('Order '+j.external_ref+' created (open it in Orders to print).');}"
        "parts.push('Added '+j.added+' line(s)'+(j.total_dollars?', total $'+j.total_dollars:''));"
        "if(j.skipped&&j.skipped.length){parts.push('Not added (add manually in Orders): '+j.skipped.join(', '));}"
        "out.innerHTML=parts.join(' &middot; ');})"
        ".catch(function(){s.textContent='';out.textContent='Could not reach the app to raise the invoice.';});}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_fee_panel.py -q`
Expected: PASS (all fee-panel tests, including the 2 new).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_report_html.py tests/test_biofield_fee_panel.py
git commit -m "feat(intake): Raise invoice button + result (print link, added/skipped) in Fee card"
```

---

### Task 5: Full-suite gate + rollout

**Files:** none (verification only)

- [ ] **Step 1: Run the biofield suites**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_invoice.py tests/test_biofield_invoice_net.py tests/test_biofield_invoice_route.py tests/test_biofield_fee_panel.py -q`
Expected: PASS (all).

- [ ] **Step 2: Open PR, merge, roll out to the local app**

```bash
git push -u origin <branch>
gh pr create --title "feat(intake): Raise invoice button on the Biofield form" --body "<summary + spec link>"
# after merge:
cd ~/deploy-chat && git checkout main && git merge --ff-only origin/main
launchctl kickstart -k "gui/$(id -u)/com.glen.biofield-local-server"
```

- [ ] **Step 3: Live-verify on the running app** (render-verify, not just inject)

Drive the real endpoint against a test intake and confirm a real proposed order + print link:
```bash
doppler run -p remedy-match -c prd -- bash -c \
 'curl -s -X POST "http://127.0.0.1:8011/author/a5/invoice?key=$CONSOLE_SECRET" | python3 -m json.tool'
```
Expected: `ok:true`, a `print_url`, Biofield + resolved remedies in `added`, elixirs in `skipped`. Then confirm the order appears on the Orders board and the print link renders `/invoice/<token>?print=1`. Note: this creates a real proposed order for the test client — delete it from the Orders board afterward if it was only a check.

## Notes for the implementer

- `authored_report(cx, tid)["layers"]` is the causal-chain rows; each row's `remedy` is the client-facing remedy name. Iterating layers yields every remedy (a layer may contribute more than one).
- The `biofield-analysis` line MUST NOT carry `unit_cents` — omitting it lets the client's `client_prices` courtesy apply on prod (verified: `effective_unit_cents` returns the courtesy).
- Keep all user-facing strings free of em dashes and ALL CAPS (Glen's copy rules).
