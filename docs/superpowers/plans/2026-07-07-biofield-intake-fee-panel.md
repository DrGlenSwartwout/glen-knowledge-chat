# Biofield Intake fee-panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Fee panel to the local Biofield Intake authoring page (`/author/<id>`, :8011) that sees and sets a client's `biofield-analysis` courtesy price in prod `client_prices`, without invoicing.

**Architecture:** A new pure-plus-thin-network module `dashboard/biofield_fee.py` talks to the existing prod `/api/console/client-prices` endpoint (GET/POST/DELETE) using `CONSOLE_SECRET` + `PUBLIC_BASE_URL` + an `X-Console-Key` header — the same best-effort cross-app pattern already used by `_default_fetch_profile`. `render_author_html` gains a `fee_state` param that renders a `render_fee_panel(state)` card after the client header. `create_app` gains injectable `fee_get`/`fee_set`/`fee_clear` (defaulting to the module functions) so routes and tests use a fake with no network.

**Tech Stack:** Python 3, Flask (the local app is a Flask factory `create_app`), `urllib` (stdlib, matching existing helpers), pytest.

## Global Constraints

- Slug is the constant `biofield-analysis` everywhere (`BIOFIELD_SLUG`).
- Standard charge = `30000` cents ($300); stated value = `99700` cents ($997) — module constants that mirror the prod `biofield-analysis` product (canonical there; update if the product changes). Deliberate simplification from the spec's "live fetch" wording: constants avoid an extra network call for two stable values.
- Prod calls are **best-effort**: any failure (missing `CONSOLE_SECRET`, unreachable, non-2xx) returns a sentinel; the panel renders "pricing unavailable" and never throws into the page.
- `$0` is a valid courtesy (comp); negative amounts are rejected.
- The panel never creates an invoice — console/QBO does that (unchanged).
- No prod-side changes: `/api/console/client-prices`, `dashboard/client_prices.py`, and the pricer are consumed as-is.
- Copy rules (author's standing preference): no em dashes, no ALL CAPS, no "Hook:" labels in any user-facing string.
- Run tests from the repo root. If pytest collection errors on missing env, prefix any command with `doppler run -p remedy-match -c prd --`.

---

### Task 1: Pure fee helpers (`dashboard/biofield_fee.py`)

**Files:**
- Create: `dashboard/biofield_fee.py`
- Test: `tests/test_biofield_fee.py`

**Interfaces:**
- Produces:
  - `BIOFIELD_SLUG = "biofield-analysis"`, `STANDARD_CENTS = 30000`, `VALUE_CENTS = 99700`
  - `dollars_to_cents(v) -> int` (accepts str/int/float dollars; raises `ValueError` on negative or non-numeric)
  - `cents_to_dollars(cents: int) -> str` (`30000 -> "300"`, `69750 -> "697.50"`)
  - `parse_courtesy(resp: dict) -> dict` → `{"courtesy_cents": int|None, "note": str}` (finds `BIOFIELD_SLUG` in `resp["prices"]`)
  - `build_fee_state(email: str, fee_get) -> dict` → the panel state (see Step 5)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_fee.py
import pytest
from dashboard import biofield_fee as bf


def test_constants():
    assert bf.BIOFIELD_SLUG == "biofield-analysis"
    assert bf.STANDARD_CENTS == 30000 and bf.VALUE_CENTS == 99700


@pytest.mark.parametrize("v,cents", [("300", 30000), (100, 10000), (0, 0), ("697.50", 69750), (97.0, 9700)])
def test_dollars_to_cents_ok(v, cents):
    assert bf.dollars_to_cents(v) == cents


@pytest.mark.parametrize("bad", [-1, "-5", "abc", "", None])
def test_dollars_to_cents_rejects(bad):
    with pytest.raises(ValueError):
        bf.dollars_to_cents(bad)


@pytest.mark.parametrize("cents,s", [(30000, "300"), (0, "0"), (69750, "697.50"), (9700, "97")])
def test_cents_to_dollars(cents, s):
    assert bf.cents_to_dollars(cents) == s


def test_parse_courtesy_found():
    resp = {"ok": True, "prices": [{"slug": "biofield-analysis", "price_cents": 10000, "note": "special"},
                                   {"slug": "other", "price_cents": 5}]}
    assert bf.parse_courtesy(resp) == {"courtesy_cents": 10000, "note": "special"}


def test_parse_courtesy_absent():
    assert bf.parse_courtesy({"ok": True, "prices": []}) == {"courtesy_cents": None, "note": ""}
    assert bf.parse_courtesy({}) == {"courtesy_cents": None, "note": ""}


def test_build_fee_state_no_email():
    st = bf.build_fee_state("", fee_get=lambda e: {"available": True, "courtesy_cents": None, "note": ""})
    assert st["has_email"] is False and st["available"] is False
    assert st["standard_cents"] == 30000 and st["value_cents"] == 99700


def test_build_fee_state_with_courtesy():
    st = bf.build_fee_state("j@x.com", fee_get=lambda e: {"available": True, "courtesy_cents": 10000, "note": "special"})
    assert st["has_email"] and st["available"] and st["courtesy_cents"] == 10000 and st["note"] == "special"


def test_build_fee_state_unavailable():
    st = bf.build_fee_state("j@x.com", fee_get=lambda e: {"available": False, "courtesy_cents": None, "note": ""})
    assert st["has_email"] and st["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_biofield_fee.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.biofield_fee'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/biofield_fee.py
"""Client fee (biofield-analysis courtesy) helpers for the local Intake app.

Pure functions + thin best-effort network calls to the prod console
`/api/console/client-prices` endpoint. The panel on /author/<id> uses these to
see and set a client's courtesy price; prod `client_prices` stays the single
source the pricer reads. Never raises into the page — network failures degrade
to an "unavailable" state.
"""
import json as _json
import os
import urllib.parse
import urllib.request

BIOFIELD_SLUG = "biofield-analysis"
STANDARD_CENTS = 30000   # mirrors the prod biofield-analysis product ($300 charge)
VALUE_CENTS = 99700      # mirrors the product's stated value ($997)


def dollars_to_cents(v):
    """Dollars (str/int/float) -> integer cents. Rejects negatives and non-numbers."""
    if v is None:
        raise ValueError("amount required")
    try:
        d = float(str(v).strip())
    except (TypeError, ValueError):
        raise ValueError("invalid amount")
    if d < 0:
        raise ValueError("amount must be non-negative")
    return int(round(d * 100))


def cents_to_dollars(cents):
    """Integer cents -> a display dollar string ('300', '697.50')."""
    cents = int(cents)
    return f"{cents // 100}" if cents % 100 == 0 else f"{cents / 100:.2f}"


def parse_courtesy(resp):
    """Pull the biofield-analysis entry out of a client-prices GET response."""
    for row in (resp or {}).get("prices", []) or []:
        if row.get("slug") == BIOFIELD_SLUG:
            return {"courtesy_cents": int(row["price_cents"]), "note": row.get("note") or ""}
    return {"courtesy_cents": None, "note": ""}


def build_fee_state(email, fee_get):
    """The state the panel renders from. Only calls fee_get when an email exists."""
    email = (email or "").strip()
    state = {"email": email, "has_email": bool(email), "available": False,
             "courtesy_cents": None, "note": "",
             "standard_cents": STANDARD_CENTS, "value_cents": VALUE_CENTS}
    if not email:
        return state
    got = fee_get(email) or {}
    state["available"] = bool(got.get("available"))
    state["courtesy_cents"] = got.get("courtesy_cents")
    state["note"] = got.get("note") or ""
    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_biofield_fee.py -v`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_fee.py tests/test_biofield_fee.py
git commit -m "feat(biofield-fee): pure helpers for courtesy fee state"
```

---

### Task 2: Default prod-call functions (best-effort network)

**Files:**
- Modify: `dashboard/biofield_fee.py`
- Test: `tests/test_biofield_fee.py`

**Interfaces:**
- Produces (all best-effort, never raise):
  - `default_fee_get(email) -> {"available": bool, "courtesy_cents": int|None, "note": str}`
  - `default_fee_set(email, cents, note="") -> {"ok": bool}`
  - `default_fee_clear(email) -> {"ok": bool}`
- Consumes: `parse_courtesy`, `BIOFIELD_SLUG` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_biofield_fee.py

def test_default_fee_get_no_secret_is_unavailable(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    got = bf.default_fee_get("j@x.com")
    assert got == {"available": False, "courtesy_cents": None, "note": ""}


def test_default_fee_get_parses_prod_response(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    class _Resp:
        def read(self): return b'{"ok":true,"prices":[{"slug":"biofield-analysis","price_cents":10000,"note":"special"}]}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(bf.urllib.request, "urlopen", lambda *a, **k: _Resp())
    got = bf.default_fee_get("j@x.com")
    assert got == {"available": True, "courtesy_cents": 10000, "note": "special"}


def test_default_fee_get_network_failure_is_unavailable(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    def boom(*a, **k): raise OSError("prod down")
    monkeypatch.setattr(bf.urllib.request, "urlopen", boom)
    assert bf.default_fee_get("j@x.com")["available"] is False


def test_default_fee_set_no_secret_is_not_ok(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    assert bf.default_fee_set("j@x.com", 10000, "n") == {"ok": False}


def test_default_fee_set_posts_ok(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    seen = {}
    class _Resp:
        def read(self): return b'{"ok":true}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def _open(req, timeout=None):
        seen["method"] = req.get_method(); seen["url"] = req.full_url
        seen["body"] = _json.loads(req.data.decode())
        return _Resp()
    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    out = bf.default_fee_set("j@x.com", 10000, "special")
    assert out == {"ok": True}
    assert seen["method"] == "POST"
    assert seen["body"] == {"email": "j@x.com", "slug": "biofield-analysis",
                            "price_cents": 10000, "note": "special"}


def test_default_fee_clear_deletes(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    seen = {}
    class _Resp:
        def read(self): return b'{"ok":true}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def _open(req, timeout=None):
        seen["method"] = req.get_method(); seen["body"] = _json.loads(req.data.decode())
        return _Resp()
    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    assert bf.default_fee_clear("j@x.com") == {"ok": True}
    assert seen["method"] == "DELETE"
    assert seen["body"] == {"email": "j@x.com", "slug": "biofield-analysis"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_biofield_fee.py -k default_fee -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'default_fee_get'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/biofield_fee.py`:

```python
def _console():
    """(base_url, key) or (None, None) when no CONSOLE_SECRET is set."""
    key = os.environ.get("CONSOLE_SECRET")
    if not key:
        return None, None
    base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
    return base, key


def _request(method, base, key, body=None):
    """POST/DELETE to the client-prices endpoint with a JSON body. Returns parsed JSON."""
    url = f"{base}/api/console/client-prices?key=" + urllib.parse.quote(key)
    data = _json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"X-Console-Key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return _json.loads(r.read().decode() or "{}")


def default_fee_get(email):
    email = (email or "").strip().lower()
    base, key = _console()
    if not email or not base:
        return {"available": False, "courtesy_cents": None, "note": ""}
    try:
        url = (f"{base}/api/console/client-prices?key=" + urllib.parse.quote(key)
               + "&email=" + urllib.parse.quote(email))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = _json.loads(r.read().decode() or "{}")
    except Exception:
        return {"available": False, "courtesy_cents": None, "note": ""}
    out = parse_courtesy(resp)
    out["available"] = True
    return out


def default_fee_set(email, cents, note=""):
    email = (email or "").strip().lower()
    base, key = _console()
    if not email or not base:
        return {"ok": False}
    try:
        resp = _request("POST", base, key, {"email": email, "slug": BIOFIELD_SLUG,
                                            "price_cents": int(cents), "note": note or ""})
        return {"ok": bool(resp.get("ok", True))}
    except Exception:
        return {"ok": False}


def default_fee_clear(email):
    email = (email or "").strip().lower()
    base, key = _console()
    if not email or not base:
        return {"ok": False}
    try:
        resp = _request("DELETE", base, key, {"email": email, "slug": BIOFIELD_SLUG})
        return {"ok": bool(resp.get("ok", True))}
    except Exception:
        return {"ok": False}
```

Delete the throwaway stub `default_fee_get` (the first one); keep only the correct version and the `_console`/`_request` helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_biofield_fee.py -v`
Expected: PASS (all Task 1 + Task 2 cases).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_fee.py tests/test_biofield_fee.py
git commit -m "feat(biofield-fee): best-effort prod client-prices get/set/clear"
```

---

### Task 3: Render the fee panel

**Files:**
- Modify: `dashboard/biofield_report_html.py` (add `render_fee_panel`; add `fee_state=None` param to `render_author_html` at line 821 and inject after `hdr` in the `_page(...)` return)
- Test: `tests/test_biofield_fee_panel.py`

**Interfaces:**
- Consumes: `biofield_fee.cents_to_dollars`, state dict from `build_fee_state` (Task 1).
- Produces: `render_fee_panel(state) -> str`; `render_author_html(report, depth_values=None, transcript="", covered_by_layer=None, narrative="", fee_state=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_fee_panel.py
from dashboard.biofield_report_html import render_fee_panel, render_author_html


def _state(**kw):
    base = {"email": "j@x.com", "has_email": True, "available": True, "courtesy_cents": None,
            "note": "", "standard_cents": 30000, "value_cents": 99700}
    base.update(kw)
    return base


def test_panel_shows_value_and_standard():
    html = render_fee_panel(_state())
    assert "$997" in html and "$300" in html
    assert "Standard" in html                       # no courtesy => standard applies
    assert "/author/" not in html or "fee" in html  # buttons target the fee routes


def test_panel_shows_courtesy_and_clear():
    html = render_fee_panel(_state(courtesy_cents=10000, note="special"))
    assert "$100" in html and "special" in html
    assert "Clear" in html                          # clear-to-standard offered


def test_panel_presets_present():
    html = render_fee_panel(_state())
    assert "697" in html and "100" in html and "0" in html   # preset buttons


def test_panel_no_email_disables():
    html = render_fee_panel(_state(email="", has_email=False, available=False))
    assert "add a client email" in html.lower()


def test_panel_unavailable():
    html = render_fee_panel(_state(available=False))
    assert "unavailable" in html.lower()


def test_author_html_without_fee_state_still_renders():
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": "j@x.com"}, "date": "2026-06-23", "layers": []}
    html = render_author_html(rep)                   # existing callers pass no fee_state
    assert "Edit Biofield Test" in html and "feepanel" not in html


def test_author_html_injects_panel_when_state_given():
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": "j@x.com"}, "date": "2026-06-23", "layers": []}
    html = render_author_html(rep, fee_state=_state())
    assert "feepanel" in html and "$300" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_biofield_fee_panel.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_fee_panel'`.

- [ ] **Step 3: Write minimal implementation**

Add to `dashboard/biofield_report_html.py` (near the other `render_*` helpers). `_e` is the existing HTML-escape helper in this module:

```python
def render_fee_panel(state):
    """The Fee card on the authoring page: value + standard + this client's fee,
    with set/clear controls. Renders from a build_fee_state() dict."""
    from dashboard.biofield_fee import cents_to_dollars
    val = cents_to_dollars(state["value_cents"])
    std = cents_to_dollars(state["standard_cents"])
    head = (f"<div class=card id=feepanel><h2>Fee</h2>"
            f"<p class=sub>Value ${val} &middot; standard charge ${std}. "
            "Set a courtesy below; it applies automatically when you raise the invoice in console. "
            "This panel does not invoice.</p>")
    if not state["has_email"]:
        return head + "<div class=food>Add a client email in the header to set a fee.</div></div>"
    if not state["available"]:
        return head + "<div class=food>Pricing unavailable (couldn't reach console).</div></div>"
    cc = state["courtesy_cents"]
    if cc is None:
        cur = f"<div class=food>This client: <b>Standard: ${std}</b></div>"
        clear = ""
    else:
        note = f" &middot; {_e(state['note'])}" if state["note"] else ""
        cur = f"<div class=food>This client: <b>Courtesy: ${cents_to_dollars(cc)}</b>{note}</div>"
        clear = ("<button class='btn ghost' onclick=clearFee()>Clear &rarr; back to standard</button>")
    controls = (
        "<div class=btnrow style='margin-top:8px'>"
        "<label>Courtesy $</label><input id=fee_amt style='width:100px' inputmode=decimal>"
        "<label>Note</label><input id=fee_note style='width:200px'>"
        "<button class=btn onclick=setFee()>Set courtesy</button>" + clear + "</div>"
        "<div class=btnrow style='margin-top:4px'>"
        "<button class='btn ghost' onclick='preFee(697)'>$697 courtesy</button>"
        "<button class='btn ghost' onclick='preFee(100)'>$100 special</button>"
        "<button class='btn ghost' onclick='preFee(0)'>$0 comp</button>"
        "<span id=feestat class=food></span></div>")
    js = (
        "<script>"
        "function preFee(v){document.getElementById('fee_amt').value=v;}"
        "function feeSwap(u){var b=document.getElementById('fee_amt');"
        "var body=u.indexOf('clear')>-1?{}:{dollars:b.value,note:document.getElementById('fee_note').value};"
        "fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})"
        ".then(r=>r.json()).then(j=>{if(j.html){document.getElementById('feepanel').outerHTML=j.html;}"
        "else{document.getElementById('feestat').textContent=j.error||'error';}});}"
        "function setFee(){feeSwap(location.pathname.replace(/\\/$/,'')+'/fee');}"
        "function clearFee(){feeSwap(location.pathname.replace(/\\/$/,'')+'/fee/clear');}"
        "</script>")
    return head + cur + controls + js + "</div>"
```

Change the `render_author_html` signature (line 821) and injection. Current:

```python
def render_author_html(report, depth_values=None, transcript="", covered_by_layer=None, narrative=""):
```
becomes:
```python
def render_author_html(report, depth_values=None, transcript="", covered_by_layer=None, narrative="", fee_state=None):
```
And in the `return _page("Edit Biofield Test", head + hdr + "<div id=e4lpanel></div>" ...)` assembly, insert the fee panel right after `hdr`. Add this line just before the `return _page(...)`:
```python
    fee_html = render_fee_panel(fee_state) if fee_state else ""
```
and change `head + hdr + "<div id=e4lpanel></div>"` to `head + hdr + fee_html + "<div id=e4lpanel></div>"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_biofield_fee_panel.py tests/test_biofield_author_html.py -v`
Expected: PASS (new panel tests + existing author-html tests still green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_report_html.py tests/test_biofield_fee_panel.py
git commit -m "feat(biofield-fee): render the fee panel on the authoring page"
```

---

### Task 4: Wire routes + page load into the local app

**Files:**
- Modify: `biofield_local_app.py` (add `fee_get`/`fee_set`/`fee_clear` to `create_app` at line 171; default them; fetch `fee_state` in `author_edit` and pass to `render_author_html`; add `POST /author/<id>/fee` and `POST /author/<id>/fee/clear`)
- Test: `tests/test_biofield_fee_routes.py`

**Interfaces:**
- Consumes: `biofield_fee.build_fee_state`, `biofield_fee.dollars_to_cents`, `biofield_fee.default_fee_*` (Tasks 1-2); `render_fee_panel` (Task 3); existing `authored_report`, `render_author_html`.
- Produces routes: `POST /author/<id>/fee` `{dollars, note}` → `{ok, html}`; `POST /author/<id>/fee/clear` → `{ok, html}`; both `{ok:false, error}` + 400 when the test has no email.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_fee_routes.py
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


def _app(db, store):
    """Inject fakes: fee_get reads `store`, fee_set/clear mutate it."""
    def fee_get(email):
        return {"available": True, "courtesy_cents": store.get(email), "note": "special" if store.get(email) else ""}
    def fee_set(email, cents, note):
        store[email] = int(cents); return {"ok": True}
    def fee_clear(email):
        store.pop(email, None); return {"ok": True}
    return create_app(db, fee_get=fee_get, fee_set=fee_set, fee_clear=fee_clear)


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_author_page_shows_fee_panel(tmp_path):
    client = _app(str(tmp_path / "c.db"), {}).test_client()
    tid = _new(client, "j@x.com")
    html = client.get(f"/author/{tid}").get_data(as_text=True)
    assert "feepanel" in html and "$300" in html and "$997" in html


def test_set_fee_route(tmp_path):
    store = {}
    client = _app(str(tmp_path / "c.db"), store).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/fee", json={"dollars": "100", "note": "special"}).get_json()
    assert j["ok"] and store["j@x.com"] == 10000
    assert "Courtesy" in j["html"] and "$100" in j["html"]


def test_clear_fee_route(tmp_path):
    store = {"j@x.com": 10000}
    client = _app(str(tmp_path / "c.db"), store).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/fee/clear", json={}).get_json()
    assert j["ok"] and "j@x.com" not in store and "Standard" in j["html"]


def test_set_fee_no_email_is_400(tmp_path):
    client = _app(str(tmp_path / "c.db"), {}).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    r = client.post(f"/author/{tid}/fee", json={"dollars": "100"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_set_fee_bad_amount_is_400(tmp_path):
    client = _app(str(tmp_path / "c.db"), {}).test_client()
    tid = _new(client, "j@x.com")
    r = client.post(f"/author/{tid}/fee", json={"dollars": "-5"})
    assert r.status_code == 400 and r.get_json()["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_biofield_fee_routes.py -v`
Expected: FAIL — `create_app` has no `fee_get` kwarg (TypeError), and the fee routes 404.

- [ ] **Step 3: Write minimal implementation**

In `biofield_local_app.py`:

(a) Add the import near the other `dashboard` imports at the top:
```python
from dashboard import biofield_fee
from dashboard.biofield_report_html import render_fee_panel
```
(If `render_author_html` etc. are imported from `dashboard.biofield_report_html` already, add `render_fee_panel` to that existing import list instead of a second import.)

(b) Extend the `create_app` signature (line 171-173) — add three kwargs:
```python
def create_app(db_path=DEFAULT_DB, complete=None, tts=None, deepgram_token=None,
               ...,   # keep every existing parameter unchanged
               fetch_runner=None, fetch_profile=None, fetch_recent_comms=None,
               fee_get=None, fee_set=None, fee_clear=None):
```
and default them alongside the other `x = x or _default_x` lines (near line 191):
```python
    fee_get = fee_get or biofield_fee.default_fee_get
    fee_set = fee_set or biofield_fee.default_fee_set
    fee_clear = fee_clear or biofield_fee.default_fee_clear
```

(c) In `author_edit` (line 402-418), compute the fee state and pass it to the render. Replace the final `return Response(render_author_html(...))` with:
```python
            c_email = ((rep.get("client") or {}).get("email") or "").strip()
        fstate = biofield_fee.build_fee_state(c_email, fee_get)
        return Response(render_author_html(rep, dv, transcript, covered_by_layer=covered,
                                           narrative=narrative, fee_state=fstate),
                        mimetype="text/html")
```
(Note the `build_fee_state` call is placed after the `with sqlite3.connect(...)` block closes, so no network happens while holding the db connection.)

(d) Add the two routes next to `author_header` (after line 441):
```python
    @app.route("/author/<test_id>/fee", methods=["POST"])
    def author_fee(test_id):
        d = request.get_json(silent=True) or {}
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"ok": False, "error": "Add a client email in the header first."}, 400
        try:
            cents = biofield_fee.dollars_to_cents(d.get("dollars"))
        except (ValueError, TypeError):
            return {"ok": False, "error": "Enter a valid non-negative amount."}, 400
        fee_set(email, cents, (d.get("note") or "").strip())
        state = biofield_fee.build_fee_state(email, fee_get)
        return {"ok": True, "html": render_fee_panel(state)}

    @app.route("/author/<test_id>/fee/clear", methods=["POST"])
    def author_fee_clear(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"ok": False, "error": "Add a client email in the header first."}, 400
        fee_clear(email)
        state = biofield_fee.build_fee_state(email, fee_get)
        return {"ok": True, "html": render_fee_panel(state)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_biofield_fee_routes.py -v`
Expected: PASS (all 5 cases).

- [ ] **Step 5: Run the full biofield suite (no regressions) + commit**

Run: `python -m pytest tests/ -k biofield_fee -v && python -m pytest tests/test_biofield_author_html.py tests/test_biofield_local_app.py -v`
Expected: PASS.

```bash
git add biofield_local_app.py tests/test_biofield_fee_routes.py
git commit -m "feat(biofield-fee): fee panel routes + page-load state on /author"
```

---

### Task 5: Live render-verify + manual cleanup

**Files:** none (verification only).

- [ ] **Step 1: Restart the local app to load the new code**

The `:8011` app has no hot-reload. Run:
`launchctl kickstart -k gui/$(id -u)/com.glen.biofield-local-server`
(or restart however `biofield_local_app` is being run locally).

- [ ] **Step 2: Drive the panel end-to-end against prod client_prices**

Use a **disposable test-client email** (not a real client). Open an authoring test that has that email set (or create one via the UI and set the header email), then load `http://127.0.0.1:8011/author/<id>?key=$CONSOLE_SECRET`. Confirm the Fee panel shows Value $997 / standard $300 / "Standard". Set a $100 courtesy, confirm the panel swaps to "Courtesy — $100 · <note>". Reload the page, confirm it persists (read back from prod). Click "Clear", confirm it returns to "Standard".

Verify headlessly with the same tool used elsewhere (curl with `?key=` or a headless browser), per [[feedback_render_verify_not_just_inject]] — actually load the page and observe the rendered panel, don't just assert the route JSON.

- [ ] **Step 3: Clean up the test row**

Remove the disposable client's courtesy so no stray pricing remains:
`curl -sS -X DELETE "$PUBLIC_BASE_URL/api/console/client-prices?key=$CONSOLE_SECRET" -H "X-Console-Key: $CONSOLE_SECRET" -H 'Content-Type: application/json' -d '{"email":"<disposable>","slug":"biofield-analysis"}'`
(run via `doppler run -p remedy-match -c prd --` so the env is present). Confirm a follow-up GET shows no biofield-analysis entry for that email.

- [ ] **Step 4: Commit any doc/notes updates** (if the verification surfaced a fix, fold it into the relevant task and re-run its tests).

---

## Self-Review

**Spec coverage:**
- See + set courtesy, writes prod client_prices, no invoice → Tasks 2 (set/clear/get) + 4 (routes). ✓
- Local→prod via CONSOLE_SECRET + PUBLIC_BASE_URL + X-Console-Key → Task 2 `_console`/`_request`. ✓
- Panel shows value/standard/current fee + set/clear + presets → Task 3. ✓
- No-email and prod-unreachable degradation → Task 3 (render) + Task 1/2 (state/best-effort) + Task 4 (400). ✓
- $0 allowed, negatives rejected → Task 1 `dollars_to_cents` + Task 4 400 case. ✓
- Reminder "applied when you invoice in console", never invoices → Task 3 copy. ✓
- Testing incl. real-prod caution + disposable email cleanup → Task 5. ✓
- Out of scope (invoicing, per-SKU/FF pricing, pricer/endpoint changes) → nothing in the plan touches them. ✓

**Placeholder scan:** No TBD/TODO; the one throwaway stub in Task 2 Step 3 is intentional (shows the failure path) and is explicitly replaced by the correct version in the same step.

**Type consistency:** `build_fee_state`, `default_fee_get/set/clear`, `render_fee_panel`, `dollars_to_cents`, `cents_to_dollars`, `parse_courtesy`, and the `fee_get/fee_set/fee_clear` injectables are named identically across Tasks 1-4. State dict keys (`has_email`, `available`, `courtesy_cents`, `note`, `standard_cents`, `value_cents`) match between builder, panel, and tests.
