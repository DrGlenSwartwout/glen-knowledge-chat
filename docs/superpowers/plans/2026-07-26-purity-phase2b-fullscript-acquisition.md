# Purity Phase 2b — Fullscript-source excipient acquisition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a Fullscript product, automatically acquire its manufacturer "Other Ingredients" from the public product page, extract it under the `verify_quotes` fabrication guard, and run the existing purity screen — so all 24 seed products become screenable without manual entry.

**Architecture:** One DB-free `acquire(product)` orchestrator with a source cascade. Phase 2b implements Source A: fetch the unauthenticated page `https://fullscript.com/catalog/products/<slug>` (browser User-Agent), hand the cleaned page text to a new text-source LLM extractor that returns the target product's Other Ingredients line, and pass that line through the shipped `dashboard/document_extract.verify_quotes` (verbatim-substring, fails-closed). A new console route runs acquire OUTSIDE the DB lock (network + model are slow), then screens and writes the existing `product_ratings` row inside the lock. The Phase-2a manual screen route is untouched and remains a fallback.

**Tech Stack:** Python 3.9 / Flask, `requests` (already a dependency), `anthropic` SDK (already used by `document_extract`), sqlite in dev + Postgres adapter in prod.

## Global Constraints

Every task's requirements implicitly include these:

- **Unrated-never-green.** If Other Ingredients cannot be obtained/verified, the row lands `unrated` (color NULL), never green. `screen_label(actives, None, avoidlist)` returns color `unrated`; pass `None` (not `[]`) whenever data is absent. `[]` means "verified to list nothing" and screens green — never use it for a failure.
- **Fabrication guard binds acquisition.** An extracted ingredient line is accepted only if `verify_quotes` confirms it is a verbatim substring of the fetched source. Any fabrication, wrong-block guess, or malformed model reply fails closed to "not found" → `parsed=None` → `unrated`.
- **Model/network calls run OUTSIDE `_db_lock`.** `acquire()` takes no `cx` and touches no DB. The route calls `acquire()` (network + model) with no lock held, then takes `_db_lock` only for the `record_screen` write. This mirrors `document_extract.call_model_for_extraction`'s contract (see its docstring: holding `_db_lock` across a model call stalls every DB-touching request in the gevent worker).
- **Browser User-Agent on the outbound fetch.** Fullscript is behind Cloudflare, which 403s python-urllib/default UAs. Send a real browser UA (mirror `dashboard/ghl_email.py`'s `_UA`).
- **No new dependencies, no new tables.** Reuse `requests`, `anthropic`, and the existing `product_ratings` table + state machine.
- **No live network or model calls in tests.** Every test injects a `fetch=` and/or `call_model=` stub. No test hits fullscript.com or the Anthropic API.
- **Console routes gate on `_portal_console_ok()`** and return `401 {"error":"unauthorized"}` when it is false, exactly like the sibling purity routes.

---

### Task 1: Fullscript public-page fetcher

**Files:**
- Create: `dashboard/fullscript_ingredients.py`
- Test: `tests/test_fullscript_ingredients.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Uses `requests` (real) via an injectable `fetch` callable so tests never hit the network.
- Produces:
  - `PRODUCT_URL = "https://fullscript.com/catalog/products/{slug}"`
  - `fetch_page_text(slug, *, fetch=None) -> str | None` — returns cleaned, human-readable page text (unicode escapes decoded, HTML tags stripped, whitespace collapsed) on HTTP 200; `None` on blank slug, non-200, or any exception. `fetch` is a callable `(url, headers) -> obj` with `.status_code` and `.text`; defaults to a thin `requests.get` wrapper (timeout 15s).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fullscript_ingredients.py
from dashboard import fullscript_ingredients as fi


class _Resp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


# The Other Ingredients text as it really appears embedded in the page payload
# (unicode-escaped, wrapped in HTML tags) — verified 2026-07-26 against Jarrow
# Magnesium Taurate.
JARROW_RAW = (
    'x\\u003cb\\u003eOther Ingredients:\\u003c/b\\u003e\\u003cbr\\u003e\\n'
    'Capsule (hydroxypropylmethylcellulose), magnesium stearate '
    '(vegetable source) and silicon dioxide.\\u003cbr\\u003e\\u003cp\\u003eKeep out.\\u003c/p\\u003e'
)


def test_fetch_returns_cleaned_text_on_200():
    calls = {}

    def fake_fetch(url, headers):
        calls["url"] = url
        calls["ua"] = headers.get("User-Agent", "")
        return _Resp(200, JARROW_RAW)

    text = fi.fetch_page_text("magnesium-taurate", fetch=fake_fetch)
    assert calls["url"] == "https://fullscript.com/catalog/products/magnesium-taurate"
    assert "Mozilla" in calls["ua"]                       # browser UA sent
    # unicode escapes decoded and tags stripped -> the ingredient line is clean,
    # contiguous, human-readable text
    assert "Other Ingredients:" in text
    assert "magnesium stearate (vegetable source)" in text
    assert "silicon dioxide" in text
    assert "\\u003c" not in text and "<b>" not in text     # cleaned


def test_fetch_returns_none_on_non_200():
    assert fi.fetch_page_text("x", fetch=lambda u, h: _Resp(403, "denied")) is None


def test_fetch_returns_none_on_exception():
    def boom(url, headers):
        raise RuntimeError("network down")
    assert fi.fetch_page_text("x", fetch=boom) is None


def test_fetch_returns_none_on_blank_slug():
    assert fi.fetch_page_text("", fetch=lambda u, h: _Resp(200, "x")) is None
    assert fi.fetch_page_text(None, fetch=lambda u, h: _Resp(200, "x")) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_fullscript_ingredients.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.fullscript_ingredients'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/fullscript_ingredients.py
"""Fetch a Fullscript public catalog product page and return its text.

Source A of the purity-acquisition cascade (spec Section 2). The page at
https://fullscript.com/catalog/products/<slug> is unauthenticated and carries
the manufacturer's Other Ingredients line verbatim; a plain GET with a browser
User-Agent returns it (Fullscript is behind Cloudflare, which 403s default
python UAs -- see reference_cloudflare_ua_ban).

This module ONLY fetches and cleans. It does not parse ingredients and makes no
model call -- extraction + the fabrication guard live in document_extract, and
orchestration in purity_acquire.
"""
import re

PRODUCT_URL = "https://fullscript.com/catalog/products/{slug}"

# A real browser UA; mirrors dashboard/ghl_email.py. Default python-requests /
# urllib UAs are 403'd by Cloudflare.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_TIMEOUT = 15


def _default_fetch(url, headers):
    import requests
    return requests.get(url, headers=headers, timeout=_TIMEOUT)


def _clean(text):
    """Decode \\uXXXX escapes, strip HTML tags, collapse whitespace.

    The ingredient text sits inside a JS/RSC string in the page payload, so it
    arrives with \\u003c-style escapes and surrounding tags. Decoding + tag
    stripping yields readable text in which the Other Ingredients line is a
    single contiguous run -- which is what the extractor quotes and
    verify_quotes checks against. We decode ONLY the \\uXXXX form (not a full
    unicode_escape pass, which would corrupt real multibyte characters)."""
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"<[^>]+>", " ", text)      # drop HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def fetch_page_text(slug, *, fetch=None):
    """Cleaned text of the public product page for `slug`, or None on any
    failure (blank slug, non-200, network/parse exception). Never raises."""
    s = (slug or "").strip()
    if not s:
        return None
    fetch = fetch or _default_fetch
    url = PRODUCT_URL.format(slug=s)
    try:
        resp = fetch(url, {"User-Agent": _UA})
        if getattr(resp, "status_code", None) != 200:
            return None
        return _clean(resp.text or "")
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_fullscript_ingredients.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/fullscript_ingredients.py tests/test_fullscript_ingredients.py
git commit -m "feat(purity): Fullscript public-page fetcher (Source A)"
```

---

### Task 2: Text-source Other-Ingredients extractor (guarded)

**Files:**
- Modify: `dashboard/document_extract.py` (append a text-source extractor beside the image/PDF `call_model_for_extraction`)
- Test: `tests/test_document_extract_other_ingredients.py`

**Interfaces:**
- Consumes: `verify_quotes(items, source_text) -> (kept, dropped)` (already in this module).
- Produces:
  - `extract_other_ingredients(source_text, *, name, brand, sku="", call_model=None) -> str | None` — returns the target product's Other Ingredients line, verified as a verbatim substring of `source_text`; `None` if the model finds nothing, returns a non-substring (fabrication), or replies malformed. `call_model` is a callable `(source_text, name, brand, sku) -> dict` for tests; defaults to a real Anthropic text call.
  - `_OTHER_ING_PROMPT` (module constant) and `_default_call_model_text(source_text, name, brand, sku)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_document_extract_other_ingredients.py
from dashboard import document_extract as dx

SOURCE = (
    "Magnesium Taurate by Jarrow Formulas SKU JAR-MAGTAU90 . "
    "Other Ingredients: Capsule (hydroxypropylmethylcellulose), magnesium "
    "stearate (vegetable source) and silicon dioxide. Keep out of reach."
)


def test_returns_verified_line():
    line = "Capsule (hydroxypropylmethylcellulose), magnesium stearate (vegetable source) and silicon dioxide"

    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": line}

    got = dx.extract_other_ingredients(SOURCE, name="Magnesium Taurate",
                                       brand="Jarrow Formulas", sku="JAR-MAGTAU90",
                                       call_model=fake_model)
    assert got == line


def test_fabricated_line_fails_closed():
    # Model invents ingredients NOT present in the source -> verify_quotes drops
    # it -> None. This is the safety guard; it must bite.
    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": "pharmaceutical glaze and titanium dioxide"}

    assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                        call_model=fake_model) is None


def test_empty_line_returns_none():
    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": ""}

    assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                        call_model=fake_model) is None


def test_malformed_reply_returns_none():
    for bad in [None, [], "not a dict", {"wrong_key": "z"}, {"other_ingredients_line": 123}]:
        assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                            call_model=lambda *a, _b=bad: _b) is None


def test_model_error_returns_none():
    def boom(source_text, name, brand, sku):
        raise RuntimeError("model down")
    assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                        call_model=boom) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_document_extract_other_ingredients.py -v`
Expected: FAIL with `AttributeError: module 'dashboard.document_extract' has no attribute 'extract_other_ingredients'`

- [ ] **Step 3: Write the implementation**

Append to `dashboard/document_extract.py` (uses the existing `verify_quotes`, `_MODEL`, `json`, `os` already imported at the top of that module):

```python
# --- Text-source Other-Ingredients extraction (purity Phase 2b) -----------
# Sibling of call_model_for_extraction, for an HTML/text source rather than an
# image/PDF blob. Same fabrication guard (verify_quotes), same fails-closed
# discipline. The page embeds several products' data, so the model is anchored
# on the target product's name/brand/SKU and instructed to return ONLY that
# product's Other Ingredients; verify_quotes then requires the returned line to
# be a verbatim substring of the fetched page, so a fabricated or wrong-block
# line fails closed to None.

_OTHER_ING_PROMPT = (
    "You are reading the text of a supplement catalog web page. It may contain "
    "several products. Find the ONE product that matches this identity:\n"
    "  name: {name}\n  brand: {brand}\n  sku: {sku}\n\n"
    "Return STRICT JSON: {{\"other_ingredients_line\": \"...\"}} where the value "
    "is the VERBATIM 'Other Ingredients' (or 'Non-Medicinal Ingredients') text "
    "for THAT product ONLY, copied exactly from the page, character for "
    "character. Do NOT include the 'Other Ingredients:' label, other products' "
    "ingredients, the Supplement/Medicinal Facts actives, dosages, or any "
    "warning text. If you cannot find this exact product's other-ingredients "
    "text on the page, return {{\"other_ingredients_line\": \"\"}}. Never guess "
    "or infer an ingredient that is not written on the page. No markdown "
    "fences, no prose outside the JSON."
)


def _default_call_model_text(source_text, name, brand, sku):
    """The real Anthropic text call. Lazy-imported so tests that inject
    call_model never pull the SDK."""
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    prompt = _OTHER_ING_PROMPT.format(name=name or "", brand=brand or "", sku=sku or "")
    resp = cli.messages.create(
        model=_MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": source_text[:120000]},
            {"type": "text", "text": prompt}]}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):                      # tolerate accidental fences
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text)


def extract_other_ingredients(source_text, *, name, brand, sku="", call_model=None):
    """Return the target product's Other Ingredients line, verified verbatim
    against source_text, or None. Fails closed on a model error, a non-dict
    reply, a non-string / empty line, or a line that is not a verbatim
    substring of source_text (fabrication)."""
    call = call_model or _default_call_model_text
    try:
        payload = call(source_text, name, brand, sku)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    line = payload.get("other_ingredients_line")
    if not isinstance(line, str) or not line.strip():
        return None
    kept, _dropped = verify_quotes([{"source_quote": line}], source_text)
    if not kept:
        return None
    return line.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_document_extract_other_ingredients.py -v`
Expected: PASS (5 passed). Note `test_malformed_reply_returns_none` exercises the non-dict, wrong-key, and non-string-line branches; `test_fabricated_line_fails_closed` proves the guard bites.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/document_extract.py tests/test_document_extract_other_ingredients.py
git commit -m "feat(purity): guarded text-source Other-Ingredients extractor"
```

---

### Task 3: `acquire()` orchestrator + ingredient splitter

**Files:**
- Create: `dashboard/purity_acquire.py`
- Test: `tests/test_purity_acquire.py`

**Interfaces:**
- Consumes:
  - `fullscript_ingredients.fetch_page_text(slug, *, fetch=None) -> str | None` (Task 1)
  - `document_extract.extract_other_ingredients(source_text, *, name, brand, sku="", call_model=None) -> str | None` (Task 2)
- Produces:
  - `split_other_ingredients(line) -> list[str]` — splits an Other Ingredients line into items on commas, semicolons, and the word "and"; strips a leading "other ingredients:" / "non-medicinal ingredients:" label and a trailing period; drops empties.
  - `acquire(product, *, fetch=None, call_model=None) -> dict` with keys `{"raw": str, "parsed": list | None, "source": str, "ok": bool}`. `product` is a dict with `product_slug`, `name`, `brand`, and optional `sku`. DB-free; makes no DB call and holds no lock. On any acquisition failure returns `{"raw": "", "parsed": None, "source": "fullscript", "ok": False}` (parsed None → caller screens `unrated`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_acquire.py
from dashboard import purity_acquire as pa

CLEAN_PAGE = (
    "Magnesium Taurate by Jarrow Formulas SKU JAR-MAGTAU90 . "
    "Other Ingredients: Capsule (hydroxypropylmethylcellulose), magnesium "
    "stearate (vegetable source) and silicon dioxide. Keep out of reach."
)
JARROW_LINE = ("Capsule (hydroxypropylmethylcellulose), magnesium stearate "
               "(vegetable source) and silicon dioxide")
PROD = {"product_slug": "magnesium-taurate", "name": "Magnesium Taurate",
        "brand": "Jarrow Formulas", "sku": "JAR-MAGTAU90"}


def test_split_on_commas_and_and():
    items = pa.split_other_ingredients(JARROW_LINE)
    assert items == ["Capsule (hydroxypropylmethylcellulose)",
                     "magnesium stearate (vegetable source)", "silicon dioxide"]


def test_split_strips_label_and_period():
    items = pa.split_other_ingredients("Other Ingredients: silica, gelatin.")
    assert items == ["silica", "gelatin"]


def test_split_empty_line():
    assert pa.split_other_ingredients("") == []


def test_acquire_success_end_to_end():
    res = pa.acquire(PROD,
                     fetch=lambda url, headers: type("R", (), {"status_code": 200, "text": CLEAN_PAGE})(),
                     call_model=lambda s, n, b, k: {"other_ingredients_line": JARROW_LINE})
    assert res["ok"] is True
    assert res["source"] == "fullscript"
    assert res["raw"] == JARROW_LINE
    assert "magnesium stearate (vegetable source)" in res["parsed"]
    assert "silicon dioxide" in res["parsed"]


def test_acquire_fetch_fails_returns_unrated_shape():
    res = pa.acquire(PROD, fetch=lambda url, headers: type("R", (), {"status_code": 404, "text": ""})(),
                     call_model=lambda *a: {"other_ingredients_line": JARROW_LINE})
    assert res == {"raw": "", "parsed": None, "source": "fullscript", "ok": False}


def test_acquire_extract_finds_nothing_returns_unrated_shape():
    res = pa.acquire(PROD,
                     fetch=lambda url, headers: type("R", (), {"status_code": 200, "text": CLEAN_PAGE})(),
                     call_model=lambda s, n, b, k: {"other_ingredients_line": ""})
    assert res["ok"] is False and res["parsed"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_purity_acquire.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.purity_acquire'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/purity_acquire.py
"""Excipient acquisition orchestrator (purity Phase 2b).

acquire(product) resolves a product's Other Ingredients through the source
cascade and returns a uniform {raw, parsed, source, ok} shape. Phase 2b ships
Source A only (the Fullscript public product page). DB-free by contract: it
makes the slow network + model calls and MUST run outside _db_lock; the caller
(the console route) does the DB write.

Failure is never fatal and never optimistic: any miss returns parsed=None, and
the caller screens None -> 'unrated' (never green). The fabrication guard lives
in document_extract.extract_other_ingredients; this module only orchestrates
and splits.
"""
import re

from dashboard import fullscript_ingredients as _fi
from dashboard import document_extract as _dx

_MISS = {"raw": "", "parsed": None, "source": "fullscript", "ok": False}

_LABEL_RE = re.compile(r"^\s*(other|non[- ]?medicinal)\s+ingredients?\s*:\s*", re.I)


def split_other_ingredients(line):
    """Split an Other Ingredients line into individual items on commas,
    semicolons, and the word 'and'. Strips a leading label and a trailing
    period; drops empties. Parenthetical descriptors are kept -- the screen's
    _normalize handles them."""
    s = _LABEL_RE.sub("", line or "").strip().rstrip(".")
    if not s:
        return []
    parts = re.split(r"\s*,\s*|\s*;\s*|\s+and\s+", s)
    return [p.strip() for p in parts if p.strip()]


def acquire(product, *, fetch=None, call_model=None):
    """Acquire Other Ingredients for `product` (dict: product_slug, name,
    brand, optional sku). Returns {raw, parsed, source, ok}. DB-free; holds no
    lock; never raises."""
    slug = (product or {}).get("product_slug")
    text = _fi.fetch_page_text(slug, fetch=fetch)
    if not text:
        return dict(_MISS)
    line = _dx.extract_other_ingredients(
        text, name=product.get("name") or "", brand=product.get("brand") or "",
        sku=product.get("sku") or "", call_model=call_model)
    if not line:
        return dict(_MISS)
    return {"raw": line, "parsed": split_other_ingredients(line),
            "source": "fullscript", "ok": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_purity_acquire.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/purity_acquire.py tests/test_purity_acquire.py
git commit -m "feat(purity): acquire() orchestrator + ingredient splitter"
```

---

### Task 4: Console acquire route

**Files:**
- Modify: `app.py` (add route after `api_console_purity_screen`, ~line 27997)
- Test: `tests/test_purity_routes.py` (append)

**Interfaces:**
- Consumes:
  - `purity_acquire.acquire(product, *, fetch=None, call_model=None) -> {raw, parsed, source, ok}` (Task 3)
  - `purity_screen.screen_label(actives, other_ingredients, avoidlist)` (existing; `None` → unrated)
  - `purity_avoidlist.load_avoidlist()` (existing)
  - `product_ratings.record_screen(cx, key, *, brand, product_name, other_ingredients_raw, other_ingredients_parsed, screen)` (existing)
  - `fullscript.product_by_slug(cx, product_slug) -> dict | None` (existing; for brand/name fallback)
  - `_portal_console_ok()`, `_db_lock`, `db`, `sqlite3`, `jsonify`, `request`, `re` (existing in app.py)
- Produces: `POST /api/console/purity/acquire` → `{"ok": True, "status", "color", "source", "raw"}` on success; `401`/`400` on auth/validation failure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_routes.py  (append; the file already has an authed console
# client fixture and imports app — follow the existing pattern in this file for
# `client` and console auth; the snippet below assumes an authed `client`.)
from dashboard import purity_acquire


def test_acquire_route_screens_red(client, monkeypatch):
    # Fullscript-sourced ingredients include a stearate (red) + silica (yellow)
    monkeypatch.setattr(purity_acquire, "acquire", lambda product, **kw: {
        "raw": "magnesium stearate (vegetable source) and silicon dioxide",
        "parsed": ["magnesium stearate (vegetable source)", "silicon dioxide"],
        "source": "fullscript", "ok": True})
    r = client.post("/api/console/purity/acquire", json={
        "product_key": "jarrow::magnesium-taurate", "product_slug": "magnesium-taurate",
        "brand": "Jarrow Formulas", "product_name": "Magnesium Taurate"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["color"] == "red"            # stearate beats silica
    assert body["status"] == "screened"
    assert body["source"] == "fullscript"


def test_acquire_route_miss_lands_unrated(client, monkeypatch):
    monkeypatch.setattr(purity_acquire, "acquire", lambda product, **kw: {
        "raw": "", "parsed": None, "source": "fullscript", "ok": False})
    r = client.post("/api/console/purity/acquire", json={
        "product_key": "brand::unknown", "product_slug": "unknown"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["color"] is None             # unrated: never green
    assert body["status"] == "unrated"


def test_acquire_route_requires_product_key(client):
    r = client.post("/api/console/purity/acquire", json={"product_slug": "x"})
    assert r.status_code == 400


def test_acquire_route_requires_product_slug(client):
    r = client.post("/api/console/purity/acquire", json={"product_key": "k"})
    assert r.status_code == 400


def test_acquire_route_unauthorized(unauth_client):
    # unauth_client = a test client with no console auth (mirror the existing
    # unauthorized-path test in this file).
    r = unauth_client.post("/api/console/purity/acquire",
                           json={"product_key": "k", "product_slug": "s"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_purity_routes.py -k acquire -v`
Expected: FAIL — the route does not exist yet, so posts return 404 (assertions on 200/400/401 fail).

- [ ] **Step 3: Write the implementation**

Insert into `app.py` immediately after the `api_console_purity_screen` function (after line ~27997, before `@app.route("/api/console/purity/tier2"...)`):

```python
@app.route("/api/console/purity/acquire", methods=["POST"])
def api_console_purity_acquire():
    """Auto-acquire a Fullscript product's Other Ingredients from its public
    catalog page, run the Phase-1 screen, and record the result. This is the
    Phase-2b automated path; /purity/screen (manual entry) remains as a
    fallback.

    acquire() makes the slow network + model calls and is called OUTSIDE
    _db_lock (holding the lock across a model call stalls the gevent worker --
    same contract as document_extract.call_model_for_extraction). The lock is
    taken only for the record_screen write.

    A failed acquisition (parsed None) screens to 'unrated' -- never green --
    per the unrated-never-green safety rule."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import (product_ratings as _pr, purity_screen as _ps,
                           purity_avoidlist as _pa, purity_acquire as _acq,
                           fullscript as _fs)
    b = request.get_json(silent=True) or {}
    key = (b.get("product_key") or "").strip()
    slug = (b.get("product_slug") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    if not slug:
        return jsonify({"error": "product_slug_required"}), 400
    brand = b.get("brand") or ""
    name = b.get("product_name") or ""
    sku = b.get("sku") or ""
    # Fill brand/name from the Fullscript catalog row if the caller omitted
    # them (read-only; needed for the extractor's product-anchoring prompt).
    if not (brand and name):
        with db.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _fs.init_tables(cx)
            frow = _fs.product_by_slug(cx, slug)
        if frow:
            brand = brand or (frow.get("brand") or "")
            name = name or (frow.get("name") or "")
    # Slow calls: OUTSIDE the lock.
    res = _acq.acquire({"product_slug": slug, "name": name, "brand": brand, "sku": sku})
    avoidlist = _pa.load_avoidlist()
    screen = _ps.screen_label(None, res["parsed"], avoidlist)   # parsed None -> unrated
    with _db_lock, db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pr.init_tables(cx)
        _pr.record_screen(cx, key, brand=brand, product_name=name,
                          other_ingredients_raw=res["raw"],
                          other_ingredients_parsed=(res["parsed"] or []),
                          screen=screen)
        row = _pr.get(cx, key)
    return jsonify({"ok": True, "status": row["status"], "color": row["color"],
                    "source": res["source"], "raw": res["raw"]})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_purity_routes.py -k acquire -v`
Expected: PASS (5 passed). If the existing file's console-auth fixtures are named differently than `client`/`unauth_client`, adapt these tests to the fixtures already in `tests/test_purity_routes.py` (read the top of that file first) — do not introduce a new auth mechanism.

- [ ] **Step 5: Run the full purity + extract + fullscript test set**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && python3 -m pytest tests/test_purity_routes.py tests/test_purity_acquire.py tests/test_fullscript_ingredients.py tests/test_document_extract_other_ingredients.py tests/test_product_ratings.py tests/test_purity_screen.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add app.py tests/test_purity_routes.py
git commit -m "feat(purity): console /purity/acquire route (Fullscript auto-screen)"
```

---

## Self-Review

**1. Spec coverage** (Section 2 acquisition, resolved 2026-07-26):
- Source A = Fullscript public page, browser-UA GET → **Task 1**. ✅
- One `acquire(product) → {raw, parsed, source, ok}` interface, source cascade shape → **Task 3**. ✅
- Guarded text-source extractor reusing `verify_quotes`, fails-closed → **Task 2**. ✅
- Synchronous operator-triggered, model call outside `_db_lock`, inline console wiring → **Task 4**. ✅
- Unrated-never-green on any acquisition failure → enforced in Tasks 3 (parsed None) + 4 (`screen_label(None,…)`), tested in both. ✅
- Sources B/C shaped-for, not built → `source` field + `_MISS` shape leave room; not implemented (YAGNI). ✅
- Reader additive badge is Phase 3 — correctly **out of scope** here. ✅

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step has complete code. ✅ One conditional instruction in Task 4 Step 4 (adapt to existing fixture names) is a real integration note, not a placeholder — the implementer must read `tests/test_purity_routes.py`'s existing auth fixtures rather than invent new ones.

**3. Type consistency:**
- `fetch_page_text(slug, *, fetch=None) -> str|None` — same in Task 1 produce, Task 3 consume. ✅
- `extract_other_ingredients(source_text, *, name, brand, sku="", call_model=None) -> str|None` — same in Task 2 produce, Task 3 consume. ✅
- `acquire(product, *, fetch=None, call_model=None) -> {raw,parsed,source,ok}` — same in Task 3 produce, Task 4 consume (route calls `acquire({...})` with defaults). ✅
- `record_screen(...)` kwargs match the real signature read from `dashboard/product_ratings.py`. ✅
- `screen_label(None, parsed, avoidlist)` → `{color,...}`; `record_screen` takes `screen=`. Route passes `res["parsed"]` (list|None) as `other_ingredients` and `(res["parsed"] or [])` as parsed — matches the manual route's own None/[] handling. ✅

Note: the model-reliability limitation (verify_quotes proves the line is *on the page* but not that it belongs to the *target* product vs. a related product) is handled by prompt anchoring on name/brand/SKU plus the human confirm gate — `record_screen` stores `other_ingredients_raw`, and the confirm step (Phase 2a) lets the operator reject a mismatch before it counts. Documented in the spec; no code change needed in this phase.
