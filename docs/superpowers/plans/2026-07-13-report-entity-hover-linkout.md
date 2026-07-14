# Report Entity Hover + Link-Out — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give clinical entities (stress patterns, remedies, functions, ingredients) on the client portal report and product/ingredient pages a hover/tap pop-up with basic info and a click-through to their full page in a new tab.

**Architecture:** One shared static frontend component (`static/entity-ref.js` + `static/entity-ref.css`) renders a single reusable popover for any element marked `.entity-ref`, driven entirely by `data-*` attributes. A pure server-side resolver (`dashboard/entity_refs.py`) turns an entity name into a `{name, info, href}` record; each surface's existing data path calls it and emits the markup. Entities with info but no destination are pop-up-only; entities with neither render as plain text.

**Tech Stack:** Python 3 / Flask (backend, `dashboard/*.py`, `app.py`), vanilla JS + inline HTML templates (`static/*.html`), SQLite (`biofield_remedy_meanings`, `topic_pages`), pytest.

## Global Constraints

- **No new JS libraries.** The popover is vanilla JS, following the existing delegated-listener pattern in `client-portal.html` (`wirePatternDetails`).
- **Escaping discipline:** `data-info`/`data-name` are HTML-attribute-escaped when written into markup and re-escaped when injected into the popover (never trust as raw HTML) — same rule the current `wirePatternDetails` follows.
- **New tabs use `rel="noopener"`** (and `target="_blank"`).
- **Gated content must never leak:** remedy info/href is only attached when the report is already unblurred (`show == True` in `portal_view._assemble_biofield`). Never resolve or emit remedy data for a blurred report.
- **No empty pop-ups:** if an entity's `info` is empty, render it as plain text (no `.entity-ref`), never an entity-ref with an empty body.
- **Never a wrong link:** a remedy/function href is emitted only on a positive existence check (a real product slug / an approved topic page). On any ambiguity, omit `href` (pop-up only).
- **Render-verify house rule:** every frontend task ends by driving the page in headless Chrome and observing behavior, not just asserting the payload.
- **deploy-chat tests need Doppler:** run app-importing / DB tests as `doppler run -p remedy-match -c dev -- python3 -m pytest <path>` (bare pytest silently skips or floods live email). Pure-helper tests that import only `dashboard.entity_refs` and stub `cx` do not need Doppler.
- **Structures are out of scope** this slice (no description text, no page) — leave them as plain text.

---

### Task 1: `entity_refs` resolver + unit tests

**Files:**
- Create: `dashboard/entity_refs.py`
- Test: `tests/test_entity_refs.py`

**Interfaces:**
- Consumes: `dashboard.biofield_meanings.get_map(cx) -> {slug: meaning}`; `dashboard.biofield_authoring.resolve_remedy_name(cx, spoken) -> str`; `dashboard.topic_pages.get_page(cx, slug) -> dict|None`; `dashboard.ingredients.slugify(name) -> str`.
- Produces:
  - `pattern_ref(name: str, description: str) -> {"name": str, "info": str, "href": None}`
  - `remedy_ref(cx, spoken: str, product_exists=None) -> {"name": str, "info": str, "href": str|None}` where `product_exists` is an optional `Callable[[str], bool]`; when `None`, `href` is always `None` (safe default).
  - `function_ref(cx, title: str) -> {"name": str, "info": str, "href": str|None}`
  - `ingredient_ref(cx, name: str, slug: str, page_getter=None) -> {"name": str, "info": str, "href": str|None}` where `page_getter` is an optional `Callable[[str], dict|None]` returning an ingredient page dict; when `None`, uses `dashboard.ingredient_pages.get_page(cx, slug)`.
  - `clip(text, sentences=2, cap=280) -> str` (exported helper, used by callers that already hold text).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_entity_refs.py
from dashboard import entity_refs as er


def test_clip_keeps_n_sentences_and_caps():
    assert er.clip("One. Two. Three.", sentences=2) == "One. Two."
    long = "x" * 400
    out = er.clip(long, sentences=1, cap=280)
    assert len(out) == 281 and out.endswith("…")


def test_pattern_ref_is_popup_only():
    r = er.pattern_ref("Heavy Metals", "Accumulated metals stress detoxification. More detail. And more.")
    assert r["name"] == "Heavy Metals"
    assert r["info"].startswith("Accumulated metals stress")
    assert r["href"] is None


def test_pattern_ref_blank_description_gives_empty_info():
    assert er.pattern_ref("ER Stress", "")["info"] == ""


class _FakeCx:
    """Minimal stand-in; entity_refs only calls the wrapped module functions,
    which we monkeypatch, so this object is never queried directly."""


def test_remedy_ref_href_only_when_product_exists(monkeypatch):
    monkeypatch.setattr(er._ba, "resolve_remedy_name", lambda cx, s: "Terrain Restore")
    monkeypatch.setattr(er._bm, "get_map", lambda cx: {"terrain-restore": "Rebuilds terrain. Second sentence. Third."})
    r = er.remedy_ref(_FakeCx(), "terain restore", product_exists=lambda slug: slug == "terrain-restore")
    assert r["href"] == "/begin/product/terrain-restore"
    assert r["info"] == "Rebuilds terrain. Second sentence."


def test_remedy_ref_no_product_exists_is_popup_only(monkeypatch):
    monkeypatch.setattr(er._ba, "resolve_remedy_name", lambda cx, s: "Mystery Remedy")
    monkeypatch.setattr(er._bm, "get_map", lambda cx: {"mystery-remedy": "A meaning."})
    r = er.remedy_ref(_FakeCx(), "mystery remedy")  # product_exists=None
    assert r["href"] is None
    assert r["info"] == "A meaning."


def test_remedy_ref_blank_name():
    assert er.remedy_ref(_FakeCx(), "")["href"] is None


def test_function_ref_uses_topic_page_and_links(monkeypatch):
    page = {"slug": "detoxification", "kind": "function", "state": "approved",
            "content_json": {"summary": "How the body clears toxins. Extra sentence here."}}
    monkeypatch.setattr(er._tp, "get_page", lambda cx, slug: page if slug == "detoxification" else None)
    r = er.function_ref(_FakeCx(), "Detoxification")
    assert r["href"] == "/learn/detoxification"
    assert r["info"].startswith("How the body clears toxins")


def test_function_ref_unapproved_or_missing_is_plain(monkeypatch):
    monkeypatch.setattr(er._tp, "get_page", lambda cx, slug: None)
    r = er.function_ref(_FakeCx(), "Nonexistent")
    assert r["href"] is None and r["info"] == ""


def test_ingredient_ref_pulls_info_and_links(monkeypatch):
    getter = lambda slug: {"content_json": {"what": "A magnesium form that crosses into the brain."}}
    r = er.ingredient_ref(_FakeCx(), "Magnesium L-Threonate", "magnesium-l-threonate", page_getter=getter)
    assert r["href"] == "/begin/ingredient/magnesium-l-threonate"
    assert r["info"].startswith("A magnesium form")


def test_ingredient_ref_no_page_is_plain():
    r = er.ingredient_ref(_FakeCx(), "Obscure Herb", "obscure-herb", page_getter=lambda slug: None)
    assert r["href"] is None and r["info"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_entity_refs.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'dashboard.entity_refs'`.

- [ ] **Step 3: Implement `dashboard/entity_refs.py`**

```python
"""Resolve a clinical entity (stress pattern, remedy, function, ingredient) to a
small {name, info, href} record used by the shared entity-ref hover component.

All functions are pure/wrapped and never raise into callers: on any miss they
return info="" and/or href=None so the frontend degrades to plain text. Gating
(a remedy is only resolved for an unblurred report) is the CALLER's job — this
module has no notion of payment state."""
import json
import re

from dashboard import biofield_meanings as _bm
from dashboard import biofield_authoring as _ba
from dashboard import topic_pages as _tp
from dashboard import ingredient_pages as _ip
from dashboard.ingredients import slugify as _slugify


def clip(text, sentences=2, cap=280):
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:sentences]).strip()
    if len(out) > cap:
        out = out[:cap].rstrip() + "…"
    return out


def _first_text(obj):
    """First meaningful string in a page's content, shape-robust. Prefers named
    summary-ish keys, then any nested string."""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for k in ("summary", "intro", "overview", "what", "what_it_is",
                  "description", "body", "text"):
            t = _first_text(obj.get(k))
            if t:
                return t
        for v in obj.values():
            t = _first_text(v)
            if t:
                return t
    if isinstance(obj, list):
        for v in obj:
            t = _first_text(v)
            if t:
                return t
    return ""


def _page_content(page):
    """Return the parsed content object from a page dict whose content_json may be
    a dict already or a JSON string."""
    if not isinstance(page, dict):
        return {}
    raw = page.get("content_json")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"text": raw}
    return raw or {}


def pattern_ref(name, description):
    """Stress pattern: pop-up only — no detail page exists yet."""
    return {"name": (name or "").strip(), "info": clip(description, sentences=3), "href": None}


def remedy_ref(cx, spoken, product_exists=None):
    """Remedy -> product page + curated meaning. href only when the resolved name
    maps to a slug that product_exists() confirms; else pop-up only. product_exists
    is required to emit any href (None => always pop-up only, the safe default)."""
    name = (spoken or "").strip()
    if not name:
        return {"name": name, "info": "", "href": None}
    try:
        resolved = _ba.resolve_remedy_name(cx, name) or name
    except Exception:
        resolved = name
    slug = _slugify(resolved)
    try:
        meaning = _bm.get_map(cx).get(slug, "")
    except Exception:
        meaning = ""
    href = None
    if slug and callable(product_exists):
        try:
            if product_exists(slug):
                href = f"/begin/product/{slug}"
        except Exception:
            href = None
    return {"name": name, "info": clip(meaning, sentences=2), "href": href}


def function_ref(cx, title):
    """Function/structure title -> /learn topic page. href + info only when an
    APPROVED topic page of kind 'function' exists for the slug; else plain."""
    name = (title or "").strip()
    if not name:
        return {"name": name, "info": "", "href": None}
    slug = _slugify(name)
    try:
        page = _tp.get_page(cx, slug)
    except Exception:
        page = None
    if not page or (page.get("kind") or "") != "function" or (page.get("state") or "") != "approved":
        return {"name": name, "info": "", "href": None}
    info = clip(_first_text(_page_content(page)), sentences=2)
    return {"name": name, "info": info, "href": f"/learn/{slug}"}


def ingredient_ref(cx, name, slug, page_getter=None):
    """Ingredient -> its ingredient page + a short 'what it is' summary. href +
    info only when an ingredient page exists for the slug; else plain."""
    name = (name or "").strip()
    slug = (slug or "").strip()
    if not slug:
        return {"name": name, "info": "", "href": None}
    getter = page_getter if callable(page_getter) else (lambda s: _safe_ing_page(cx, s))
    try:
        page = getter(slug)
    except Exception:
        page = None
    if not page:
        return {"name": name, "info": "", "href": None}
    info = clip(_first_text(_page_content(page)), sentences=2)
    return {"name": name, "info": info, "href": f"/begin/ingredient/{slug}"}


def _safe_ing_page(cx, slug):
    try:
        return _ip.get_page(cx, slug)
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_entity_refs.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/entity_refs.py tests/test_entity_refs.py
git commit -m "feat: entity_refs resolver for hover pop-up + link-out"
```

---

### Task 2: Shared `entity-ref` popover component (JS + CSS)

**Files:**
- Create: `static/entity-ref.js`
- Create: `static/entity-ref.css`
- Test (manual/headless): drive a throwaway HTML fixture (see Step 4).

**Interfaces:**
- Produces: a global `window.wireEntityRefs(root)` that binds ONE delegated listener to `root` (default `document`). Any element matching `.entity-ref[data-info]` gains: hover/focus → shared popover `#entityPop` shows `data-name` (bold) + `data-info`; if `data-href` present the element is/behaves as a new-tab link on click, and on touch the popover includes an "Open full page ↗" anchor. Idempotent (safe to call more than once per root).
- Markup contract emitted by later tasks:
  `<a class="entity-ref" data-name="…" data-info="…(escaped)…" href="/begin/…" target="_blank" rel="noopener">Label</a>` (link-out)
  or `<span class="entity-ref" tabindex="0" data-name="…" data-info="…">Label</span>` (pop-up only).

- [ ] **Step 1: Write `static/entity-ref.css`**

```css
/* Shared hover/tap pop-up for clinical entities (stress patterns, remedies,
   functions, ingredients) across the portal report and product/ingredient pages. */
.entity-ref{
  cursor:pointer; text-decoration:none; color:inherit;
  border-bottom:1px dotted currentColor; }
.entity-ref[href]{ border-bottom-style:dashed; }
.entity-ref:focus-visible{ outline:2px solid var(--brand,#2f6f5e); outline-offset:2px; }
#entityPop{
  position:absolute; z-index:1000; max-width:320px; max-height:50vh; overflow:auto;
  padding:10px 12px; border-radius:10px;
  background:var(--card,#12332b); color:var(--cream,#f4efe6);
  box-shadow:0 6px 24px rgba(0,0,0,.28); font-size:13.5px; line-height:1.45;
  opacity:0; pointer-events:none; transition:opacity .12s ease; }
#entityPop.show{ opacity:1; pointer-events:auto; }
#entityPop .ep-name{ font-weight:700; margin-bottom:3px; }
#entityPop .ep-link{ display:inline-block; margin-top:8px; font-weight:600;
  color:var(--brand-lt,#8fd0bd); text-decoration:none; }
```

- [ ] **Step 2: Write `static/entity-ref.js`**

```javascript
/* Shared entity-ref popover. No dependencies. One shared #entityPop element,
   one delegated listener per root. data-info/data-name are treated as PLAIN TEXT
   (assigned via textContent) — never injected as HTML. */
(function(){
  "use strict";
  var pop, hideT, isTouch = ("ontouchstart" in window) || navigator.maxTouchPoints > 0;

  function ensurePop(){
    if(pop) return pop;
    pop = document.createElement("div");
    pop.id = "entityPop";
    pop.setAttribute("role","tooltip");
    pop.innerHTML = '<div class="ep-name"></div><div class="ep-info"></div>'
                  + '<a class="ep-link" target="_blank" rel="noopener" hidden>Open full page ↗</a>';
    document.body.appendChild(pop);
    // Keep the popover open while the pointer is inside it (so the link is clickable).
    pop.addEventListener("mouseenter", function(){ clearTimeout(hideT); });
    pop.addEventListener("mouseleave", hide);
    return pop;
  }

  function show(el){
    var p = ensurePop();
    clearTimeout(hideT);
    p.querySelector(".ep-name").textContent = el.getAttribute("data-name") || el.textContent || "";
    p.querySelector(".ep-info").textContent = el.getAttribute("data-info") || "";
    var href = el.getAttribute("href") || el.getAttribute("data-href") || "";
    var link = p.querySelector(".ep-link");
    // The link row is for touch (no hover-click). On desktop the element itself
    // is the link, so we only surface the row when there's no hover affordance.
    if(href && isTouch){ link.href = href; link.hidden = false; }
    else { link.hidden = true; link.removeAttribute("href"); }
    p.classList.add("show");
    position(p, el);
  }

  function position(p, el){
    var r = el.getBoundingClientRect();
    var sx = window.pageXOffset, sy = window.pageYOffset;
    p.style.left = "0px"; p.style.top = "0px";  // measure at origin first
    var pw = p.offsetWidth, ph = p.offsetHeight, vw = document.documentElement.clientWidth;
    var left = Math.min(Math.max(8, r.left + sx), sx + vw - pw - 8);
    var below = r.bottom + sy + 6, above = r.top + sy - ph - 6;
    // Prefer below; flip above when it would overflow the viewport bottom.
    var top = (r.bottom + ph + 6 > document.documentElement.clientHeight && above > sy) ? above : below;
    p.style.left = left + "px"; p.style.top = top + "px";
  }

  function hide(){ hideT = setTimeout(function(){ if(pop) pop.classList.remove("show"); }, 120); }

  window.wireEntityRefs = function(root){
    root = root || document;
    if(root.__entityWired) return;
    root.__entityWired = true;
    root.addEventListener("mouseover", function(e){
      var el = e.target.closest && e.target.closest(".entity-ref[data-info]");
      if(el) show(el);
    });
    root.addEventListener("mouseout", function(e){
      if(e.target.closest && e.target.closest(".entity-ref[data-info]")) hide();
    });
    root.addEventListener("focusin", function(e){
      var el = e.target.closest && e.target.closest(".entity-ref[data-info]");
      if(el) show(el);
    });
    root.addEventListener("focusout", hide);
    // Touch: first tap shows the popover (and its link) instead of navigating.
    root.addEventListener("click", function(e){
      var el = e.target.closest && e.target.closest(".entity-ref[data-info]");
      if(!el) return;
      if(isTouch && pop && pop.classList.contains("show") &&
         pop.querySelector(".ep-name").textContent === (el.getAttribute("data-name")||el.textContent)){
        return; // second tap on the same ref: let the native link (if any) proceed
      }
      if(isTouch){ e.preventDefault(); show(el); }
      // desktop: native <a target=_blank> handles the new tab; span refs do nothing
    }, true);
    document.addEventListener("keydown", function(e){ if(e.key === "Escape") hide(); });
    // Dismiss on outside tap (touch).
    document.addEventListener("click", function(e){
      if(pop && pop.classList.contains("show") && !e.target.closest(".entity-ref") && !e.target.closest("#entityPop")) hide();
    });
  };
})();
```

- [ ] **Step 3: Write a throwaway fixture to drive it**

Create `/tmp/entity-ref-fixture.html` (NOT committed):

```html
<!doctype html><meta charset=utf-8>
<link rel="stylesheet" href="/static/entity-ref.css">
<p>A remedy <a class="entity-ref" data-name="Terrain Restore" data-info="Rebuilds terrain." href="/begin/product/terrain-restore" target="_blank" rel="noopener">Terrain Restore</a> and a pattern <span class="entity-ref" tabindex="0" data-name="Heavy Metals" data-info="Accumulated metals.">Heavy Metals</span>.</p>
<script src="/static/entity-ref.js"></script><script>wireEntityRefs();</script>
```

- [ ] **Step 4: Render-verify in headless Chrome**

Serve the worktree's `static/` and load the fixture; confirm by observation:
- Hovering "Terrain Restore" shows a popover reading **Terrain Restore** / "Rebuilds terrain."; moving the pointer away hides it.
- Hovering "Heavy Metals" shows the popover with no link.
- The popover never overflows the right edge (shrink the window and re-check).
- Focus the span via keyboard (Tab) → popover appears; `Esc` hides it.

Use the project's render-verify approach (headless Chrome via the `claude-in-chrome` tools or a scripted Playwright/puppeteer run against `python3 -m http.server` rooted at the worktree). Capture a screenshot of the open popover as evidence.

- [ ] **Step 5: Commit**

```bash
git add static/entity-ref.js static/entity-ref.css
git commit -m "feat: shared entity-ref hover/tap popover component"
```

---

### Task 3: Wire the client portal report (patterns, remedies, functions)

**Files:**
- Modify: `dashboard/portal_view.py:101-116` (`_assemble_biofield`)
- Modify: `static/client-portal.html` — include the shared assets in `<head>`; replace the pattern-chip render (~1157-1168) and the layer render (~1208-1217) with entity-ref markup; replace the `wirePatternDetails()` call (line ~1725) and definition (~754-785) with `wireEntityRefs()`.

**Interfaces:**
- Consumes: `entity_refs.remedy_ref(cx, spoken, product_exists) -> {name,info,href}`, `entity_refs.function_ref(cx, title) -> {name,info,href}` (Task 1). A `product_exists` callable backed by the product catalog (see Step 3).
- Produces: per-layer payload now carries `remedy_info`, `remedy_href` (strings; `remedy_href` may be `""`), and `function_href`, `function_info` for the layer title. Findings already carry `description`.

- [ ] **Step 1: Write the failing backend test**

```python
# tests/test_portal_view_entity_refs.py
import dashboard.portal_view as pv

def test_assemble_attaches_remedy_ref_when_shown(monkeypatch):
    monkeypatch.setattr(pv, "entity_refs_remedy", lambda name: {"name": name, "info": "Rebuilds terrain.", "href": "/begin/product/terrain-restore"})
    monkeypatch.setattr(pv, "entity_refs_function", lambda title: {"name": title, "info": "", "href": None})
    content = {"layers": [{"n": 1, "title": "Liver", "meaning": "m", "remedy": "Terrain Restore", "dosing": "2/day"}]}
    out = pv._assemble_biofield(content, "confirmed", scan_date=None, scan_dates=[], actionable=False, unlocked=True)
    L = out["layers"][0]
    assert L["remedy_info"] == "Rebuilds terrain."
    assert L["remedy_href"] == "/begin/product/terrain-restore"

def test_assemble_omits_remedy_ref_when_blurred(monkeypatch):
    called = {"n": 0}
    def _boom(name): called["n"] += 1; return {}
    monkeypatch.setattr(pv, "entity_refs_remedy", _boom)
    content = {"layers": [{"n": 1, "title": "Liver", "remedy": "Terrain Restore"}]}
    out = pv._assemble_biofield(content, "confirmed", scan_date=None, scan_dates=[], actionable=False, unlocked=False)
    assert out["blurred"] is True
    assert "remedy" not in out["layers"][0] and "remedy_info" not in out["layers"][0]
    assert called["n"] == 0  # never resolve a remedy for a blurred report
```

- [ ] **Step 2: Run it to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_view_entity_refs.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.portal_view' has no attribute 'entity_refs_remedy'`.

- [ ] **Step 3: Add the resolver seam + wiring in `portal_view.py`**

Add near the top of `dashboard/portal_view.py` (module scope), thin wrappers so tests can monkeypatch without a live catalog:

```python
from dashboard import entity_refs as _er

def _product_exists(slug):
    """True when `slug` is a live, sellable product page. Backed by the product
    catalog; wrapped so a catalog miss/exception is treated as 'no link'."""
    try:
        from dashboard.products import all_products  # {slug: {...}} of live products
        return slug in all_products()
    except Exception:
        return False

def entity_refs_remedy(name):
    return _er.remedy_ref(_db(), name, product_exists=_product_exists)

def entity_refs_function(title):
    return _er.function_ref(_db(), title)
```

> Note for implementer: `_db()` is however this module already obtains a DB connection (grep the file for the existing connection helper it uses for `get_portal_content_by_email`; reuse that exact accessor). If the module has no such helper, thread `cx` into `_assemble_biofield` from its caller (`portal_data`, which already holds `cx`) instead of opening a new one. `dashboard.products.all_products` — confirm the exact function name that returns the live product map; if it differs (e.g. `load()['products']`), use that. These are the only two local lookups to confirm.

Then in `_assemble_biofield`, replace the layer loop (lines 106-112) with:

```python
    layers = []
    for L in (content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        fn = entity_refs_function(item["title"])
        item["function_info"] = fn["info"]
        item["function_href"] = fn["href"] or ""
        if show:  # unconfirmed OR unpaid remedies NEVER leave the server
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
            rr = entity_refs_remedy(item["remedy"]) if item["remedy"] else {"info": "", "href": None}
            item["remedy_info"] = rr["info"]
            item["remedy_href"] = rr["href"] or ""
        layers.append(item)
```

- [ ] **Step 4: Run the backend test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_view_entity_refs.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Include shared assets + swap the frontend render**

In `static/client-portal.html` `<head>` (after the favicon link, line ~4) add:

```html
  <link rel="stylesheet" href="/static/entity-ref.css">
```

Before `</body>` (near the existing scripts) add:

```html
  <script src="/static/entity-ref.js"></script>
```

Replace the pattern chip render (lines 1160-1167) — turn every finding with a description into an entity-ref span (pop-up only), keeping the plain span for the rest, and drop the old `#patDetail` panel:

```javascript
         <div class="pat-wrap">${findings.map((f,i)=>{
           const nm = esc(f.name||f.code||"");
           const ds = (f.description||"").trim();
           return ds
             ? `<span class="pat-chip entity-ref${rvlCls}"${patDelay(i)} tabindex="0" data-name="${esc(f.name||f.code||"")}" data-info="${esc(ds)}">${nm}</span>`
             : `<span class="pat-chip${rvlCls}"${patDelay(i)}>${nm}</span>`;
         }).join("")}</div>`
```

(Delete the trailing `<div class="pat-detail" id="patDetail" hidden></div>` line — the shared `#entityPop` replaces it. Keep the `anyDetail` "tap one to learn more" hint.)

Replace the layer title + remedy render (lines 1212, 1215) so the title becomes a function entity-ref when linkable and the remedy becomes an entity-ref:

```javascript
            <h3>${(L.function_info||L.function_href)
                  ? `<span class="entity-ref"${L.function_href?"":' tabindex="0"'} data-name="${esc(L.title||"")}" data-info="${esc(L.function_info||"")}"${L.function_href?` role="link"`:""} data-href="${esc(L.function_href||"")}">${esc(L.title||"")}</span>`
                  : esc(L.title||"")}</h3>
            ${L.meaning?`<div class="mean">${esc(L.meaning)}</div>`:""}
            ${(L.stresses&&L.stresses.length)?`<div class="lstress">${L.stresses.map(s=>`<span class="lstress-chip">${esc(s.label||s.code||"")}</span>`).join("")}</div>`:""}
            ${L.remedy?`<div class="rx">${
              (L.remedy_info||L.remedy_href)
                ? (L.remedy_href
                    ? `<a class="entity-ref" data-name="${esc(L.remedy)}" data-info="${esc(L.remedy_info||"")}" href="${esc(L.remedy_href)}" target="_blank" rel="noopener"><b>${esc(L.remedy)}</b></a>`
                    : `<span class="entity-ref" tabindex="0" data-name="${esc(L.remedy)}" data-info="${esc(L.remedy_info||"")}"><b>${esc(L.remedy)}</b></span>`)
                : `<b>${esc(L.remedy)}</b>`
              }${L.dosing?`<span class="dose">${esc(L.dosing)}</span>`:""}</div>`:""}
```

> Function title with an href but no info still links (the `data-info` is empty → the popover shows just the name; acceptable for a titled link). If you prefer strict "no empty pop-up", gate the function entity-ref on `L.function_info` being non-empty; leave title plain otherwise. Keep it simple: render the entity-ref when EITHER info or href is present, matching the remedy rule.

- [ ] **Step 6: Remove `wirePatternDetails`, call `wireEntityRefs`**

Delete the `wirePatternDetails` function (lines ~754-785) and its `let _patWired=false;` guard, and replace the `wirePatternDetails();` call (line ~1725) with:

```javascript
  wireEntityRefs(document.getElementById("app") || document);
```

- [ ] **Step 7: Render-verify the portal in headless Chrome**

Drive a confirmed, unblurred portal (a `/portal/<token>` served by a locally-running app under Doppler, OR a saved payload rendered into the static file). Confirm by observation:
- A described stress-pattern chip shows its pop-up on hover; an undescribed one does not.
- A layer remedy that resolved to a product shows the pop-up AND opens `/begin/product/<slug>` in a NEW tab on click.
- A remedy with no product resolution shows the pop-up but is not a link.
- A blurred (unpaid) report shows NO remedy text at all (unchanged gating) and the payload contains no `remedy_*` fields (check the `/api/portal/<token>` JSON).
- A layer title that matches an approved function topic page links to `/learn/<slug>` in a new tab.

- [ ] **Step 8: Commit**

```bash
git add dashboard/portal_view.py tests/test_portal_view_entity_refs.py static/client-portal.html
git commit -m "feat: entity-ref hover + link-out on the client portal report"
```

---

### Task 4: Wire product-page ingredients

**Files:**
- Modify: `app.py:6259-6266` (ingredient objects in `begin_product_page_data`) — attach `info` per ingredient.
- Modify: `static/begin-product.html` — include shared assets; add entity-ref attributes to the existing ingredient anchor (`renderIngredientsBody`, lines ~401-415); call `wireEntityRefs()`.

**Interfaces:**
- Consumes: `entity_refs.ingredient_ref(cx, name, slug) -> {name,info,href}` (Task 1); the shared component (Task 2).
- Produces: each ingredient object in the `ingredients` section body gains an `info` string (may be `""`). The anchor already carries `href=/begin/ingredient/<slug>` + `target=_blank`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_product_ingredient_info.py
import json, app as appmod

def test_product_page_data_ingredient_has_info(monkeypatch):
    client = appmod.app.test_client()
    # A product with one dict ingredient; stub the product loader + ingredient_ref.
    monkeypatch.setattr(appmod, "_get_product", lambda slug: {
        "name": "Test Formula", "ingredients": [{"name": "Magnesium L-Threonate", "dose": "2000mg"}]})
    from dashboard import entity_refs
    monkeypatch.setattr(entity_refs, "ingredient_ref",
        lambda cx, name, slug, **k: {"name": name, "info": "Brain-penetrant magnesium.", "href": f"/begin/ingredient/{slug}"})
    r = client.get("/begin/product-page-data/test-formula")
    body = r.get_json()
    ings = next(s["body"]["ingredients"] for s in body["sections"] if s["id"] == "ingredients")
    assert ings[0]["info"] == "Brain-penetrant magnesium."
```

> Confirm the response JSON key that wraps the sections list (grep `begin_product_page_data` for the final `jsonify(...)` — it may be `{"sections": [...]}` or a bare list; adjust `body["sections"]` to match).

- [ ] **Step 2: Run it to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_product_ingredient_info.py -q`
Expected: FAIL — `KeyError: 'info'`.

- [ ] **Step 3: Attach `info` in `app.py`**

In `begin_product_page_data` (lines 6259-6266), after each ingredient's `slug` is set, attach its info:

```python
    from dashboard.ingredients import slugify as _slugify
    from dashboard import entity_refs as _er
    _raw_ingredients = p.get("ingredients") or card.get("ingredients", [])
    ingredients = []
    for _ing in _raw_ingredients:
        if isinstance(_ing, dict):
            _ing = dict(_ing)
            _ing["slug"] = _slugify(_ing.get("name") or "")
        else:
            _ing = {"name": str(_ing), "dose": "", "slug": _slugify(str(_ing))}
        _ing["info"] = _er.ingredient_ref(_db_conn(), _ing.get("name",""), _ing["slug"]).get("info", "")
        ingredients.append(_ing)
```

> `_db_conn()` — reuse whatever connection accessor this route already has in scope for other DB reads (grep the surrounding function/module for the existing `cx`/connection). `ingredient_ref` swallows its own errors and returns `info=""` on any miss, so a bad connection degrades to plain text, not a 500.

- [ ] **Step 4: Run the test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_product_ingredient_info.py -q`
Expected: PASS.

- [ ] **Step 5: Frontend — shared assets + entity-ref attributes**

In `static/begin-product.html` `<head>` add `  <link rel="stylesheet" href="/static/entity-ref.css">`. Near the existing `<script src="/static/theme-toggle.js"></script>` (line 233) add `  <script src="/static/entity-ref.js"></script>`.

In `renderIngredientsBody` (lines 406-410), add the entity-ref class + data attributes to the anchor that already exists (it already has href/target/rel):

```javascript
          var a = document.createElement('a');
          a.setAttribute('href', '/begin/ingredient/' + ingSlug);
          a.setAttribute('target', '_blank');
          a.setAttribute('rel', 'noopener');
          a.className = 'sp-ing-link entity-ref';
          a.setAttribute('data-name', ingName);
          if (ing.info) a.setAttribute('data-info', ing.info);
```

At the end of the page's init (where the page first renders its sections — grep for where sections are rendered / the DOMContentLoaded handler), call `wireEntityRefs();` once. (An ingredient with empty `info` has no `data-info`, so the component ignores it and it stays a normal link — matching the "no empty pop-up" rule.)

- [ ] **Step 6: Render-verify in headless Chrome**

Load `/begin/product/<a real formulation slug>`; open the "What's inside" section. Confirm: hovering an ingredient with info shows the pop-up; clicking opens `/begin/ingredient/<slug>` in a NEW tab; an ingredient without info is a plain link (no pop-up). Screenshot the open pop-up.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_product_ingredient_info.py static/begin-product.html
git commit -m "feat: entity-ref hover on product-page ingredients"
```

---

### Task 5: Wire ingredient-page related links (remedies/products)

**Files:**
- Modify: `app.py` — `begin_ingredient_page_data` (route at 6646): attach `info` to related-forms and related-products entries.
- Modify: `static/begin-ingredient.html` — include shared assets; add entity-ref attributes in `renderRelatedForms` (anchor at line ~340) and `renderFormulations`/related-products (anchor at line ~387); call `wireEntityRefs()`.

**Interfaces:**
- Consumes: `entity_refs.ingredient_ref` (for related forms, which are ingredients) and `entity_refs.remedy_ref` (for related formulations/products); the shared component.
- Produces: related-form entries gain `info` (via `ingredient_ref`), related-formulation entries gain `info` (via `remedy_ref` with a `product_exists` check).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingredient_related_info.py
import app as appmod

def test_ingredient_page_data_related_have_info(monkeypatch):
    client = appmod.app.test_client()
    # Stub the ingredient-page loader to return one related form + one formulation.
    monkeypatch.setattr(appmod, "_ingredient_page_payload",
        lambda slug: {"slug": slug, "name": "Vitamin C",
                      "related_forms": [{"slug": "sodium-ascorbate", "name": "Sodium Ascorbate"}],
                      "formulations": [{"slug": "immune-support", "name": "Immune Support"}]})
    from dashboard import entity_refs
    monkeypatch.setattr(entity_refs, "ingredient_ref",
        lambda cx, name, slug, **k: {"name": name, "info": "A buffered form.", "href": f"/begin/ingredient/{slug}"})
    monkeypatch.setattr(entity_refs, "remedy_ref",
        lambda cx, name, **k: {"name": name, "info": "A daily formula.", "href": "/begin/product/immune-support"})
    r = client.get("/begin/ingredient-page-data/vitamin-c")
    body = r.get_json()
    assert body["related_forms"][0]["info"] == "A buffered form."
    assert body["formulations"][0]["info"] == "A daily formula."
```

> Adjust the stub target (`_ingredient_page_payload`) and the response keys to match the real `begin_ingredient_page_data` implementation — grep the route for how it assembles `related_forms` and `formulations`, and stub the function that produces them.

- [ ] **Step 2: Run it to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_ingredient_related_info.py -q`
Expected: FAIL — `KeyError: 'info'`.

- [ ] **Step 3: Attach `info` in `begin_ingredient_page_data`**

In the route, after `related_forms` and `formulations` are assembled, enrich each entry (using the same in-scope DB connection the route already uses):

```python
    from dashboard import entity_refs as _er
    for _f in related_forms:
        _f["info"] = _er.ingredient_ref(cx, _f.get("name",""), _f.get("slug","")).get("info","")
    for _p in formulations:
        _p["info"] = _er.remedy_ref(cx, _p.get("name",""), product_exists=lambda s: True).get("info","")
```

> `related_forms` / `formulations` here name the local lists the route builds — match the actual variable names. For `product_exists` on this surface, the entry already came from the catalog so it is known-live; passing `lambda s: True` is safe. Reuse the route's existing `cx`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_ingredient_related_info.py -q`
Expected: PASS.

- [ ] **Step 5: Frontend — shared assets + entity-ref attributes**

In `static/begin-ingredient.html` `<head>` add `  <link rel="stylesheet" href="/static/entity-ref.css">` and near the page scripts add `  <script src="/static/entity-ref.js"></script>`.

In `renderRelatedForms` (anchor at line ~340) and `renderFormulations` (anchor at line ~387), add to each anchor:

```javascript
        a.className = (a.className ? a.className + ' ' : '') + 'entity-ref';
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener');
        a.setAttribute('data-name', f.name || '');
        if (f.info) a.setAttribute('data-info', f.info);
```

(These related links currently open same-tab; adding `target="_blank"` matches the "full page in a new tab" behavior. Confirm this is desired here too — it is consistent with the product page.) After the accordion sections render, call `wireEntityRefs();` once.

- [ ] **Step 6: Render-verify in headless Chrome**

Load `/begin/ingredient/<a real ingredient slug>` that has related forms/formulations; expand those accordions. Confirm hovering a related form/formulation shows its pop-up and clicking opens the target page in a new tab. Screenshot.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_ingredient_related_info.py static/begin-ingredient.html
git commit -m "feat: entity-ref hover on ingredient-page related links"
```

---

## Self-Review

**Spec coverage:**
- Hover pop-up + click new-tab for the five entity types → patterns (T3), remedies (T3), functions (T3), ingredients (T4, T5); structures explicitly deferred per spec ✓.
- One reusable component, no library → `static/entity-ref.{js,css}` (T2), used by all three surfaces ✓.
- Desktop hover / touch tap-with-link / keyboard + Esc → T2 component ✓.
- Pop-up-only for entities without a page (patterns) → `pattern_ref` returns `href=None` (T1), rendered as span (T3) ✓.
- Gated remedy content never leaks → `show`-guarded, resolver never called when blurred, test asserts it (T3) ✓.
- No empty pop-up / no wrong link → `info==""` → plain text; href only on positive existence check (T1 tests) ✓.
- `rel="noopener"` on all new tabs → T2/T3/T4/T5 ✓.
- Print/PDF + chat report unchanged → not touched ✓.

**Placeholder scan:** No "TBD/TODO". The three "confirm the exact accessor/key" notes (`_db()`, `all_products`, response-key names) are explicit, bounded verification steps with concrete fallbacks, not open-ended work — acceptable because they name exactly what to grep and what to do either way.

**Type consistency:** `entity_refs.*` all return `{name, info, href}` with `href` either a string or `None`; callers coerce `href or ""` before putting it in the payload; the frontend treats empty-string href as "no link". `wireEntityRefs(root)` signature consistent across T2/T3/T4/T5. Field names `remedy_info`/`remedy_href`/`function_info`/`function_href`/`info` consistent between the backend that sets them and the frontend that reads them.
