# Stress-Pattern Glossary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public E4L stress-pattern glossary (index + per-pattern detail pages showing description + body structures/functions) that portal pop-up patterns click through to.

**Architecture:** A pure read-only reader over `e4l.db` (`dashboard/pattern_glossary.py`), four Flask routes mirroring `/begin/ingredient`, two static HTML shells, and a portal wire-up that attaches `pattern_href` to findings.

**Tech Stack:** Python 3 / Flask, SQLite (`e4l.db`, read-only), vanilla JS + inline-HTML static pages, pytest.

## Global Constraints

- **Read `e4l.db` read-only** via `biofield_e4l._connect_ro(biofield_e4l._db_path())`. Never open it read-write; never create it.
- **Never raise into a route:** glossary functions degrade (empty index / None detail / `pattern_href=""`) on any DB error.
- **Escape all DB text on render** (textContent or escaped template) — same discipline as `static/entity-ref.js`.
- **New tabs use `rel="noopener"`.** Reuse `static/entity-ref.css/js` on the portal (already included there).
- **Slug = `dashboard.ingredients.slugify(code)`** (code is the `e4l_items` PK).
- **deploy-chat tests need Doppler** for app-importing tests: `doppler run -p remedy-match -c dev -- python3 -m pytest <path>`. Pure `pattern_glossary` tests build their own temp sqlite and need no Doppler.
- **A pattern is linkable/listed only when it `page_exists`** = has a description OR ≥1 mapped structure.

---

### Task 1: `pattern_glossary` reader + unit tests

**Files:**
- Create: `dashboard/pattern_glossary.py`
- Test: `tests/test_pattern_glossary.py`

**Interfaces:**
- Consumes: `dashboard.ingredients.slugify`; `dashboard.biofield_e4l._connect_ro`, `._db_path`.
- Produces:
  - `slug_for(code) -> str`
  - `open_ro(db_path=None) -> sqlite3.Connection|None` (read-only e4l.db; None on failure)
  - `get_pattern(cx, slug) -> dict|None`
  - `page_exists(cx, slug) -> bool`
  - `list_patterns(cx) -> list[dict]`
  - `STYPE_LABELS: dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pattern_glossary.py
import sqlite3
import pytest
from dashboard import pattern_glossary as pg


def _seed(cx):
    cx.executescript(
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, category TEXT NOT NULL, "
        " subcategory TEXT, name TEXT NOT NULL, full_name TEXT, e4l_description TEXT, "
        " clinical_notes TEXT, sort_order INTEGER);"
        "CREATE TABLE e4l_pattern_structures (code TEXT NOT NULL, structure TEXT NOT NULL, "
        " stype TEXT, is_primary INTEGER DEFAULT 0, source_phrase TEXT, PRIMARY KEY(code,structure));"
    )
    cx.executemany("INSERT INTO e4l_items VALUES (?,?,?,?,?,?,?,?)", [
        ("ED1", "ED", "", "Source", "Source Driver", "Supports the body's fundamental energy source.", "", 1),
        ("Lead", "Environmental", "Heavy Metals", "Lead", "Lead", "", "", 2),   # structures-only
        ("ES9", "ES", "", "Ghost", "Ghost", "", "", 3),                          # neither -> excluded
    ])
    cx.executemany("INSERT INTO e4l_pattern_structures VALUES (?,?,?,?,?)", [
        ("ED1", "Heart", "organ", 1, ""),
        ("ED1", "Energy & Stamina", "function", 0, ""),
        ("Lead", "Nervous System", "system", 1, ""),
    ])
    cx.commit()


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    _seed(c); return c


def test_slug_for_uses_code(cx):
    assert pg.slug_for("Heavy Metals") == "heavy-metals"
    assert pg.slug_for("ED1") == "ed1"


def test_get_pattern_shape_and_structure_order(cx):
    p = pg.get_pattern(cx, "ed1")
    assert p["code"] == "ED1" and p["name"] == "Source" and p["full_name"] == "Source Driver"
    assert p["description"].startswith("Supports the body")
    # primary structure first, then by stype/structure
    assert [s["structure"] for s in p["structures"]] == ["Heart", "Energy & Stamina"]
    assert p["structures"][0]["is_primary"] == 1
    assert p["has_page"] is True


def test_get_pattern_unknown_slug_is_none(cx):
    assert pg.get_pattern(cx, "nope") is None


def test_page_exists_rules(cx):
    assert pg.page_exists(cx, "ed1") is True          # described + structures
    assert pg.page_exists(cx, "lead") is True          # structures only
    assert pg.page_exists(cx, "ghost") is False        # neither
    assert pg.page_exists(cx, "missing") is False


def test_list_patterns_groups_and_excludes_empty(cx):
    groups = pg.list_patterns(cx)
    flat = {p["slug"]: p for g in groups for p in g["patterns"]}
    assert "ed1" in flat and "lead" in flat
    assert "ghost" not in flat                          # excluded (no page)
    assert flat["ed1"]["has_desc"] is True and flat["ed1"]["n_structures"] == 2
    assert flat["lead"]["has_desc"] is False and flat["lead"]["n_structures"] == 1
    cats = [g["category"] for g in groups]
    assert cats == sorted(set(cats), key=cats.index)   # each category once, stable
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_pattern_glossary.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.pattern_glossary'`.

- [ ] **Step 3: Implement `dashboard/pattern_glossary.py`**

```python
"""Read-only reader over e4l.db for the public stress-pattern glossary. Pure; never
writes; never raises into callers (open failure -> None / empty)."""
import sqlite3

from dashboard.ingredients import slugify as _slugify
from dashboard import biofield_e4l as _be

STYPE_LABELS = {
    "organ": "Organs", "system": "Body systems", "function": "Functions",
    "emotion": "Emotions", "substance": "Substances", "immune_cell": "Immune cells",
    "other": "Other",
}

# Fixed display order for category sections in the index; unknown categories append.
_CATEGORY_ORDER = ["ED", "EI", "ET", "ES", "ER", "MB", "MR", "BFA",
                   "Nutrition", "Environmental", "Sensitivity", "LifeJourney"]


def slug_for(code):
    return _slugify(code or "")


def open_ro(db_path=None):
    try:
        cx = _be._connect_ro(_be._db_path(db_path))
        cx.row_factory = sqlite3.Row
        return cx
    except Exception:
        return None


def _slug_map(cx):
    out = {}
    for r in cx.execute("SELECT code FROM e4l_items"):
        out.setdefault(slug_for(r["code"]), r["code"])
    return out


def _code_for(cx, slug):
    return _slug_map(cx).get((slug or "").strip().lower())


def _structures(cx, code):
    rows = cx.execute(
        "SELECT structure, stype, is_primary FROM e4l_pattern_structures WHERE code=? "
        "ORDER BY is_primary DESC, stype, structure", (code,)).fetchall()
    return [{"structure": r["structure"], "stype": (r["stype"] or "other"),
             "is_primary": r["is_primary"] or 0} for r in rows]


def get_pattern(cx, slug):
    code = _code_for(cx, slug)
    if not code:
        return None
    r = cx.execute(
        "SELECT code, category, subcategory, name, full_name, e4l_description "
        "FROM e4l_items WHERE code=?", (code,)).fetchone()
    if not r:
        return None
    structures = _structures(cx, code)
    desc = (r["e4l_description"] or "").strip()
    return {
        "code": r["code"], "name": (r["name"] or r["code"]).strip(),
        "full_name": (r["full_name"] or "").strip(),
        "category": r["category"] or "", "subcategory": (r["subcategory"] or "").strip(),
        "description": desc, "structures": structures,
        "has_page": bool(desc or structures),
    }


def page_exists(cx, slug):
    code = _code_for(cx, slug)
    if not code:
        return False
    r = cx.execute("SELECT TRIM(COALESCE(e4l_description,'')) d FROM e4l_items WHERE code=?",
                   (code,)).fetchone()
    if r and r["d"]:
        return True
    n = cx.execute("SELECT COUNT(*) n FROM e4l_pattern_structures WHERE code=?", (code,)).fetchone()
    return bool(n and n["n"])


def list_patterns(cx):
    rows = cx.execute(
        "SELECT i.code, i.category, i.name, i.full_name, i.sort_order, "
        "  TRIM(COALESCE(i.e4l_description,'')) AS d, "
        "  (SELECT COUNT(*) FROM e4l_pattern_structures s WHERE s.code=i.code) AS n "
        "FROM e4l_items i ORDER BY i.category, COALESCE(i.sort_order, 9999), i.name").fetchall()
    by_cat = {}
    for r in rows:
        if not (r["d"] or r["n"]):
            continue  # no page -> exclude
        by_cat.setdefault(r["category"] or "", []).append({
            "slug": slug_for(r["code"]), "name": (r["name"] or r["code"]).strip(),
            "full_name": (r["full_name"] or "").strip(),
            "has_desc": bool(r["d"]), "n_structures": r["n"] or 0,
        })
    ordered = [c for c in _CATEGORY_ORDER if c in by_cat]
    ordered += [c for c in by_cat if c not in _CATEGORY_ORDER]
    return [{"category": c, "patterns": by_cat[c]} for c in ordered]
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_pattern_glossary.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/pattern_glossary.py tests/test_pattern_glossary.py
git commit -m "feat: pattern_glossary read-only reader over e4l.db"
```

---

### Task 2: Glossary routes

**Files:**
- Modify: `app.py` — add four routes near the `/learn` family (~line 6835).

**Interfaces:**
- Consumes: `dashboard.pattern_glossary` (Task 1); `STATIC`, `send_from_directory`, `jsonify` (already imported).
- Produces: `GET /learn/patterns`, `GET /learn/patterns-data`, `GET /learn/pattern/<slug>`, `GET /learn/pattern-data/<slug>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pattern_routes.py  (Doppler required: imports app)
import app as appmod


def test_pattern_data_known_and_unknown(monkeypatch):
    client = appmod.app.test_client()
    from dashboard import pattern_glossary as pg
    monkeypatch.setattr(pg, "open_ro", lambda db_path=None: object())
    monkeypatch.setattr(pg, "get_pattern",
        lambda cx, slug: {"code": "ED1", "name": "Source", "structures": [], "has_page": True} if slug == "ed1" else None)
    ok = client.get("/learn/pattern-data/ed1").get_json()
    assert ok["code"] == "ED1"
    missing = client.get("/learn/pattern-data/nope")
    assert missing.status_code == 404


def test_patterns_index_data(monkeypatch):
    client = appmod.app.test_client()
    from dashboard import pattern_glossary as pg
    monkeypatch.setattr(pg, "open_ro", lambda db_path=None: object())
    monkeypatch.setattr(pg, "list_patterns", lambda cx: [{"category": "ED", "patterns": []}])
    body = client.get("/learn/patterns-data").get_json()
    assert body["groups"][0]["category"] == "ED"
```

- [ ] **Step 2: Run to verify failure**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_pattern_routes.py -q`
Expected: FAIL — routes return 404 (not registered).

- [ ] **Step 3: Add the routes in `app.py`** (place immediately BEFORE `@app.route("/learn/<slug>")` at ~6778 so the literal `pattern`/`patterns` segments are registered; Flask static-vs-dynamic precedence also protects this, but ordering makes intent clear)

```python
@app.route("/learn/patterns")
def learn_patterns_index():
    return send_from_directory(STATIC, "patterns-index.html")


@app.route("/learn/patterns-data")
def learn_patterns_data():
    from dashboard import pattern_glossary as _pg
    cx = _pg.open_ro()
    if cx is None:
        return jsonify({"groups": []})
    try:
        return jsonify({"groups": _pg.list_patterns(cx)})
    finally:
        try: cx.close()
        except Exception: pass


@app.route("/learn/pattern/<slug>")
def learn_pattern_page(slug):
    return send_from_directory(STATIC, "pattern-page.html")


@app.route("/learn/pattern-data/<slug>")
def learn_pattern_data(slug):
    from dashboard import pattern_glossary as _pg
    cx = _pg.open_ro()
    if cx is None:
        return jsonify({"state": "unknown"}), 404
    try:
        p = _pg.get_pattern(cx, slug)
        if not p:
            return jsonify({"state": "unknown"}), 404
        return jsonify(p)
    finally:
        try: cx.close()
        except Exception: pass
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_pattern_routes.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_pattern_routes.py
git commit -m "feat: stress-pattern glossary routes (index + detail)"
```

---

### Task 3: Static pages (detail + index)

**Files:**
- Create: `static/pattern-page.html`
- Create: `static/patterns-index.html`

**Interfaces:**
- Consumes: `/learn/pattern-data/<slug>` and `/learn/patterns-data` (Task 2).

- [ ] **Step 1: Create `static/pattern-page.html`**

A self-contained page mirroring `begin-ingredient.html`'s head (favicon, fonts, `theme-toggle.js`) and card styling. Slug from `location.pathname.split('/').pop()`. Fetch the data endpoint; render title, category badge, description, and structures grouped by `stype`. All text via `textContent`.

```html
<!DOCTYPE html><html lang="en"><head>
  <link rel="icon" type="image/png" href="/static/favicon.png">
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stress Pattern · Dr. Glen Swartwout</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&family=Raleway:wght@600;700&display=swap" rel="stylesheet">
  <style>
    :root{--bg:#0b241d;--card:#12332b;--cream:#f4efe6;--brand:#2f6f5e;--muted:#9fb7ad;--line:#21472d}
    body{margin:0;background:var(--bg);color:var(--cream);font-family:'Open Sans',sans-serif;line-height:1.55}
    .wrap{max-width:760px;margin:0 auto;padding:32px 20px 80px}
    h1{font-family:Raleway,sans-serif;font-size:1.7rem;margin:.2rem 0}
    .full{color:var(--muted);font-size:1rem;margin:0 0 .6rem}
    .badge{display:inline-block;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;
      background:var(--brand);color:#fff;border-radius:10px;padding:2px 9px;margin-bottom:14px}
    .desc{font-size:1.02rem;margin:.4rem 0 1.4rem}
    .sgroup{margin:1rem 0}
    .sgroup h3{font-family:Raleway,sans-serif;font-size:1rem;margin:.2rem 0 .5rem;color:var(--cream)}
    .schips{display:flex;flex-wrap:wrap;gap:6px}
    .schip{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:4px 11px;font-size:.86rem}
    .schip.primary{border-color:var(--brand)}
    .back{display:inline-block;margin-bottom:18px;color:var(--muted);text-decoration:none;font-size:.9rem}
    .muted{color:var(--muted)} .hidden{display:none}
  </style>
  <script src="/static/theme-toggle.js"></script>
</head><body>
  <div class="wrap">
    <a class="back" href="/learn/patterns">← All stress patterns</a>
    <div id="pat-loading" class="muted">Loading…</div>
    <div id="pat-notfound" class="hidden muted">We couldn't find that stress pattern. <a href="/learn/patterns">Browse the glossary</a>.</div>
    <div id="pat-body" class="hidden">
      <span id="pat-badge" class="badge hidden"></span>
      <h1 id="pat-title"></h1>
      <p id="pat-full" class="full hidden"></p>
      <p id="pat-desc" class="desc hidden"></p>
      <div id="pat-structs"></div>
    </div>
  </div>
  <script>
    var STYPE_LABELS = {organ:"Organs",system:"Body systems",function:"Functions",
      emotion:"Emotions",substance:"Substances",immune_cell:"Immune cells",other:"Other"};
    var STYPE_ORDER = ["organ","system","function","emotion","substance","immune_cell","other"];
    function show(id){document.getElementById(id).classList.remove('hidden');}
    function hide(id){document.getElementById(id).classList.add('hidden');}
    var slug = decodeURIComponent(location.pathname.split('/').filter(Boolean).pop() || '');
    fetch('/learn/pattern-data/' + encodeURIComponent(slug), {credentials:'same-origin'})
      .then(function(r){ if(!r.ok) return null; return r.json(); })
      .then(function(d){
        hide('pat-loading');
        if(!d || d.state === 'unknown'){ show('pat-notfound'); return; }
        show('pat-body');
        document.title = (d.name || 'Stress Pattern') + ' · Dr. Glen Swartwout';
        document.getElementById('pat-title').textContent = d.name || d.code || '';
        if(d.full_name && d.full_name !== d.name){ var f=document.getElementById('pat-full'); f.textContent=d.full_name; show('pat-full'); }
        if(d.category){ var b=document.getElementById('pat-badge'); b.textContent=d.category; show('pat-badge'); }
        if(d.description){ var p=document.getElementById('pat-desc'); p.textContent=d.description; show('pat-desc'); }
        var byType = {};
        (d.structures || []).forEach(function(s){ (byType[s.stype||'other']=byType[s.stype||'other']||[]).push(s); });
        var host = document.getElementById('pat-structs');
        var any = false;
        STYPE_ORDER.forEach(function(t){
          var items = byType[t]; if(!items || !items.length) return;
          any = true;
          var g = document.createElement('div'); g.className='sgroup';
          var h = document.createElement('h3'); h.textContent = STYPE_LABELS[t] || t; g.appendChild(h);
          var chips = document.createElement('div'); chips.className='schips';
          items.forEach(function(s){
            var c = document.createElement('span'); c.className='schip'+(s.is_primary?' primary':'');
            c.textContent = s.structure; chips.appendChild(c);
          });
          g.appendChild(chips); host.appendChild(g);
        });
        if(any){
          var head = document.createElement('h3'); head.style.marginTop='1.4rem';
          head.textContent = 'Body structures & functions involved';
          host.insertBefore(head, host.firstChild);
        }
      })
      .catch(function(){ hide('pat-loading'); show('pat-notfound'); });
  </script>
</body></html>
```

- [ ] **Step 2: Create `static/patterns-index.html`**

Mirrors the same head/styles; fetches `/learn/patterns-data`; renders one section per category with a list of links.

```html
<!DOCTYPE html><html lang="en"><head>
  <link rel="icon" type="image/png" href="/static/favicon.png">
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stress Pattern Glossary · Dr. Glen Swartwout</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&family=Raleway:wght@600;700&display=swap" rel="stylesheet">
  <style>
    :root{--bg:#0b241d;--card:#12332b;--cream:#f4efe6;--brand:#2f6f5e;--muted:#9fb7ad;--line:#21472d}
    body{margin:0;background:var(--bg);color:var(--cream);font-family:'Open Sans',sans-serif;line-height:1.5}
    .wrap{max-width:820px;margin:0 auto;padding:32px 20px 80px}
    h1{font-family:Raleway,sans-serif;font-size:1.8rem;margin:.2rem 0 .3rem}
    .lead{color:var(--muted);margin:0 0 1.4rem}
    .cat{font-family:Raleway,sans-serif;font-size:1.1rem;margin:1.4rem 0 .5rem;border-bottom:1px solid var(--line);padding-bottom:4px}
    ul.plist{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px 18px}
    ul.plist a{color:var(--cream);text-decoration:none;border-bottom:1px dashed var(--brand)}
    ul.plist li{padding:5px 0;font-size:.95rem}
    .hint{color:var(--muted);font-size:.8rem;margin-left:6px}
    .muted{color:var(--muted)} .hidden{display:none}
  </style>
  <script src="/static/theme-toggle.js"></script>
</head><body>
  <div class="wrap">
    <h1>Stress Pattern Glossary</h1>
    <p class="lead">The bioenergetic stress patterns an E4L scan can identify, and the body structures and functions each one involves.</p>
    <div id="idx-loading" class="muted">Loading…</div>
    <div id="idx-empty" class="hidden muted">The glossary is being prepared.</div>
    <div id="idx-body"></div>
  </div>
  <script>
    function hide(id){document.getElementById(id).classList.add('hidden');}
    fetch('/learn/patterns-data', {credentials:'same-origin'})
      .then(function(r){ return r.json(); })
      .then(function(d){
        hide('idx-loading');
        var groups = (d && d.groups) || [];
        if(!groups.length){ document.getElementById('idx-empty').classList.remove('hidden'); return; }
        var host = document.getElementById('idx-body');
        groups.forEach(function(g){
          if(!g.patterns || !g.patterns.length) return;
          var h = document.createElement('div'); h.className='cat'; h.textContent = g.category; host.appendChild(h);
          var ul = document.createElement('ul'); ul.className='plist';
          g.patterns.forEach(function(p){
            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '/learn/pattern/' + encodeURIComponent(p.slug);
            a.textContent = p.name; li.appendChild(a);
            if(p.n_structures){ var s=document.createElement('span'); s.className='hint'; s.textContent = p.n_structures + ' structures'; li.appendChild(s); }
            ul.appendChild(li);
          });
          host.appendChild(ul);
        });
      })
      .catch(function(){ hide('idx-loading'); document.getElementById('idx-empty').classList.remove('hidden'); });
  </script>
</body></html>
```

- [ ] **Step 3: Render-verify in headless Chrome**

Serve the app under Doppler (or serve `static/` and stub the two fetches). Confirm:
- `/learn/pattern/<described+structured slug>` shows title, category badge, description, and grouped structures with the "Body structures & functions involved" heading; primary chips are outlined.
- A structures-only pattern shows no description paragraph but still the structures block.
- An unknown slug shows the not-found state.
- `/learn/patterns` lists categories with links that navigate to detail pages.
Capture one screenshot of a populated detail page.

- [ ] **Step 4: Commit**

```bash
git add static/pattern-page.html static/patterns-index.html
git commit -m "feat: stress-pattern glossary pages (detail + index)"
```

---

### Task 4: Portal wire-up (patterns click through)

**Files:**
- Modify: `dashboard/biofield_e4l.py` — `_findings` (~84-100) attaches `pattern_href`.
- Modify: `static/client-portal.html` — described pattern chip becomes a link when `pattern_href` present.
- Modify: `dashboard/entity_refs.py` — `pattern_ref` gains optional `href`.
- Test: `tests/test_findings_pattern_href.py`

**Interfaces:**
- Consumes: `pattern_glossary.slug_for`, `.page_exists`, and the read-only `cx` already open in `_findings`.
- Produces: each finding dict gains `"pattern_href"` (str, `""` when no page).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_findings_pattern_href.py
import sqlite3
from dashboard import biofield_e4l as be


def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.executescript(
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, category TEXT, subcategory TEXT, "
        " name TEXT, full_name TEXT, e4l_description TEXT, clinical_notes TEXT, sort_order INTEGER);"
        "CREATE TABLE e4l_pattern_structures (code TEXT, structure TEXT, stype TEXT, is_primary INTEGER, source_phrase TEXT, PRIMARY KEY(code,structure));"
    )
    cx.execute("INSERT INTO e4l_items VALUES ('ED1','ED','','Source','Source Driver','A description.','',1)")
    cx.execute("INSERT INTO e4l_items VALUES ('ER5','ER','','Bare','Bare','','',2)")  # no desc, no struct
    cx.commit()
    return cx


def test_findings_attach_pattern_href(monkeypatch):
    cx = _mk()
    # Build a minimal scan the real _findings query can read, or call the helper it uses.
    hrefs = be._pattern_hrefs(cx, ["ED1", "ER5"])
    assert hrefs["ED1"] == "/learn/pattern/ed1"
    assert hrefs["ER5"] == ""     # no page
```

> This tests a small extracted helper `_pattern_hrefs(cx, codes)` so we don't have to stand up a full scan row. `_findings` calls it and copies the result onto each finding.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_findings_pattern_href.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.biofield_e4l' has no attribute '_pattern_hrefs'`.

- [ ] **Step 3: Implement in `dashboard/biofield_e4l.py`**

Add the helper and call it in `_findings`:

```python
def _pattern_hrefs(cx, codes):
    """{code: '/learn/pattern/<slug>' or ''} — link only patterns that have a page."""
    from dashboard import pattern_glossary as _pg
    out = {}
    for code in codes:
        try:
            slug = _pg.slug_for(code)
            out[code] = ("/learn/pattern/" + slug) if _pg.page_exists(cx, slug) else ""
        except Exception:
            out[code] = ""
    return out
```

In `_findings`, after building `out` (the list of finding dicts), attach the href:

```python
    hrefs = _pattern_hrefs(cx, [f["code"] for f in out])
    for f in out:
        f["pattern_href"] = hrefs.get(f["code"], "")
    return out
```

(Locate the existing `return out` at the end of `_findings` and insert these three lines before it. `out` items already carry `code`.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_findings_pattern_href.py -q`
Expected: PASS.

- [ ] **Step 5: Update `entity_refs.pattern_ref` for consistency**

```python
def pattern_ref(name, description, href=""):
    """Stress pattern: pop-up + optional link-out to its glossary page."""
    return {"name": (name or "").strip(), "info": clip(description, sentences=3),
            "href": href or None}
```

(Update the one existing `pattern_ref` test to still pass: it asserts `href is None` when called without `href`, which still holds.)

- [ ] **Step 6: Wire the portal chip in `static/client-portal.html`**

In the patterns block render (the `findings.map` that currently emits `<span class="pat-chip entity-ref" …>` for described findings), branch on `f.pattern_href`:

```javascript
         <div class="pat-wrap">${findings.map((f,i)=>{
           const nm = esc(f.name||f.code||"");
           const ds = (f.description||"").trim();
           if(!ds) return `<span class="pat-chip${rvlCls}"${patDelay(i)}>${nm}</span>`;
           const href = (f.pattern_href||"").trim();
           return href
             ? `<a class="pat-chip entity-ref${rvlCls}"${patDelay(i)} data-name="${esc(f.name||f.code||"")}" data-info="${esc(ds)}" href="${esc(href)}" target="_blank" rel="noopener">${nm}</a>`
             : `<span class="pat-chip entity-ref${rvlCls}"${patDelay(i)} tabindex="0" data-name="${esc(f.name||f.code||"")}" data-info="${esc(ds)}">${nm}</span>`;
         }).join("")}</div>`
```

- [ ] **Step 7: Update the pattern hint copy**

The `anyDetail` hint currently reads "tap one to learn more". Leave as-is (still accurate — hover/tap shows the pop-up; described chips also link).

- [ ] **Step 8: Render-verify + unit tests**

Run: `python3 -m pytest tests/test_entity_refs.py tests/test_pattern_glossary.py tests/test_findings_pattern_href.py -q` → all pass.
Headless Chrome on a portal analysis payload: a described pattern chip with a `pattern_href` shows the hover pop-up AND opens `/learn/pattern/<slug>` in a new tab; a described chip without an href stays pop-up-only; an undescribed finding stays plain.

- [ ] **Step 9: Commit**

```bash
git add dashboard/biofield_e4l.py dashboard/entity_refs.py static/client-portal.html tests/test_findings_pattern_href.py
git commit -m "feat: portal stress patterns link to their glossary page"
```

---

## Self-Review

**Spec coverage:** index page (Task 2/3), detail page with description + structures grouped by stype (Task 1/3), `page_exists` rule (Task 1), portal click-through wiring (Task 4), read-only e4l.db + graceful degradation (Task 1/2), remedies/structure-pages/missing-descriptions deferred (not built) — all covered.

**Placeholder scan:** No TBD/TODO. The one "locate the existing `return out`" instruction in Task 4 Step 3 is a precise anchor with the exact lines to insert, not open-ended work.

**Type consistency:** `get_pattern`/`list_patterns` return the exact keys the static pages read (`name`, `full_name`, `category`, `description`, `structures[].{structure,stype,is_primary}`, `slug`, `n_structures`). `pattern_href` is a string set in `_findings` and read as `f.pattern_href` in the template. `slug_for`/`page_exists` signatures match between `pattern_glossary`, the routes, and `_pattern_hrefs`.
