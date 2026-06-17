# E4L Synthesis Fidelity + Draft Ergonomics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make E4L auto-drafts propose only active products with real FMP dosing, clean schema, and no invented names; add a "Sync order list" editor button; and learn `pattern → product` from confirmations so future drafts need fewer edits.

**Architecture:** Fidelity fixes are PURE functions in the local vault synthesis (`02 Skills/e4l_synthesis.py`), unit-tested with mocks; I/O (FMP dosing, corrections fetch) lives in the importer (`e4l-portal-import.py`) so the pure functions stay testable. The editor button is one client-side change in deploy-chat. The learning loop reuses PR #157's `biofield_corrections` log + `/api/console/biofield/corrections` endpoint.

**Tech Stack:** Python (vault synthesis + psycopg2 for FMP), Flask/JS (deploy-chat editor). Spec: `docs/superpowers/specs/2026-06-17-e4l-synthesis-fidelity-design.md`.

---

## Two repos — where each task runs

- **VAULT** `~/AI-Training/02 Skills/` — `e4l_synthesis.py`, `e4l-portal-import.py`. Edit in place (auto-snapshots, no git/PR). Tests: `02 Skills/tests/test_e4l_synthesis.py`.
  Run vault tests: `cd "/Users/remedymatch/AI-Training/02 Skills" && PYTHONPATH=. ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_e4l_synthesis.py -q`
- **DEPLOY-CHAT** worktree `/tmp/wt-deploy-chat-5326cc61` (branch `sess/5326cc61-synthfidelity`) — `static/console-biofield-portal.html` only. Commit + PR.
  Run deploy-chat suite: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`

**Pure-function signatures (consistent across tasks):**
- `load_catalog(path) -> [{slug,name,price_cents}]` (ACTIVE only)
- `resolve_ff_slug(name, catalog) -> slug|None` (exists, unchanged)
- `fmp_dosing_map(names, conn_factory=None) -> {name_lower: "dosing"}`
- `build_overrides(corrections) -> {item_code: {"product": str, "ts": str}}`
- `to_portal_content(synth, catalog, dosing_map=None, overrides=None) -> portal-content dict`

---

## Task 1: active-only catalog — `load_catalog`

**Files:** Modify `02 Skills/e4l_synthesis.py`; Test `02 Skills/tests/test_e4l_synthesis.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_load_catalog_drops_inactive(tmp_path):
    import json
    from e4l_synthesis import load_catalog
    p = tmp_path / "products.json"
    p.write_text(json.dumps({"products": {
        "active-one": {"name": "Active One", "price_cents": 100},
        "dead-one":   {"name": "Dead One", "price_cents": 200, "inactive": True},
    }}))
    cat = load_catalog(str(p))
    names = [c["name"] for c in cat]
    assert "Active One" in names and "Dead One" not in names
```

- [ ] **Step 2: Run → FAIL** (`... -m pytest tests/test_e4l_synthesis.py::test_load_catalog_drops_inactive -q`)

- [ ] **Step 3: Implement** — change the comprehension in `load_catalog`:

```python
    return [{"slug": k, "name": v.get("name", ""), "price_cents": v.get("price_cents")}
            for k, v in products.items()
            if isinstance(v, dict) and v.get("name") and not v.get("inactive")]
```

- [ ] **Step 4: Run → PASS.** **Step 5:** Run the full vault test file (existing tests still green). **Step 6: Commit** — vault auto-snapshots; no git. Note completion.

---

## Task 2: FF validation + clean schema + retain patterns — `to_portal_content`

**Files:** Modify `02 Skills/e4l_synthesis.py`; Test `02 Skills/tests/test_e4l_synthesis.py`

- [ ] **Step 1: Failing test**

```python
def test_to_portal_content_validates_ffs_cleans_schema_keeps_patterns():
    from e4l_synthesis import to_portal_content
    catalog = [{"slug": "nous-energy", "name": "Nous Energy", "price_cents": 100}]
    synth = {"greeting": "Aloha", "layers": [
        {"n": 1, "title": "Calm", "meaning": "m", "patterns": ["ES2", "MB6"],
         "ffs": ["Nous Energy", "Invented Formula"], "dosing": "take as directed"},
        {"n": 2, "title": "X", "meaning": "m2", "patterns": ["ED9"],
         "ffs": ["Invented Only"], "dosing": "whatever"}]}
    out = to_portal_content(synth, catalog)
    L1, L2 = out["layers"]
    assert L1["remedy"] == "Nous Energy"            # invented dropped, not in remedy
    assert "Invented" not in L1["remedy"]
    assert L1["patterns"] == ["ES2", "MB6"]         # patterns retained
    assert L1["dosing"] == ""                        # no dosing_map -> blank, never the LLM's
    assert L2["remedy"] == ""                        # no resolvable FF -> blank
    assert {"slug": "nous-energy", "qty": 1} in out["reorder_items"]
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — replace `to_portal_content` with:

```python
def to_portal_content(synth, catalog, dosing_map=None, overrides=None):
    """LLM synth -> editor portal-content. Each layer's remedy = only FFs that
    resolve to an ACTIVE catalog product (invented/inactive names dropped, blank if
    none). dosing comes from dosing_map (FMP), never the LLM. Learned `overrides`
    (item_code -> {product, ts}) pin a layer's FF when its patterns are known.
    Patterns are retained per layer for the learning loop."""
    dosing_map = dosing_map or {}
    layers, reorder, seen = [], [], set()
    for L in synth.get("layers") or []:
        patterns = [str(p) for p in (L.get("patterns") or [])]
        # learned override: most-recent mapping among this layer's patterns wins
        ov, ov_ts = None, None
        for code in patterns:
            o = (overrides or {}).get(code)
            if o and (ov_ts is None or (o.get("ts") or "") >= ov_ts):
                ov, ov_ts = o.get("product"), (o.get("ts") or "")
        ffs = [p.strip() for p in ov.split(" + ") if p.strip()] if ov \
              else [f for f in (L.get("ffs") or []) if (f or "").strip()]
        resolved = [f for f in ffs if resolve_ff_slug(f, catalog)]
        for f in resolved:
            slug = resolve_ff_slug(f, catalog)
            if slug and slug not in seen:
                seen.add(slug)
                reorder.append({"slug": slug, "qty": 1})
        doses = [dosing_map.get(f.strip().lower(), "") for f in resolved]
        layers.append({"n": L.get("n"), "title": L.get("title", ""),
                       "meaning": L.get("meaning", ""),
                       "remedy": " + ".join(resolved),
                       "dosing": "; ".join(d for d in doses if d),
                       "patterns": patterns})
    return {"greeting": synth.get("greeting", ""),
            "video": {"url": "", "label": "Watch your message from Dr. Glen"},
            "layers": layers, "reorder_items": reorder, "pricing_note": ""}
```

- [ ] **Step 4: Run → PASS.** **Step 5:** Run the full vault test file. **If an existing `to_portal_content` test now fails** because it asserted the old dosing/remedy behavior, update it to the new contract (remedy = resolved-only, dosing from `dosing_map`, patterns retained) — that's the intended change; note which tests you updated. **Step 6: Commit** (vault auto-snapshot).

---

## Task 3: FMP dosing + drop LLM dosing — `fmp_dosing_map`, `synthesize`

**Files:** Modify `02 Skills/e4l_synthesis.py`; Test `02 Skills/tests/test_e4l_synthesis.py`

- [ ] **Step 1: Failing test** (mock the DB — never hit live FMP)

```python
def test_fmp_dosing_map_concats_and_blanks_on_miss():
    from e4l_synthesis import fmp_dosing_map

    class FakeCur:
        def __init__(self, rows): self.rows = rows; self._r = None
        def execute(self, q, args): self._r = self.rows.get((args[0] or "").lower())
        def fetchone(self): return self._r
    class FakeConn:
        def __init__(self, rows): self.rows = rows
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return FakeCur(self.rows)

    rows = {"nous energy": ("2 capsules", "twice daily", "with food")}
    cf = lambda: FakeConn(rows)
    m = fmp_dosing_map(["Nous Energy", "Unknown Product"], conn_factory=cf)
    assert m["nous energy"] == "2 capsules twice daily with food"
    assert m.get("unknown product", "") == ""        # miss -> blank/absent
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — add to `e4l_synthesis.py`:

```python
def fmp_dosing_map(names, conn_factory=None):
    """Real dosing per product from fmp_newapp.products (dosage + dosage_freq +
    dosage_timing), keyed by lowercased product name. Missing/unreachable -> blank.
    conn_factory injectable for tests; defaults to psycopg2 on SUPABASE_DB_URL."""
    wanted = [str(n).strip() for n in (names or []) if str(n).strip()]
    if not wanted:
        return {}
    if conn_factory is None:
        def conn_factory():
            import psycopg2  # lazy: module imports without psycopg2 installed
            return psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    out = {}
    try:
        with conn_factory() as conn:
            cur = conn.cursor()
            for nm in wanted:
                cur.execute("SELECT dosage, dosage_freq, dosage_timing "
                            "FROM fmp_newapp.products WHERE lower(product_name)=lower(%s) "
                            "LIMIT 1", (nm,))
                row = cur.fetchone()
                if row:
                    parts = [str(p).strip() for p in row if p and str(p).strip()]
                    out[nm.lower()] = " ".join(parts)
    except Exception as ex:
        print(f"[fmp-dosing] failed: {ex!r}", flush=True)
        return {}
    return out
```

  Then in `synthesize`, **drop `dosing` from the LLM schema** so it never produces it: change the JSON shape in `sys_prompt` from `...\"ffs\":[str],\"dosing\":str}]}` to `...\"ffs\":[str]}]}`, and remove any mention of dosing in the instructions. (The LLM now returns `{title,meaning,patterns,ffs}` per layer.)

- [ ] **Step 4: Run → PASS.** **Step 5:** Full vault test file green. **Step 6: Commit** (vault auto-snapshot).

---

## Task 4: learned overrides — `build_overrides`

**Files:** Modify `02 Skills/e4l_synthesis.py`; Test `02 Skills/tests/test_e4l_synthesis.py`

- [ ] **Step 1: Failing test**

```python
def test_build_overrides_pattern_to_product_recent_wins():
    from e4l_synthesis import build_overrides
    corrections = [
        {"created_at": "2026-05-01T00:00:00Z", "content": {"layers": [
            {"patterns": ["ES2"], "remedy": "Old Product"}]}},
        {"created_at": "2026-06-01T00:00:00Z", "content": {"layers": [
            {"patterns": ["ES2", "MB6"], "remedy": "New Product"},
            {"patterns": ["ED9"], "remedy": ""}]}},   # blank remedy -> ignored
    ]
    ov = build_overrides(corrections)
    assert ov["ES2"]["product"] == "New Product"      # most-recent wins
    assert ov["MB6"]["product"] == "New Product"
    assert "ED9" not in ov                             # blank remedy not learned
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — add to `e4l_synthesis.py`:

```python
def build_overrides(corrections):
    """From confirmed corrections, learn item_code -> {product, ts}. Each confirmed
    layer maps all its patterns to its (non-blank) remedy; most-recent confirmation
    wins. `corrections` = [{created_at|scan_date, content:{layers:[{patterns,remedy}]}}]."""
    out = {}
    for c in corrections or []:
        ts = c.get("created_at") or c.get("scan_date") or ""
        for L in (c.get("content") or {}).get("layers") or []:
            remedy = (L.get("remedy") or "").strip()
            if not remedy:
                continue
            for code in (L.get("patterns") or []):
                code = str(code)
                prev = out.get(code)
                if prev is None or ts >= (prev.get("ts") or ""):
                    out[code] = {"product": remedy, "ts": ts}
    return out
```

- [ ] **Step 4: Run → PASS** (also add an integration assert: with these overrides, `to_portal_content` pins a layer whose `patterns` include `ES2` to `New Product` — stub a synth whose layer has `ffs:["Whatever"]` + `patterns:["ES2"]` and a catalog containing `New Product`; assert `remedy == "New Product"`). **Step 5:** Full vault test file. **Step 6: Commit** (vault auto-snapshot).

---

## Task 5: importer wiring — FMP dosing + corrections → overrides

**Files:** Modify `02 Skills/e4l-portal-import.py` (vault); no unit test (manual smoke)

- [ ] **Step 1:** Add a corrections fetcher near `fetch_history`:

```python
def fetch_corrections():
    """Pull confirmed corrections from the live console (the learning signal)."""
    try:
        key = os.environ["CONSOLE_SECRET"]
        url = "https://illtowell.com/api/console/biofield/corrections?since=2000-01-01"
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        return json.load(urllib.request.urlopen(req, timeout=20)).get("corrections", [])
    except Exception as ex:
        print(f"[corrections] skipped: {ex!r}", flush=True)
        return []
```

- [ ] **Step 2:** In `main()`, after `synth["layers"] = E.order_layers_by_pattern_count(...)` and BEFORE `content = E.to_portal_content(...)`, build the dosing map + overrides and pass them:

```python
    overrides = E.build_overrides(fetch_corrections())
    # resolved FF names across all layers (after overrides) -> FMP dosing
    ff_names_used = []
    for L in synth.get("layers") or []:
        for f in (L.get("ffs") or []):
            if f and f not in ff_names_used:
                ff_names_used.append(f)
    for o in overrides.values():
        for p in (o.get("product") or "").split(" + "):
            if p.strip() and p.strip() not in ff_names_used:
                ff_names_used.append(p.strip())
    dosing_map = E.fmp_dosing_map(ff_names_used)
    content = E.to_portal_content(synth, catalog, dosing_map=dosing_map, overrides=overrides)
```

- [ ] **Step 3:** Syntax check (do NOT run the live importer): `/Library/Developer/CommandLineTools/usr/bin/python3 -m py_compile "/Users/remedymatch/AI-Training/02 Skills/e4l-portal-import.py"` → expect `py OK` (no output = success).

- [ ] **Step 4: Commit** (vault auto-snapshot). Note: a real end-to-end run happens in Task 7.

---

## Task 6: editor "Sync order list" button + retain patterns — `console-biofield-portal.html`

**Files:** Modify `static/console-biofield-portal.html` (DEPLOY-CHAT worktree); no unit test (JS check + manual)

- [ ] **Step 1: Catalog name→slug map.** In `loadCatalog()`, after building the datalist, store a global map. Add near the top of the `<script>`: `let CAT_BY_NAME = {};` and inside `loadCatalog`, for each product `p`: `CAT_BY_NAME[(p.name||'').trim().toLowerCase()] = p.slug;` (the `/catalog` response items carry `name` + `slug`).

- [ ] **Step 2: Retain patterns through edit.** The layer form drops `patterns`. In `addLayer(L)`, stash them on the element: when building the layer element, set `el.dataset.patterns = JSON.stringify(L && L.patterns || []);`. In `collectLayers()`, include them: add `patterns: JSON.parse(el.dataset.patterns || '[]')` to each collected layer object. (So load → edit → publish preserves `patterns`, which flows into the confirmed content + the corrections log.)

- [ ] **Step 3: Sync button.** Add a button next to "+ Add remedy" (near `id="reitems"`): `<button class="btn ghost sm" onclick="syncOrderFromLayers()">Sync order list from layers</button>`. Implement:

```javascript
function syncOrderFromLayers(){
  const layers = collectLayers();
  const existing = {}; collectItems().forEach(it => existing[it.slug] = it);  // keep qty/price
  const slugs = [];
  layers.forEach(L => (L.remedy||'').split('+').forEach(part => {
    const slug = CAT_BY_NAME[part.trim().toLowerCase()];
    if(slug && !slugs.includes(slug)) slugs.push(slug);
  }));
  $('reitems').innerHTML = '';
  slugs.forEach(slug => addItem(existing[slug] || {slug, qty: 1}));
  if($('reitems').children.length === 0) { /* leave empty */ }
  setStatus(`Order list synced from layers — ${slugs.length} item(s).`, true);
}
```

(`collectLayers`, `collectItems`, `addItem`, `setStatus`, `$` already exist. `addItem(it)` takes `{slug, qty, price_cents?}`.)

- [ ] **Step 4: Verify.** Extract the main `<script>` and `node --check` it (balanced braces / valid template literals). Run the full deploy-chat suite (a served-page test must still 200): baseline green. Brand rules: no emojis, "Order" not "Reorder".

- [ ] **Step 5: Commit**
```bash
cd /tmp/wt-deploy-chat-5326cc61
git add static/console-biofield-portal.html
git commit -m "editor: Sync order list from layers + retain layer patterns"
```

---

## Task 7: validate on Othon + finish

**Files:** none (validation)

- [ ] **Step 1:** Re-run the synthesis on Othon locally (dry — write the seed, do NOT auto-publish over his confirmed report):
  `cd "/Users/remedymatch/AI-Training" && doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python "02 Skills/e4l-portal-import.py" --email backdoc.molina@gmail.com` (no `--publish-draft`).
- [ ] **Step 2:** Open the written seed JSON in `~/AI-Training/05 Clients/`. Confirm: no invented names (every `remedy` resolves to an active product or is blank); `dosing` fields carry real FMP protocol (or blank, never "take as directed"); `patterns` present per layer; the previously-confirmed patterns (e.g. L2/L6) now reuse Glen's BFA / Macular Wellness choices via the override map. Report the before/after editing burden.
- [ ] **Step 3:** Push the deploy-chat branch + open a PR for the editor change (Task 6); the vault changes ship via auto-snapshot.
```bash
cd /tmp/wt-deploy-chat-5326cc61 && git push -u origin sess/5326cc61-synthfidelity
gh pr create --base main --title "Editor: Sync order list from layers + retain patterns" --body "Part of the E4L synthesis-fidelity work (vault synthesis changes ship via auto-snapshot). Adds the Sync-order-list button + retains layer patterns for the learning loop."
```

---

## Self-Review notes
- **Spec coverage:** active catalog (T1), FF validation + clean schema + retained patterns (T2), FMP dosing + LLM stops dosing (T3), learned overrides (T4), importer wiring (T5), editor sync button + patterns preservation (T6), Othon validation (T7). Blank-on-no-match + most-recent-conflict rule encoded in T2/T4.
- **Type consistency:** `to_portal_content(synth, catalog, dosing_map=None, overrides=None)`; `fmp_dosing_map(names, conn_factory=None) -> {name_lower: str}`; `build_overrides(corrections) -> {item_code: {product, ts}}`; overrides applied by splitting `product` on " + "; dosing joined with "; "; layer dict gains `patterns`.
- **Verify during impl:** the existing `02 Skills/tests/test_e4l_synthesis.py` may assert the OLD `to_portal_content` dosing/remedy behavior — update those to the new contract (T2). Confirm `deploy-chat311` venv import path for the vault tests (use `PYTHONPATH=.`). Confirm `/catalog` response items expose `name`+`slug` (T6). The corrections endpoint returns confirmed content WITH `patterns` only for reports confirmed AFTER T6 ships (Othon's existing confirmation predates patterns-retention — his override learning starts from his next confirmation, OR re-confirm him once T6 is live).
