# Purity Phase 3 (slice 1) — Fullscript card purity badge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Annotate each product on the client's Fullscript portal card with its confirmed purity color — a green/yellow dot+tooltip, and a greyed "contains excipients we avoid" note for confirmed reds — gated behind a new `PURITY_BADGES_ENABLED` flag so it ships dark.

**Architecture:** Backend `_fullscript_for` enriches each product dict with a `purity: {color}` field, looked up from confirmed `product_ratings` rows by `product_key = "fullscript::" + product_slug`, only when the flag is on. The static portal JS (`fullscriptBodyHtml` in `static/client-portal.html`) reads that field and renders the dot/tooltip/greyed treatment. `fullscriptBodyHtml` is a pure `fsData → HTML string` function, so the frontend is verified by evaluating it headlessly with a synthetic payload.

**Tech Stack:** Python 3.9 / Flask (sqlite dev, Postgres adapter prod), vanilla JS in a static HTML portal, headless Chrome for render-verify.

## Global Constraints

- **Flag default OFF, byte-identical when off.** `PURITY_BADGES_ENABLED` mirrors `_fullscript_enabled()` exactly (env var, default off). When off, `_fullscript_for` adds NO `purity` key to any product — the payload is byte-identical to today. (Verify with a flag-off test.)
- **Confirmed only.** A color badge appears ONLY for a `product_ratings` row at `status == 'confirmed'`. Screened-but-unconfirmed, unrated, or missing rows produce no `purity` field → no badge → the card behaves exactly as it does pre-rating.
- **Reds are shown greyed + noted, NOT suppressed** (spec Section 3, refined 2026-07-27): a confirmed red product stays in the list, greyed, labeled "contains excipients we avoid", keeping its `best_ff` pairing. Green/yellow get a color dot + tooltip. Tooltip copy VERBATIM: green = "Meets our purity standard", yellow = "Minor filler only".
- **Brand colors** (see reference_remedy_match_brand_colors): green dot `#2f6f5e` (brand green-teal), yellow dot `#c9a227` (gold).
- **Best-effort, never break the card.** The purity enrichment lives inside `_fullscript_for`'s existing `try/except … return None` and must not raise; a lookup failure leaves products un-badged but the card still renders. Reads are Postgres-safe (scalar `fetchone()`, no PRAGMA/lastrowid).
- **No new dependencies.**

---

### Task 1: `product_ratings.confirmed_color`

**Files:**
- Modify: `dashboard/product_ratings.py` (append a function)
- Test: `tests/test_product_ratings.py` (append)

**Interfaces:**
- Produces: `confirmed_color(cx, product_key) -> str | None` — returns the row's `color` iff it exists AND `status == 'confirmed'`; otherwise `None`. Row-factory-agnostic (indexes the first column of a scalar select), so it is safe whether or not the caller set `cx.row_factory`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_product_ratings.py — append
def test_confirmed_color_only_for_confirmed_rows():
    import sqlite3
    from dashboard import product_ratings as pr
    cx = sqlite3.connect(":memory:")
    pr.init_tables(cx)
    # a screened (not confirmed) green row -> no color yet
    pr.record_screen(cx, "fullscript::a", brand="B", product_name="A",
                     other_ingredients_raw="cellulose", other_ingredients_parsed=["cellulose"],
                     screen={"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    assert pr.confirmed_color(cx, "fullscript::a") is None      # screened, not confirmed
    # a confirmed red row -> its color
    pr.record_screen(cx, "fullscript::r", brand="B", product_name="R",
                     other_ingredients_raw="magnesium stearate", other_ingredients_parsed=["magnesium stearate"],
                     screen={"color": "red", "red_hits": ["stearate"], "yellow_hits": [], "avoidlist_version": "v1"})
    pr.confirm(cx, "fullscript::r")                              # red: screened -> confirmed
    assert pr.confirmed_color(cx, "fullscript::r") == "red"
    # a missing key -> None
    assert pr.confirmed_color(cx, "fullscript::missing") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_product_ratings.py::test_confirmed_color_only_for_confirmed_rows -v`
Expected: FAIL — `AttributeError: module 'dashboard.product_ratings' has no attribute 'confirmed_color'`

- [ ] **Step 3: Write the implementation**

Append to `dashboard/product_ratings.py`:

```python
def confirmed_color(cx, product_key):
    """The confirmed color for a product, or None. Returns the row's color ONLY
    when status == 'confirmed'; a screened/ai_draft/unrated/missing row returns
    None. The Fullscript badge reader (Phase 3) calls this so an unconfirmed or
    unrated product carries no badge. Scalar select (indexes column 0), so it is
    correct with any row_factory and Postgres-safe."""
    row = cx.execute(
        "SELECT color FROM product_ratings WHERE product_key=? AND status='confirmed'",
        (product_key,)).fetchone()
    return row[0] if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_product_ratings.py::test_confirmed_color_only_for_confirmed_rows -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/product_ratings.py tests/test_product_ratings.py
git commit -m "feat(purity): confirmed_color(cx, key) reader for the badge"
```

---

### Task 2: `PURITY_BADGES_ENABLED` flag + enrich `_fullscript_for`

**Files:**
- Modify: `app.py` — add `_purity_badges_enabled()` near `_fullscript_enabled()` (~app.py:18269 area); enrich products in `_fullscript_for` (the `g["products"].append({...})` block, ~app.py:23239, then a post-build enrichment pass before `return {...}`)
- Test: `tests/test_purity_badge_enrich.py` (create)

**Interfaces:**
- Consumes: `product_ratings.confirmed_color(cx, product_key) -> str|None` (Task 1).
- Produces:
  - `_purity_badges_enabled() -> bool` — env `PURITY_BADGES_ENABLED`, default off, same truthy set as `_fullscript_enabled`.
  - `_fullscript_for` products gain `"purity": {"color": <"red"|"yellow"|"green">}` ONLY when the flag is on AND a confirmed rating exists for `fullscript::<slug>`. Otherwise no `purity` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_purity_badge_enrich.py
import sqlite3, pytest, app as app_mod
from dashboard import product_ratings as pr, fullscript as fs


@pytest.fixture
def cx_db(monkeypatch, tmp_path):
    db = str(tmp_path / "b.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); fs.init_tables(cx)
    # one confirmed-green product keyed fullscript::slug-green
    pr.record_screen(cx, "fullscript::slug-green", brand="B", product_name="G",
                     other_ingredients_raw="cellulose", other_ingredients_parsed=["cellulose"],
                     screen={"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    # green must pass through ai_draft before confirm
    pr.set_tier2(cx, "fullscript::slug-green", None, "{}")
    pr.confirm(cx, "fullscript::slug-green")
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    return db


def _enrich(groups):
    # drive just the enrichment helper the task adds (see impl): given groups of
    # products with product_slug, attach purity when flag on.
    return app_mod._enrich_fullscript_purity(groups)


def test_flag_off_adds_no_purity(monkeypatch, cx_db):
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: False)
    groups = [{"products": [{"product_slug": "slug-green"}]}]
    _enrich(groups)
    assert "purity" not in groups[0]["products"][0]


def test_flag_on_adds_confirmed_color(monkeypatch, cx_db):
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: True)
    groups = [{"products": [{"product_slug": "slug-green"},
                            {"product_slug": "slug-none"}]}]
    _enrich(groups)
    assert groups[0]["products"][0]["purity"] == {"color": "green"}   # confirmed
    assert "purity" not in groups[0]["products"][1]                    # no confirmed row -> no badge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_badge_enrich.py -v`
Expected: FAIL — `AttributeError: … has no attribute '_enrich_fullscript_purity'` (and `_purity_badges_enabled`).

- [ ] **Step 3: Write the implementation**

Add the flag helper right after `_fullscript_enabled()` in `app.py`:

```python
def _purity_badges_enabled():
    """Default OFF. When off, _fullscript_for adds no `purity` key, so the
    portal payload stays byte-identical. Mirrors _fullscript_enabled."""
    return (os.environ.get("PURITY_BADGES_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _enrich_fullscript_purity(groups):
    """Attach `purity: {color}` to each product carrying a CONFIRMED rating,
    when the flag is on. In place; best-effort -- any failure leaves products
    un-badged and never raises (the card must still render)."""
    if not _purity_badges_enabled():
        return groups
    try:
        from dashboard import product_ratings as _pr
        with db.connect(LOG_DB) as cx:
            _pr.init_tables(cx)
            for g in groups:
                for p in g.get("products", []):
                    slug = (p.get("product_slug") or "").strip()
                    if not slug:
                        continue
                    color = _pr.confirmed_color(cx, "fullscript::" + slug)
                    if color:
                        p["purity"] = {"color": color}
    except Exception:
        pass
    return groups
```

Then, in `_fullscript_for`, enrich immediately before the return. Change:

```python
        return {"dispensary_url": _fullscript_dispensary_url(), "groups": groups}
```

to:

```python
        _enrich_fullscript_purity(groups)
        return {"dispensary_url": _fullscript_dispensary_url(), "groups": groups}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_badge_enrich.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add app.py tests/test_purity_badge_enrich.py
git commit -m "feat(purity): PURITY_BADGES_ENABLED flag + confirmed-color enrichment in _fullscript_for"
```

---

### Task 3: Render the badge in `fullscriptBodyHtml`

**Files:**
- Modify: `static/client-portal.html` — the product `.map(p => …)` inside `fullscriptBodyHtml` (~line 806-811)
- Verify: headless eval of `fullscriptBodyHtml` with a synthetic payload (no pytest; see Step 2/4)

**Interfaces:**
- Consumes: each product `p` may now carry `p.purity.color` (`"red"|"yellow"|"green"`) from Task 2.
- Produces: rendered `<li>` markup — a color dot with a `title` tooltip for green/yellow; a greyed `<li>` plus a "contains excipients we avoid" note for red; unchanged markup when `p.purity` is absent.

- [ ] **Step 1: Write the implementation**

In `static/client-portal.html`, replace the product-map body (the `const brand = …` through the `return \`<li>…\`;` lines inside `fullscriptBodyHtml`) with:

```javascript
      const brand = p.brand ? `<span class="small muted"> — ${esc(p.brand)}</span>` : "";
      const ff = (p.ff && p.ff.name)
        ? `<span class="small muted"> · you also have <strong>${esc(p.ff.name)}</strong></span>` : "";
      const why = p.reason ? `<div class="small muted">${esc(p.reason)}</div>` : "";
      // Purity badge (Phase 3, flag-gated server-side): a color dot+tooltip for
      // green/yellow; a greyed row + note for a confirmed red. Absent p.purity
      // renders exactly as before.
      const pc = p.purity && p.purity.color;
      const dot = (bg, label) => `<span title="${esc(label)}" style="display:inline-block;width:.6em;height:.6em;border-radius:50%;background:${bg};margin-right:.4em;vertical-align:middle"></span>`;
      let badge = "", liStyle = "", note = "";
      if (pc === "green") badge = dot("#2f6f5e", "Meets our purity standard");
      else if (pc === "yellow") badge = dot("#c9a227", "Minor filler only");
      else if (pc === "red") { liStyle = ' style="opacity:.55"'; note = `<div class="small muted">contains excipients we avoid</div>`; }
      return `<li${liStyle}>${badge}<a href="/fs/${fsToken}/${esc(p.product_slug||"")}" target="_blank" rel="noopener">${esc(p.name||"")}</a>${brand}${ff}${why}${note}</li>`;
```

- [ ] **Step 2: Verify it fails first (markup absent) — headless eval**

Serve the file and eval the function with a synthetic payload. Run a local static server from the worktree root, open the page headless (Chrome via the mcp browser tools, or any headless runner), and evaluate:

```javascript
window.token = "TESTTOK";
window.fullscriptBodyHtml({
  dispensary_url: "https://us.fullscript.com/x",
  groups: [{ heading: "From your scan", products: [
    { name: "Clean One", product_slug: "clean-1", purity: { color: "green" } },
    { name: "Filler One", product_slug: "filler-1", purity: { color: "yellow" } },
    { name: "Dirty One", product_slug: "dirty-1", purity: { color: "red" } },
    { name: "Unrated One", product_slug: "unrated-1" } ]}]
});
```

Before Step 1's edit, the returned string contains none of the badge markers below — that is the failing baseline. Record it, then apply Step 1.

- [ ] **Step 3: (implementation already applied in Step 1)**

- [ ] **Step 4: Verify it passes — assert on the returned HTML string**

Re-run the Step 2 eval after the edit and assert the returned string satisfies ALL of:
- contains `title="Meets our purity standard"` (green dot present)
- contains `title="Minor filler only"` (yellow dot present)
- contains `#2f6f5e` and `#c9a227` (brand green + gold)
- the "Dirty One" `<li>` carries `opacity:.55` AND the string contains `contains excipients we avoid`
- the "Unrated One" `<li>` has NO dot/opacity/note (its slice of the string, between `unrated-1` link and the next `</li>`, contains none of the markers)

Expected: all assertions hold. If driving via the mcp Chrome tools, capture the eval result and check these substrings; if using a node+jsdom runner, load the file's script and call the global `fullscriptBodyHtml`. Either way the assertion target is the returned HTML string.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add static/client-portal.html
git commit -m "feat(purity): render color dot+tooltip / greyed-red note on the Fullscript card"
```

---

## Post-implementation (controller, after all tasks reviewed)

1. **Merge → deploy** (ships dark; `PURITY_BADGES_ENABLED` unset, so no client sees a change).
2. **Render-verify on prod against a real portal** that has confirmed-rated Fullscript matches: drive the live client portal headless with the flag temporarily on in a scratch check (or verify the payload carries `purity` for a known confirmed product via `/api/portal/<token>` with the flag on), confirming the dot/tooltip/greyed markup renders and mobile layout is intact (CSSOM, not window-resize — see feedback_portal_render_verify).
3. **Flip** `PURITY_BADGES_ENABLED=1` in `doppler … -p remedy-match -c prd`, restart Render, re-verify the live card. This is the client-facing go-live and is Glen's call (flag flip = a deploy/restart).

## Self-Review

**Spec coverage (Section 3 — badge slice):**
- Confirmed-color lookup keyed `fullscript::<slug>`, confirmed-only → Task 1 (`confirmed_color`) + Task 2 (enrichment). ✅
- Flag-gated, ships dark, byte-identical when off → Task 2 (`_purity_badges_enabled`, flag-off test). ✅
- Green/yellow dot + tooltip (verbatim copy), reds greyed + "contains excipients we avoid" note, unrated unchanged → Task 3. ✅
- Decoupled from the seed file (reads ratings, not the seed) → enrichment reads `product_ratings`. ✅
- Render-verify then flag-flip → Post-implementation steps. ✅
- Aggregate stat (slice 2) is explicitly out of scope here. ✅

**Placeholder scan:** none — every code step carries complete code; Task 3's verification names exact substrings, not "assert it looks right."

**Type consistency:** `confirmed_color(cx, product_key) -> str|None` (Task 1) is called with `"fullscript::" + slug` in Task 2; the `purity` field shape `{"color": <str>}` produced in Task 2 is read as `p.purity.color` in Task 3. Flag helper name `_purity_badges_enabled` is consistent across Task 2 and the frontend gate note. Consistent.

**Note on Task 2's green-confirm test:** `set_tier2(..., None, "{}")` then `confirm(...)` mirrors exactly how the 24 live rows were confirmed (green/yellow pass screened→ai_draft→confirmed; the tier-2 score is a deferred placeholder). Reds would `confirm()` directly from screened.
