# Product Page Images — Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a visitor tap a heart on their favorite image in each type's grid; record the vote in `sales_page_votes` with the chosen image's `prompt_variant_id` + `model_id` captured at pick time, so Phase C can aggregate per-variation and per-model across products.

**Architecture:** Reuse the existing `sales_page_votes` table + `record_pick` upsert (one vote/session/type). Denormalize the two attribution tags onto the vote (looked up from `sales_page_images` in a new vote endpoint). The data endpoint adds the visitor's current picks to the grouped payload; the template renders a heart overlay per tile with optimistic voting. All new behavior behind a new flag `SALES_PAGES_IMAGE_VOTE`, shipping dark.

**Tech Stack:** Python 3 / Flask, SQLite (`LOG_DB`), vanilla JS template (`static/begin-product.html`), pytest.

## Global Constraints

- New behavior gated by env flag `SALES_PAGES_IMAGE_VOTE` (truthy = `1`/`true`/`yes`); read as module global `_SALES_IMAGE_VOTE_ENABLED` in `app.py`. OFF = no `picks` payload, no heart UI, vote endpoint 404s. Independent of `_SALES_IMAGE_PICK_ENABLED` (the dark pairwise pick).
- A vote is a positive favorite: `variant` must be an int ≥ 1 (NO "neither"/0 option — that belonged to the pairwise A/B).
- Attribution tags (`prompt_variant_id`, `model_id`) are denormalized onto the vote **at pick time**, captured from the chosen image's tags. They refresh on a re-vote.
- One vote per (session, product, kind) — the existing `UNIQUE(session_id, product_slug, kind)` upsert. Anonymous via cookie `amg_session`; email from `get_authenticated_user` if logged in.
- Picks only — NO impression/view logging (deferred to Phase C).
- Tests: pytest, `sqlite3.connect(":memory:")` per test, import `dashboard.*` directly, no live network. New file `tests/test_sales_pages_phase_b.py`. Follow `tests/test_sales_pages_phase4b.py` style.
- Sandbox: use `python3` (no `python`). `import app` CANNOT run here (Pinecone client built at import → network auth) — verify app.py edits with `python3 -m py_compile app.py` plus the unit-tested helpers they call. Full route/template behavior = manual verification with real env.
- Work in worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at Phase A tip `0a85ac1` which is already in `main`). Commit per task. No edits to `main`.

---

### Task 1: Vote attribution columns + tagged record_pick

**Files:**
- Modify: `dashboard/sales_votes.py`
- Test: `tests/test_sales_pages_phase_b.py`

**Interfaces:**
- Produces: `record_pick(cx, slug, kind, variant, session_id, email="", prompt_variant_id=None, model_id=None)` — persists the two tags on insert AND refreshes them on the upsert update. `init_table` migrates the two columns idempotently.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase_b.py
import sqlite3
from dashboard import sales_votes as sv

def _cx(): return sqlite3.connect(":memory:")

def _row(cx, slug, kind, session_id):
    return cx.execute("SELECT chosen_variant, prompt_variant_id, model_id FROM sales_page_votes "
                      "WHERE product_slug=? AND kind=? AND session_id=?", (slug, kind, session_id)).fetchone()

def test_record_pick_persists_tags():
    cx = _cx()
    sv.record_pick(cx, "p", "botanical", 2, "s1", prompt_variant_id=7, model_id="imagen-4")
    assert _row(cx, "p", "botanical", "s1") == (2, 7, "imagen-4")

def test_record_pick_revote_updates_variant_and_tags():
    cx = _cx()
    sv.record_pick(cx, "p", "botanical", 1, "s1", prompt_variant_id=3, model_id="flux-1.1-pro")
    sv.record_pick(cx, "p", "botanical", 4, "s1", prompt_variant_id=9, model_id="recraft-v3")
    assert _row(cx, "p", "botanical", "s1") == (4, 9, "recraft-v3")
    # still one row for this (session, product, kind)
    n = cx.execute("SELECT COUNT(*) FROM sales_page_votes WHERE product_slug='p' AND kind='botanical' AND session_id='s1'").fetchone()[0]
    assert n == 1

def test_record_pick_backward_compatible_without_tags():
    cx = _cx()
    sv.record_pick(cx, "p", "mechanism", 1, "s1")   # 6-arg legacy call (Phase-4 style)
    assert _row(cx, "p", "mechanism", "s1") == (1, None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_b.py -q`
Expected: FAIL (no such column prompt_variant_id / record_pick has no tag kwargs).

- [ ] **Step 3: Write minimal implementation**

In `dashboard/sales_votes.py`, extend `init_table` (after the CREATE TABLE, before `cx.commit()`):

```python
    for _col, _decl in (("prompt_variant_id", "INTEGER"), ("model_id", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE sales_page_votes ADD COLUMN {_col} {_decl}")
        except Exception:
            pass
```

Replace `record_pick` with:

```python
def record_pick(cx, slug, kind, variant, session_id, email="", prompt_variant_id=None, model_id=None):
    init_table(cx); now = _now(); email = (email or "").strip().lower()
    cx.execute("INSERT INTO sales_page_votes (product_slug, kind, chosen_variant, session_id, email, "
               "created_at, updated_at, prompt_variant_id, model_id) "
               "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id, product_slug, kind) DO UPDATE SET "
               "chosen_variant=excluded.chosen_variant, "
               "email=CASE WHEN excluded.email!='' THEN excluded.email ELSE sales_page_votes.email END, "
               "updated_at=excluded.updated_at, "
               "prompt_variant_id=excluded.prompt_variant_id, model_id=excluded.model_id",
               (slug, kind, int(variant), session_id, email, now, now, prompt_variant_id, model_id))
    if email and session_id:
        cx.execute("UPDATE sales_page_votes SET email=? WHERE session_id=? AND email=''", (email, session_id))
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_b.py -q`
Expected: PASS (3 tests). Also confirm no regression in the Phase-4 vote tests: `python3 -m pytest tests/test_sales_pages_phase4b.py -q` → the 8 pure-logic tests still pass (7 pre-existing Pinecone-import failures are environmental, not from this change).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_votes.py tests/test_sales_pages_phase_b.py
git commit -m "feat(sales-img): vote attribution columns + tagged record_pick (Phase B task 1)"
```

---

### Task 2: tags_for image lookup

**Files:**
- Modify: `dashboard/sales_images.py`
- Test: `tests/test_sales_pages_phase_b.py` (append)

**Interfaces:**
- Consumes: `record_image` (Phase A).
- Produces: `tags_for(cx, slug, kind, variant) -> (prompt_variant_id, model_id)` — tags of the ready image at that slot, or `(None, None)` if absent.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_b.py
from dashboard import sales_images as si

def test_tags_for_returns_image_tags():
    cx = _cx()
    si.record_image(cx, "p", "botanical", 3, "botanical-3.png", prompt_variant_id=5, model_id="recraft-v3")
    assert si.tags_for(cx, "p", "botanical", 3) == (5, "recraft-v3")

def test_tags_for_missing_slot_returns_none_pair():
    cx = _cx()
    assert si.tags_for(cx, "p", "botanical", 2) == (None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_b.py -q`
Expected: FAIL (`tags_for` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_images.py`:

```python
def tags_for(cx, slug, kind, variant):
    """(prompt_variant_id, model_id) for the ready image at (slug, kind, variant), or (None, None)."""
    init_tables(cx)
    r = cx.execute("SELECT prompt_variant_id, model_id FROM sales_page_images "
                   "WHERE product_slug=? AND kind=? AND variant=? AND state='ready' "
                   "ORDER BY id DESC LIMIT 1", (slug, kind, int(variant))).fetchone()
    return (r[0], r[1]) if r else (None, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_b.py -q`
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_images.py tests/test_sales_pages_phase_b.py
git commit -m "feat(sales-img): tags_for image lookup (Phase B task 2)"
```

---

### Task 3: Vote flag + vote endpoint

**Files:**
- Modify: `app.py` (flag near line 2543; new route near the existing `begin_product_image_pick`, ~line 3814)

**Interfaces:**
- Consumes: `_SALES_IMAGE_VOTE_ENABLED`, `sales_images.tags_for` (Task 2), `sales_votes.record_pick` (Task 1) + `sales_votes.get_picks` (existing), `sales_image_prompts.IMAGE_KINDS`, `get_authenticated_user`, `_get_product`.

- [ ] **Step 1: Add the flag**

In `app.py`, right after the line `_SALES_IMAGE_VARIATIONS_ENABLED = ...` (≈ line 2543), add:

```python
_SALES_IMAGE_VOTE_ENABLED = os.environ.get("SALES_PAGES_IMAGE_VOTE", "").strip().lower() in ("1", "true", "yes")
```

- [ ] **Step 2: Add the vote endpoint**

Add near the existing `@app.route("/begin/product-image-pick/<slug>", ...)` handler (so the image routes stay together):

```python
@app.route("/begin/product-image-vote/<slug>", methods=["POST"])
def begin_product_image_vote(slug):
    from dashboard import sales_image_prompts as _sip
    if not _SALES_IMAGE_VOTE_ENABLED or not _get_product(slug):
        return ("", 404)
    data = request.get_json(silent=True) or {}
    kind = (data.get("kind") or "").strip()
    if kind not in _sip.IMAGE_KINDS:
        return jsonify({"ok": False}), 400
    try:
        variant = int(data.get("variant"))
    except (TypeError, ValueError):
        return jsonify({"ok": False}), 400
    if variant < 1:
        return jsonify({"ok": False}), 400
    session_id = request.cookies.get("amg_session", "")
    au = get_authenticated_user(request)
    email = ((au or {}).get("email") or "").strip().lower() if au else ""
    from dashboard import sales_images as _si, sales_votes as _sv
    with sqlite3.connect(LOG_DB) as cx:
        pv, mid = _si.tags_for(cx, slug, kind, variant)
        _sv.record_pick(cx, slug, kind, variant, session_id, email, prompt_variant_id=pv, model_id=mid)
        picks = _sv.get_picks(cx, slug, session_id=session_id, email=email)
    return jsonify({"ok": True, "picks": picks})
```

- [ ] **Step 3: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds (no syntax error). DO NOT run `import app` (Pinecone-at-import fails in this sandbox).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): SALES_PAGES_IMAGE_VOTE flag + vote endpoint (Phase B task 3)"
```

---

### Task 4: Data endpoint — include current picks

**Files:**
- Modify: `app.py` — the grouped branch inside `if _SALES_IMAGE_VARIATIONS_ENABLED:` (≈ lines 3266-3269)

**Interfaces:**
- Consumes: `_SALES_IMAGE_VOTE_ENABLED`, `sales_votes.get_picks`, `get_authenticated_user`.

- [ ] **Step 1: Add picks to the grouped body**

In `app.py`, the grouped branch currently reads:

```python
                    if _SALES_IMAGE_VARIATIONS_ENABLED:
                        _grouped = _si2.display_images_grouped(_cx2, slug)
                        _state = _si2.images_grouped_state(_cx2, slug)
                        _img_sec["body"] = {"grouped": _grouped, "state": _state, "target": 8}
```

Append, immediately after the `_img_sec["body"] = {...}` line (same indentation, still inside the `if _SALES_IMAGE_VARIATIONS_ENABLED:` block):

```python
                        if _SALES_IMAGE_VOTE_ENABLED:
                            from dashboard import sales_votes as _sv2
                            _vsess = request.cookies.get("amg_session", "")
                            _vau = get_authenticated_user(request)
                            _vem = ((_vau or {}).get("email") or "").strip().lower() if _vau else ""
                            _img_sec["body"]["picks"] = _sv2.get_picks(_cx2, slug, session_id=_vsess, email=_vem)
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds. (Do NOT run `import app`.)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): data endpoint includes visitor picks (Phase B task 4)"
```

---

### Task 5: Template — heart overlay + optimistic vote

**Files:**
- Modify: `static/begin-product.html` — `renderGrouped` (+ its two call sites) and a `vote()` helper + CSS

**Interfaces:**
- Consumes: the grouped payload's optional `picks: {botanical: variant|null, mechanism: variant|null}` (Task 4); the vote endpoint `POST /begin/product-image-vote/<slug>` (Task 3).

- [ ] **Step 1: Replace `renderGrouped` and add `vote()`**

In `static/begin-product.html`, replace the entire `function renderGrouped(grouped, target){ ... }` with the version below, which (a) takes a `picks` arg, (b) renders a heart per tile when picks is provided, (c) remembers the last grouped/target, and adds a `vote()` helper right after it. (`wrap`, `slug`, `BASE`, `KIND_LABEL` are already in scope.)

```javascript
      var _picks = {}, _lastGrouped = null, _lastTarget = 8, _voteOn = false;

      function renderGrouped(grouped, target, picks){
        if (picks !== undefined && picks !== null){ _picks = picks; _voteOn = true; }
        _lastGrouped = grouped; _lastTarget = target;
        wrap.className = 'sp-img-groups'; wrap.innerHTML = '';
        ['botanical', 'mechanism'].forEach(function(kind){
          var tiles = (grouped && grouped[kind]) || [];
          var sec = document.createElement('div'); sec.className = 'sp-img-group';
          var h = document.createElement('div'); h.className = 'sp-img-group-title';
          h.textContent = (KIND_LABEL[kind] || kind) + (_voteOn ? ' — tap your favorite' : '');
          sec.appendChild(h);
          var grid = document.createElement('div'); grid.className = 'sp-img-grid';
          tiles.forEach(function(t){
            var fig = document.createElement('figure'); fig.className = 'sp-img-tile';
            var chosen = (_voteOn && _picks[kind] === t.variant);
            if (chosen){ fig.className += ' sp-img-picked'; }
            var el = document.createElement('img'); el.className = 'sp-product-img';
            el.src = t.url; el.alt = kind + ' image of the product';
            fig.appendChild(el);
            if (_voteOn){
              var btn = document.createElement('button'); btn.type = 'button';
              btn.className = 'sp-img-heart' + (chosen ? ' on' : '');
              btn.setAttribute('aria-label', 'Pick this as your favorite');
              btn.textContent = chosen ? '♥' : '♡';
              (function(k, v){ btn.addEventListener('click', function(){ vote(k, v); }); })(kind, t.variant);
              fig.appendChild(btn);
            }
            if (t.model_label){
              var cap = document.createElement('figcaption'); cap.className = 'sp-img-cap';
              cap.textContent = 'made with ' + t.model_label; fig.appendChild(cap);
            }
            grid.appendChild(fig);
          });
          for (var i = tiles.length; i < 4; i++){
            var ph = document.createElement('div'); ph.className = 'sp-img-tile sp-img-ph';
            ph.textContent = '…'; grid.appendChild(ph);
          }
          sec.appendChild(grid); wrap.appendChild(sec);
        });
      }

      function vote(kind, variant){
        _picks[kind] = variant;                                  // optimistic
        renderGrouped(_lastGrouped, _lastTarget, _picks);
        fetch(BASE + '/begin/product-image-vote/' + slug, {
          method: 'POST', credentials: 'same-origin',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({kind: kind, variant: variant})
        }).then(function(r){ return r.json(); })
          .then(function(d){ if (d && d.picks){ renderGrouped(_lastGrouped, _lastTarget, d.picks); } })
          .catch(function(){});
      }
```

- [ ] **Step 2: Pass `picks` at the two call sites**

There are two `renderGrouped(` calls in `renderImagesBody`. Update both to pass the picks:
- The direct render (was `renderGrouped(grouped, body.target);`) → `renderGrouped(grouped, body.target, body.picks);`
- Inside `startPoll` (was `if (b.grouped){ renderGrouped(b.grouped, b.target); }`) → `if (b.grouped){ renderGrouped(b.grouped, b.target, b.picks); }`

(When the vote flag is off, `body.picks`/`b.picks` is `undefined` → hearts stay hidden; existing behavior unchanged.)

- [ ] **Step 3: Add CSS**

Add to the page's `<style>` block, near the other `.sp-img-*` rules:

```css
.sp-img-tile{position:relative}
.sp-img-heart{position:absolute;top:6px;right:6px;border:none;background:rgba(0,0,0,.38);
  color:#fff;border-radius:50%;width:30px;height:30px;font-size:16px;line-height:30px;
  text-align:center;cursor:pointer;padding:0}
.sp-img-heart.on{background:rgba(220,40,80,.92)}
.sp-img-picked .sp-product-img{outline:3px solid rgba(220,40,80,.92);outline-offset:-3px;border-radius:8px}
```

- [ ] **Step 4: Verify**

Run:
```bash
grep -c "function renderGrouped" static/begin-product.html   # expect 1
grep -n "function vote(\|sp-img-heart\|body.picks\|b.picks\|sp-img-picked" static/begin-product.html
```
Expected: exactly one `renderGrouped`; `vote(` present; both call sites pass picks; CSS classes present. (Browser verification — hearts render, tapping moves the fill, reload keeps the pick — is manual, deferred to the human, like Phase A's template task.)

- [ ] **Step 5: Commit**

```bash
git add static/begin-product.html
git commit -m "feat(sales-img): heart-pick overlay + optimistic vote (Phase B task 5)"
```

---

### Task 6: Regression + flag-off parity

**Files:** none (verification)

- [ ] **Step 1: Run Phase B + Phase A unit suites**

Run: `python3 -m pytest tests/test_sales_pages_phase_b.py tests/test_sales_pages_phase_a.py -q`
Expected: all PASS (5 Phase B + 18 Phase A).

- [ ] **Step 2: Confirm flag-off parity**

With `SALES_PAGES_IMAGE_VOTE` unset: the data endpoint adds no `picks` key (the `if _SALES_IMAGE_VOTE_ENABLED:` block is skipped); the vote endpoint 404s; `renderGrouped` sees `picks === undefined` → no hearts. `record_pick`'s 6-arg legacy path (Phase-4) still works (covered by Task 1's backward-compat test). `python3 -m py_compile app.py` clean.

- [ ] **Step 3: Commit (if any fixups)**

```bash
git add -A && git commit -m "test(sales-img): Phase B regression pass" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** vote columns + tagged record_pick (T1), tags_for (T2), flag + vote endpoint (T3), data-endpoint picks (T4), heart overlay + optimistic vote + CSS (T5), regression + flag-off (T6). Spec sections 1-6, data flow, testing, and the "shared table / mutually-exclusive flags" + app-import notes all map to tasks. Impressions are explicitly out of scope (Phase C) — correctly no task. ✔

**Placeholder scan:** all code blocks complete; verifications are concrete commands. The only deferred item is the browser check of the heart UI (manual, like Phase A) — not a code gap. No "TODO/handle edge cases" in code.

**Type consistency:** `record_pick(..., prompt_variant_id=None, model_id=None)` (T1) matches the endpoint's call (T3) and the `_row`/test reads; `tags_for(cx, slug, kind, variant) -> (pv, mid)` (T2) matches the endpoint unpack `pv, mid = _si.tags_for(...)` (T3); `get_picks` returns `{botanical, mechanism}` consumed as `body.picks` in the template (T5) and emitted by the data endpoint (T4); `renderGrouped(grouped, target, picks)` (T5) matches both updated call sites (T5 step 2). The vote endpoint returns `{ok, picks}` consumed by `vote()` as `d.picks` (T5).
