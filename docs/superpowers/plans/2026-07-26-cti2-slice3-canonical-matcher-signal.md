# CTI-2 Slice 3 — canonical record as a gated signal at remedy-match review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the console FF-match review page, show the client's canonical record (conditions, terrain concerns, body systems, challenges, goals) as a distinct labeled block beside their scan-driven matches — without touching the ranking or the published draft, and never shown to the client.

**Architecture:** Backend — `api_console_ff_match_drafts_list` enriches each draft with a `canonical` block from `canonical_tags.get_person(email)` minus `tags`. Frontend — a new `static/js/ff-draft-canon.js` exports `renderCanonBlock(canonical)`; `static/console-ff-drafts.html` loads it and calls it in `card()` above the match items. Read-through, display-only.

**Tech Stack:** Python 3, Flask (`app.py`), `dashboard/canonical_tags.py`, `dashboard/ff_match_drafts.py`, vanilla JS. Tests mirror the app-reload harness and the `portal-documents.js` node-test pattern.

## Global Constraints

- **The `canonical` block is NEVER merged into `items`** and never fed to `ff_matcher`/`_make_ff_items_for`/the Pinecone query/`_ff_llm_rank`. The scan-driven ranking and the published `items` stay byte-for-byte unchanged.
- **`tags` is not surfaced** (CRM/GHL bucket). Fields = `conditions`, `terrain_concerns`, `body_systems` (discrete lists), `challenges`, `goals` (scalar strings).
- **Gated to the console.** Only `/api/console/ff-match-drafts` + `static/console-ff-drafts.html` change. The client card `/api/portal/<token>/ff-matches` is untouched.
- **Read-through, best-effort.** No writes; `get_person` is read-only; any canonical failure → empty block, the endpoint returns 200 unchanged.
- **Display-only.** The rendered block must NOT be an `.item`, so `collectItems`/`publish` can never pick it up.
- Console gate on the endpoint is unchanged. NEVER `cur.lastrowid` in new code.
- **Running tests:** run targeted files; do NOT run a bare full suite from a shell with real creds (sends real email) — use `bash ci/run-tests.sh` for the gate.

---

## File Structure

**Modify:**
- `app.py` — add `_ff_draft_canonical` helper; enrich `api_console_ff_match_drafts_list` (~line 22512).
- `static/console-ff-drafts.html` — load the new JS; call `renderCanonBlock(d.canonical)` in `card()`.

**Create:**
- `static/js/ff-draft-canon.js` — `renderCanonBlock(canonical)` (exported).
- Tests: `tests/test_ff_draft_canonical.py`, `tests/test_ff_draft_canon_render.js`

---

### Task 1: Backend — enrich the console draft list with `canonical`

**Files:**
- Modify: `app.py` — `_ff_draft_canonical` helper (near `api_console_ff_match_drafts_list`, ~22510); call it in the list endpoint.
- Test: `tests/test_ff_draft_canonical.py`

**Interfaces:**
- Consumes: `canonical_tags.get_person(cx, email)`, `ff_match_drafts.list_by_status`.
- Produces: `_ff_draft_canonical(cx, email) -> dict` (canonical fields minus `tags`, `{}` on failure); `GET /api/console/ff-match-drafts` drafts each gain a `canonical` key.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ff_draft_canonical.py
"""Backend: the console FF-draft review list enriches each draft with a
`canonical` block (canonical_tags.get_person minus tags). Env-gated like the
other api tests."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires app env (use doppler run / CI)", allow_module_level=True)

import app  # noqa: E402
from dashboard import canonical_tags as ct  # noqa: E402
from dashboard import ff_match_drafts as ffd  # noqa: E402


def _cx_mem():
    cx = sqlite3.connect(":memory:")
    ct.init_tables(cx)
    return cx


def _seed_canon(cx, email, **fields):
    for f, vals in fields.items():
        for v in (vals if isinstance(vals, (list, tuple)) else [vals]):
            ct.set_attr(cx, email, f, v, source="test")
    cx.commit()


# --- helper unit ----------------------------------------------------------

def test_helper_returns_fields_minus_tags():
    cx = _cx_mem()
    _seed_canon(cx, "c@x.com", conditions="glaucoma", terrain_concerns="oxidative",
                body_systems="liver", tags="vip")
    out = app._ff_draft_canonical(cx, "c@x.com")
    assert out.get("conditions") == ["glaucoma"]
    assert out.get("terrain_concerns") == ["oxidative"]
    assert out.get("body_systems") == ["liver"]
    assert "tags" not in out                       # CRM bucket dropped


def test_helper_includes_scalars():
    cx = _cx_mem()
    _seed_canon(cx, "c@x.com", challenges="fatigue", goals="more energy")
    out = app._ff_draft_canonical(cx, "c@x.com")
    assert out.get("challenges") == "fatigue" and out.get("goals") == "more energy"


def test_helper_best_effort_on_raise(monkeypatch):
    cx = _cx_mem()
    monkeypatch.setattr(ct, "get_person",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app._ff_draft_canonical(cx, "c@x.com") == {}     # no raise


def test_helper_blank_email():
    assert app._ff_draft_canonical(_cx_mem(), "") == {}


# --- endpoint integration -------------------------------------------------

@pytest.fixture
def app_env(tmp_path, monkeypatch):
    p = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", p)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return p


def _seed_draft(db, email, scan_date, items):
    with sqlite3.connect(db) as cx:
        ffd.init_table(cx)
        ffd.get_or_create(cx, email, scan_date, lambda: items)
        cx.commit()


def _seed_canon_db(db, email, **fields):
    with sqlite3.connect(db) as cx:
        ct.init_tables(cx)
        for f, vals in fields.items():
            for v in (vals if isinstance(vals, (list, tuple)) else [vals]):
                ct.set_attr(cx, email, f, v, source="test")
        cx.commit()


def test_list_endpoint_attaches_canonical_and_leaves_items(app_env):
    items = [{"name": "Neuro-Mag", "slug": "neuro-mag", "url": "/x", "meaning": "m"}]
    _seed_draft(app_env, "c@x.com", "2026-07-01", items)
    _seed_canon_db(app_env, "c@x.com", conditions="glaucoma", tags="vip")
    r = app.app.test_client().get("/api/console/ff-match-drafts?status=draft",
                                  headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    draft = r.get_json()["drafts"][0]
    assert draft["canonical"]["conditions"] == ["glaucoma"]
    assert "tags" not in draft["canonical"]
    assert draft["items"] == items                 # items byte-identical


def test_list_endpoint_no_canonical_row(app_env):
    _seed_draft(app_env, "c@x.com", "2026-07-01",
                [{"name": "X", "slug": "x", "url": "/x", "meaning": ""}])
    r = app.app.test_client().get("/api/console/ff-match-drafts?status=draft",
                                  headers={"X-Console-Key": "testkey"})
    draft = r.get_json()["drafts"][0]
    assert draft["canonical"] == {} or all(not v for v in draft["canonical"].values())


def test_list_endpoint_requires_console_key(app_env):
    assert app.app.test_client().get("/api/console/ff-match-drafts").status_code == 401
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (with dummy keys so the module doesn't skip): `PINECONE_API_KEY=pc-x OPENAI_API_KEY=sk-x ANTHROPIC_API_KEY=sk-x CONSOLE_SECRET=x python3 -m pytest tests/test_ff_draft_canonical.py -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_ff_draft_canonical'`

- [ ] **Step 3: Write the implementation**

Insert into `app.py` just before `def api_console_ff_match_drafts_list` (~line 22512):

```python
_FF_SIGNAL_FIELDS = ("conditions", "terrain_concerns", "body_systems",
                     "challenges", "goals")


def _ff_draft_canonical(cx, email):
    """The client's canonical record for the FF-review signal: canonical_tags
    get_person restricted to the clinical fields (tags dropped -- that is the
    CRM/GHL bucket). Best-effort: {} on any failure. Read-only; NEVER merged
    into the draft's match items."""
    email = (email or "").strip()
    if not email:
        return {}
    try:
        from dashboard import canonical_tags as _ct
        p = _ct.get_person(cx, email)
    except Exception:
        return {}
    return {f: p.get(f) for f in _FF_SIGNAL_FIELDS
            if (p.get(f) if isinstance(p.get(f), list) else str(p.get(f) or "").strip())}
```

In `api_console_ff_match_drafts_list`, after `drafts = ff_match_drafts.list_by_status(...)`, enrich each draft (inside the `with` block, so `cx` is open):

```python
        ff_match_drafts.init_table(cx)
        drafts = ff_match_drafts.list_by_status(cx, request.args.get("status"))
        for d in drafts:
            d["canonical"] = _ff_draft_canonical(cx, d.get("email") or "")
    return jsonify({"drafts": drafts})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PINECONE_API_KEY=pc-x OPENAI_API_KEY=sk-x ANTHROPIC_API_KEY=sk-x CONSOLE_SECRET=x python3 -m pytest tests/test_ff_draft_canonical.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_ff_draft_canonical.py
git commit -m "feat(cti2): FF-draft review list carries a canonical signal (get_person minus tags)"
```

---

### Task 2: Frontend — render the canonical block in the review card

**Files:**
- Create: `static/js/ff-draft-canon.js`
- Modify: `static/console-ff-drafts.html` (load the script; call `renderCanonBlock` in `card()`)
- Test: `tests/test_ff_draft_canon_render.js`

**Interfaces:**
- Consumes: the `canonical` block from Task 1.
- Produces: `renderCanonBlock(canonical) -> htmlString` (exported for node).

- [ ] **Step 1: Write the failing test**

```javascript
// tests/test_ff_draft_canon_render.js
// Run: node tests/test_ff_draft_canon_render.js
const assert = require('assert');
const { renderCanonBlock } = require('../static/js/ff-draft-canon.js');

// empty / absent -> nothing (no empty section)
assert.strictEqual(renderCanonBlock(null), '');
assert.strictEqual(renderCanonBlock({}), '');
assert.strictEqual(renderCanonBlock({ conditions: [], challenges: '' }), '');

// populated -> a labeled block that is NOT an .item, with the values
const html = renderCanonBlock({
  conditions: ['glaucoma', 'ocular hypertension'],
  terrain_concerns: ['oxidative stress'],
  challenges: 'fatigue'
});
assert.ok(/class="canon"/.test(html));          // distinct class
assert.ok(!/class="item"/.test(html));          // NOT an .item (never collected/published)
assert.ok(/records/i.test(html));               // a "from the records" label
assert.ok(html.includes('glaucoma') && html.includes('ocular hypertension'));
assert.ok(html.includes('oxidative stress') && html.includes('fatigue'));
// omit empty fields
assert.ok(!/body.?systems/i.test(html));

// escaping
const evil = renderCanonBlock({ conditions: ['<script>alert(1)</script>'] });
assert.ok(!evil.includes('<script>alert(1)</script>'));
assert.ok(evil.includes('&lt;script&gt;'));

console.log('ok - ff-draft canon render');
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node tests/test_ff_draft_canon_render.js`
Expected: FAIL — `Cannot find module '../static/js/ff-draft-canon.js'`

- [ ] **Step 3: Write the implementation**

Create `static/js/ff-draft-canon.js`:

```javascript
// static/js/ff-draft-canon.js
// The canonical-record signal on the console FF-match review card: the client's
// canonical attributes (conditions, terrain, systems, challenges, goals) shown
// as a distinct, labeled block SEPARATE from the scan-driven match items. It is
// NOT an .item, so collectItems()/publish never touch it, and it never reaches
// the client. Display-only, review-only.
var _FF_CANON_LABELS = {
  conditions: 'Conditions',
  terrain_concerns: 'Terrain concerns',
  body_systems: 'Body systems',
  challenges: 'Challenges',
  goals: 'Goals'
};
var _FF_CANON_ORDER = ['conditions', 'terrain_concerns', 'body_systems',
                       'challenges', 'goals'];

function _ffCanonEsc(s) {
  return (s == null ? '' : String(s)).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

function renderCanonBlock(canonical) {
  if (!canonical) return '';
  var lines = [];
  _FF_CANON_ORDER.forEach(function (f) {
    var v = canonical[f];
    var text = Array.isArray(v)
      ? v.map(function (x) { return String(x).trim(); }).filter(Boolean).join(', ')
      : String(v == null ? '' : v).trim();
    if (text) {
      lines.push('<div class="canon-line"><span class="canon-k">'
        + _ffCanonEsc(_FF_CANON_LABELS[f]) + '</span> '
        + _ffCanonEsc(text) + '</div>');
    }
  });
  if (!lines.length) return '';
  return '<div class="canon"><div class="canon-h">From the client’s records '
    + '(context for your review — not part of the match)</div>'
    + lines.join('') + '</div>';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderCanonBlock: renderCanonBlock };
}
```

In `static/console-ff-drafts.html`:
1. Add the script tag beside the existing `op-nav.js` script (top of the page):
   `<script src="/static/js/ff-draft-canon.js"></script>`
2. In `card(d, di)`, insert the block **between the `dhead` div and `itemsHtml`** so it renders above the match items:
   ```javascript
   function card(d, di){
     var items = d.items || [];
     var itemsHtml = items.map(function(it, ii){ return itemBlock(di, ii, it); }).join('');
     var canonHtml = renderCanonBlock(d.canonical);   // distinct block, above items
     return '<div class="dcard" data-di="'+di+'">'
       + '<div class="dhead"><div><b>FF matches</b><span class="email">'+esc(d.email)+'</span></div>'
       + '<div class="scandate">scan '+esc(d.scan_date)+'</div></div>'
       + canonHtml
       + itemsHtml
       + '<div class="actions"><button class="publish" data-di="'+di+'">Publish</button>'
       + '<span class="status" data-di-status="'+di+'"></span></div>'
       + '</div>';
   }
   ```
3. Add minimal CSS for `.canon` / `.canon-h` / `.canon-line` / `.canon-k` in the page's `<style>` so the block reads as a distinct, muted context panel (a light border/background, smaller heading) — visually clearly NOT a match item. Match the page's existing style vocabulary.

- [ ] **Step 4: Run the node test to verify it passes**

Run: `node tests/test_ff_draft_canon_render.js`
Expected: `ok - ff-draft canon render`

- [ ] **Step 5: Verify in a real browser (cannot be faked)**

Boot the app against a seeded DB: a `ff_match_drafts` draft (status `draft`) for an email that ALSO has canonical attributes (`canonical_tags.set_attr` for conditions/terrain/etc.). Open `/static/console-ff-drafts.html?key=<console secret>` (or the console route that serves it). Confirm by eye: the draft card shows the "From the client's records" block ABOVE the match items, styled distinctly; a draft whose email has no canonical record shows NO such block; editing/publishing still works and the published items are unchanged (the canon block is not collected). A green node test is NOT this evidence. If the app cannot boot in your environment, say so plainly and leave this for a human.

- [ ] **Step 6: Commit**

```bash
git add static/js/ff-draft-canon.js static/console-ff-drafts.html tests/test_ff_draft_canon_render.js
git commit -m "feat(cti2): render the canonical record block on the FF-match review card"
```

---

### Task 3: Full-suite gate

- [ ] **Step 1: Run the whole gated suite**

Run: `bash ci/run-tests.sh`
Expected: PASS (ratchets against `tests/known_failures.txt`, fails only on a NEW failure).

- [ ] **Step 2: If a NEW failure appears, fix the cause**

Most likely an existing test asserting the exact `/api/console/ff-match-drafts` draft shape (now with an added `canonical` key). Confirm the new key is additive and correct before adjusting the assertion; never add to `known_failures.txt`.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(cti2): keep the CI ratchet green"
```

---

## Self-Review notes (for the implementer)

- The `canonical` block is additive to each draft — it must NOT change `items`. If a diff touches `ff_matcher.py`, `_make_ff_items_for`, `_ff_llm_rank`, or the published-items path, it's out of scope.
- `renderCanonBlock` output must carry `class="canon"` and must NOT carry `class="item"` — `collectItems` selects `.item`, so a wrong class would leak the signal into the published draft.
- Never surface `tags`.
- Task 2 Step 5's browser check is real: a green node test proves the markup, not that the page renders. Seed a client WITH canonical attributes.

## Completes CTI-2

Support program (Slice 1) + biofield narrative (Slice 2) + this gated matcher-review signal (Slice 3) = all three canonical readers wired.
