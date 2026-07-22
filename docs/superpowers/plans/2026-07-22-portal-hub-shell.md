# Portal Hub Shell (Phase 0 + 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the practitioner finder on for all portals, and reorganize the portal landing from one overloaded "Current Analysis" scroll into a journey-grouped hub grid of service tiles — shipped dark behind a flag.

**Architecture:** The portal is a client-side SPA: `render(d, v)` in `static/client-portal.html` builds all UI from a JSON payload. `d` = scan data (`/api/portal/<token>`); `v` = the account/feature payload from `get_portal_view()` (`dashboard/portal_view.py`). We add a `v.hub_enabled` flag; when on, `render()` emits a status banner + a tile grid whose tiles reuse the **existing** `showTab()` / `[data-panel]` mechanism to route between panels. No engine is rebuilt — cards move between panels; two duplicate cards are removed. This slice wires only tiles whose destination content already exists today (My Analysis, Scan History, Orders, Find a Practitioner, Body Map, MasterClasses, Ask Dr. Glen, Refer a Friend, Referrals, Account). The genuinely-new pages (My Health Profile, My Healing Oasis, My Remedies external list) are later phases and their tiles are omitted until then.

**Tech Stack:** Python 3.9 + Flask (`app.py`, `dashboard/portal_view.py`), vanilla client-side JS + CSS in `static/client-portal.html`, SQLite payload sources. Server-side changes get pytest coverage; client-side changes get a headless render-verification (drive the real page, assert on the DOM — never assert on the payload alone).

## Global Constraints

- **Feature flags gate everything, default OFF.** New flag `PORTAL_HUB_ENABLED` mirrors the existing `PORTAL_FINDER_ENABLED` env pattern (`app.py:6106`): accepted truthy values are `("1","true","yes","on")`.
- **Flag state is verified in PROD, not dev.** An env var unset in dev is not proof of its prod value.
- **Single source of truth.** This slice adds no new client data; it only relocates existing cards. Do not fork any record.
- **Client copy rules (Glen's voice):** no em dashes, no ALL CAPS words, no "Hook:"-style labels. Sentence case in UI copy.
- **Brand colors:** green-teal `#2f6f5e` + gold `#d4a843` (match existing portal CSS variables; do not introduce a new palette).
- **Render-verify, don't just inject.** A payload field being present is not proof the client renders it. Every client task ends by loading the real page headless and asserting the DOM.
- **Reuse before build.** The finder (`/practitioner-finder`), `showTab()`, `[data-panel]` panels, and `buildScanHistoryHtml`/`buildOrdersHtml` already exist. Extend them.

---

### Task 1: Turn the practitioner finder on (Phase 0)

Independent of the hub. The finder feature and its embed card already exist; the card is gated by `PORTAL_FINDER_ENABLED` (`app.py:6106` → `dashboard/portal_view.py:267` `_practitioner_finder_block`). This task confirms the gate behaves and enables it.

**Files:**
- Config: `PORTAL_FINDER_ENABLED` env var (Doppler/Render) — no code change
- Test: `tests/test_portal_finder_block.py` (create)

**Interfaces:**
- Consumes: `_practitioner_finder_block(address, enabled)` → `{"enabled": bool, "location": str, "country": str}` (`dashboard/portal_view.py:267`)
- Produces: nothing new; asserts existing behavior

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_finder_block.py
from dashboard.portal_view import _practitioner_finder_block

def test_finder_block_enabled_prefills_zip_over_city():
    b = _practitioner_finder_block({"zip": "96720", "city": "Hilo", "country": "US"}, True)
    assert b == {"enabled": True, "location": "96720", "country": "US"}

def test_finder_block_disabled_flag_off():
    b = _practitioner_finder_block({"zip": "96720"}, False)
    assert b["enabled"] is False

def test_finder_block_absent_address_empty_location_defaults_us():
    b = _practitioner_finder_block(None, True)
    assert b == {"enabled": True, "location": "", "country": "US"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_portal_finder_block.py -v`
Expected: FAIL initially only if the import path is wrong; if the function already behaves, tests PASS immediately — that is the acceptable outcome for a characterization test. If it ERRORs on import, fix the import path before proceeding.

- [ ] **Step 3: Confirm no code change is needed**

`_practitioner_finder_block` already implements this. No implementation edit. The "work" is the config flip in Step 5.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_portal_finder_block.py -v`
Expected: 3 passed.

- [ ] **Step 5: Enable the flag in prod and render-verify**

Set `PORTAL_FINDER_ENABLED=on` in the prod config (Doppler `-c prd`, or Render env — whichever holds portal flags; confirm which by checking where `PORTAL_FINDER_ENABLED` is currently defined/absent in prod, not dev). Then load a real portal URL headless and assert the finder card is present:

```
Load https://illtowell.com/portal/<a-real-token>
Assert: an element containing "Find a Practitioner" is visible (the embedded finder iframe card, client-portal.html:1287).
```

Expected: the finder card renders. If the flag write to prd is classifier-blocked, stage the exact value and hand the one-line set command to Glen to apply.

- [ ] **Step 6: Commit**

```bash
git add tests/test_portal_finder_block.py
git commit -m "test: characterize practitioner-finder block; enable finder (Phase 0)"
```

---

### Task 2: Add the `PORTAL_HUB_ENABLED` flag and thread it into the payload

**Files:**
- Modify: `app.py:6106` region (define `_PORTAL_HUB_ENABLED`)
- Modify: `app.py:25342` region (pass `hub_enabled=_PORTAL_HUB_ENABLED` to `get_portal_view`)
- Modify: `dashboard/portal_view.py:318` (signature default) and `:340`-region (add `"hub_enabled"` to the view dict, next to `"practitioner_finder"` at `:350`)
- Test: `tests/test_portal_hub_flag.py` (create)

**Interfaces:**
- Consumes: `get_portal_view(cx, person_id, *, offers_enabled_keys, quiz_url, public_base_url, finder_enabled, biofield_unlocked, supplement_review_enabled)` (`app.py:25342`)
- Produces: `get_portal_view(..., hub_enabled: bool = False)`; adds `view["hub_enabled"]: bool` to the returned payload (the `v` object consumed by `render(d, v)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_hub_flag.py
import inspect
from dashboard import portal_view

def test_get_portal_view_accepts_hub_enabled():
    sig = inspect.signature(portal_view.get_portal_view)
    assert "hub_enabled" in sig.parameters
    assert sig.parameters["hub_enabled"].default is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_portal_hub_flag.py -v`
Expected: FAIL — `assert "hub_enabled" in sig.parameters` is False.

- [ ] **Step 3: Add the flag env in app.py (next to the finder flag)**

At `app.py:6106`, immediately after the `_PORTAL_FINDER_ENABLED = ...` line, add:

```python
# Journey-grouped hub landing for the client portal (banner + tile grid).
# Ships dark; flip to route the portal through the hub instead of the single
# Current-Analysis scroll. Same truthy set as the finder flag.
_PORTAL_HUB_ENABLED = os.environ.get("PORTAL_HUB_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Thread it into the get_portal_view call**

At the `get_portal_view(...)` call (`app.py:25342`), add the keyword argument alongside `finder_enabled`:

```python
        view = _pv.get_portal_view(cx, ident.person_id,
                                   offers_enabled_keys=_enabled_offer_keys(),
                                   quiz_url=QUIZ_URL, public_base_url=PUBLIC_BASE_URL,
                                   finder_enabled=_PORTAL_FINDER_ENABLED,
                                   hub_enabled=_PORTAL_HUB_ENABLED,
                                   biofield_unlocked=_portal_biofield_unlocked(ident.email),
                                   supplement_review_enabled=_sr.enabled())
```

- [ ] **Step 5: Add the parameter + payload key in portal_view.py**

In `get_portal_view`'s signature (`dashboard/portal_view.py:318` region), add `hub_enabled=False` alongside `finder_enabled=False`. Then in the returned view dict (near `:350`, beside `"practitioner_finder"`), add:

```python
        "hub_enabled": bool(hub_enabled),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_portal_hub_flag.py tests/test_portal_finder_block.py -v`
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add app.py dashboard/portal_view.py tests/test_portal_hub_flag.py
git commit -m "feat: add PORTAL_HUB_ENABLED flag, thread hub_enabled into portal payload"
```

---

### Task 3: Render the hub banner + tile grid (structure only, behind the flag)

Add the landing hub to `render(d, v)`. When `v.hub_enabled` is true, prepend a status banner and a journey-grouped tile grid above the existing content; when false, render exactly as today (zero behavior change for un-flagged portals).

**Files:**
- Modify: `static/client-portal.html` — `render(d, v)` (starts `:1131`); add CSS in the page's `<style>`; add a global `buildHubHtml(d, v)` helper near `showTab` (`:638`)

**Interfaces:**
- Consumes: `v.hub_enabled` (bool), existing `showTab(name)` (`:638`), existing `[data-panel]` convention
- Produces: global `buildHubHtml(d, v) -> string` (banner + grid markup); a `.portal-hub` block injected at the top of the render output when the flag is on

- [ ] **Step 1: Add the hub CSS**

In the `<style>` block of `static/client-portal.html`, add (uses the existing brand variables; if the file uses literal hex, match `#2f6f5e`/`#d4a843`):

```css
.portal-hub { margin: 0 0 1.25rem; }
.hub-banner { display:flex; gap:1rem; align-items:center; flex-wrap:wrap;
  background:var(--brand-soft,#e5efea); border:1px solid #cdd9d2; border-radius:14px; padding:1rem 1.15rem; }
.hub-banner .where { flex:1; min-width:12rem; }
.hub-banner .eyebrow { font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:#2f6f5e; font-weight:700; }
.hub-banner h2 { font-size:1.15rem; margin:.15rem 0 .2rem; }
.hub-banner .next { background:#f6ecd2; border:1px solid #cdd9d2; border-radius:10px; padding:.6rem .75rem; max-width:16rem; }
.hub-grid-group { margin-top:1rem; }
.hub-grid-group .glabel { font-size:.68rem; letter-spacing:.11em; text-transform:uppercase; color:#7c8a83; font-weight:700; margin:.1rem 0 .5rem; }
.hub-tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; }
@media (max-width:760px){ .hub-tiles { grid-template-columns:repeat(2,1fr); } }
@media (max-width:480px){ .hub-tiles { grid-template-columns:1fr; } }
.hub-tile { text-align:left; cursor:pointer; background:#fff; border:1px solid #dde6e0; border-radius:12px;
  padding:.8rem; min-height:5.5rem; display:flex; flex-direction:column; gap:.35rem; font:inherit; color:inherit; }
.hub-tile:hover,.hub-tile:focus-visible { border-color:#2f6f5e; box-shadow:0 6px 18px rgba(24,36,32,.08); outline:none; }
.hub-tile .tname { font-weight:650; font-size:.95rem; }
.hub-tile .tdesc { font-size:.78rem; color:#7c8a83; }
```

- [ ] **Step 2: Add the `buildHubHtml` helper (global scope, near showTab)**

Immediately after `showTab` (`static/client-portal.html:~645`), add. Tiles are data-driven so later phases append entries. Only tiles whose panel exists in this slice are listed:

```javascript
// Journey-grouped hub landing. Each tile routes via the existing showTab() to a
// [data-panel] section. Groups encode the healing journey. Later phases add tiles.
function buildHubHtml(d, v){
  const groups = [
    ["Understand", [
      ["current", "My Analysis", "Your current scan, matches and healing path"],
    ]],
    ["Act", [
      ["finder", "Find a Practitioner", "Certified practitioners near you"],
      ["refer",  "Refer a Friend", "Share healing and earn as an Ambassador"],
    ]],
    ["Learn & Ask", [
      ["ask",     "Ask Dr. Glen", "Chat about your terrain and remedies"],
      ["bodymap", "Body Map", "Your systems, personalized"],
      ["classes", "MasterClasses", "Free courses and teachings"],
    ]],
    ["Track", [
      ["history", "Scan History", "Every scan over time"],
      ["orders",  "Orders & Invoices", "Receipts and anything unpaid"],
      ["referrals","Referrals", "Who you've referred and your rewards"],
    ]],
  ];
  const tile = ([panel,name,desc]) =>
    `<button type="button" class="hub-tile" onclick="showTab('${panel}')">
       <span class="tname">${esc(name)}</span><span class="tdesc">${esc(desc)}</span></button>`;
  const grp = ([label,tiles]) =>
    `<div class="hub-grid-group"><p class="glabel">${esc(label)}</p>
       <div class="hub-tiles">${tiles.map(tile).join("")}</div></div>`;
  const next = (d.next_step_label)
    ? `<div class="next"><div class="eyebrow">Your next step</div><div>${esc(d.next_step_label)}</div></div>`
    : "";
  return `<div class="portal-hub">
    <div class="hub-banner"><div class="where">
      <div class="eyebrow">Where you are</div>
      <h2>${esc(d.hub_headline || "Your healing home")}</h2></div>${next}</div>
    ${groups.map(grp).join("")}</div>`;
}
```

Note: `refer`, `ask`, `classes`, `referrals` panels are wired in Task 4/5; if a panel is not yet present, its tile is still shown but routing to a missing panel is a no-op (safe). Only list a tile here once its panel exists — add `refer`/`referrals`/`ask`/`classes` tiles in the same commit that adds their panels (Task 5).

- [ ] **Step 3: Inject the hub at the top of render output when the flag is on**

In `render(d, v)`, find where the output `html` string is first assembled (the welcome card near `:1153`). Prepend the hub when enabled. Add near the start of `render`:

```javascript
  // Hub landing (flagged). Prepended above existing cards; panels below unchanged.
  let hubHtml = (v && v.hub_enabled) ? buildHubHtml(d, v) : "";
```

and ensure the final returned/assigned markup is `hubHtml + html` (locate the single place where `html` is written to the DOM — typically `container.innerHTML = html` — and change it to `container.innerHTML = hubHtml + html`).

- [ ] **Step 4: Render-verify (flag ON and OFF)**

Drive the real page headless (webapp-testing skill):

```
Case A (flag OFF, un-flagged portal): load /portal/<token>.
  Assert: NO element with class "portal-hub" exists. Page looks exactly as before.
Case B (flag ON): with PORTAL_HUB_ENABLED=on for the test env, load /portal/<token>.
  Assert: ".portal-hub" exists; ".hub-tile" count >= 6; the "My Analysis" tile text is present;
          clicking the "Scan History" tile calls showTab('history') and reveals [data-panel="history"].
```

Expected: A unchanged; B shows the grid and tiles route.

- [ ] **Step 5: Commit**

```bash
git add static/client-portal.html
git commit -m "feat: portal hub banner + journey tile grid behind PORTAL_HUB_ENABLED"
```

---

### Task 4: Make panels always available under the hub (decouple from the tab-bar gate)

Today the three panels + tab bar render only when `d.scan_history_enabled` (`client-portal.html:2114`-region), and `showTab` toggles `[data-panel]`. Under the hub, tiles are the navigation, so the same `[data-panel]` sections must exist even when the old tab bar does not. This task ensures the `current`/`history`/`orders` panel wrappers are emitted whenever `v.hub_enabled` is on (independently of the old `scan_history_enabled` tab-bar path), and that the hub hides all panels except the active one on load.

**Files:**
- Modify: `static/client-portal.html` — the panel-wrapping region (`:2114`-`:2128`) and `render`'s post-inject step

**Interfaces:**
- Consumes: `showTab(name)` (`:638`), `buildScanHistoryHtml(d)` (`:674`), `buildOrdersHtml(d)` (`:709`)
- Produces: `[data-panel="current"]`, `[data-panel="history"]`, `[data-panel="orders"]` present whenever `v.hub_enabled`; default active panel = `current`

- [ ] **Step 1: Emit panels under the hub flag too**

At the panel-wrapping region (`:2114`), change the condition that currently gates on `d.scan_history_enabled` to also fire under the hub. Replace the gate expression:

```javascript
  // was: if (d.scan_history_enabled) { ...wrap into panels + tab bar... }
  const wrapPanels = d.scan_history_enabled || (v && v.hub_enabled);
  if (wrapPanels) {
    // existing panel-wrap body, BUT only emit the old tab bar (#portalTabs)
    // when d.scan_history_enabled; under hub-only, tiles are the nav.
  }
```

Keep the existing `<section data-panel="current">…</section>`, `history`, and `orders` wrapping. Guard only the `#portalTabs` button-bar markup with `if (d.scan_history_enabled)` so the hub does not show both a tab bar and the grid.

- [ ] **Step 2: Set the default active panel on load under the hub**

After `container.innerHTML = hubHtml + html`, initialize the panel visibility. Add:

```javascript
  if (v && v.hub_enabled) {
    const last = (()=>{ try { return sessionStorage.getItem('rm_portal_tab'); } catch(e){ return null; } })();
    showTab(last || 'current');
  }
```

`showTab` already sets `p.hidden` for every `[data-panel]` and persists the choice, so this both hides inactive panels and restores the last tile.

- [ ] **Step 3: Render-verify panel routing without the old tab bar**

```
With PORTAL_HUB_ENABLED=on and a portal where scan_history_enabled is OFF:
  Load /portal/<token>. Assert:
    - "#portalTabs" does NOT exist (no old tab bar)
    - [data-panel="current"] is visible; [data-panel="history"] and [data-panel="orders"] are hidden
    - clicking the "Orders & Invoices" tile shows [data-panel="orders"] and hides the others
```

Expected: tiles drive panels with no tab bar present.

- [ ] **Step 4: Commit**

```bash
git add static/client-portal.html
git commit -m "feat: hub emits current/history/orders panels + default panel on load"
```

---

### Task 5: Move stray cards into panels; remove the two duplicate cards; wire remaining tiles

Relocate cards so each lands in its journey panel, delete the two known duplicates, and add the tiles/panels for existing-content destinations not yet wired (Refer a Friend, Referrals, Ask Dr. Glen, MasterClasses, Body Map link, Account).

**Files:**
- Modify: `static/client-portal.html` — `render(d, v)` card assembly, the `buildHubHtml` tile list, the panel wrappers

**Interfaces:**
- Consumes: the ambassador cards ("Your Ambassador links" `:1987`, "Become an Ambassador" `:2006`, "Your share page" `:2017`), the scan-history teaser (`:1761`), the legacy "History & receipts" card (`:1394`), Ask Dr. Glen card (`:1270`), Body Map card (`:1294`), MasterClasses card (`:1298`), account snapshot (`:1307`)
- Produces: panels `refer`, `referrals`, `ask`, `classes`, `account`; deletions of the two duplicate cards

- [ ] **Step 1: Remove the two duplicate cards**

Delete the scan-history teaser card block (`static/client-portal.html:1761`, the small "Scan history" quiet card) and the legacy combined "History & receipts" card block (`:1394`). Both are superseded by the Scan History and Orders panels. Excise each `html += \`<div class="card"…>…\`` block in full; leave surrounding cards intact.

- [ ] **Step 2: Group the ambassador cards into a `refer` panel and a `referrals` panel**

Wrap the "Your Ambassador links" + "Become an Ambassador" + "Your share page" card strings into `<section data-panel="refer">…</section>` (the doing surface), and create a `<section data-panel="referrals">` that shows the referral dashboard content if present in `d` (referred contacts / rewards). If no dashboard data exists yet, render a single card: `Your referrals will appear here once someone joins.` Keep it factual, no placeholder TODO.

- [ ] **Step 3: Group Ask Dr. Glen / Body Map / MasterClasses / Account into panels**

Wrap the Ask Dr. Glen card (`:1270`) in `<section data-panel="ask">`, MasterClasses (`:1298`) in `<section data-panel="classes">`, and the account snapshot (`:1307`) + notification prefs + scan-prefs footer (`:2106`-region) in `<section data-panel="account">`. The Body Map tile routes to the existing `/portal/<token>/bodymap` page: make its hub tile a link rather than a `showTab`, i.e. add a variant in `buildHubHtml` where `bodymap` uses `onclick="location.href='/portal/'+encodeURIComponent(TOKEN)+'/bodymap'"` (reuse the token the page already holds; confirm the existing global var name for the token near the top of the script).

- [ ] **Step 4: Add the now-valid tiles to `buildHubHtml`**

The `groups` array in Task 3 already lists `refer`, `ask`, `classes`, `referrals`. Confirm each now has a matching panel (or link, for `bodymap`). Add an Account entry to the top bar area — not the grid — per spec (a small `account` tile or a header link; render a header link `<a onclick="showTab('account')">Account</a>` above the grid).

- [ ] **Step 5: Render-verify the full grid end to end**

```
With PORTAL_HUB_ENABLED=on, load /portal/<token>. Assert:
  - No scan-history teaser card and no "History & receipts" card anywhere in [data-panel="current"]
  - Tiles present: My Analysis, Find a Practitioner, Refer a Friend, Ask Dr. Glen, Body Map,
    MasterClasses, Scan History, Orders & Invoices, Referrals
  - Clicking Refer a Friend shows [data-panel="refer"] with the Ambassador-links card
  - Clicking Referrals shows [data-panel="referrals"]
  - Clicking Body Map navigates to /portal/<token>/bodymap
  - Account link shows [data-panel="account"] with the account snapshot + preferences
```

Expected: every tile routes to real content; no duplicates remain.

- [ ] **Step 6: Commit**

```bash
git add static/client-portal.html
git commit -m "feat: relocate cards into hub panels, remove duplicate history cards, wire remaining tiles"
```

---

### Task 6: Final verification and rollout note

**Files:**
- Create: `docs/superpowers/plans/2026-07-22-portal-hub-shell-rollout.md` (short rollout/verification log)

- [ ] **Step 1: Full server test run (scoped, no email side effects)**

Run only the portal tests to avoid the full suite's live-email side effects:

Run: `python3 -m pytest tests/test_portal_hub_flag.py tests/test_portal_finder_block.py -v`
Expected: all passed.

- [ ] **Step 2: Render-verify both flag states one more time on a staging/prod token**

Confirm: flag OFF → today's portal unchanged; flag ON → hub grid with all wired tiles; finder card present (Task 1).

- [ ] **Step 3: Write the rollout note**

Record: the two new env flags (`PORTAL_FINDER_ENABLED`, `PORTAL_HUB_ENABLED`), their prod values, the deploy that shipped them, and the fact that every portal deploy 502s illtowell for ~4-8 min (batch + warn). Note that turning `PORTAL_HUB_ENABLED` on is a config flip, reversible by unsetting it.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-07-22-portal-hub-shell-rollout.md
git commit -m "docs: portal hub shell rollout + verification note"
```

---

## Deferred to later plans (own spec sections already written)

- **Phase 2 — My Health Profile:** editable record over the existing intake store + AI chat-suggestion confirm/edit/dismiss queue.
- **Phase 3 — My Healing Oasis:** Replenish (reorder) + Build Out (owned tools incl. external + analysis-ordered roadmap led by Harmony / Water Ionizer / Kloud).
- **Phase 4 — My Remedies external list + Ambassador dashboard:** the "remedies from other companies" list (brand/product/reason/importance/upgrade) — **reuse the existing `REVIEWS_ENABLED` + `supplement_reviews` module** (`_supplement_reviews_block`, `dashboard/supplement_reviews.py`) for the request-review + hosted-review half — and the Referrals dashboard data.

## Self-Review

- **Spec coverage (this slice):** Phase 0 finder-on → Task 1. Hub flag/payload → Task 2. Banner + tile grid + journey groups → Task 3. Panel routing without tab bar → Task 4. Card migration + two duplicate removals + remaining tiles → Task 5. Rollout/verify → Task 6. New-page tiles (Health Profile, Healing Oasis, My Remedies external) correctly deferred, not silently dropped.
- **Placeholder scan:** the one "no dashboard data yet" empty-state renders real factual copy, not a TODO. No "implement later" steps.
- **Type consistency:** `hub_enabled` (Python bool param) → `view["hub_enabled"]` → `v.hub_enabled` (JS) used consistently; `buildHubHtml(d, v)` and `showTab(name)` names match across tasks; panel names (`current/history/orders/refer/referrals/ask/classes/account/finder`) are consistent between `buildHubHtml` tiles and the `[data-panel]` wrappers.
