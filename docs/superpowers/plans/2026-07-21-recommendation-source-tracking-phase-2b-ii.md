# Recommendation Source Tracking — Phase 2b-ii (client portal UI) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Implementers of the UI tasks should ALSO load the frontend-design skill.

**Goal:** Add a client-facing "My Recommendations" section to `static/client-portal.html` that renders the collapsible, per-source recommendation sections from `GET /api/portal/<token>/recommendations` — icon rows with counts, top-5 + show-more, a per-product hide control, the operator note (read-only) and the client's own editable note — and persists hide / client-note / section-collapse through the live token-authed write endpoints.

**Architecture:** All work is in the single vanilla-JS file `static/client-portal.html` (4321 lines, no framework). The section is a new `.card` block appended into the page's `html` string (rendered via `app.innerHTML`), styled with the existing CSS-variable design system (`.card`, `.btn`, `.pill`, `.muted`, `.small`). It fetches `/api/portal/<token>/recommendations` and POSTs to the three `/api/portal/<token>/recommendation/{hide,client-note,section}` endpoints (all live, from 2b-i). No backend changes.

**Tech Stack:** Vanilla JS + inline CSS in one HTML file; Flask serves it at `/portal/<token>`. No JS test runner exists — automated coverage is a serve/substring test; correctness is confirmed by a rendered browser check (webapp-testing / claude-in-chrome).

## Global Constraints

- **The client note is client-authored free text — it MUST be escaped** with the page's `esc()` (line 411; escapes `& < > "`, safe for text and double-quoted attributes). Every interpolated value (product name, url, notes, product_key, icons) goes through `esc()`. Use **double-quoted** HTML attributes so `esc()` (which escapes `"` but not `'`) is sufficient. (Backend already caps notes at 4000 chars and escapes on the console side.)
- **Identity is the token in the path** — reuse the page's existing `token`/`seg` var and `encodeURIComponent(token)`; never send an email. Writes are `fetch(..., {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({...})})`, mirroring the existing `client-fact` / `recommendation/dismiss` handlers.
- **Match the existing page** — reuse `.card`/`.btn`/`.btn.ghost`/`.pill`/`.muted`/`.small` classes and the CSS variables; do not introduce a new design language. Source icons are the emoji already returned by the endpoint (`icon` field). Place the section with the other product/remedy cards (after the wishlist card, ~line 1691).
- **No backend changes** — the endpoints are live and unchanged. Do not modify `app.py` or `dashboard/*`.
- **Collapse persistence:** the section header toggles the body's visibility AND POSTs `/recommendation/section` `{section_key, collapsed}`; on load, honor each section's `collapsed` from the payload.
- **Failure-safe UI:** a failed fetch shows a quiet empty/error state, never a broken page (mirror the page's `fetchJson`→null convention and the `#recoErr` pattern).
- **CI known_failures ratchet; never run the bare full suite (sends live email).**

---

### Task 1: Render the "My Recommendations" section (fetch + collapsible sections + product rows)

**Files:**
- Modify: `static/client-portal.html`
- Test: `tests/test_client_portal_recommendations_ui.py`

**Interfaces:**
- Consumes: `GET /api/portal/<token>/recommendations` → `{ok, sections:[{source,label,icon,collapsed,total,shown,products:[{product_key,name,url,icons:[{source,count,icon,first_touch}],operator_note,client_note}]}]}`.
- Produces: a rendered `.card` "My Recommendations" with one collapsible sub-section per source; each product row shows its icon row (icon+count), a name linking to `url`, the operator note (read-only) and the client note (as an editable field, wired in Task 2), and a hide control (wired in Task 2). Empty/absent → the section renders nothing or a quiet empty state.

**Implementer: load the frontend-design skill** for the visual treatment (hierarchy, the icon-row + count chips, the collapse affordance, spacing), matching the existing portal cards.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client_portal_recommendations_ui.py
import app as app_module


def test_portal_page_has_recommendations_section_and_fetch():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/portal/anytoken").get_data(as_text=True)  # page is static; token resolved client-side
    assert "/recommendations" in body                 # the page fetches the endpoint
    assert "My Recommendations" in body                # the section heading
    assert "renderRecommendations" in body             # the render function exists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_client_portal_recommendations_ui.py -q`
Expected: FAIL (markers absent).

- [ ] **Step 3: Implement**

In `static/client-portal.html`:
- Add a `SRC_LABEL`/icon fallback is unnecessary (the endpoint returns `label` + `icon`). Add a fetch of `/api/portal/${encodeURIComponent(token)}/recommendations` (via the existing `fetchJson` helper) — either add it to the `Promise.all` in `load()` (~line 823) or fetch lazily; pass its `sections` into `render`.
- Add `renderRecommendations(sections)` returning an HTML string for a `<div class="card"><h2>My Recommendations</h2>…</div>`, appended into the page's `html` (after the wishlist card, ~line 1691). For each section: a header row (label + icon + a "N products" count + a collapse toggle button carrying `data-section` and reflecting `collapsed`), and a body `<div>` (hidden when `collapsed`) listing up to `shown` product rows plus a "Show all N" affordance when `total > shown`. Each product row:
  - the icon row: for each `icons[]` entry, a chip `<span class="pill">{icon}{count}</span>` (title = `esc(source) + " ×" + count + " · first " + esc(first_touch)`);
  - the name as `<a href="{esc(url)}">{esc(name)}</a>` when `url`, else `esc(name)`;
  - the operator note (when present) as a read-only `<p class="muted small">{esc(operator_note)}</p>`;
  - a client-note field: `<input class="rec-cnote" data-pk="{esc(product_key)}" value="{esc(client_note)}" placeholder="your note">` + a small save button (wired in Task 2);
  - a hide button `<button class="btn ghost small rec-hide" data-pk="{esc(product_key)}">hide</button>` (wired in Task 2).
- If `sections` is empty/null, render nothing (or a single quiet `<p class="muted">No recommendations yet.</p>` inside the card — pick one and be consistent).
- ALL interpolated dynamic values via `esc()`; all attributes double-quoted.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_client_portal_recommendations_ui.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add static/client-portal.html tests/test_client_portal_recommendations_ui.py
git commit -m "feat(rec): client portal My Recommendations section (render collapsible source sections)"
```

---

### Task 2: Wire the interactions (hide, client-note save, section-collapse persist)

**Files:**
- Modify: `static/client-portal.html`
- Test: `tests/test_client_portal_recommendations_ui.py` (extend)

**Interfaces:**
- Produces: click/save handlers that POST to the three write endpoints and update the UI, using the page's one-shot-latch convention (disable control before `await`, re-enable + show error on failure).

- [ ] **Step 1: Write the failing test** (extend the file)

```python
def test_portal_page_wires_recommendation_writes():
    import app as app_module
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/portal/anytoken").get_data(as_text=True)
    assert "/recommendation/hide" in body
    assert "/recommendation/client-note" in body
    assert "/recommendation/section" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_client_portal_recommendations_ui.py -q`
Expected: FAIL (write paths absent).

- [ ] **Step 3: Implement** (wire, after render, mirroring `acceptRecommendation()`/`dismissRecommendation()` and the `client-fact` toggle):
- **Hide:** on `.rec-hide` click → disable button → `POST /api/portal/<token>/recommendation/hide` `{product_key, hidden:true}` → on ok, remove the row (or `await load()` to re-render, dropping the hidden product) → on error re-enable + show a quiet error.
- **Client-note save:** on the note save control → `POST .../recommendation/client-note` `{product_key, note: input.value}` → on ok, a brief saved affirmation; on error, error text. Read the LIVE `input.value` at save time (do not cache the pre-render string).
- **Section-collapse:** on the section header toggle → flip the body's hidden state immediately (optimistic) → `POST .../recommendation/section` `{section_key, collapsed}` (fire-and-forget is acceptable; on error, leave the optimistic state — it's a display pref). On next `load()`, the payload's `collapsed` re-establishes the persisted state.
- Wire via event delegation (like the wishlist rows) or per-id after render; guard against double-wiring on re-render (the page re-renders `app.innerHTML` each `load()`, so prefer delegation on a stable ancestor or re-attach each render — match whatever the surrounding code does).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_client_portal_recommendations_ui.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add static/client-portal.html tests/test_client_portal_recommendations_ui.py
git commit -m "feat(rec): wire portal recommendation writes (hide, client-note, section-collapse)"
```

---

### Task 3: Rendered verification (no code — real browser round-trip)

Because the page has no JS unit-test harness, verify the section actually works end to end against a real portal token.

- [ ] **Step 1: Seed a client with recommendations + a portal token**

Locally (or against a scratch db), create a `client_portals` row + a few `recommendation_events` for that email (biofield/self/purchased), and obtain the raw token (via `client_portal.upsert_portal`). OR use a real prod token for a client known to have recommendation events (read-only visual check — do not mutate a real client's notes).

- [ ] **Step 2: Render + interact**

Load the frontend `run` / claude-in-chrome / webapp-testing skill. Open `/portal/<token>`, confirm:
- the "My Recommendations" card renders the expected sections with icon+count chips, top-5 + show-more, operator note read-only, client-note field prefilled;
- toggling a section collapses/expands and persists across reload (the `collapsed` state round-trips);
- hiding a product removes it and it stays hidden on reload;
- saving a client note persists (reload shows it), and it does NOT alter the operator note;
- an apostrophe in a client note round-trips (no truncation) and no markup injection (try a `<b>`/`'` in the note — it renders as text).

- [ ] **Step 3: Record the result** in the task report (screenshots / observations). No commit unless a fix was needed.

---

## Self-review checklist (controller, before dispatch)

- Every dynamic value (esp. the client-authored note) escaped via `esc()`; attributes double-quoted.
- Identity is the token only; writes match the existing POST convention; failure-safe UI.
- Visual matches the existing portal cards (frontend-design); collapse persists; hide removes; note saves.
- No backend changes.

## After 2b-ii

This completes the client-facing portal. Remaining roadmap: **2c** (product-page "add to my portal" → `self`, needs product-page-visitor→client identity) and **2d** (reveal-click + on-page engagement capture so biofield/scan/chat accrue on real actions). Marketing channels (email/newsletter/ads/social) are Phase 3.
