# Console Role-Based Navigation (Sub-project A) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved (architecture + layout + rollout). Ready for implementation plan.

First of three sub-projects from the console/Business OS IA brainstorm. **A** = role-aware
navigation (this spec). **B** = consolidate/de-sprawl (merge duplicate boards, fold orphans).
**C** = act-in-place inline actions. B and C are separate spec→plan→build cycles, out of scope here.

## Goal

Make the internal console show each person the right navigation: **Glen** (owner) sees everything
with Settings consolidated into a sub-row; **Rae** sees the same structure with her rarely-used
boards tucked into a single "More ▾" overflow; **Shaira** keeps her existing separate
`/workspace/shaira` page, untouched. The split is **organizational, not a restriction** — Rae has
full access (OWNER-class); the role only reshapes the nav and identifies her so the system can act
on her identity (for actions now, for true restrictions later).

A side benefit: by assigning every board a slot in the role→visibility map, A also **de-orphans the
~16 pages** that are reachable today only by direct URL or search.

## Current state (reuse-first)

The role/permission plumbing already largely exists — A wires it together, it does not invent it.

- **`_auth(...)`** (`app.py` ~18570) resolves a request's key/token to a context
  `{scope, user_name, user_id}`: `CONSOLE_SECRET` → `scope="admin"`; a row in the `access_tokens`
  table → `scope="workspace:<owner>"` (e.g. `workspace:shaira`). Single shared `CONSOLE_SECRET`
  (`dashboard/__init__.py:7`); no per-user identity at the legacy `require_console_key` layer.
- **`dashboard/rbac.py`** defines roles `OWNER` (Glen), `OPS` (Rae), `VA` (Shaira), `AGENT`,
  `SYSTEM`; a `POLICY` matrix (role × risk-tier → AUTO/CONFIRM/QUEUE/DENY); `actor_for_scope(scope)`
  (`"admin"`→OWNER, `"workspace:rae"`→OWNER, `"workspace:shaira"`→VA via `SCOPE_ROLES`); and
  `resolve_actor(...)`.
- **`access_tokens` + `workspace_users`** tables (created in `_init_workspace_schema`,
  `app.py` ~14925) back Shaira's scoped token; tokens are minted via
  `POST /api/access-tokens` (`app.py` ~19194), returned once with a ready URL.
- **`static/op-nav.js`** renders all 7 primary tabs + the 13-board Business OS sub-row
  **unconditionally** whenever a key is present. No per-role filtering exists.
- **`static/shaira-workspace.html`** at `GET /workspace/<owner>` (`app.py` ~18680) is Shaira's
  page; it does **not** load `op-nav.js`.

**The two gaps A closes:**
1. There is no way for the front-end to learn the caller's role → no per-role nav.
2. `/api/action/*` (`_bos_actor`, `app.py` ~23972, via `rbac.resolve_actor`) only recognizes
   `CONSOLE_SECRET`; scoped tokens are **not** wired in (the deferred "RBAC-UX" task). So a token
   holder like Rae could see a reshaped nav but could not perform BOS actions.

## Design

### Component 1 — `GET /api/me`

A small read-only endpoint returning the caller's identity for the front-end:

```
GET /api/me   (key via X-Console-Key header or ?key=)
200 → {"role": "owner"|"ops"|"va", "name": "Glen"|"Rae"|"Shaira"|null,
       "nav": "glen"|"rae"|"va"|null, "scope": "<scope>"}
```

Built on the existing `_auth()` + `rbac.actor_for_scope()`. Two distinct fields, because **`role`
governs permissions and `nav` governs layout** — and Glen and Rae are *both* OWNER-class:
- `role` — the lowercased `rbac` role used by the action layer. `CONSOLE_SECRET` → `owner`;
  `workspace:rae` → `owner`; `workspace:shaira` → `va`.
- `nav` — the layout-profile key `op-nav.js` selects its visibility map by, computed server-side
  from `(scope, name)`: `CONSOLE_SECRET` → `"glen"`; `workspace:rae` → `"rae"`;
  `workspace:shaira` → `"va"` (Shaira is not on `op-nav` anyway). This is what lets two OWNER-class
  identities get different bars without the front-end string-matching names.

No key / unresolved → `200 {"role": null, "name": null, "nav": null}` (the front-end treats a null
`nav` as "show everything" — see Error handling).

### Component 2 — Rae's access token

Mint one `workspace_users` row (`name="rae"`, `scope="workspace:rae"`) and one `access_tokens`
row via the existing `POST /api/access-tokens` path (or a one-off script that calls the same
helper). Output: Rae's personal bookmark URL (`/dashboard?key=<token>`), persisted in her browser
exactly as the console key is today. No new table or auth model. Done once at rollout; the token is
long-lived and revocable via the existing `revoked_at` column.

### Component 3 — Wire scoped tokens into the action layer

Extend the BOS action entry point so a request authenticated by an `access_tokens` token resolves
to the correct `rbac` role, not just `CONSOLE_SECRET`. Concretely: `_bos_actor()` /
`resolve_actor(...)` gains a `role_for_token` resolver that looks the token up via `_auth()` →
`actor_for_scope(scope)`. Rae's `workspace:rae` → `Actor(OWNER)` (full power); a `workspace:shaira`
token → `Actor(VA)` (still bound by the existing `POLICY` matrix — A does **not** loosen Shaira's
permissions). `CONSOLE_SECRET` behavior is unchanged.

### Component 4 — Role-aware `op-nav.js`

On load, `op-nav.js` calls `GET /api/me`, then renders from a **single declarative visibility map**
keyed by the `nav` profile (`"glen"` | `"rae"`; null → `"glen"` fallback). Each board carries a
slot per profile: `primary` (shown on the bar), `more` (in a "More ▾" overflow), or `hidden`.
`nav="glen"` → all `primary` (plus the Settings sub-row, below) except the owner-More group.
`nav="rae"` → her daily set `primary`, everything else in `more`. The "More ▾" is a dropdown appended to the bar
(top level) and to the Business OS sub-row. The map is a small config object at the top of
`op-nav.js` — retuning the split later is a data edit, not a render-logic change.

The map (approved):

**Top-level tabs.** Owner-primary: Dashboard, Console, Business OS, Projects, Inbox, Settings,
Funnel. Rae-primary: Dashboard, Console, Business OS, Inbox. Rae-More: Projects, Settings, Funnel.

**Business OS boards.** Rae-primary: Orders, Payments, Finance, CRM, Reviews, Shipping, New Order.
Rae-More (owner sees these primary, except the owner-More group): Products, Biofield, Sales Pages,
Ingredient Pages, Topic Pages, Biofield Reveals, Biofield Intake. **Owner-More group** (de-orphaned,
rarely-daily for everyone): Practitioners, Cert, Coaching Cohort, Top Products, Atlas, Wholesale,
Clips, Studio Credits, Membership, Remedy Meanings, Topic Suggestions. For Rae, the entire
owner-More group plus the owner-primary clinical/content boards are all in her More.

### Component 5 — Settings as a sub-tab parent

The Settings tab becomes a parent with its own sub-row (the same pattern Business OS already uses):
**Pricing** (`/console/pricing-settings`), **Shipping-config** (`/admin/shipping`), **Tax**
(`/admin/tax`), **Write-Mac** (the active-write-Mac control on `/console/settings`). This gives the
config orphans a nav home. **No page merging** in A — the sub-tabs link to the existing pages;
consolidating their contents is sub-project B.

## Data flow

Page load → `op-nav.js` reads the resolved key (URL `?key=` or `localStorage('console_key')`, as
today) → `fetch('/api/me', {key})` → `{role, name, nav}` → pick the visibility map for that `nav`
profile → render primary tabs + "More ▾" overflow + (for `nav="glen"`) the Settings sub-row. A BOS action click →
`/api/action/*` with the same key → `_bos_actor` resolves the token to an `rbac` role → `POLICY`
gate → execute or queue/deny.

## Error handling (safe by construction)

The streamlined view activates **only** when `/api/me` positively returns `nav="rae"`. Any of:
`/api/me` errors, times out, returns `nav: null`, or returns `nav="glen"` → `op-nav.js` renders the
**full bar**. The owner can never have a tab hidden by accident; `CONSOLE_SECRET` always resolves to
`nav="glen"`. The change is therefore inert for Glen by design. `op-nav.js` must apply this fallback on fetch
rejection and on any unexpected shape.

## Out of scope

- **B** — merging duplicate boards (Money = Payments+Finance; one Pages editor; one Approvals
  queue), folding/retiring orphan *pages*, killing Settings stubs. A only gives boards a nav slot.
- **C** — inline action affordances (Create-PO from the reorder report, retry/contact a failed
  charge, record-payment in Finance, record-level dashboard deep-links).
- Tightening Shaira's permissions or changing her workspace page.
- Any new restriction on Rae (she is full-access OWNER-class).

## Dependencies

- Existing `_auth()`, `rbac.py`, `access_tokens`/`workspace_users`, `POST /api/access-tokens`.
- `op-nav.js` is loaded on every internal page; the `/api/me` fetch must be fast and cached per
  page load (one call). The nav must render its fallback (full bar) without waiting indefinitely.

## Testing (run via [reference_deploy_chat_local_tests])

- **`/api/me`:** `CONSOLE_SECRET` → `role=owner, nav=glen`; a seeded `workspace:rae` token →
  `role=owner, nav=rae`; a seeded `workspace:shaira` token → `role=va, nav=va`; no key →
  `{role:null, nav:null}`; revoked token → `{role:null, nav:null}`.
- **Action wiring:** a `workspace:rae` token is authorized for an OWNER-level BOS action; a
  `workspace:shaira` token is still denied an IRREVERSIBLE action per the unchanged `POLICY`;
  `CONSOLE_SECRET` behavior unchanged.
- **Render-verify (headless, per the render-verify lesson):** load an internal page (e.g.
  `/dashboard`) three ways and assert the rendered bar + zero JS console/page errors:
  1. owner key → full top bar + full Business OS sub-row + Settings sub-row.
  2. Rae token → streamlined top bar (Dashboard/Console/Business OS/Inbox + "More ▾"), Business OS
     sub-row shows only her primary boards + a working "More ▾" dropdown.
  3. `/api/me` forced to fail (offline/blocked) → full bar (fallback).
- Confirm Shaira's `/workspace/shaira` is unchanged (still no `op-nav.js`).

## Rollout

Additive: new `/api/me` endpoint, a `role_for_token` resolver on the action layer, a declarative
map + "More ▾" + Settings sub-row in `op-nav.js`, and a one-time Rae token mint. No page moves, no
table changes. Owner-fallback makes it safe without a feature flag; mint Rae's token after the
headless render-verify passes, then hand Glen her bookmark URL.
