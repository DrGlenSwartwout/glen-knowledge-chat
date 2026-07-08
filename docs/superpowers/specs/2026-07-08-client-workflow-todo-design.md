# Client workflow "To-Do" board — design (teed up 2026-07-08)

**Status:** spec ready to build. Origin: the per-client flow spans several surfaces across two apps (scan → reveal → approve → intake → invoice → portal publish → email), and there is no single place that shows what's done and what's next. The operator loses the thread. This is the central checklist that ties the whole session's work together.

**Supersedes:** most of `2026-07-08-cross-app-client-navigation-design.md` (#707) — the To-Do board IS the navigation + status, so build this instead of the standalone nav buttons (keep only the composer "← Edit in Intake" button if still wanted).

## Goal

One central board that, for each client in flight, shows the ordered workflow steps, **auto-checks each step off from live data**, and **deep-links each open step to the exact action** (in whichever app). The operator opens one page and always knows the next click.

## Where it lives

The **local Intake app `:8011`** (operator hub, already has the client list + PHI + a console key to reach prod). A new page `GET /todo` aggregates, per client, local state (authored tests) + prod state (reveals, orders, portal) via the existing console-key calls.

## The workflow steps (each: label · status source · action link)

Ordered checklist per client (email is the join key):

1. **Scan received** — a `biofield_reveals` row exists → Reveals console.
2. **Reveal approved** — `first_approved` on the latest reveal → Reveals console ("Approve & send link").
3. **Intake authored** — an authored test exists (`biofield_auth_tests` by email) → local `/author/a<id>`.
4. **Invoice raised** — an `orders` row for the email → local `/author/a<id>/invoice-view`.
5. **Invoice published to portal** — `orders.portal_published` → the invoice card (portal).
6. **Invoice paid** — `orders.pay_status='paid'` → (terminal; shows amount).
7. **Portal published** — portal `biofield_status='confirmed'` → prod composer (`/console/biofield-portal?email=`).
8. **Client notified** — portal `notified_at` set / reveal `notified_at` → (terminal).

Each row renders: a checkbox (checked = done, from the status source), the label, and — when NOT done — a **"→ do it"** link to the deep-linked action for that step.

## Status derivation

A best-effort aggregator `workflow_state(email)` returns `{step_key: {done: bool, detail: str, action_url: str}}` for the 8 steps, reading:
- local `biofield_auth_tests` / `biofield_auth_chain` (authored + invoice-ready),
- prod `/api/console/biofield-reveals` (scan + approved),
- prod orders (raised / published / paid — via a small console lookup by email; may need a new `/api/console/orders?email=` filter),
- prod `/api/console/biofield-portal?email=` (portal status + notified).

Never raises — a source that errors renders that step as "unknown" rather than breaking the board.

## Which clients appear

The board lists **clients with at least one incomplete step** (in-flight), newest activity first, with a search box to pull up any client by name/email. A "show completed" toggle reveals fully-done clients. (Reuse the existing client type-ahead.)

## Auto-check vs manual

- **Auto** for every step that has a data signal (all 8 above).
- **Manual override** (a persisted "operator marked done") only for steps a client legitimately skips (e.g., a client who won't be invoiced) — a small `workflow_overrides(email, step_key, done)` table so the board can show "N/A / skipped" without faking the data.

## Open decisions (resolve at build)

1. **Board home** — local `:8011/todo` (recommend, operator hub) vs a prod console tab. Confirm.
2. **Per-client card vs one big table** — recommend a compact per-client card (8 checkboxes + next-action link) so the "next click" is obvious. Confirm.
3. **Orders lookup** — add `/api/console/orders?email=` (needed; there's no per-email order filter today) vs derive from `/api/people`. Recommend the filter.
4. **How far back** — only clients active in the last N days on the default board (recommend 30) to keep it fast; search reaches everyone.
5. **Localhost links** — same wrinkle as #707: steps that deep-link to `:8011` are operator-Mac-only; steps to prod work anywhere. The board itself lives on `:8011`, so this is consistent (it's already an operator tool).

## Testing

- `workflow_state(email)` for a client mid-flow returns correct done/undone per step with valid action URLs; a client with nothing returns all-undone; errors in one source degrade that step to "unknown," others still resolve.
- The board lists only in-flight clients by default; search finds a completed one; "show completed" toggles them.
- A step with a `workflow_overrides` row renders as skipped, not undone.

## Out of scope (v1)

- Cross-client bulk actions.
- Notifications/reminders when a step stalls (a later cron could nudge).
- Editing the workflow step set from the UI (steps are fixed in code for v1).
