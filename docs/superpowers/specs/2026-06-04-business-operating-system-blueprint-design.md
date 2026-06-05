# Business Operating System Blueprint

**Date:** 2026-06-04
**Status:** Master design / blueprint (whole-map). Phase 1 specified in build detail; later phases scoped, each gets its own spec + plan.
**Repo:** deploy-chat (Flask app, `app.py` + `dashboard/` modules + `static/` console)
**Author note:** No em dashes (Glen's standing rule).

---

## 1. Goal

Turn the existing console/dashboard/admin surfaces into a single **business operating system**: one place to SEE the whole business, DECIDE with intelligence that drives action, and ACT across every function, with routine work handled autonomously and exceptions surfaced for a human.

The decided interaction model is **agent plus actionable panels over one shared action layer**: every business action exists once and is invokable both by clicking a panel button and by asking Justus. Autonomy is governed by an actor-by-risk policy matrix, not a single switch.

---

## 2. Current state (the starting point)

This is already a proto-OS, not just a dashboard. Verified inventory (2026-06-04):

**Surfaces:** `/console` (Glen/Rae/Shaira tabs, todos+capture, people/households CRM, referrals, Justus AI sidebar, calendar), `/console/inbox` (Gmail), `/console/projects` (kanban), `/console/settings`, `/dashboard` (money, GHL pipeline, ScoreApp, HeyGen, health, intelligence briefings), `/admin/membership`, `/admin/shipping`, `/admin/atlas`, `/admin/clips`.

**Interactivity today:** Justus is a 28-tool Haiku agent (multi-turn, SSE) acting on three domains only: projects, todos, households. Inbox reply/triage, shipping admin, membership creation, Atlas/clips approvals, and household-merge are interactive. Most dashboard panels are read-only.

**Integrations:** QBO (full CRUD), Stripe (gated), Authorize.net (read), Wise (read), Practice Better (read + webhook), GHL (full CRUD via curl workaround past Cloudflare WAF), Rebrandly (write), GrooveKart (webhook), ScoreApp (webhook), Gmail (read/write), SMTP, ElevenLabs, HeyGen (read), Pinecone (read), Supabase (CRUD), Anthropic, OpenAI, USPS (email parse + rate scrape), Mapbox, Cloudflare R2, Meta Ads (read, verification pending).

**Automation spine:** Render crons (personal email daily, reply-watcher 15m, USPS rate weekly, briefings daily, PB tag sync daily), launchd CNS tracking watcher 15m, in-process hourly console-push triage orchestrator, membership renewals, workspace backup, Shaira daily report. Separate scheduled reconciler agents (Rae weekly close, siloed-systems reconciler) run outside the console.

**Data spine:** ~50 SQLite tables in `/data/chat_log.db` (users, auth, query_log, people, households, todos and workspace, affiliate/referral, memberships, calendar_events, shipments, oauth_tokens, and more), Supabase Postgres (journal_entries, practitioners), and JSON config files in `/data` (products, aliases, atlas, trusted-links, pairings, intelligence briefings).

**Auth model:** single shared `CONSOLE_SECRET` via `@require_console_key`, plus a seed of per-user scoped access tokens; `CRON_SECRET` for bulk jobs.

### The core gap

The system can SEE far more than it can DO. Panels are read-heavy and Justus acts on only three domains. "Fully interactive" means closing the read-to-act gap across every module and letting Justus act across all of it, through one consistent, audited, permission-aware layer.

---

## 3. Architecture: the spine

Chosen approach: **Action Registry + Event/Audit stream + RBAC.** Rejected alternatives: two parallel implementations (panels and agent drift, double maintenance, today's pattern) and agent-only (slow for routine ops, single point of failure).

### 3.1 Action Registry

Every business action is declared once as a typed record:

```
Action {
  key            # e.g. "finance.refund_order", "orders.create_label", "crm.move_deal"
  module         # finance | orders | crm | marketing | products | content | comms | tasks
  title          # human label for the panel button + Justus tool
  description     # one line, also used as the Justus tool description
  params         # typed schema (name, type, required) for both form fields and tool input
  risk_tier      # read | low_write | money_send | irreversible
  permission     # roles allowed: {owner, ops, va, agent, system}
  executor(params, ctx) -> result   # performs the action, usually wrapping an existing route/API client
  confirm_summary(params) -> str     # "You are about to refund $84.00 to Jane R." for the confirm gate
  reversible     # bool
  undo(event) -> result              # optional, for reversible actions
}
```

Actions are registered in code next to the logic they wrap (a decorator `@action(...)` populating a module-level `ACTION_REGISTRY` dict). Executors call the internal Flask route handlers or API clients that already exist (QBO, GHL, Gmail, shipping, etc.), so the registry is a thin, uniform front over capability the app already has.

### 3.2 Single dispatch path

Both front-ends call one function:

```
dispatch_action(key, params, actor, attended) -> Result | PendingApproval | Denied
  1. resolve Action from registry (404 if unknown)
  2. validate params against Action.params
  3. permission check: actor.role in Action.permission, else Denied
  4. policy lookup: POLICY[actor.role][Action.risk_tier] in {auto, confirm, queue, deny}
       - deny    -> Denied
       - queue   -> write Event(status=pending_approval); return PendingApproval
       - confirm -> if not confirmed flag, return needs-confirmation (with confirm_summary); else execute
       - auto    -> execute
  5. execute: Action.executor(params, ctx); write Event(status=done|failed, result)
  6. return Result
```

Panel buttons POST to a generic `/api/action/<key>` that calls `dispatch_action(..., attended=True)`. Justus tools are thin wrappers that call the same `dispatch_action`. New capability appears in both front-ends the moment an action is registered. Justus tool definitions can be generated from the registry so the agent and panels never drift.

### 3.3 Event / Audit stream

One append-only `events` table is the OS spine and the activity feed:

```
events(
  id, ts, actor, source,            # source: panel | justus | cron | webhook | system
  action_key, module, risk_tier,
  params_json, result_json,
  status,                           # done | failed | pending_approval | confirmed | cancelled
  reversible, undo_token,
  ref_type, ref_id                  # optional link to an order, invoice, contact, etc.
)
```

It records two kinds of entries: **operator/agent actions** (every dispatch) and **business events** ingested from existing webhooks and crons (order paid, lead arrived, invoice overdue, email received). That makes the stream a single timeline of everything happening in the business, and the audit log of everything the OS and its operators did. Pending approvals are `status=pending_approval` rows that render as action cards in Command Home; approving one dispatches its action.

### 3.4 RBAC / actor identity

Replace the single `CONSOLE_SECRET` gate with actor-scoped identity, building on the existing `access_tokens` (already seeded with scoped tokens). Each request resolves to an actor with a role:

- **owner** (Glen) : full access
- **ops** (Rae) : full operations
- **va** (Shaira) : scoped read + low-risk writes
- **agent** (Justus, unattended/scheduled) : non-interactive runs
- **system** : crons, webhooks

`CONSOLE_SECRET` remains the owner master key for backward compatibility during migration.

### 3.5 Autonomy policy matrix

Policy is a function of (actor role x action risk tier), stored as config:

| Role | read | low_write (tags, todos, drafts, notes) | money_send (refunds, payouts, outbound email, bulk) | irreversible (deletes, merges) |
|------|------|----------------------------------------|-----------------------------------------------------|--------------------------------|
| owner (Glen) | auto | auto | confirm (phased, see below) | confirm |
| ops (Rae) | auto | auto | confirm | confirm |
| va (Shaira) | auto (scoped) | auto | queue for owner/ops approval | deny |
| agent (Justus unattended) | auto | auto | **always queue** for approval | deny |
| system (cron/webhook) | auto | auto | queue for approval | deny |

A scheduled Justus run can triage, tag, and draft on its own; a refund or outbound send it wants becomes a pending-approval card rather than executing. **Unattended Justus never moves money: money_send always queues for human approval, regardless of amount (decided 2026-06-04).**

**Phased owner autonomy (decided 2026-06-04).** The owner money_send cell starts at **confirm on every action** (manual approval for a break-in period, so we watch the registry behave on real money before loosening it). Once it has proven reliable, flip a config flag to enable **auto under $50, confirm at or above $50**. The policy matrix reads its owner money_send mode from config (`OWNER_MONEY_AUTO_THRESHOLD`, default 0 = confirm everything; set to 50 after the break-in). No code change to graduate, just config.

---

## 4. The unified shell and the Home signal board

One console shell with a left-nav. The landing view is the **Home signal board**, and behind it sits **one dedicated page per module** (the 9 functional modules in section 5). Justus is cross-cutting and present on every page, not a page of its own. Existing pages (`/dashboard`, `/console/inbox`, `/console/projects`, `/admin/*`) become module views inside the shell rather than separate destinations; we wrap, not rebuild. The shell carries actor identity (login/token), so every panel and Justus call inherit the actor.

### 4.1 Home signal board

A grid of one cell per functional module (a 3x3 mission-control view). Each cell shows:

- the module name,
- a **color-coded priority signal** for "how much does this need me right now",
- a one-line summary (for example "4 invoices overdue, $612"),
- the **top actions needed now** (1 to 3), each either a one-click dispatch (through `dispatch_action`) or a jump into the module page,
- click-through to the module's full page.

**Signal colors** (priority for action):

| Color | Meaning |
|-------|---------|
| red | urgent, act now (overdue, blocked, money or client at risk) |
| amber | needs attention soon (due today, queue building) |
| green | healthy, nothing required |
| gray | idle / no data |

### 4.2 The module signal contract

Each module implements one function:

```
signal(actor) -> {
  level,            # red | amber | green | gray
  summary,          # one line for the Home cell
  top_actions,      # [ {label, action_key?, params?, href?} ]  up to 3
  count             # optional badge number (e.g. items needing attention)
}
```

The Home board calls every module's `signal()` and renders the grid; the worst level floats the cell up visually. Pending-approval events from the spine (section 3.3) feed straight into the relevant module's signal, so anything queued for the operator shows as red/amber on Home. Priority rules start as **seed heuristics per module** and are refined from real data over time (Glen's seed-then-learn rule), for example: Money red on overdue AR or cash below a floor; Orders red on an unfulfilled order aging past 24h; Comms red on a flagged email unanswered past N hours; Content amber when approvals are piling up. The heuristics live next to each module so they evolve independently.

---

## 5. Module map (end-state)

There are **nine functional module pages**, each with its own page, each contributing **views** (panels, mostly already exist as reads), **actions** (new registry entries that make them actionable), and a **signal()** for the Home board (section 4.2). The nine: Money & Finance, Sales & CRM, Orders & Fulfillment, Marketing & Growth, Products & Inventory, Content & Knowledge, Comms & Calendar, Team & Tasks, and Practitioner & B2B. The Home signal board (section 4) is the landing view, not a module. Justus (section 5.10) is cross-cutting, present on every page. Listed below with the current-to-target delta (subsection order is not priority order).

> Reconciling "the 9": with the Home board now its own landing view (not a module), the ninth functional module is **Practitioner & B2B**, which also closes the noted B2B/Fullscript gap. If your nine differ (for example you want Command/Strategy as a page and B2B folded elsewhere), say so and the set adjusts; it does not affect the Phase 1 spine.

### 5.1 Practitioner & B2B Channel
- **Integrations:** GHL, QBO, `wholesale_orders` / `dispensary_orders` / `practitioners` tables, Practice Better, future Fullscript.
- **Purpose:** the wholesale and practitioner channel, a stated growth priority with no operational home today.
- **Views:** practitioner accounts, wholesale/dispensary orders, application and approval pipeline, ASH-certified tiers.
- **Actions:** `b2b.approve_practitioner`, `b2b.create_wholesale_order`, `b2b.set_pricing_tier`, `b2b.invite_practitioner`.
- **Delta:** the tables exist but are unsurfaced and the practitioner portal is "future"; this module gives the B2B/Fullscript priority a home.

### 5.2 Money & Finance
- **Integrations:** QBO (CRUD), Stripe, Wise, Authorize.net, Practice Better.
- **Views:** cash position, AR aging / unpaid invoices, weekly close, reconciliation results.
- **Actions:** `finance.issue_invoice`, `finance.refund_order`, `finance.void_invoice`, `finance.record_payment`, `finance.run_reconciliation`, `finance.send_payment_reminder`.
- **Delta:** money is observe-only today; QBO can already write, so this is mostly registering actions and pulling the existing reconciler agents into the console as `run_reconciliation`.

### 5.3 Sales & CRM
- **Integrations:** GHL (CRUD), people/households tables (strong already), PB.
- **Views:** people/households (keep), GHL pipeline (already read).
- **Actions:** `crm.move_deal`, `crm.add_tag`, `crm.log_outreach`, `crm.enroll_workflow`, `crm.create_opportunity`, plus existing household merge actions migrated into the registry.
- **Delta:** pipeline is read-only today; add deal/opportunity/workflow actions. Household tools already exist in Justus; re-home them as registry actions.

### 5.4 Orders & Fulfillment
- **Integrations:** GrooveKart (webhook), USPS, Gmail, shipments table, QBO.
- **Views:** order lifecycle board (new, packed, shipped, done) built from webhook events + shipments + QBO.
- **Actions:** `orders.create_label`, `orders.mark_packed`, `orders.mark_shipped`, `orders.send_tracking`, `orders.refund` (delegates to finance).
- **Delta:** today orders are webhook-to-GHL-plus-email and USPS is email-parsed with manual drafts. Needs a real order object and a label/tracking integration. Highest manual-effort loop to close.

### 5.5 Marketing & Growth
- **Integrations:** Meta Ads (read, pending verification), ScoreApp, affiliate/referral tables, funnel (`/begin`), Rebrandly.
- **Views:** spend-to-lead-to-order-to-LTV attribution; campaign performance; affiliate/referral and partner-link performance (the `/begin/tools` work surfaces here).
- **Actions:** `growth.pause_campaign` (when Meta API clears), `growth.approve_affiliate`, `growth.create_referral_source`, `growth.adjust_offer`.
- **Delta:** data is scattered and read-only; this module is mostly assembling attribution and surfacing affiliate controls that already exist as raw routes.

### 5.6 Products & Inventory
- **Integrations:** products.json, Pinecone, the Numbers master sheet, GrooveKart, QBO items.
- **Views:** one catalog master, stock levels, pricing across channels, the 150-formulation launch pipeline as a board.
- **Actions:** `products.adjust_price`, `products.update_stock`, `products.publish`, `products.advance_launch_stage`.
- **Delta:** catalog is fragmented with no single source of truth; this is the largest data-reconciliation effort and likely a later phase.

### 5.7 Content & Knowledge
- **Integrations:** Pinecone, Atlas, clips/R2, HeyGen, video ingestion, case studies.
- **Views:** a content factory: drafted, approved, scheduled, published across atlas concepts, clips, videos, case studies.
- **Actions:** `content.approve_concept`, `content.approve_clip`, `content.trigger_render` (HeyGen), `content.publish`.
- **Delta:** Atlas and clips approvals exist as separate admin pages; unify them and add HeyGen render-trigger (currently read-only).

### 5.8 Comms & Calendar
- **Integrations:** Gmail (CRUD), Google Calendar, Practice Better scheduling.
- **Views:** inbox (keep), calendar (keep).
- **Actions:** `comms.send_reply`, `comms.draft_reply`, `comms.archive`, `comms.book_appointment` (PB), `comms.create_event`.
- **Delta:** inbox is already interactive; add booking/scheduling beyond the current view-and-suppress.

### 5.9 Team & Tasks
- **Integrations:** todos/workspace tables.
- **Views:** todos, delegation, time tracking (keep).
- **Actions:** existing todo/project tools migrated into the registry, now permission-aware and audited.
- **Delta:** functionally complete; mainly gains RBAC and audit by moving onto the spine.

### 5.10 Justus, the cross-cutting operator
- Justus's tools become thin wrappers over `dispatch_action`. As actions register across modules, Justus gains the ability to act on all of them with consistent permissions, confirmation, and audit. Unattended (scheduled) Justus runs at `agent` role and queues anything risky for approval.

---

## 6. Sequencing

Each phase is its own spec and implementation plan. The blueprint above is the fixed target they all aim at.

- **Phase 1, the spine (specified below in build detail):** Action Registry, `dispatch_action`, Event/Audit stream, RBAC/actor identity + policy matrix, generic `/api/action/<key>`, Justus tools re-homed onto the registry, the **module signal() contract**, and the **Home signal board** (the 9-cell mission-control landing view) rendering signals, pending approvals, the activity stream, and briefings-as-actions. Migrate the three existing Justus domains (todos, projects, households) onto the registry as the proof, and ship real signals for the modules whose data already exists. This makes today's read-only dashboard actionable without a rebuild.
- **Phase 2, Orders & Fulfillment (decided 2026-06-04):** close the order-to-ship-to-track loop end to end. Proves the panel-plus-agent action model on a high-value, currently-manual workflow.
- **Phase 3+, remaining modules** on the spine, in rough ROI order: Money & Finance, Sales & CRM pipeline, Marketing & Growth, Comms/Calendar scheduling, Content factory, Products & Inventory (largest data effort, last).

---

## 7. Phase 1 build detail (the spine)

**New modules/files (proposed):**
- `dashboard/actions.py` : the `ACTION_REGISTRY`, the `@action` decorator, `dispatch_action`, the policy matrix, risk tiers, and the `Result/PendingApproval/Denied` types.
- `dashboard/events.py` : the `events` table init, append helpers, ingest hooks for webhooks/crons, and the activity-stream query API.
- `dashboard/rbac.py` : actor resolution from token/`CONSOLE_SECRET`, roles, and the policy config.
- Route `/api/action/<key>` (POST) : generic attended dispatch for panel buttons.
- Route `/api/events` (GET) : the activity stream + pending approvals for Command Home.
- Route `/api/events/<id>/approve` and `/cancel` (POST) : resolve pending approvals.
- `dashboard/signals.py` : the `signal()` contract, a registry of per-module signal functions, and the aggregation that powers the Home board. Phase 1 ships real signals for modules whose data already exists (Team & Tasks from todos; read-derived signals for Money, Orders, Comms from existing reads) and gray placeholders for not-yet-built modules; each later module phase fills in its own `signal()`.
- Route `/api/home/signals` (GET) : aggregate every module `signal()` for the Home board.
- `static/console-home.html` : the Home signal board (9-cell grid: color, summary, top actions, click-through) plus the activity stream and pending-approval cards.
- Justus refactor: replace the 3 hardcoded tool groups with registry-generated tool defs that call `dispatch_action`.

**Migration strategy:** wrap existing capability, do not rewrite it. The first registered actions are the todo/project/household operations Justus already performs, re-expressed as registry actions, proving the path with zero new business logic. The existing `/admin/*` and `/dashboard` pages keep working unchanged and get folded into the shell incrementally.

**Backward compatibility:** `CONSOLE_SECRET` continues to authorize as the owner role throughout Phase 1, so nothing breaks while RBAC is introduced.

---

## 8. Testing

- Unit: `dispatch_action` permission and policy matrix (every actor x risk tier cell), param validation, confirm/queue/deny branches, audit-event writes.
- Unit: registry registration and Justus tool generation from the registry.
- Unit: event stream append + query, pending-approval lifecycle (create, approve, cancel).
- Route: `/api/action/<key>` attended dispatch (auto, confirm-required, queued, denied) and `/api/events` rendering, gated on actor identity.
- Migration: the three re-homed domains (todos/projects/households) behave identically through the registry as they do today (characterization tests against current behavior).
- Guard: no action executes a money_send or irreversible tier for `va`/`agent`/`system` without a pending-approval row. Guard: unattended-agent money_send always queues regardless of amount; owner money_send obeys `OWNER_MONEY_AUTO_THRESHOLD` (0 = confirm all; 50 = auto under $50).
- Signals: each module `signal()` returns the contract shape; the worst level floats up on the Home board; a pending-approval event raises the owning module's level; `/api/home/signals` aggregates all nine.

---

## 9. Non-goals (this blueprint)

- No change to commission logic, the affiliate program schema, or GHL workflow definitions.
- No new payment processor; Products & Inventory catalog unification is explicitly deferred to a later phase.
- No mobile app in Phase 1 (a notification/mobile action surface is a candidate later, once the event spine exists).
- Phase 1 does not migrate every `/admin/*` page into the shell at once; it establishes the spine and Command Home and folds pages in incrementally.

---

## 10. Risks and open questions

- **GHL Cloudflare WAF:** GHL writes still require the curl/local-Mac workaround. Order/CRM actions that write to GHL inherit that constraint; the registry executor must route GHL writes through the working path.
- **Catalog source of truth:** Products & Inventory depends on choosing one master (Numbers sheet vs GrooveKart vs a new table). Deferred, but it blocks that module.
- **Render cron statelessness:** cron containers do not share the web `/data` disk; unattended-agent actions must dispatch via the web service (the established pattern), not write the DB directly.
- **Identity rollout:** moving from one shared secret to per-actor RBAC needs a low-friction login for Rae and Shaira; scoped tokens are the seed but the UX needs design in Phase 1.

**Resolved 2026-06-04:** Phase 2 is Orders & Fulfillment. Owner money_send starts at confirm-everything (manual break-in) and graduates to auto-under-$50 via config, no code change. Unattended Justus always queues money_send for human approval.
