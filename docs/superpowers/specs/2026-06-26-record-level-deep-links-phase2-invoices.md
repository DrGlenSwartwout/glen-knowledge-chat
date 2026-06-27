# Record-Level Dashboard Deep-Links — Phase 2: Invoice / AR (QBO)

**Date:** 2026-06-26
**Status:** Design approved, ready for implementation plan
**Repo:** deploy-chat (`glen-knowledge-chat`)
**Builds on:** Phase 1 (PR #349 + #350, merged) — the citation-token registry mechanism.

## Problem

Phase 1 made person/client mentions in the dashboard briefings clickable (→
`/console/crm?email=`). Phase 2 makes **accounts-receivable / invoice** mentions
in the money-cash briefing clickable, landing on the specific invoice row of the
`/console/money#receivables` board (which already has per-row record-payment
actions).

**Key constraint discovered during scoping:** the money briefing today reports
**Practice Better** invoices (PB `invoiceNumber`), but the console `#receivables`
board is **QuickBooks AR** (`finance.open_invoices()`, QBO `Id`). These are
different systems with **no shared identifier** — a PB invoice cannot be matched
to a QBO row. **Decision (Glen): add QBO AR to the briefing** so the briefing
reports the same receivables the board shows, making per-row deep-links possible.
PB invoices remain person-linked (Phase 1, unchanged).

## Scope

**In scope:**
- Add QBO accounts-receivable to the money-cash snapshot.
- Mint **invoice** linkables for QBO open invoices → `/console/money?invoice=<id>#receivables`.
- Highlight the targeted row on the `#receivables` board (scroll-to + flash) via a
  new `?invoice=` autoload.
- Reword the money-cash prompt so QBO AR is the AR/overdue/collections focus
  (linkable) and PB stays clinical-billing activity (person-linked) — no
  double-counting of receivables.

**Out of scope:**
- Orders and payments deep-links (Phase 3).
- Any change to `static/dashboard.html`, `resolveRefLinks`, or `recNavigate` —
  invoice links reuse them verbatim.
- Changing PB invoice handling (stays Phase-1 person links).

## Mechanism reuse

Invoice links are just another `ref` in the registry resolving to a URL. The
client side is unchanged:
- `mdRender` → `resolveRefLinks` resolves `ref:rN` → the invoice URL (unknown →
  plain text), exactly as for people.
- `recNavigate` already produces the correct URL for a query+hash href:
  `/console/money?invoice=123#receivables` → splits the hash, appends the key
  with `&`, re-appends the hash → `/console/money?invoice=123&key=<KEY>#receivables`.
  The board reads `?key=`, `?invoice=` from `location.search` and `#receivables`
  from the hash.

## Components

### 1. Snapshot — add QBO AR (`dashboard/briefing_runner.py`)

In `gather_snapshot()`, add to the `money` block:
`"qbo_ar": _safe(<finance.open_invoices>, label="qbo_ar")`. Rows are
`{id, doc, customer, email, total, balance, due_date, days_overdue}`. The
existing `_safe()` wrapper turns a missing-QBO-token failure (off-prod) into an
`_error` marker — no crash, no links locally.

### 2. `briefing_links.py` — invoice linkables

- `invoice_url(qbo_id)` → `"/console/money?invoice=" + quote(id) + "#receivables"`.
  Single source of truth; no console key.
- An invoice pass in `build_linkables`: for each `money.qbo_ar` row with an `id`,
  mint `{type:"invoice", display: customer or ("Invoice " + doc/id), url:
  invoice_url(id)}` and stamp `rec["ref"]`. Refs share the existing counter;
  dedup by url. `_error` blocks skipped.
- Person pass (PB invoices + inbox senders) unchanged.

### 3. money-cash prompt (`dashboard/briefing_runner.py`)

Reword `SLUG_PROMPTS["money-cash"]` so **QBO AR (`money.qbo_ar`) is the accounts
receivable / overdue / collections** source (name the customer + amount + age),
and **Practice Better (`money.practice_better`) is clinical billing activity**
(collected/outstanding). Explicitly tell it not to report the same receivable
twice. The Phase-1 RECORD LINKS instruction (already present) makes the LLM link
any record carrying a `ref`, so QBO rows get linked once discussed.

### 4. `static/console-money.html` — row highlight

- Add `data-inv="<id>"` to each `.ar-row` in the receivables `rowHtml`.
- After `MoneyReceivables.load()` renders, read `?invoice=` from
  `location.search`; if present, find `.ar-row[data-inv="<id>"]`, scroll it into
  view and add a transient highlight class (CSS flash). Mirrors the Phase-1 CRM
  `?email=` autoload. The `#receivables` hash already activates the tab; the
  autoload must run after the receivables rows are in the DOM.
- Graceful: missing/unknown `?invoice=` → no-op.

### 5. No change to the dashboard renderer

`static/dashboard.html`, `resolveRefLinks`, `recNavigate` untouched.

## Data flow

```
gather_snapshot → money.qbo_ar = finance.open_invoices() (rows w/ QBO id)
  → build_linkables: invoice pass stamps ref on each AR row, registry[ref] =
    {type:invoice, display, url:/console/money?invoice=<id>#receivables}
  → LLM money-cash card cites overdue invoices as [Acme, $5,000](ref:r3)
  → persisted to money-cash.md + money-cash.links.json (Phase-1 path, unchanged)
serve → {markdown, links}
dashboard mdRender → resolveRefLinks → <a href="/console/money?invoice=..#receivables" class=rec-link>
click → recNavigate appends key → /console/money?invoice=<id>&key=<KEY>#receivables
console-money: #receivables tab active, ?invoice= autoload scrolls+flashes the row
```

## Backward compatibility

Additive. If `qbo_ar` is absent or `_error` (off-prod, or QBO token failure), no
invoice linkables are minted and the card simply omits AR links — exactly today's
behavior. Existing person links and PB handling unchanged.

## Testing

- **Unit:** `invoice_url`; `build_linkables` mints invoice refs from a `qbo_ar`
  block (id → invoice_url, ref stamped); `_error` qbo_ar safe; a mixed
  person+invoice snapshot yields both types with shared ref counter + url dedup.
- **Real-shape mock:** qbo_ar rows shaped like `finance.aging()` output
  (`id, doc, customer, email, total, balance, due_date, days_overdue`).
- **Source-assert:** money-cash prompt references `qbo_ar`; `console-money.html`
  has `data-inv` + `?invoice=` autoload + highlight.
- **Render-verify (browser):** seed a receivables payload with a known row id,
  load `/console/money?invoice=<id>&key=…#receivables`, assert the receivables
  tab is active, the row scrolls into view and gets the highlight class, and
  there are zero console errors.
- **Prod go-live:** after merge + deploy, trigger a regen and confirm the
  money-cash registry contains `type:"invoice"` entries with
  `/console/money?invoice=…#receivables` urls (only when QBO AR has open
  invoices).

## Open items resolved during planning

- Exact `finance.open_invoices` import path + call style in `gather_snapshot`
  (mirror the other `_safe(...)` money entries).
- Whether the receivables tab’s `load()` is async such that the `?invoice=`
  autoload must hook its completion (verify the `MoneyReceivables.load()` /
  `_moneyLoaded` flow) rather than run on a bare DOMContentLoaded.
