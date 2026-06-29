# Trial-Membership Category + Trial-Credit Upgrade Incentive + `/console/members`

**Date:** 2026-06-28
**Status:** Approved (design) — ready for implementation plan
**Author:** Glen + Claude
**Builds on:** PR #394 (member-gated quantity pricing), project_membership_pause_loyalty, points engine (`dashboard/points.py`)

---

## Problem

The `$1` Biofield trial currently does three things at once (`_fulfill_biofield_trial`, app.py): unlocks the Biofield Analysis, grants a **31-day `memberships` access row** (`source='biofield_trial'`), and creates a `kind=membership` `$99/mo` subscription that auto-bills after one month unless cancelled.

Because `_is_paid_member` counts *any* live membership grant, a trial buyer **already receives the member volume discount** on remedy reorders during those 31 days. That makes the intended conversion carrot ("trial buyers pay regular; we hand back the missed discount as credit when they upgrade") impossible — there is no missed discount to accrue.

**Decision (Glen, 2026-06-28):** trial pricing = **regular price + accrue credit**, implemented by making **trial membership a distinct category** that does NOT count as a paid member for the volume-discount gate, while keeping all content/biofield access.

---

## Goals

1. **Trial = its own category.** A `$1`-trial buyer is a *trial member*, not a *full paid member*, until their first `$99` charge clears.
2. **Trial members price at regular** on remedy orders (volume discount is a paid-member perk).
3. **Accrue the missed discount as credit** over a 30-day window; **drop it into points on conversion to full**, where it auto-applies to the next remedy order.
4. **`/console/members`** page: Trial / Full / Paused columns; the Trial column shows each person's accrued credit = the upgrade call-list.
5. **Show the credit as the upgrade hook** in the client portal and on the invoice.

## Non-goals

- No change to content/biofield/ingredient access for trial members (they keep everything).
- No retroactive clawback of discounts already given to current mid-trial members.
- Credit is delivered as **points** (option b), never applied to the membership charge itself (that billing path isn't points-wired).
- No CVC / card storage (PCI — see reference_member_gated_qty_pricing).

---

## 1. Membership-category model (foundation)

A single classifier is the source of truth. Proposed home: `dashboard/subscriptions.py` (pure module) as `category_for(...)`, with a thin app-level wrapper `membership_category(email)` that opens `LOG_DB`.

`membership_category(email)` → one of `none` | `trial` | `full` | `paused`, derived from the member's `kind=membership` subscription (+ the biofield-trial grant). **No schema change.**

| Category | Rule |
|---|---|
| **paused** | active `kind=membership` sub with `skip_next=1` |
| **full** | active `kind=membership` sub, `skip_next=0`, `order_count ≥ 1` (≥1 real `$99` charge cleared) |
| **trial** | active `kind=membership` sub, `skip_next=0`, `order_count = 0` (still in the free first month) |
| **none** | no active membership sub |

Notes:
- `order_count` is the existing billing source of truth: `create_membership` inserts `order_count=0`; `advance_after_charge` bumps it to 1 on the first successful `$99` charge. So `order_count=0 ⇒ trial`, `≥1 ⇒ full` is exact and needs no new flag.
- A buyer with only a biofield-trial *grant* but somehow no membership sub is still `trial` (defensive: fall back to an active `source='biofield_trial'` grant with no paid sub).

### Gate change

Redefine **`_is_paid_member(email)` ⇒ `membership_category(email) == 'full'`**.

Audit confirms `_is_paid_member`'s only callers are the four pricing sites #394 touched:
- `_qty_unit_cents` callers (app.py ~6471, ~6478, ~13971)
- `_portal_priced_lines(... member=_is_paid_member(email))` (app.py ~11934)

All other membership checks (biofield reveal visibility ~2315, biofield-trial "already member" ~2567, `_ingredient_paid_ok` ~5055, portal `out["paid"]` ~10602) call `_active_membership_for_email` **directly** and are intentionally left unchanged — trial members keep content/biofield/ingredient access.

**Behavior change (intended):** trial members now price remedy orders at regular instead of the member volume rate. Go-forward only at deploy; no retroactive change to past orders.

### Tests
- `category_for` truth table: none/trial/full/paused across order_count 0/1/N, skip_next 0/1, no-sub.
- `_is_paid_member` returns False for a trial email, True for a full email, False for paused/none.
- Pricing regression: a trial email gets regular `_qty_unit_cents`; a full email gets the discounted tier.

---

## 2. Part A — `/console/members`

- **Route:** `@app.route("/console/members")` `@require_console_key` → `send_from_directory(STATIC, "console-members.html")`.
- **API:** `@app.route("/api/console/members")` `@require_console_key` → JSON of three buckets.
- **Nav:** add `{ id:"members", label:"Members", href:"/console/members"+qs }` to `bosMods` in `static/op-nav.js`.
- **Template:** new `static/console-members.html`, three columns **Trial / Full / Paused**, following the existing console page pattern.

Each member row (all from existing data):
- name + email (name via people/customers lookup)
- plan: `$99/mo`
- started: membership `granted_at` (or sub `created_at`)
- **next-charge** (`next_charge_date`) for trial/full, or **paused-until** (`resume_date = add_months(next_charge_date, cadence_months)`) for paused
- loyalty tier: `tier_for(order_count)`
- **Trial rows only:** accrued credit `$X` from `trial_credit.accrued_credit_cents` (the call-list).

Data sources: `kind=membership` subscriptions (all statuses active/paused) + `memberships` grants + `membership_category`.

### Tests
- `/api/console/members` requires console key (401 without).
- Buckets a seeded trial / full / paused member into the right column with the right fields; trial row carries `credit_cents`.

---

## 3. Part B — trial-credit engine

New pure module `dashboard/trial_credit.py` (sqlite connection passed in; no Flask/Stripe).

### Accrual
`accrued_credit_cents(cx, email, *, products, settings, window_days=30) -> int`:
1. Find the buyer's `biofield_trial` order; window start = its `created_at`, end = start + 30 days.
2. For each of the buyer's orders with `created_at` in `[start, end]`, for each volume-eligible line:
   - `regular = _qty_unit_cents/_inhouse_ff_unit_cents(..., member=False)`
   - `member  = same fn(..., member=True)`
   - add `max(0, regular − member) × qty`
3. Return the sum (clamped ≥ 0).

Implementation note: the pricing fns live in `app.py`; `trial_credit.py` takes a pricing callback (or app passes pre-resolved `(regular_unit, member_unit, qty)` per line) to keep the module pure and testable. Decide in the plan; default = pass a `price_member` callable.

### Grant on conversion (the trigger)
In `cron_charge_subscriptions` (app.py ~21366), at the point a `kind=membership` sub's **first** successful charge calls `advance_after_charge` and `order_count` goes 0→1:
```
credit = trial_credit.accrued_credit_cents(cx, email, ...)
if credit > 0:
    points.credit(cx, email, value_cents=credit,
                  reason='trial_upgrade_credit',
                  order_ref=f'trial-credit:{email}')   # idempotent via has_entry
```
- Idempotent: deterministic `order_ref` so a re-run never double-credits.
- Lands in the points balance → auto-applies to the next remedy order via the existing `points.redeem` path. No membership-billing change.
- 30-day window caps the liability.

### Tests
- Accrual sums `(regular − member) × qty` over only in-window, volume-eligible lines; ignores out-of-window orders and non-eligible items; returns 0 with no trial order.
- Conversion grant: simulating the first `$99` charge writes one `trial_upgrade_credit` points entry of the accrued amount; a second run is a no-op (idempotent); a cancelled-before-charge buyer gets nothing.

---

## 4. Part C — display

### Portal
- `api_client_portal` adds `membership_category` + `trial_credit_cents` to its JSON.
- `static/client-portal.html`: for `category == 'trial'`, render a banner:
  > **Your membership credit so far: $X** — it becomes points toward your next remedy order when your membership continues.
- Hidden for non-trial categories.

### Invoice
- `_invoice_summary` adds `member_credit_cents` for a non-full member = the discount missed on **this** order (Σ over this invoice's volume-eligible lines of `(regular − member) × qty`).
- `static/invoice.html`: a line under the total:
  > Become a member and get **$X** back toward your next order.
- Shown only when `member_credit_cents > 0` and the buyer is not already full. Stays out of the print/PDF `#print-root` isolation as appropriate (match existing invoice print rules).

### Tests
- Portal API returns `trial_credit_cents` for a trial buyer, `category` correct.
- `_invoice_summary` returns `member_credit_cents` for a non-full buyer with eligible lines, 0 for a full member.

---

## 5. Build sequence (incremental PRs; render-verify each on prod)

1. **PR1 — category model + gate change.** `category_for`/`membership_category`, `_is_paid_member = full-only`, tests. Render-verify: trial email prices regular, full member prices discounted.
2. **PR2 — trial-credit engine.** `dashboard/trial_credit.py` + accrual + grant-on-conversion hook in the charge cron, tests.
3. **PR3 — `/console/members`.** Route + API + template + nav, reading category + credit.
4. **PR4 — display.** Portal banner + invoice line.

Each PR: own branch off `sess/c0fdf59f` worktree → tests → opus review → PR → render-verify on prod before the next.

---

## Reuse map (existing code)

- `dashboard/subscriptions.py`: `tier_for`, `add_months`, `create_membership`, `active_memberships_by_email`, `advance_after_charge`, `pause_membership_by_email`, schema (`kind`, `order_count`, `skip_next`, `next_charge_date`, `cadence_months`).
- app.py: `_active_membership_for_email` (8573), `_is_paid_member` (4238), `_qty_unit_cents` (4189), `_inhouse_ff_unit_cents` (4202), `_fulfill_biofield_trial` (6527), `cron_charge_subscriptions` (21366), `api_client_portal` (11691), `_invoice_summary` (25531), `_portal_priced_lines` (11641).
- `dashboard/points.py`: `credit`, `has_entry`, `balance`, `redeem`.
- `static/op-nav.js` (nav), `dashboard/__init__.py` `require_console_key`.
- Tests: `tests/`, `conftest.py` (`DATA_DIR` override; `:memory:` for pure modules).

## Open items deferred to the plan

- Exact pure/impure boundary for the pricing callback into `trial_credit.py`.
- Whether `/console/members` also surfaces a manual "grant credit now" action (default: no — conversion is automatic).
- Copy polish on the two display strings (approved as written above).
