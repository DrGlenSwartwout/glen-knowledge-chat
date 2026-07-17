# Settlement Follow-ups — Group-Bundle Atomicity + Skipped-Settler Visibility — Design

**Date:** 2026-07-17
**Status:** Approved (design), pending spec review
**Depends on:** Settlement durability (PR #958). `_grant_group_bundle` (app.py), `dashboard/order_settlement.settle_paid_order_effects`, the `todos` console board.
**Owner:** Glen / RemedyMatch

## Problem

The #958 whole-branch review flagged that the widened settlement window (redirect + webhook can now run the full settler chain concurrently) exposes non-atomic settlers. Investigation narrowed this to **one real gap** plus a **visibility gap**:

- **Wallet margin — NOT a bug (dropped).** `wallet.earn_dropship_margin` writes to a **Postgres** ledger that already has a partial UNIQUE index `(qbo_invoice_id, entry_type) WHERE qbo_invoice_id IS NOT NULL` and a `FOR UPDATE` row lock. A concurrent double-credit is impossible: the loser's insert rolls back on the constraint. No change. (Aside, out of scope: the ledger's `entry_type` CHECK constraint omits several types actually written — pre-existing schema drift.)

- **Group-bundle double-grant — REAL and LIVE (`GROUP_BUNDLE_ENABLED=true` in prod).** `_grant_group_bundle` (app.py) is a SQLite check-then-insert on `group_bundle_grants` (PK `invoice_id`): it runs `create_membership` + `_member_join_welcome` **before** inserting the marker, and `create_membership` is a plain non-idempotent INSERT. Two concurrent runs (redirect + webhook for the same order) both pass the `SELECT`, both create a membership and send a welcome email → **two membership rows + two welcome emails**; only the loser's marker INSERT raises (swallowed).

- **Silent skipped settler — visibility gap.** `settle_paid_order_effects` returns `{"kind","settled","skipped"}`, but both callers (redirect + webhook) ignore it. A transient per-settler failure is marked `settled_at` (attempted) and never retried — invisibly. The highest-value silent loss is a stranded subscription row.

## Design

### Part 1 — group-bundle claim-first (atomicity)
Reorder `_grant_group_bundle` so the marker is claimed **atomically before** any side effect:

```
cur = _gcx.execute(
    "INSERT INTO group_bundle_grants (invoice_id, created_at) VALUES (?,?) "
    "ON CONFLICT(invoice_id) DO NOTHING", (g_invoice, _now_utc().isoformat()))
_gcx.commit()
if cur.rowcount == 0:
    return  # another run already claimed this invoice — do not re-grant
# only the claim winner reaches here:
existing = active_memberships_by_email(...)
if existing: set_next_charge_date(...)
elif g_cus and g_pm: create_membership(...); _member_join_welcome(...)
```

`ON CONFLICT(invoice_id) DO NOTHING` on the PK makes the claim atomic across processes — exactly one run wins, so exactly one membership + one welcome. Mirrors the cashout-review todo's `ON CONFLICT(dedup_key) DO NOTHING` claim pattern already in the codebase.

**Trade-off (documented, acceptable):** a hard crash in the tiny window between the marker commit and `create_membership` would strand that grant (marker present, membership never made, retry skips on the claim). This is strictly better than the current double-grant + double-email, and is the same rare crash-strand class the settlement design already accepts for I1. The lost effect is one free group-window grant, not money.

### Part 2 — skipped-settler → console todo
Both callers capture the return and, when `skipped` is non-empty, raise ONE deduped todo on the `todos` board via a new best-effort helper `_raise_settlement_skip_todo(order_ref, kind, skipped)`:

- `dedup_key = f"settle-skip:{order_ref}"` — collapses the redirect+webhook duplicates for the same order into one board item.
- Insert mirrors `_maybe_raise_cashout_review`: `INSERT INTO todos (created_at, owner, category, title, body, priority, source, dedup_key) VALUES (...) ON CONFLICT(dedup_key) DO NOTHING`, `owner="glen"`, `category="Fulfillment"` (or existing nearest), `priority="high"`, `source="settlement-skip"`, body naming the order_ref, kind, and the skipped effect names.
- Best-effort: the helper never raises into the request path (own try/except), matching the existing settlement best-effort contract.

Both call sites:
```
_res = _osx.settle_paid_order_effects(...)
if _res and _res.get("skipped"):
    _raise_settlement_skip_todo(order_ref, _res.get("kind"), _res["skipped"])
```

## Out of scope
- Wallet margin (already atomic). The wallet `entry_type` CHECK-constraint drift (separate schema issue).
- Auto-retrying a skipped settler (the todo makes it visible + manually actionable; auto-retry would need the settlement-claim machinery, a larger change).

## Testing
- **Group-bundle:** a second `_grant_group_bundle` for the same invoice is a no-op (one membership row, `create_membership`/welcome NOT called the second time); a simulated concurrent race (two runs, marker pre-claimed by the first) → the second bails on `rowcount==0`. Existing group-bundle grant tests stay green.
- **Skipped todo:** `settle_paid_order_effects` returning a non-empty `skipped` → one todo row with `dedup_key=settle-skip:<ref>`; a second call (same order) → still one row (ON CONFLICT); empty `skipped` → no todo; the helper swallows a DB error without raising.
- **No regression:** redirect + webhook settlement paths unchanged when `skipped` is empty.

## Files
- **Modify:** `app.py` (`_grant_group_bundle` reorder; new `_raise_settlement_skip_todo`; two call sites capture the result).
- **Create:** `tests/test_group_bundle_atomic.py`, `tests/test_settlement_skip_todo.py`.
