# Per-Client Pickup Flag — Design

**Date:** 2026-07-08
**Status:** Approved (design); implementation not started
**Follows:** PR #734 (`b405022c`) — *hand-off invoices are not pickups*

---

## Problem

PR #734 established the rule: **a Biofield hand-off is not a pickup.** Hand-off invoices now compute shipping normally, because the `biofield-analysis` service line no longer counts as a bottle.

But Glen's rule has a second half that the codebase cannot express:

> orders … should have shipping added **unless that client is marked as picking up products**

**There is no per-client pickup marker anywhere in the codebase.** Today, a client who genuinely collects in person must have the "Pickup (no shipping)" checkbox ticked by hand on every order, and a Biofield hand-off — which renders no checkbox at all — can never be a pickup.

## Goal

Let a client be marked as collecting in person, so that:

1. In-house order entry **pre-checks** the pickup box when that client is selected (Glen can always override).
2. A Biofield hand-off, which posts no `pickup` key, **resolves the flag server-side**.
3. Everything else is unchanged.

## Scope

**In scope:** in-house order entry (`/api/orders/manual`) and the Biofield hand-off (`dashboard/biofield_invoice.py`).

**Explicitly out of scope:** the client's own self-serve checkout — portal reorder and funnel. A flag set months earlier in a different screen must not silently zero shipping on a purchase with no human in the loop. This is a deliberate constraint, not an oversight; it is the same class of failure PR #734 unwound.

**Also out of scope:** backfilling orders already stored with `channel='pickup'`.

---

## Current state (verified 2026-07-08)

- `people` table DDL: `app.py:23899` (`_init_people_table()`, runs at import).
- **Two migrations exist for one table.** `_init_people_table()` contains an inline `ALTER TABLE people ADD COLUMN` loop for `("address1", "address2", "zip")` — *this is the one prod runs*. `dashboard/customers.py:add_people_address_columns()` is a near-duplicate **referenced only by `tests/test_inhouse_order_entry.py:32`**. Adding a column to the customers.py copy alone would leave every test green while prod never gains the column.
- `dashboard/customers.py:PICKER_COLS` is an allowlist. `get_person()` / `find_people()` project onto it, so a new column is invisible to the UI until it is added there.
- `/api/orders/manual` reads `pickup = bool(body.get("pickup"))`. `static/order-new.html` **always** posts the checkbox (create and edit). `biofield_invoice.default_create_order` posts **no** `pickup` key (as of #734).
- There is **no customers console page**. The customer picker on `static/order-new.html` is the only place a client record is selected.
- `dashboard/portal_identity.py:46` also runs `CREATE TABLE IF NOT EXISTS people` with a **narrower** column set. In the web app `_init_people_table()` wins by import order. This is an ordering accident, not a guarantee. **Noted, not fixed here.**

---

## Design

### 1. Schema — one migration, not two

Add to `people`:

```
pickup_default INTEGER DEFAULT 0
```

Move the column list into `dashboard/customers.py` as a single idempotent migration, and have `_init_people_table()` call it:

```python
# dashboard/customers.py
_ADDED_COLS = (("address1", "TEXT DEFAULT ''"),
               ("address2", "TEXT DEFAULT ''"),
               ("zip",      "TEXT DEFAULT ''"),
               ("pickup_default", "INTEGER DEFAULT 0"))

def add_people_columns(cx):
    """Additively migrate `people`. Idempotent. THE migration — app.py calls this."""
```

`add_people_address_columns` is **renamed** to `add_people_columns` (it no longer adds only address columns), and its single caller — `tests/test_inhouse_order_entry.py:32` — is updated. No alias is kept: a leftover alias would let a future caller re-introduce the divergence. `app.py:_init_people_table()` drops its inline `ALTER` loop and calls `customers.add_people_columns(cx)` instead. One code path, exercised by both prod and tests.

**Rationale:** this is the defect that would have silently sunk the feature. It is in scope because it is the thing being modified, not a drive-by refactor.

### 2. Module API — `dashboard/customers.py`

```python
def set_pickup_default(cx, person_id, on: bool) -> None
def pickup_default_for_email(cx, email) -> bool   # unknown/blank email -> False
```

Add `"pickup_default"` to `PICKER_COLS` so the picker payload carries it.

**Unknown email resolves to `False`.** Guessing wrong toward `True` ships goods for free; guessing wrong toward `False` charges shipping Glen can refund. Fail toward charging.

### 3. Resolution rule — exactly one place

In `/api/orders/manual` only:

```python
pickup = bool(body["pickup"]) if "pickup" in body else customers.pickup_default_for_email(cx, email)
```

| caller | posts `pickup`? | result |
|---|---|---|
| order entry (create) | always | checkbox wins |
| Biofield hand-off | never | client's `pickup_default` |
| unknown/new client | never | `False` (charge shipping) |

### 4. The edit route does NOT consult the flag

`/api/orders/<id>/edit` stays purely checkbox-driven, via `orders.channel_on_edit(pickup, existing_channel)` from #734.

**Why this matters:** if edit re-resolved the client default, then unchecking pickup on a pickup-client's order would snap back to `pickup` on save. That is the latch bug of #734 rebuilt in a new place. This is a hard constraint with a test guarding it.

### 5. UI — `static/order-new.html`

- **Create mode:** when a client is chosen from the picker, `$("pickup").checked = !!p.pickup_default`. Set it both ways (`= !!x`, never `if (x) … = true`), so picking a pickup client and then a normal one clears the box.
- **Edit mode:** unchanged. Prefills from the order's own `channel`, never from the client flag.
- **Setting the flag:** a small "Always picks up (no shipping)" toggle beside the customer panel, posting to a new console-gated endpoint.

### 6. New endpoint

```
POST /api/console/customers/pickup   {email | person_id, pickup: bool}
```

Console-key gated, mirroring the existing `POST /api/console/customers/rename` (`app.py:33085`). Owner role only.

---

## Data flow

```
Biofield hand-off ──(no pickup key)──► /api/orders/manual ──► pickup_default_for_email(email)
                                                                      │
Order entry (create) ──(pickup: bool)──► /api/orders/manual ──────────┤ explicit wins
                                                                      ▼
                                                        channel = "pickup" | "retail"
                                                        shipping = effective_shipping_cents(...)

Order entry (edit) ──(pickup: bool)──► /api/orders/<id>/edit ──► channel_on_edit(pickup, existing)
                                                                 [flag NEVER consulted]
```

## Error handling

| Case | Behavior |
|---|---|
| Unknown / blank email | `False` — charge shipping |
| `people` row missing `pickup_default` (pre-migration) | `False` (column read defensively) |
| Console endpoint, non-owner | 401, flag unchanged |
| Migration re-run | No-op (idempotent `ALTER` in try/except) |

## Testing

Each with a watched RED before its fix.

1. `pickup_default_for_email` → `False` for unknown email, blank email, `None`.
2. `set_pickup_default` / `pickup_default_for_email` round-trip, both directions (on → off → on).
3. `add_people_columns` is idempotent; a fresh DB gains `pickup_default`; a second call is a no-op.
4. **`_init_people_table()` produces a `people` table containing `pickup_default`** — guards the two-migrations divergence by asserting against the path prod actually runs.
5. `/api/orders/manual` with **no** `pickup` key + flagged client → `channel='pickup'`, `shipping_cents == 0`.
6. `/api/orders/manual` with explicit `pickup: false` + flagged client → `channel='retail'`, shipping charged. Explicit wins.
7. `/api/orders/manual` with no `pickup` key + unflagged client → `channel='retail'`.
8. **Edit-route guard:** editing a flagged client's order with `pickup: false` yields `channel='retail'` — the flag must not resurrect the latch.
9. `PICKER_COLS` includes `pickup_default`, and `find_people` returns it.

## Follow-ups (not this spec)

- Backfill of already-latched `channel='pickup'` rows. Counting script: `/tmp/pickup-count.sh` (aggregate only, no PII).
- `portal_identity.py` creating a narrower `people` table under the same name.
- Bundles (`dry-eye-relief-program` et al.) resolve to the `"default"` bottle type but hold physical components; their packing is likely wrong.
