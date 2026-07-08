# Reveal spend-eligibility — design

**Date:** 2026-07-08
**Author:** Glen + Claude
**Status:** approved (design), building

## Goal

A **free member** who places a **paid order of $100 or more since their last biofield reveal** earns their **next reveal fully un-blurred** (all layers + remedies, same content as a paid member sees). Loyalty loop: buying remedies keeps a free member's next reading unlocked without a paid membership.

## Rules (confirmed with Glen 2026-07-08)

- **Who:** intended for free members; a paid member already sees full reveals so the grant is inert for them.
- **Trigger:** a **single paid order with `total_cents >= 10000`** (not cumulative).
- **Order sources:** any (in-house, GrooveKart, funnel) — all land in the `orders` table via `_ingest_order`. Must be **paid** (`pay_status='paid'` or `paid_cents >= 10000`).
- **"Since their last reveal":** order `created_at` is **after** the email's previous `biofield_reveals` row.
- **Reward:** the **single next reveal is fully un-blurred** (A: full reveal, not just top remedy). **No banking, no stacking** — any qualifying spend in the period unlocks exactly the one next reveal; each reveal independently evaluates its own preceding window.

## Why NOT reuse `biofield_free_unlocks`

`biofield_free_unlocks(email PRIMARY KEY, reveal_id, granted_at)` is the existing **one-time, one-per-email lifetime** unlock ("your one free reveal"), and it only un-blurs the **top remedy** (`top_unlocked = fu_rid == row.id`; `free_available = fu_rid is None`). It is INSERT-OR-IGNORE, so it can't repeat per period, and it's the wrong scope. Reusing it would collide with that feature. So we add a **separate per-reveal** unlock.

## Mechanism

### New table + accessors (`dashboard/biofield_reveals.py`)

```sql
CREATE TABLE IF NOT EXISTS biofield_reveal_spend_unlocks (
  reveal_id INTEGER PRIMARY KEY,   -- one row per fully-unlocked reveal
  email TEXT,
  granted_at TEXT
);
```
- `init_spend_unlocks(cx)`
- `record_spend_unlock(cx, reveal_id, email)` — `INSERT OR IGNORE` (idempotent)
- `is_spend_unlocked(cx, reveal_id) -> bool`

### Grant — derived check at reveal creation

`upsert()` (`biofield_reveals.py:80`) is the single reveal-insert chokepoint and returns `(id, is_new)`. When `is_new` is True, best-effort call `maybe_unlock_for_spend(cx, email, new_id)`:

1. Find the previous reveal's timestamp: `SELECT created_at FROM biofield_reveals WHERE email=? AND id<? ORDER BY id DESC LIMIT 1`; if none, use `''` (open-ended — any prior qualifying order counts toward the first reveal).
2. `EXISTS` a qualifying order: `SELECT 1 FROM orders WHERE lower(email)=? AND total_cents>=10000 AND (lower(coalesce(pay_status,''))='paid' OR paid_cents>=10000) AND created_at > ? LIMIT 1`.
3. If found → `record_spend_unlock(cx, new_id, email)`.

Best-effort: wrapped so it never breaks reveal creation. No `_ingest_order` hook, no credit lifecycle.

### Consume — full un-blur via the existing `paid` gate

Everything that un-blurs a reveal already keys off `flags["paid"]` (`_biofield_unlock_flags` ~`app.py:2904`, consumed by `_biofield_visible_slugs` and the reveal payload builder). So in `_biofield_unlock_flags`, after computing `paid`, OR in the per-reveal spend unlock:

```python
paid = paid or _br.is_spend_unlocked(cx, row.get("id"))
```

This renders that one reveal with full content through all existing consumers with no other edits. It flips only the **local, per-reveal view flag** (not real membership / `is_member` / `_active_membership_for_email`), so it affects only that reveal's payload. Member-only funnel flags bundled in the paid branch (`trial_enabled`, `program_enabled`, `cart_enabled`) are all OFF in prod and, if on, showing a $100 buyer a reorder path is acceptable.

## Edge cases

- **No prior reveal:** open-ended window; any prior qualifying paid order unlocks the first reveal.
- **Actual paid member:** grant is harmless (already `paid`); the OR is a no-op for them.
- **Multiple qualifying orders in the period:** `EXISTS` — one or many both unlock exactly the one next reveal (no banking).
- **Refund before the reveal:** the check reads live `orders` state, so a refund that clears `pay_status`/`paid_cents` naturally stops qualifying.
- **Re-synthesis of the same reveal (upsert update path):** `is_new` is False on updates, so the grant runs only on genuine creation; `record_spend_unlock` is `INSERT OR IGNORE` regardless.

## Testing (`tests/test_biofield_reveal_spend_unlock.py`)

- Free-ish email + paid $100 order after prior reveal ⇒ new reveal `is_spend_unlocked` True; `_biofield_unlock_flags` returns `paid=True` for it.
- Order < $100, or unpaid, or dated before the prior reveal ⇒ not unlocked.
- Two qualifying orders ⇒ still exactly one unlock on the next reveal.
- No-prior-reveal + qualifying order ⇒ first reveal unlocked.
- `maybe_unlock_for_spend` re-run for the same reveal ⇒ idempotent (one row).
- `upsert(is_new=True)` with a qualifying order ⇒ spend-unlock row created; `upsert` update path (is_new False) ⇒ none.

## Out of scope (v1)

- Forcing/scheduling a new E4L scan on qualification.
- Cumulative sub-$100 spend.
- Banking multiple free reveals.
- Refund claw-back beyond the live-state read.
