# Spec — Email-Sourcing Collector (supplier-quote auto-ingest)

> **Staging location:** written to the vault because the deploy-chat worktree is busy with the core-editing build. Move to `deploy-chat/docs/superpowers/specs/` on the build branch once core-editing (E1) merges.

## Context

Glen wants supplier **pricing + sourcing data collected automatically from email** instead of manual notes (trigger: the HydroCurc $334/kg · MOQ 25 kg · 7–10 day quote arrived by email and had to be hand-entered). The data target now exists — the `ingredient_sources` table from the FMP migration (`price_per_unit`, `unit_size`, `minimum_order`, `lead_time_days`, supplier) — and the core-editing build (E1) makes those fields editable + override-protected.

**Confirmed by Glen:** **review-queue model** — quotes land in a staging inbox the human approves; never auto-write to a source.

## Architecture

```
Render daily cron  →  IMAP read drglenswartwout@gmail.com  →  filter (sender/keyword + LLM classify)
   →  Haiku structured-output extract {supplier, ingredient, $/unit, currency, MOQ, lead time, confidence}
   →  INSERT-OR-IGNORE into supplier_quotes (prod chat_log.db), keyed by gmail message-id (idempotent)
   →  Console "Sourcing inbox" lists pending quotes + best-match ingredient/supplier (human-adjustable)
   →  Approve  →  writes/updates an ingredient_sources row + marks the quote applied   (Reject/dismiss too)
```

**Why server-side (Render cron), not local:** the email creds are headless-ready — **`GMAIL_DRGLEN_APP_PASSWORD` is in Doppler prd** → IMAP (`imaplib.IMAP4_SSL("imap.gmail.com")`, app-password login), the exact pattern in `02 Skills/email-bounce-scan.py`. The write target (`supplier_quotes` + `ingredient_sources` + the console) is the prod DB on Render, so a Render cron reads mail and writes the DB directly — no local→prod hop. Mirrors the existing hourly `glen-qbo-reconcile` Render cron.

## Scope

**In (v1):**
- `supplier_quotes` staging table (prod chat_log.db).
- `scripts/scan_supplier_quotes.py` — IMAP read + heuristic/LLM filter + LLM extraction + idempotent stage. Dry-run default, `--write`; runnable as a Render cron.
- `dashboard/sourcing.py` — reads (pending/all), fuzzy match helpers (ingredient/supplier by name), `approve_quote` (write an `ingredient_sources` row from a quote + mark applied), `dismiss_quote`.
- `/api/sourcing/*` endpoints (console-gated): list, patch-match (adjust the matched ingredient/supplier/fields before approving), approve, dismiss.
- Console **"Sourcing inbox"** tab in `/admin/ingredients`: pending quotes with extracted fields + matched ingredient/supplier (editable), Approve / Dismiss.
- Render cron `glen-sourcing-scan` (daily).

**Out (deferred / dependencies):**
- **Auto-write** — review-only by design.
- **New-ingredient / new-supplier creation** — approve writes a source to an **existing** ingredient. A quote for an ingredient not yet in the DB (e.g. HydroCurc today) stages as **"unmatched — needs ingredient"**; fully applying it needs the ingredient to exist (manual create now, or the **E2 create-capability** later). v1 surfaces + matches + applies-to-existing. This is the main dependency: E2 (create) makes new-ingredient onboarding one-click.
- **CoA / spec-attachment ingestion** (auto-populate spec fields — curcuminoid %, particle size, heavy metals — from an attached CoA PDF, like the HydroCurc CoA we hand-ingested) = **v2**. Note the hook: the scanner can capture attachment refs now; parsing them into spec fields is the next layer.
- Multi-currency conversion (capture `currency`, don't convert); non-ingredient procurement; PO automation.

## Data model (SQLite, prod `chat_log.db`)

```sql
CREATE TABLE IF NOT EXISTS supplier_quotes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  gmail_msg_id TEXT,                 -- idempotency key
  received_at TEXT, from_email TEXT, subject TEXT, raw_snippet TEXT,
  -- extracted (LLM):
  supplier_name TEXT, ingredient_name TEXT,
  price REAL, price_unit TEXT, currency TEXT,
  moq REAL, moq_unit TEXT, lead_time_days INTEGER,
  confidence REAL,                   -- 0..1 from the extractor
  -- matched (fuzzy → human-confirmable):
  supplier_id INTEGER REFERENCES suppliers(id),
  ingredient_id INTEGER REFERENCES ingredients(id),
  -- lifecycle:
  status TEXT DEFAULT 'pending',     -- pending | applied | dismissed
  applied_source_id INTEGER REFERENCES ingredient_sources(id),
  has_attachments INTEGER DEFAULT 0, -- v2 hook (CoA etc.)
  extras TEXT, notes TEXT,           -- notes = curated
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_quotes_msg ON supplier_quotes(gmail_msg_id) WHERE gmail_msg_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_quotes_status ON supplier_quotes(status);
```

## Approve flow (the one that writes real sourcing data)

On **Approve** of a pending quote whose `ingredient_id` is matched: insert (or update) an `ingredient_sources` row for that ingredient — `supplier_id`, `supplier_name`, `price_per_unit = price`, `unit_size`/`unit_type` from `price_unit`, `minimum_order = moq`, `minimum_order_unit = moq_unit`, `lead_time_days` — then set the quote `status='applied'`, `applied_source_id`. **Respects the E1 override system:** if that source's `price_per_unit` is console-overridden, approving flags a conflict rather than silently overwriting (human chooses). Dismiss → `status='dismissed'`.

## Email selection (avoid scanning everything)
- IMAP search the last N days of inbox (config), skip already-staged `gmail_msg_id`s.
- Pre-filter on sender heuristics + keywords (price, /kg, MOQ, lead time, quote, COA) to cut LLM volume.
- LLM returns `is_supplier_quote` + `confidence`; stage only `is_supplier_quote=true`. Low-confidence still stages (flagged) — human decides; nothing is silently dropped.
- Optional: honor a Gmail label (e.g. `Sourcing`) Glen applies to force-include.

## Reuse
- `02 Skills/email-bounce-scan.py` — IMAP4_SSL + `GMAIL_DRGLEN_APP_PASSWORD` headless read; the daily-scanner skeleton.
- The QBO-reconciler Render cron pattern (`glen-qbo-reconcile`) for the prod cron.
- `ingredient_sources` + the curated/override write model (E1); the console tab + `api()` patterns.
- Haiku structured-output extraction (force tool-use — the journal lesson: never trust free-text JSON).

## Verification
1. Unit tests: extraction parse, staging idempotency (re-scan same msg → 0 new), fuzzy match, approve writes an `ingredient_sources` row + marks applied, dismiss.
2. Route tests (Pinecone-skip) for `/api/sourcing/*`.
3. Dry-run the scanner against a fixture mailbox / sample emails → review staged rows.
4. End-to-end on a real recent quote (e.g. the HydroCurc email) → stages → matches → approve → source row appears.

## Build approach
Own spec→plan→SDD, branch off `main`, **after core-editing (E1) merges**. ~4 tasks: (1) `supplier_quotes` schema + `dashboard/sourcing.py` reads/approve/dismiss/match + tests; (2) `scripts/scan_supplier_quotes.py` (IMAP + LLM extract, idempotent) + tests; (3) `/api/sourcing/*` endpoints + tests; (4) console Sourcing-inbox tab; + Render cron wiring. Whole-branch review. Highest-risk: the extraction accuracy + the approve→ingredient_sources write (override-conflict handling).

## Dependencies / sequencing
- **Soft-depends on E1 (core-editing)** for the editable source fields + override model — finishing now.
- **Pairs with E2 (create new ingredients/sources)** for one-click new-ingredient onboarding from a quote; without E2, unmatched-ingredient quotes stage but apply only after the ingredient exists.
- Recommend: build this **after E1 merges**; E2 can come before or after (it only improves the unmatched-quote path).
