-- Wellness Credit wallet (Phase 2 of the practitioner wholesale portal).
-- Spec: docs/superpowers/specs/2026-06-01-practitioner-wholesale-portal-design.md
-- Additive + idempotent. Apply manually to the intelligence-engine Supabase:
--     psql "$SUPABASE_DB_URL" < migrations/wallet.sql

-- One credit balance per practitioner + their certification progress (the
-- floor F in dashboard/wholesale_pricing.py = 40 - modules_completed*1.25).
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS wallet_balance_cents bigint NOT NULL DEFAULT 0;
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS modules_completed smallint NOT NULL DEFAULT 0
    CHECK (modules_completed BETWEEN 0 AND 12);

-- Audit log. amount_cents is signed (earns +, spends -); balance_after_cents
-- snapshots the running balance for reconciliation.
CREATE TABLE IF NOT EXISTS wallet_ledger (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  practitioner_id uuid NOT NULL REFERENCES practitioners(id) ON DELETE CASCADE,
  entry_type      text NOT NULL CHECK (entry_type IN
                    ('earn_order','earn_dropship','spend_order','spend_module')),
  amount_cents        bigint NOT NULL,
  balance_after_cents bigint NOT NULL,
  qbo_invoice_id  text,   -- links an earn/spend to the invoice it came from
  module_slug     text,   -- for spend_module
  earn_period     text,   -- 'YYYY-MM', used for the monthly module-redemption gate
  note            text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS wallet_ledger_prac
  ON wallet_ledger (practitioner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS wallet_ledger_period
  ON wallet_ledger (practitioner_id, entry_type, earn_period);

-- Idempotency backstop: at most one earn/spend per invoice per entry_type, so a
-- retried checkout cannot double-credit or double-debit.
CREATE UNIQUE INDEX IF NOT EXISTS wallet_ledger_invoice_uniq
  ON wallet_ledger (qbo_invoice_id, entry_type)
  WHERE qbo_invoice_id IS NOT NULL;

-- IMPORTANT: do NOT re-create v_practitioners_public here. It is a stored
-- SELECT * that does not auto-expose columns added later, so wallet_balance_cents
-- and modules_completed stay OUT of the public finder view. Any future re-create
-- of that view MUST continue to exclude these two columns.
