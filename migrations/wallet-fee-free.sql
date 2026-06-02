-- Allow the 'earn_fee_free' ledger entry type (3% credit on Zelle/Wise-paid orders).
-- Apply: psql "$SUPABASE_DB_URL" < migrations/wallet-fee-free.sql
ALTER TABLE wallet_ledger DROP CONSTRAINT IF EXISTS wallet_ledger_entry_type_check;
ALTER TABLE wallet_ledger ADD CONSTRAINT wallet_ledger_entry_type_check
  CHECK (entry_type IN ('earn_order','earn_dropship','spend_order','spend_module','earn_fee_free'));
