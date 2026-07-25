-- Reciprocal client/practitioner portal linking and first-entry qualification gate.
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS qualification_type text
  CHECK (qualification_type IN ('licensed','coach_certification','resale'));
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS certification_details text;
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS qualification_completed_at timestamptz;
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS manual_approval_requested_at timestamptz;
ALTER TABLE practitioners
  ADD COLUMN IF NOT EXISTS manual_approval_note text;
