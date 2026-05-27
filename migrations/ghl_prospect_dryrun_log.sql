-- GHL prospecting sync — dry-run audit log
-- Spec: docs/superpowers/specs/2026-05-26-practitioner-finder-design.md (Phase 2)
-- Tag-only first-run posture: dry-run writes intended upserts here for Glen to
-- review before the real-run flag flips.

CREATE TABLE IF NOT EXISTS ghl_prospect_dryrun_log (
  id                       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  practitioner_id          uuid REFERENCES practitioners(id) ON DELETE CASCADE,
  intended_email           text NOT NULL,
  intended_first_name      text,
  intended_last_name       text,
  intended_phone           text,
  intended_tags            text[] NOT NULL DEFAULT '{}',
  intended_custom_fields   jsonb NOT NULL DEFAULT '{}',
  source_org               text,
  logged_at                timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ghl_prospect_dryrun_log_logged_at
  ON ghl_prospect_dryrun_log (logged_at DESC);
CREATE INDEX IF NOT EXISTS ghl_prospect_dryrun_log_source_org
  ON ghl_prospect_dryrun_log (source_org);
