-- Practitioner Finder orchestrator run log
-- Spec: docs/superpowers/specs/2026-05-26-practitioner-finder-design.md (Phase 2)

CREATE TABLE IF NOT EXISTS scraper_runs (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  adapter_name    text NOT NULL,
  started_at      timestamptz NOT NULL DEFAULT now(),
  finished_at     timestamptz,
  rows_scraped    integer NOT NULL DEFAULT 0,
  rows_inserted   integer NOT NULL DEFAULT 0,
  rows_updated    integer NOT NULL DEFAULT 0,
  status          text NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running','success','partial','failure')),
  error_message   text
);

CREATE INDEX IF NOT EXISTS scraper_runs_adapter_started
  ON scraper_runs (adapter_name, started_at DESC);
