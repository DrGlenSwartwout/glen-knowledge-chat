# Sourcing-scan cron — ops note

## How it's wired (as shipped)

The scan runs **inside the web container** via a cron-gated endpoint, NOT as a standalone
Render cron. A separate cron container has its own ephemeral disk and could not see the
web service's `LOG_DB` where `supplier_quotes` lives — it would stage quotes into a
throwaway database the Sourcing inbox never reads (the same disk-isolation trap the
reply-watcher hit). So:

- **Endpoint:** `POST /api/cron/sourcing-scan` (in `app.py`) — auth `X-Cron-Secret == CRON_SECRET`
  (falls back to `CONSOLE_SECRET`). Calls `scripts/scan_supplier_quotes.py::scan(write=True,
  db_path=str(LOG_DB))`. Pass `?dry=1` for a no-write dry run, `?days=N` to widen the window.
- **Trigger:** folded into the always-on daily cron `scripts/run_personal_email_cron.py`
  (`run_daily_piggybacks()`), alongside the testimonial / pb-sync / triage-digest jobs —
  best-effort, fires ~7am HST daily. No new Render service to provision.
- **Env (already set on the web service):** `GMAIL_DRGLEN_APP_PASSWORD` (IMAP),
  `ANTHROPIC_API_KEY` (Haiku extraction), `CRON_SECRET` (auth). `GMAIL_DRGLEN_USER`
  defaults to drglenswartwout@gmail.com.

## Safety

- Only stages to a **review queue** — nothing is auto-approved into `ingredient_sources`;
  Glen matches/approves or dismisses each quote in the Sourcing inbox at `/admin/ingredients`.
- **Idempotent** by `gmail_msg_id`, so re-runs never double-stage.

## Manual dry run / verify

```
curl -sS -X POST "https://illtowell.com/api/cron/sourcing-scan?dry=1&days=30" \
  -H "X-Cron-Secret: $CRON_SECRET"
```
Returns `{"ok":true,"scanned":N,"staged":M,"mode":"dry_run"}`. Then open the Sourcing
inbox tab at `/admin/ingredients` to review anything a live (non-dry) run staged.
