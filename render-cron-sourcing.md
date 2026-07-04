# Sourcing-scan Render cron — ops note

## Cron job: `glen-sourcing-scan`

**Schedule:** daily (e.g. `0 8 * * *` — 8 AM UTC)

**Command:** `python scripts/scan_supplier_quotes.py --write`

**Required env vars** (set in the Render service environment or via Doppler):
- `GMAIL_DRGLEN_APP_PASSWORD` — Gmail app password for drglenswartwout@gmail.com (IMAP access)
- `ANTHROPIC_API_KEY` — Claude/Haiku extraction

Mirrors the pattern used by `glen-qbo-reconcile`.

---

## Go-live checklist

1. **Dry run first** — omit `--write` to review counts without staging any quotes:
   ```
   python scripts/scan_supplier_quotes.py
   ```
   Check the output: how many emails scanned, how many look-like-quotes, how many extracted.

2. **Review staged quotes** — open the Sourcing inbox tab at `/admin/ingredients` (Sourcing inbox tab) and inspect a few rows. Confirm supplier names and prices look reasonable.

3. **Enable the cron** — add `glen-sourcing-scan` in the Render dashboard (Cron Jobs section) with `--write`. Set the schedule and the two env vars above.

4. **First live run** — watch the Render cron logs. Idempotent by `gmail_msg_id` so re-runs are safe.
