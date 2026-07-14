# Reply-watcher Gmail token hardening — design

**Date:** 2026-07-13
**Status:** approved (design), pending implementation plan
**Author:** Claude (with Glen)

## Problem

On 2026-07-12 the reply-watcher cron (`glen-reply-watcher`, every 15 min) failed
every run with `HTTP 500: "No Gmail token at /opt/render/.config/google/token.json"`.
The cron itself is thin: it curls the web service's `/api/cron/reply-watch`, which
runs `reply_watcher.process_inbox_replies()`. That path loads the Gmail OAuth token
from a **single file** (`/data/google-token.json`, fallback `~/.config/google/token.json`).
When that file was absent, every run 500'd → cron exited 1 → a 15-min flood of Render
"Server failure" emails. It self-healed only once the token file was restored.

Two fragilities:
1. **Single-file source of truth.** If `/data/google-token.json` is missing (fresh
   disk, wipe, or never written), all Gmail-dependent paths hard-fail. The file is
   the only copy the reply-watcher / inbox loaders know about.
2. **No refreshed-token persistence.** `Credentials.from_authorized_user_file` refreshes
   the access token in memory but the loaders never write it back, so a rotated refresh
   token would silently drift stale.

There is **already a durable pattern in the repo** that avoids both: `_run_cron()`
(app.py) runs the console-push cron using tokens stored in the `oauth_tokens` DB table
(`glen_gmail`, `rae_gmail`, `calendar`), hydrated to temp files per run, with refreshed
tokens **written back to the DB**. The reply-watcher and `dashboard/inbox.py` simply never
adopted it. The `oauth_tokens` table lives on the same disk but is part of `chat_log.db` —
the app's most-backed-up asset — and is read/written via `/api/tokens/<name>` GET/PUT.

## Goal

The reply-watcher and the shared inbox Gmail loader can't silently lose their token.
Common case self-heals with no human action; the rare true-re-auth case produces exactly
one clear alert (never a 15-min flood), delivered over a channel that does not depend on
the Gmail token itself.

## Design

### 1. Shared durable token loader (`dashboard/gmail_token.py`, new)

A single helper both `reply_watcher.py` and `dashboard/inbox.py` call, so the two
Gmail-loading code paths stop diverging.

```
load_gmail_credentials(db_path, name="inbox_gmail", scopes=...) -> (creds, source)
```

Resolution precedence:
1. **`oauth_tokens` DB row** where `name = "inbox_gmail"` — durable source of truth.
2. **File fallback:** `GMAIL_TOKEN_PATH` env → `/data/google-token.json` → `~/.config/google/token.json`.
3. If none yields usable creds → raise `GmailTokenMissing` (a typed exception the caller maps to the alert path).

Scope handling keeps the existing behavior in `dashboard/inbox.py`: read the token's
granted `scopes`, intersect with requested, pass the effective set (avoids `invalid_scope`
on refresh).

**Self-heal backfill:** if creds were loaded from a file but the DB row was absent,
write the file's JSON into `oauth_tokens[name]` (guarded by the app's `_db_lock`). So a
present file repopulates the durable store automatically, and future loads prefer the DB.

### 2. Refreshed-token write-back

```
persist_refreshed_credentials(db_path, name, creds) -> bool
```

Called after a run completes. If `creds` were refreshed during the run (token value or
expiry changed vs. what was loaded), serialize `creds.to_json()` and upsert into
`oauth_tokens[name]` under `_db_lock` — mirrors the write-back loop in `_run_cron`.
Best-effort: a write-back failure is logged, never fatal to the run.

### 3. One deduped alert, over SMTP (not Gmail)

When `load_gmail_credentials` raises `GmailTokenMissing`, `cron_reply_watch` (app.py):
- Sends **one** alert email to Glen via the existing **SMTP** transactional path
  (`smtplib` + `SMTP_HOST/USER/PASS`, the same machinery behind `send_magic_link_email`) —
  independent of the dead Gmail token.
- **Dedup + health state:** a reserved `oauth_tokens` row named `inbox_gmail_health` whose
  `token_json` holds a small JSON blob `{healthy: bool, last_ok: iso, last_alert: iso}`
  (satisfies the table's `NOT NULL token_json` with no new table/migration). Suppress the
  alert if `last_alert` is within the outage window (default 6h). Prevents a new 15-min flood.
- Still returns non-200 so the run is recorded as failed (Render logs remain accurate),
  but Glen now gets a single actionable "re-auth the Gmail token" message.
- On every successful load, update the same `inbox_gmail_health` row (`healthy=true`,
  `last_ok=now`, clear `last_alert`) so a console page could later surface
  "Gmail token healthy: yes/no, last ok: …" without new infra.

### 4. Seed (one-time, at rollout)

The token file is present now (watcher green). PUT its contents into `oauth_tokens` as
`inbox_gmail` via the existing `PUT /api/tokens/inbox_gmail` (auth: `X-Console-Key`), so the
DB is the source of truth from first deploy. Documented as a rollout step, not code.

## Components / files

- **new** `dashboard/gmail_token.py` — `load_gmail_credentials`, `persist_refreshed_credentials`,
  `GmailTokenMissing`; owns precedence, self-heal backfill, scope intersection, write-back.
- `reply_watcher.py` — replace `_resolve_token_path` / `_get_gmail_service` with the shared
  loader; call `persist_refreshed_credentials` after `process_inbox_replies`.
- `dashboard/inbox.py` — replace its `_resolve_token_path` / `_get_gmail_service` with the
  shared loader (keeps scope-intersection behavior).
- `app.py` `cron_reply_watch` — catch `GmailTokenMissing`, fire the deduped SMTP alert, set
  health flag.

## Error handling / edge cases

- DB and file both present but disagree → **DB wins** (source of truth); file is fallback only.
- Loaded from DB, refresh rotates token → write-back updates DB; file left as-is (stale file is
  harmless since DB is preferred).
- Loaded from file (DB empty), refresh rotates → write-back seeds DB with the rotated token
  (self-heal + freshness in one).
- Write-back failure → logged, run still succeeds.
- Alert-send failure (SMTP down) → logged; do not crash the endpoint.
- Concurrency → all DB writes under `_db_lock`, matching `_run_cron`.
- `invalid_grant` (refresh token revoked/expired) surfaces as a load failure → same alert path
  as a missing token (both mean "human must re-auth").

## Testing (run under `doppler run -p remedy-match -c dev -- python3 -m pytest`)

- **Loader precedence:** DB row present → uses DB, no file read. DB empty + file present →
  uses file AND backfills DB row. Both empty → raises `GmailTokenMissing`.
- **Self-heal:** after a file-sourced load, assert `oauth_tokens[inbox_gmail]` now exists and
  equals the file JSON.
- **Write-back:** feed creds whose `to_json()` differs post-run → assert DB updated; identical →
  no write.
- **Alert dedup:** two consecutive `GmailTokenMissing` within the window → exactly one SMTP send
  (monkeypatch `app.smtplib.SMTP` / the send helper); outside the window → a second send.
- **Endpoint:** `cron_reply_watch` with token missing → 500 + alert fired once + health flag set;
  with token present → unchanged behavior (regression).
- App-importing tests SILENTLY SKIP without doppler (no PINECONE_API_KEY) — must run under doppler dev.

## Out of scope (YAGNI)

- No change to the Google OAuth app / consent-screen publishing status or re-auth cadence.
- No new console UI beyond the persisted health flag (a page can read it later).
- No changes to the `glen_gmail` / `rae_gmail` / `calendar` console-push path in `_run_cron`.
- Not migrating other Gmail callers; only the reply-watcher + shared inbox loader.

## Rollout

1. Merge behind no flag (additive; DB-first with file fallback is backward compatible — if the
   DB row is absent it behaves exactly as today via the file).
2. Seed: `PUT /api/tokens/inbox_gmail` with the current token JSON.
3. Confirm live: a reply-watch run logs source=`db` and (if rotated) a write-back; the tokenless
   probe path is exercised by the tests, not in prod.
