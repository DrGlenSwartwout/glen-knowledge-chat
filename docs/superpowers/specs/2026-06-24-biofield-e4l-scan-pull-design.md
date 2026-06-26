# Biofield Analysis — pull recent E4L scan on client identification

**Date:** 2026-06-24
**Status:** Approved
**Scope:** Local-only Biofield Analysis tool (`biofield_local_app.py` + `dashboard/biofield_*`). No server changes.

## Problem

The new manual remote causal **spoken** Biofield Analysis (FMP replacement) does not see the
client's E4L voice-scan history. When Glen identifies a client at the start of a remote
spoken session, the most recent E4L results should be pulled in automatically — and if there
is no *fresh* voice scan (within the past 2 weeks), he should be told, and shown how many days
ago the last scan was done.

## Decisions (confirmed)

1. **Pull behavior:** Reference panel **+** narrative context. Show the scan's ranked findings
   read-only beside the authoring screen, AND make them available to the narrative/video-script
   LLM as corroborating context. The causal chain is NOT auto-seeded — Glen's spoken testing
   stays the source of truth.
2. **Stale scans (> 2 weeks):** Always show the most recent scan with its age. Fresh scans get a
   green badge; stale scans get an amber `⚠ No fresh voice scan — last scan N days ago (stale)`
   badge but the findings are **still shown** (clearly marked stale). No scan at all → grey
   `No E4L scan on file`.
3. **Window:** 14 days (2 weeks). `fresh = days_ago <= 14`.

## Architecture

E4L scans live in a **separate** database, `~/AI-Training/e4l.db` (kept fresh by the
`e4l-daily-watch` cron). The local Biofield app uses `chat_log.db` and has no E4L connection
today. The proven query (`latest_scan` + `pull_patterns`, with identity-merge handling) lives in
the vault tool `02 Skills/e4l_synthesis.py`, which is NOT importable by the app. We **vendor**
those ~25 lines into a new self-contained `dashboard/biofield_e4l.py` (rejected alternative:
`sys.path`-inject the vault file — fragile, path has a space, off the import path).

### 1. `dashboard/biofield_e4l.py` (new — only piece touching `e4l.db`)

```
scan_context(email, today, *, db_path=None, window_days=14, limit=12) -> dict
```

Returns:
```
{status: "fresh"|"stale"|"none", found: bool, scan_id, scan_date,
 days_ago: int|None, fresh: bool, window_days: 14, findings: [...], message: str}
```
- `findings[i]` = `{rank, code, name, description}` (ranked by `priority_rank`).
- Opens `e4l.db` **read-only** (`file:{path}?mode=ro`). `E4L_DB` env overrides; default
  `~/AI-Training/e4l.db`. **Never raises** — missing DB / blank email / no client / no scan all
  return `status:"none"` (mirrors the `quote()` never-raises pattern).
- Vendors `_merge_group` (split duplicate accounts read as one person), `_latest_scan`,
  `_findings`.
- `today` is injected (YYYY-MM-DD) for testability; the route passes the Mac's local date.
  `days_ago = (today - scan_date).days`, clamped ≥ 0. `fresh = days_ago <= window_days`.

### 2. `biofield_local_app.py` routes (client becomes known at the header step)

- `POST /author/<id>/header` — after `update_header`, return
  `{"ok": true, "e4l": <context>, "html": <panel>}` so the panel renders the instant the email
  is entered.
- `GET /author/<id>/e4l` — same context for the test's stored email, so re-opening an existing
  test loads the panel. `status:"none"` when no email on file.
- `E4L_DB` constant + a `_today()` helper (`datetime.date.today().isoformat()`).

### 3. UI — `render_e4l_panel(ctx)` in `biofield_report_html.py` (server-rendered, testable)

- Green `Recent E4L scan · N days ago` / amber `⚠ No fresh voice scan — last scan N days ago
  (stale)` / grey `No E4L scan on file`, then the compact ranked findings list
  (`rank. code — name — description`).
- `render_author_html` gets a `<div id=e4lpanel>` placeholder below the header card; author JS
  fetches `GET /author/<id>/e4l` on load and re-renders from the `header` POST response.

### 4. Narrative/video LLM context — `biofield_narrative.py`

- `_user_block(report, notes, scan=None)` and `generate_narrative` / `generate_video_script` /
  `build_*_prompt` gain an **optional** `scan=None` param. Default reproduces today's output
  exactly (existing tests unchanged). When present, appends a
  `RECENT E4L VOICE SCAN (N days ago, fresh|stale)` block with the ranked findings.
- One system-prompt line added to both prompts: *"If a RECENT E4L VOICE SCAN block is present,
  you may reference what the scan showed as corroborating context; do not invent scan findings."*
  Observation language preserved.
- The two generate routes fetch `scan_context` for the client email and pass it in.

## Testing

`tests/test_biofield_e4l.py` against a seeded temp `e4l.db`: fresh (≤14d), stale (>14d), none,
missing-DB, identity-merge, days-ago math (incl. future-date clamp), blank email. Plus
`render_e4l_panel` rendering states, narrative `_user_block` with/without `scan` (existing tests
must stay green), and the header/`e4l` route responses (with an injected `scan_context`).

## Guardrails / non-goals

- No auto-seeding of the causal chain.
- No deployed-server change — `biofield_e4l.py` is imported only by the local app.
- `scan` param is optional everywhere it is added — nothing existing breaks.
