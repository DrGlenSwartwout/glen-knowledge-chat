# SP2b-4 — ASH ally on /member/scan-analysis/chat (the last surface)

**Date:** 2026-06-29
**Status:** Design / spec
**Repo:** deploy-chat (illtowell.com)
**Parent:** SP2b (cross-surface ASH ally memory). SP2b-1/2/3 shipped (#424, #426, #428, flag-dark
`ASH_ALLY_ENABLED`).

## Context

`/member/scan-analysis/chat` (app.py:13545) is the last client chat surface in the SP2b inventory: a
single-turn, non-streaming member chat about their longitudinal voice-scan analysis. SP2b-4 wires the
ASH ally memory layer into it with the same two touches as every prior surface. Same flag, dark.

## Surface

- `/member/scan-analysis/chat` (app.py:13545) — POST, single-turn (no history), returns
  `{answer, access, upsell}`. Subject email = the `rm_member_email` cookie (line 13555) — the member
  themselves (speaker == subject). No IDOR surface (a member's own cookie email keys their own map; the
  page is already tier-gated server-side).

## Design — two touches

1. **Overlay.** The handler assembles a local `system` string (`_SCAN_CHAT_SYSTEM` + either the member's
   analysis facts or the educate-only policy, lines 13574-13578). After that assembly and before the
   `_cl.messages.create(...)` call (line 13580), prepend the ally overlay when non-empty:
   ```python
   _ally_ov = ash_ally.ally_overlay(LOG_DB, email)
   if _ally_ov:
       system = _ally_ov + "\n\n" + system
   ```
2. **Record.** After `answer` is produced and only when `email` is present, fire the record on a
   background daemon thread (try/except-wrapped), just before the response returns (after the
   `resp = jsonify(...)` at line 13588):
   ```python
   if email:
       try:
           import threading as _t
           _t.Thread(target=ash_ally.record_turn,
                     args=(LOG_DB, _db_lock, email, q, answer),
                     daemon=True).start()
       except Exception:
           pass
   ```

`email`, `q`, `answer`, `LOG_DB`, `_db_lock`, and `ash_ally` are all already in scope. The overlay is
fail-open ("" when disabled / no cookie / error); `record_turn` is fail-open and lock-split. With no
member cookie, `email=""` → no overlay, no record (unchanged behavior). Single-turn surface → one record
per Q&A, which is correct.

## Testing

No new testable logic (the `ash_ally` helper is already unit-tested). This is one additive wiring on one
handler, gated by `python3 -c "import ast; ast.parse(open('app.py').read())"` + grep that the overlay +
record touches are present in `member_scan_analysis_chat` (overlay/record counts become 8/8 across
app.py). Behavioral proof is the go-live render-verify (no route harness in this app — same boundary as
SP2b-1/2/3).

## Verification (go-live)

Same `ASH_ALLY_ENABLED` flag. After enabling, render-verify on a coaching magic-link member (the
`rm_member_email` cookie present): ask a scan-analysis question, confirm continuity if the member has
prior ASH memory, zero console errors, and that the stored map advances; confirm a non-signed-in visitor
(no cookie) is unchanged.

## Out of scope

- Anything beyond this one surface. With SP2b-4, all 8 client chat surfaces carry the ally layer.
- The Glendalf voice/video premium tier (separate, later).
- The whole-feature go-live render-verify of #424/#426/#428/this (flip the flag, render each surface).
