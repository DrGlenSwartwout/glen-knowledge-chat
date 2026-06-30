# SP2b-2 — ASH ally on the scoped_reply client surfaces

**Date:** 2026-06-29
**Status:** Design / spec
**Repo:** deploy-chat (illtowell.com)
**Parent:** SP2b (cross-surface ASH ally memory). SP2b-1 (helper `dashboard/ash_ally.py` +
the 4 streaming surfaces) shipped (#424, flag-dark `ASH_ALLY_ENABLED`).

## Context

SP2b-1 wired the ally memory layer into the 4 streaming (SSE) chat endpoints. SP2b-2 extends it to the
**non-streaming `scoped_reply` client surfaces**: the dispensary client widget and the invoice pay-link
chat. Both call the shared `dashboard/practitioner_chat.scoped_reply`. The third caller of
`scoped_reply` — the practitioner formulation chat — is **out of scope here** (it needs a client search
to pick the subject; that's SP2b-3). Same flag (`ASH_ALLY_ENABLED`), same fail-open helper, dark until
go-live render-verify.

## Surfaces

| Surface | app.py | subject_email | notes |
|---|---|---|---|
| `/api/client/<code>/chat` (dispensary client widget) | ~11572 | `email` (already resolved + `is_member`-gated, non-empty at the call site) | speaker == subject |
| `/api/invoice/<token>/chat` (invoice pay-link chat) | ~26421 | the order's email (NOT loaded today — add a lookup) | speaker == subject |
| `/api/practitioner/chat` | ~11543 | — | **out of scope (SP2b-3)**; left unchanged |

## Design

### 1. `scoped_reply` gains an `overlay` param (in `dashboard/practitioner_chat.py`)

`scoped_reply(message, history, catalog)` builds its model system prompt inline as `_SYSTEM + cat_txt`
(line ~42). `practitioner_chat.py` imports only `json, re` (+ a deferred `import app as _app` inside
`_llm_json`), so importing `ash_map`/`ash_ally` here would risk a `dashboard → app → dashboard` cycle.

**Therefore the overlay is computed in app.py (where `email` + `ash_ally` already live) and passed in.**
Add a defaulted param:

```python
def scoped_reply(message, history, catalog, overlay=""):
    ...
    system = _SYSTEM + (overlay + "\n\n" if overlay else "") + cat_txt
    out = _llm_json(system, msgs)
```

- The overlay string is the full framed block from `ash_ally.ally_overlay` (already self-contained;
  `""` when inert). Prepended before the catalog, after `_SYSTEM`.
- **Backward-compatible:** the practitioner caller (and any other) that passes only the original 3 args
  is unaffected — `overlay` defaults to `""`, so `system` is byte-identical to today.

### 2. Wire the 2 client call sites (in `app.py`)

`from dashboard import ash_ally` is already imported (SP2b-1). At each of the 2 client surfaces, two
touches:

**a. `/api/client/<code>/chat`** (~11594): `email` is already resolved and gated.
- Before the call: `_ally_ov = ash_ally.ally_overlay(LOG_DB, email)`.
- Change the call to: `result = _chat.scoped_reply(message, history, catalog, overlay=_ally_ov)`.
- After the call (before/after the suggestions loop), fire the record in a background daemon thread,
  try/except-wrapped: `ash_ally.record_turn(LOG_DB, _db_lock, email, message, result.get("reply", ""))`.

**b. `/api/invoice/<token>/chat`** (~26421): the email is not loaded today. Replace the bare token guard
```python
if not _pp.order_id_from_invoice_token(token):
    return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
```
with an order lookup that also yields the email (`_invoice_order_for_token` already calls
`order_id_from_invoice_token` internally, so this removes the double call):
```python
order = _invoice_order_for_token(token)
if not order:
    return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
email = (order.get("email") or "").strip().lower()
```
Then the same two touches: `_ally_ov = ash_ally.ally_overlay(LOG_DB, email)`, pass `overlay=_ally_ov`
into `scoped_reply`, and fire `record_turn(LOG_DB, _db_lock, email, <message>, result.get("reply",""))`
in a background daemon thread after the call.

### 3. The practitioner surface is untouched

`/api/practitioner/chat` keeps calling `scoped_reply(message, history, catalog)` (3 args) — `overlay`
defaults to `""`, so no overlay, no record. SP2b-3 will give it a client-search subject.

## Reuse / consistency

- The helper (`ally_overlay`/`record_turn`), the flag, the fail-open contract, and the background
  daemon-thread record dispatch are all exactly as SP2b-1 — only the host endpoints differ (non-stream).
- `record_turn`'s lock-split (Haiku extract runs unlocked) is unchanged; it still runs off the request
  path via a daemon thread so the synchronous `jsonify` return is not delayed.
- Subject = the speaker on both surfaces (client widget / invoice payer), so `subject_email` is just the
  resolved `email`.

## Testing — `tests/test_practitioner_chat_overlay.py` (plain pytest, no app import)

`dashboard/practitioner_chat.py` does its `import app` lazily inside `_llm_json`, so the module imports
cleanly without secrets/network. Monkeypatch `practitioner_chat._llm_json` to capture the `system`
argument:

- `scoped_reply(msg, hist, cat, overlay="OVERLAY-TEXT")` → the captured `system` CONTAINS `OVERLAY-TEXT`,
  positioned after `_SYSTEM` and before the catalog text.
- `scoped_reply(msg, hist, cat)` (no overlay, default) → the captured `system` does NOT contain any
  overlay and equals the legacy `_SYSTEM + cat_txt` (backward-compatible — the practitioner caller is
  unaffected).
- The returned dict shape is unchanged (`{"reply", "suggested_slugs"}`).

The 2 app.py wirings are thin additive edits verified by `ast.parse` + grep (no SSE/route harness in
this app, same boundary as SP2b-1); behavioral proof is the go-live render-verify.

## Verification (go-live)

Same flag as SP2b-1 (`ASH_ALLY_ENABLED`), already dark. After enabling in Render, render-verify the 2
new surfaces as an identified user with prior memory: the dispensary client widget (`/dispensary/<code>`
with chat enabled) and the invoice chat (`/invoice/<token>` or wherever the widget mounts) — confirm
continuity, zero console errors, and that a follow-up turn advanced the stored map; confirm an anonymous
/ ungated visitor is unchanged.

## Out of scope

- `/api/practitioner/chat` + the client name/email search (SP2b-3).
- `/member/scan-analysis/chat` (SP2b-4).
- The Glendalf voice/video premium tier (separate, later).
