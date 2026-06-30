# SP2b-3 — ASH ally: practitioner client-in-focus (search + scoped memory)

**Date:** 2026-06-29
**Status:** Design / spec
**Repo:** deploy-chat (illtowell.com)
**Parent:** SP2b (cross-surface ASH ally memory). SP2b-1 (helper + 4 SSE surfaces) and SP2b-2 (the 2
client `scoped_reply` surfaces + the `overlay=""` param) shipped (#424, #426, flag-dark
`ASH_ALLY_ENABLED`).

## Context

On the **practitioner** chat surface (`/api/practitioner/chat`, the `#scoped-chat` widget on
`static/practitioner-dropship.html`), the person typing is the *practitioner*, not the health-seeker.
SP2b-3 lets the practitioner **search for one of their own clients and bring that client into focus**,
so the assistant reads/writes **that client's** ASH memory (`ash_map`, keyed by the client's email).
The practitioner builds a persistent picture of each client across sessions.

Two facts from the code (mapped 2026-06-29) shape this slice:
1. **The practitioner chat is currently broken from the UI:** `/api/practitioner/chat` authenticates via
   a `token` (query or body), but the dropship widget POSTs only `{message, history}` — no token — so
   every send 401s. SP2b-3 fixes this (front-end sends the token, mirroring the working
   `/api/practitioner/assist` pattern) as part of making client-focus work.
2. **The only practitioner→client link is `dispensary_orders`** (`practitioner_id TEXT` + `customer_email`
   in LOG_DB). There is no `practitioner_id` on `people` and no portal→practitioner attribution. So the
   client set for SP2b-3 = **clients who have a dispensary order under this practitioner.**
   ("Portal-published clients" is deferred — it needs publish-time practitioner attribution, which does
   not exist yet.)

## Authorization (the critical property — non-negotiable)

`ash_map`/`ash_ally` are keyed purely by email with no scoping of their own. So **the app layer must
verify the in-focus client belongs to the authenticated practitioner before any ASH read or write.**
Without this, a practitioner could read/write an arbitrary email's memory (IDOR). The guard is a single
query against `dispensary_orders`. It runs in BOTH the search (results are already scoped to the
practitioner) and the chat handler (re-checked server-side on every turn — the client_email arrives from
the client and must never be trusted).

`practitioner_id` is a Supabase UUID **string** everywhere — never `int()` it.

## Backend — `dashboard/practitioner_portal.py` (pure, unit-testable; mirrors `dispensary_order_history`)

Both functions take an optional `db_path` (default `_db_path()`), open `sqlite3.connect`, and call
`_ensure_dispensary_table(cx)` first — exactly like the existing dispensary functions. Both query LOG_DB
(where `dispensary_orders` and `people` both live).

- `client_belongs_to_practitioner(practitioner_id, email, *, db_path=None) -> bool` — the ownership guard:
  ```sql
  SELECT 1 FROM dispensary_orders
  WHERE practitioner_id = ? AND lower(customer_email) = lower(?) LIMIT 1
  ```
  Returns `True`/`False`. Empty/None email → `False`. `practitioner_id` coerced via `str(...)`.

- `search_clients(practitioner_id, q, *, limit=8, db_path=None) -> list[dict]` — the practitioner-scoped
  client search. Empty `q` → `[]`. Otherwise (LEFT JOIN `people` for a display name):
  ```sql
  SELECT DISTINCT d.customer_email AS email, COALESCE(p.name,'') AS name
  FROM dispensary_orders d
  LEFT JOIN people p ON lower(p.email) = lower(d.customer_email)
  WHERE d.practitioner_id = ?
    AND d.customer_email IS NOT NULL AND d.customer_email <> ''
    AND (lower(d.customer_email) LIKE ? OR lower(COALESCE(p.name,'')) LIKE ?)
  ORDER BY name, email
  LIMIT ?
  ```
  `LIKE` arg = `f"%{q.strip().lower()}%"`. Returns `[{"email": ..., "name": ...}]`, deduped by email
  (DISTINCT). Results are inherently scoped to the practitioner — no client of another practitioner can
  appear.

## Backend — endpoints (`app.py`)

- **New `GET /api/practitioner/clients/search`** (~near the other `/api/practitioner/*` routes):
  `pid = _practitioner_session_pid()`; if no `pid` → `401 {"ok": False, "error": "authentication required"}`.
  Else `q = request.args.get("q","")`, return `{"ok": True, "clients": _pp.search_clients(pid, q)}`.

- **Wire `/api/practitioner/chat`** (~app.py:11576): after `pid` is resolved, read
  `client_email = (body.get("client_email") or "").strip().lower()`. Compute a guarded subject:
  ```python
  subject = client_email if (client_email and _pp.client_belongs_to_practitioner(pid, client_email)) else ""
  _ally_ov = ash_ally.ally_overlay(LOG_DB, subject)   # "" when no/unowned client
  result = _chat.scoped_reply(message, history, catalog, overlay=_ally_ov)
  ```
  Then, only when `subject` is non-empty, fire the record on a background daemon thread (try/except):
  `ash_ally.record_turn(LOG_DB, _db_lock, subject, message, result.get("reply", ""))`.
  When no client is in focus (or an unowned email is sent), `subject=""` → no overlay, no record: the
  endpoint behaves exactly as today (a general formulation assistant). The `scoped_reply` `overlay`
  param already exists (SP2b-2).

## Frontend — `static/practitioner-dropship.html` (the `#scoped-chat` widget)

- **Fix auth:** include the practitioner `token` on the chat request — change the chat fetch to
  `'/api/practitioner/chat?token=' + encodeURIComponent(TOKEN)` (mirror the `quote`/`portal-data` calls),
  so `_practitioner_session_pid()` resolves and the chat stops 401ing.
- **Client search + focus:** above `#scoped-chat-log`, add a small search `<input>` and a results list
  that calls `GET /api/practitioner/clients/search?token=<TOKEN>&q=<typed>` (debounced). Picking a result
  sets a JS `currentClientEmail` and shows a **"Client in focus: <name> (<email>)"** badge with a
  "change/clear" affordance (clearing sets `currentClientEmail=""`).
- **Send focus:** the chat POST body gains `client_email: currentClientEmail`. With no client in focus it
  is `""` and the chat is the plain assistant.
- Fail-soft: a failed search request just shows no results; it never blocks sending a message.

## Known limitation (documented, not solved here)

`ash_map._haiku_extract`'s prompt is written for a person describing *their own* health ("their own
words"). A practitioner describing a client is third-person, so dimensions + notes are still captured
correctly, but the stored `opened_excerpt` will be the practitioner's phrasing rather than the client's.
Acceptable for v1; a practitioner-mode extract prompt is a future refinement.

## Testing

- `tests/test_practitioner_clients.py` (plain pytest, no app import — `practitioner_portal` functions take
  `db_path`): seed a temp LOG_DB via `record_dispensary_order(pid, invoice_id=..., customer_email=...)`
  for two practitioners + insert `people` rows; then:
  - `client_belongs_to_practitioner`: True for an own client (case-insensitive on email); False for a
    different practitioner's client, an unknown email, and empty/None.
  - `search_clients`: empty `q` → `[]`; matches by email substring and by joined name; **never returns
    another practitioner's client** (the security assertion); dedupes repeat-order emails (DISTINCT);
    respects `limit`.
- The endpoint + chat wiring are app.py edits, gated by `ast.parse` + grep (no route harness; same
  boundary as SP2b-1/2). The front-end is verified by go-live render-verify.

## Verification (go-live)

Same flag (`ASH_ALLY_ENABLED`). After enabling in Render, render-verify on a real practitioner token:
load `/practitioner/dropship?token=<...>`, search a known dispensary client, put them in focus, send a
turn that mentions a health detail, confirm zero console errors and that the client's `ash_map` advanced;
confirm a NON-client email (typed directly / another practitioner's client) does NOT load memory
(authorization holds); confirm the chat works with no client in focus (general assistant) and that the
401 bug is gone.

## Out of scope

- Portal-published clients as a second ownership source (needs publish-time practitioner attribution).
- A practitioner-mode (third-person) extract prompt.
- `/member/scan-analysis/chat` (SP2b-4).
- The Glendalf voice/video premium tier.
