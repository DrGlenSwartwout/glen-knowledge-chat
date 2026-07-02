# Two-Door "See & Choose" Surface — Implementation Plan

**Goal:** A flag-dark `/begin/choose` page presenting Door A (à la carte) vs Door B
(Continuous Care), reached from a reveal-page handoff CTA.

**Architecture:** New `TWO_DOOR_ENABLED` flag + `GET /begin/choose` route serving a cloned
`static/begin-choose.html` with a server-injected, token-verified `window.__CHOOSE__` payload;
plus a `choose_enabled`-gated CTA on the reveal page. No pricing/checkout/DB changes.

## Global Constraints
- `TWO_DOOR_ENABLED` off (default) ⇒ prod byte-for-byte unchanged (route redirects, CTA hidden).
- Email never placed in the client payload or URL; only the reveal token is carried forward.
- Do NOT touch `journey_state.path` / `paid_fork` semantics — use a distinct `care_fork` trigger.
- Copy: em-dash-free (Glen's rule); Raleway/Open Sans + dark healing palette (clone `begin-path.html`).

---

### Task 1: Flag + `/begin/choose` route
**Files:** Modify `app.py` (flag near ~4507; route after `begin_path` ~2503). Test: `tests/test_two_door_choose.py`.

Flag: `TWO_DOOR_ENABLED = os.environ.get("TWO_DOOR_ENABLED","").strip().lower() in ("1","true","yes","on")`

Route: flag off → `redirect("/")`. Flag on → verify `?token=` via `_biofield_verify_token(_hash_token(token))`;
valid → `reveal_url=f"/begin/biofield/{token}"`, `token` echoed; invalid/absent → `reveal_url="/begin"`,
`token=None`. Inject `window.__CHOOSE__ = {token, reveal_url, program_enabled(PROGRAM_CARE_TASTER_ENABLED),
program_tier(PROGRAM_SCALABLE_TIER), prepay_enabled(PREPAY_LADDER_ENABLED)}` before `</head>` (escape `<>&`);
no-store headers.

Tests: flag off→302 to `/`; flag on no token→200 generic (`reveal_url":"/begin"`, `token": null`, no-store);
valid token (monkeypatch `_biofield_verify_token`→(True,{...}))→`reveal_url":"/begin/biofield/TOK"`;
invalid token→generic fallback.

### Task 2: `static/begin-choose.html`
**Files:** Create `static/begin-choose.html`.

Clone `begin-path.html` chrome. Eyebrow "Choose how you continue"; two `.card`s: Door A "On your own"
(à la carte, → `__CHOOSE__.reveal_url`), Door B "Continuous Care" ($99/mo, → `/prepay`). On load + each
door click fire `POST /begin/unlock {trigger:'care_fork', care_choice:'solo'|'care', ref}` (no `path` kwarg).
Include `theme-toggle.js`, `ref-capture.js`, `widget.js`. Served by Task 1's route (assert door hooks present).

### Task 3: Reveal-page handoff CTA
**Files:** Modify `static/begin-biofield.html` (add CTA in the member render IIFE) + `app.py`
`begin_biofield_reveal` (add `"choose_enabled": TWO_DOOR_ENABLED` to both member payload branches).

CTA (anchor, `.buy-btn` class) "Choose how you want to continue" → `/begin/choose?token=<token>`, appended to
`root` only when `data.choose_enabled`. Test: patched `TWO_DOOR_ENABLED=True` + monkeypatched member reveal →
payload contains `"choose_enabled": true`.
