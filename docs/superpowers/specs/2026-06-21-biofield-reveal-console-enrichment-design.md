# Biofield Reveal Console Review Enrichment (sub-project A)

**Date:** 2026-06-21
**Status:** Approved (design); ready for implementation plan
**Repos:** deploy-chat (Flask, illtowell.com) for the console endpoint/page/store/delete; the AI-Training vault (`02 Skills/e4l-reveal-push.py`) for the bridge change.
**Parent:** Follows the matcher activation ([[project_e4l_reveal_push]]). Real drafts now flow, but the `/console/biofield-reveals` review page lacks the context Glen needs to review one. This enriches that page. Two sibling sub-projects follow as their own specs: B (auto-on-scan-arrival) and C (batch backfill).

---

## Problem

Reviewing a pushed draft in the console, several things are missing: the draft has no visible id or scan date (you cannot reference a specific one), no client name or tags (only the email), and each layer shows no stress factors (the patterns that drove it). The stress factors are stored, but as raw E4L codes (`ER26`, `MB1`) the console does not render; the readable item names live in `e4l.db` (vault), not the deploy-chat DB. Separately, leftover test drafts (placeholder `m`/`m2` meanings) clutter the pending queue, and there is no way to delete a draft.

## Goal

Make `/console/biofield-reveals` show, per draft: an id + scan-date header, the client name + tags, and each layer's stress factors as a readable list. Add a way to delete a draft, and use it to clear the old test drafts. The member reveal is unchanged and must keep not showing patterns.

## Scope

- **Bridge (vault):** the reveal-push attaches a readable `pattern_labels` list to each pushed layer (derived from the patterns it already pulls). Re-push id=6 and all future drafts then carry readable factors.
- **Store (deploy-chat):** none needed - `pattern_labels` rides inside the existing `layers_json` (arbitrary layer fields already round-trip through `_row`).
- **Console endpoint (deploy-chat):** enrich each draft with `client_name` + `tags` via a best-effort `people` lookup by email.
- **Delete (deploy-chat):** a `biofield_reveal.delete(id)` console action + a Delete button.
- **Console page (deploy-chat):** id+scan-date+name+tags header per card; a readable stress-factor list under each layer title (falling back to raw codes for pre-existing drafts).
- **Cleanup (operational):** delete the leftover test drafts after listing their ids for Glen.

**Out of scope:** sub-projects B and C; any member-facing reveal change; any change to slug resolution, canonical meanings, the trial/cart/cancel flow, approval logic. Member patterns stay hidden.

---

## Confirmed decisions (Glen, 2026-06-21)

- Include all four: draft id + scan date; readable stress factors per layer; client name + tags; delete the old test drafts.
- Name + tags come from a **server-side People join** (covers every draft, no bridge dependency). Stress factors come from the **bridge** (the deploy-chat DB has no E4L item names).
- The stored layer gains `pattern_labels`; the member payload still emits only title/summary/remedy (verified - `_biofield_layer_payload` never serializes patterns).
- Deleting drafts needs a new console action (none exists); it doubles as a general discard control.
- No emoji, no em dashes.

---

## Architecture

### 1. Bridge - `02 Skills/e4l-reveal-push.py` (vault)
- In `run`, build a label map from the patterns already pulled: `label_map = {p["item_code"]: (p.get("full_name") or p.get("name") or p["item_code"]) for p in patterns if p.get("item_code")}`.
- `build_payload(content, email, scan_date, label_map=None)`: for each layer, set `pattern_labels = [label_map.get(c, c) for c in (L.get("patterns") or [])]` (readable name, falling back to the raw code). Keep `patterns` (codes) as well. All other mapping unchanged.
- A re-push of an existing (email, scan_date) updates the pending draft's layers (the server upsert allows not-approved updates), so re-pushing id=6 backfills its `pattern_labels`.

### 2. Store - `dashboard/biofield_reveals.py` (deploy-chat)
- No schema change. `pattern_labels` is a layer field carried inside `layers_json`; `_row` already `json.loads` the layers verbatim, so it round-trips. Add `def delete(cx, rid)` (`DELETE FROM biofield_reveals WHERE id=?`, commit).

### 3. Console endpoint - `api_console_biofield_reveals` (app.py)
- Add a best-effort helper `_people_brief(cx, email) -> {"name": str, "tags": [..]}`: query `people` by `lower(email)` for `name` + `tags`; parse `tags` (a JSON-array string) defensively; return empty fields on miss or error.
- After loading `drafts` + `approved`, enrich each dict in place with `client_name` and `tags`. Wrapped so enrichment never fails the listing.

### 4. Delete action - `dashboard/biofield_reveal_actions.py` + app.py wiring
- Register `biofield_reveal.delete`: validate the id, call `biofield_reveals.delete`, return `{"deleted": id}`. Console-gated like the existing `biofield_reveal.approve` / `.edit` (CONSOLE_SECRET via the dispatch spine). Idempotent (deleting a gone row is a no-op success).

### 5. Console page - `static/console-biofield-reveals.html`
- Card header: render `#<id> - <scan_date> - <client_name>` and a tag row (small chips/text) from `d.client_name` / `d.tags`. XSS-safe (`textContent`).
- Under each layer's title, render a `Stress factors:` line listing `L.pattern_labels` (fallback to `L.patterns` when labels are absent - old drafts). Read-only display; the existing title/summary/remedy editors are unchanged.
- Add a Delete button per card -> `biofield_reveal.delete` -> on success reload the list (mirrors `doApprove`).

### Reuse / untouched
- `_biofield_layer_payload` (member), `_resolve_remedy_slug`, `biofield_meanings`, the approve/edit actions, the trial/cart/cancel flow - all unchanged.
- The `people` table + lookup pattern (`SELECT ... FROM people WHERE lower(email)=?`).

---

## Data flow
1. Bridge pushes layers each carrying `patterns` (codes) + `pattern_labels` (readable). Server stores them in `layers_json` (no schema change).
2. Console endpoint lists drafts + approved, enriching each with `client_name` + `tags` from `people`.
3. Console page shows the id/scan-date/name/tags header and the per-layer stress-factor list, plus Delete.
4. The member reveal serves exactly as before (no patterns, no labels) - verified by an anti-leak test.

## Error handling
- People lookup miss or malformed `tags` -> empty name/tags, no error; the listing still returns.
- A layer with no `pattern_labels` (legacy draft) -> the page falls back to raw `patterns` codes; a layer with neither -> the stress-factor line is omitted.
- A code missing from the bridge's `label_map` -> the code itself is used as the label.
- Delete of a non-existent id -> success no-op (idempotent); never 500.

## Testing
Deploy-chat (`tests/`), run with the project harness (doppler + venv, tmp `LOG_DB`):
- **Endpoint enrichment:** seed a `people` row + a pending reveal -> the draft dict has `client_name` + parsed `tags`; with no `people` row -> empty fields, still listed; malformed `tags` does not raise.
- **Member anti-leak (regression):** a stored layer with `patterns` + `pattern_labels` -> the served member reveal payload contains neither (assert on `window.__REVEAL__` / `_biofield_layer_payload`).
- **Delete action:** `biofield_reveal.delete` removes the row; deleting a gone id succeeds; console-gated (401 without the secret).
- **Console page serve:** the page ships the id/scan-date header markers, the stress-factor container, and the Delete control.

Vault (`02 Skills/tests/test_e4l_reveal_push.py`):
- `build_payload` attaches `pattern_labels` per layer from `label_map`; a code missing from the map falls back to the code; layers still carry `patterns`.
- The existing 6 tests stay green (the new arg is optional).

Run (deploy-chat): `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py tests/test_biofield_trial.py -v`
Run (vault): `cd ~/AI-Training && ~/.venvs/deploy-chat311/bin/python -m pytest "02 Skills/tests/test_e4l_reveal_push.py" -v`

## Notes
- Console-facing changes ship live on merge (the console is CONSOLE_SECRET-gated); no new public flag; the member reveal is untouched so there is no member regression.
- After merge + deploy: re-push id=6 (bridge) to backfill its `pattern_labels`; then list and delete the old test drafts via the new Delete action (ids confirmed with Glen first).
- Sub-projects B (auto-on-scan-arrival) and C (batch backfill) wrap the same bridge engine and inherit the `pattern_labels` enrichment.
