# Biofield Reveal: Matched Stress-Pattern Layers

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Extends the Begin Biofield reveal (#4a reveal, #4b $1 trial, #4c match-to-order cart, and the canonical-remedy-meanings + ingest guardrail). The reveal today shows a flat `interpretation` (greeting + body) plus a ranked flat `remedies` list. This restructures it around the matched **stress-pattern layers** the local matcher already computes, each titled by its functional pattern name.

---

## Problem

The portal causal-chain side (`e4l_synthesis.py`) already models a scan as **layers** - each a functional **title**, its grouped **stress patterns**, and a matched FF remedy (Layer 1 = surface/most-recent, higher n = deeper root). The begin-reveal does not carry this: its console review page cannot show the matched stress patterns, and the member sees only a flat remedy list with no sense of the causal chain. Glen needs to see and name the matched patterns when reviewing, and the member should see their reading as titled stress-pattern layers, not a bare list. Separately, once a reveal is approved it cannot be re-edited (the editor is pending-only), so a layer title cannot be corrected after approval.

## Goal

Carry the matcher's stress-pattern **layers** into the begin-reveal end to end: stored on the reveal, editable/namable in the console (pending AND approved), and rendered to the member as titled layers - each layer's **title + summary always shown**, its **remedy gated** by the existing free-top / $1-rest model. Make approved reveals re-editable.

## Scope

A `layers` field on `biofield_reveals`; the matcher push contract gains `layers` (back-compatible with old `remedies`-only pushes); the ingest guardrail/canonical-override applies per layer remedy; a derived flat `remedies` keeps #4b/#4c working; the reveal payload emits titles+summaries always and gates per-layer remedies; the console editor renders/edits layers and lists+edits approved reveals; the member reveal renders titled layers.

**Out of scope:** the local matcher itself (a separate agent - this defines the push contract the server consumes); the portal causal-chain system (`e4l_synthesis` / `portal_biofield_reports`); any server-side scan synthesis or Pinecone ranking; layer reordering in the console (the matcher sets `n`).

---

## Confirmed decisions (Glen, 2026-06-20)

- **Both surfaces:** the matcher pushes structured layers; the console shows/edits/names them; the member reveal renders titled layers.
- **Gating:** layer **titles + pattern summaries are always shown** (like the interpretation); only each layer's **remedy** is gated - top layer's remedy free after approval + the member's one-time free unblock, the rest behind the $1 trial. The reading is the gift; the remedies are the paywall.
- **Re-editable approved reveals folded in:** the console lists pending AND approved; edits (interpretation/layers/remedies) are allowed post-approval and flow to the member.
- Per-layer guardrail: a non-catalog remedy is dropped from its layer (the layer keeps its title + summary); the dropped name is recorded. The canonical-meaning override applies to each layer's remedy.
- No emoji, no em dashes.

---

## Architecture

### 1. Store - `dashboard/biofield_reveals.py`
Add a `layers TEXT` column (JSON, default `'[]'`) via an idempotent additive `ALTER` (same pattern as the `dropped` column). A layer is `{n:int, title:str, summary:str, patterns:[str], remedy:{name,slug,meaning}}` (`patterns` = raw codes, retained for the learning loop, NOT member-shown). `_row` parses `layers`. New `set_layers(cx, rid, layers)`. The flat `remedies` column stays (derived; see ingest) so #4b/#4c are unchanged.
**Re-editability:** drop the `AND first_approved=0` guard from `set_interpretation` / `set_remedies` / `set_layers` so an approved reveal can be corrected. `list_pending` stays (pending queue); add `list_approved(cx, limit)` for the approved section.

### 2. Matcher push contract - `/api/e4l/reveal-draft` (app.py)
The push JSON gains an optional `layers` array (the matcher reuses its `e4l_synthesis` layer output: `n`, `title`, a plain-English `summary`, `patterns`, and the matched FF as `remedy`). Server handling:
- If `layers` present: use them. For each layer's `remedy`, run the existing guardrail (`_resolve_remedy_slug` -> catalog slug, else **drop the remedy from the layer** and record its name in `dropped`; apply the canonical-meaning override when a slug resolves). Store the cleaned `layers`.
- **Derive the flat `remedies`** = the ordered list of each layer's surviving `remedy` (so `_biofield_visible_slugs`, #4b, #4c read it unchanged). Store both `layers` and the derived `remedies`.
- **Back-compat:** a push with only `remedies` (old matcher) is wrapped into titleless single-remedy layers (`title:""`, `summary:""`), so the new render still works and Glen names them in the console; the existing flat-remedies path is preserved.
- The token-mint-on-first-insert + email + the `(email, scan_date)` upsert are unchanged.

### 3. Reveal payload - `begin_biofield_reveal` (app.py)
Emit `layers` with **title + summary ALWAYS** (regardless of paid/approval), plus the interpretation. For each layer's **remedy**: include `{name, slug, meaning, buy_url, page_url}` only when visible (paid -> all; free + first_approved + the member's free-unlock on the TOP layer -> the top layer's remedy), else emit the layer with `remedy: null, remedy_blurred: true` (the withheld remedy details never leave the server - anti-bypass, as #4a/#4c). `_biofield_visible_slugs` reads `layers[].remedy.slug`. `blurred_count` = number of layers whose remedy is withheld. `cart_enabled` / `trial_enabled` unchanged.

### 4. Member render - `static/begin-biofield.html`
After the interpretation, render each layer as a titled section: the **title** (functional pattern name), the **summary** (plain text), then the remedy (name -> product page link + Order button when present, else the blurred placeholder + the unlock CTA). The #4c order bar collects the visible per-layer remedies (slugs from the payload, logic unchanged). XSS-safe (textContent/setAttribute; no innerHTML of dynamic data). Falls back to the existing flat-remedy render when a reveal has no layers (old data).

### 5. Console editor - `console-biofield-reveals.html` + `dashboard/biofield_reveal_actions.py`
Render each reveal's **layers**: per layer an editable `title`, `summary`, and `remedy` (name/slug/meaning) with the per-remedy "remember" canonical toggle; show the "dropped: not in catalog" notice. The edit action (`biofield_reveal.edit`) accepts `layers` (and keeps interpretation), writes via `set_layers` + `set_interpretation`, **re-derives the flat `remedies`** from the layers, and promotes "remember"-flagged remedy meanings to canonical (as today). The page lists **pending** (default) and an **approved** section (via `list_approved`); editing an approved reveal is allowed (the guards were dropped) and re-derives + persists. Approve (`biofield_reveal.approve`) un-blurs the top layer's remedy (unchanged).

### Reuse / untouched
- `_resolve_remedy_slug`, `biofield_meanings` (canonical override), the `dropped` column, `_biofield_visible_slugs` / `_biofield_unlock_flags`, the reveal token/magic-link, `is_member`, the #4b trial + #4c cart endpoints (they read the derived flat `remedies`/slugs).
- Untouched: the matcher agent, the portal causal-chain, pricing/billing.

---

## Data flow
1. Matcher pushes interpretation + `layers` (each: n, title, summary, patterns, remedy).
2. Ingest: per layer, resolve the remedy to a catalog slug (drop + record non-catalog; apply canonical override); store `layers` + the derived flat `remedies` + `dropped`.
3. Glen opens the reveal (pending or approved) in console: sees titled layers, edits titles/summaries/remedies, toggles "remember", approves (un-blurs the top layer's remedy). Edits are allowed post-approval.
4. Member opens the reveal: interpretation + every layer's title + summary shown; each layer's remedy shown if visible (paid all / free top), else blurred + $1 CTA. The #4c cart orders the visible remedies.

## Error handling
- Old `remedies`-only push -> titleless single-remedy layers (render works; Glen names them).
- A layer whose remedy is non-catalog -> the remedy is dropped, the layer keeps its title + summary, the name is in `dropped`.
- Reveal with no layers (legacy) -> the member/console fall back to the existing flat-remedy render.
- All store/ingest/override paths wrapped; ingest never rejects a push; the member render never emits a withheld remedy.
- Re-editing an approved reveal never deletes member state (it updates content; the member's free-unlock ledger and the paid grant are untouched).

## Testing
`tests/test_biofield_layers.py` (+ keep `test_biofield_trial` / `test_biofield_cart` green):
- **Store:** `set_layers` + `_row` round-trip; `set_interpretation`/`set_remedies`/`set_layers` persist on an `first_approved=1` row (re-editability); `list_approved` returns approved rows.
- **Ingest:** a push with `layers` stores them + derives the flat `remedies` (ordered layer remedies); a non-catalog layer remedy is dropped from its layer + recorded in `dropped` + the layer keeps title/summary; a catalog remedy gets the canonical-meaning override; a `remedies`-only push is wrapped into titleless layers.
- **Payload:** layer titles+summaries are emitted for a NON-paid, NON-approved member (always shown); the per-layer remedy is withheld (`remedy_blurred`) for a non-visible layer and present for the visible one (paid -> all; free top-approved -> top); `_biofield_visible_slugs` returns the visible layer slugs; a withheld remedy's name/slug never appears in the non-paid payload (anti-bypass).
- **Console:** the edit action accepts `layers`, re-derives `remedies`, promotes "remember" meanings; an approved reveal is editable; `list_approved` powers the approved section.
- **Member serve:** the reveal page ships the layer-render markers; XSS-safe.
- LLM/Stripe/GHL mocked; tmp `LOG_DB`; no emoji; no em dashes. Front-end visual pass manual.

## Notes
- Console-facing change is live on merge (the console is gated). The member-facing layer render only appears for reveals that have approved drafts (still dark until the matcher pushes layered drafts and Glen approves) - so no regression for current (zero approved-in-prod) reveals.
- The matcher must start pushing `layers` to populate the new structure; until it does, pushes are remedies-only and wrap into titleless layers (Glen can still name them).
- Ties to [[reference_biofield_causal_chain_skill]] (the layer/causal-chain model) and [[project_e4l_scan_ingestion]].
