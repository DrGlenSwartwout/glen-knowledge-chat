# Biofield Intake — Balancing Loop B3a: Profile / Tag Mining

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop (`2026-06-25-biofield-intake-balancing-loop-NOTES.md`). Builds on B1 (stress engine, #295) and B2 (two-phase voice, #297). First half of B3 (the "health/wellness tags + challenges/goals → stresses, woven into reports" requirement); B3b (recent-communication mining) and B4 (minimal-remedy set-cover) remain parked.

## Problem

Glen's requirement: the reveal + intake process should "pull in any relevant health/wellness tags as stresses for balancing and reporting," and itemize the client's stated symptoms / issues / challenges / goals. B1 seeds stresses from the E4L scan and B2 captures them by voice, but the client's consolidated profile — the tags, conditions, challenges, goals, terrain concerns, body systems, and clinical notes the practice already maintains — never reaches the intake stress list.

That profile is already exposed by the prod People hub `/api/people?q=<email>` endpoint (the same call `e4l-reveal-push.py:fetch_history` makes), returning `tags, conditions, challenges, goals, terrain_concerns, body_systems, notes, medications, history`. B3a mines it into the master stress list and weaves it into the report. (Real recent-communication mining — chat/email/GHL/PB messages — needs new prod read endpoints/connectors and is deferred to B3b.)

## Goal

For an intake session's client, pull the consolidated profile from `/api/people`, turn its health content into stresses (merged by normalized label so nothing duplicates a scan/voice/earlier-tag stress), and feed that profile context into the generated narrative so the report addresses the client's own stated concerns.

## Non-goals (deferred)

- Mining actual recent communications — chat (`query_log`), inquiries, ScoreApp quiz raw_json, email, GHL/PB messages (B3b).
- Minimal-remedy set-cover (B4).
- Changing how scan/voice stresses work, or the Phase-2 balancing flow.

## Design

### Profile fetch (injectable, HTTP)

New `fetch_profile(email) -> dict` in `biofield_local_app.py`, mirroring `e4l-reveal-push.py:fetch_history`:
- GETs `{PUBLIC_BASE_URL or https://illtowell.com}/api/people?q=<email>&key=<CONSOLE_SECRET>` with the `X-Console-Key` header.
- Returns the person dict whose email matches (case-insensitive), or `{}` on no match / any error (best-effort, never raises).
- Injected into `create_app(..., fetch_profile=None)` like `scan_lookup`/`client_search`, defaulting to the real HTTP implementation, so tests pass a stub and run offline. PHI: the profile is fetched live and used in-process; nothing new is persisted beyond the derived stress labels.

### Mining → stress labels

New pure module `dashboard/biofield_profile.py`:

`mine_profile_stresses(profile, extract) -> list[str]`
- **Discrete fields** — `tags`, `conditions`, `terrain_concerns`, `body_systems`: each may be a list or a comma/semicolon-separated string; split, strip, and take each item as a stress label directly.
- **Free-text fields** — `challenges`, `goals`, `notes`: concatenate and pass to `extract(text)` — the injected B2 parser `interpret_stresses` (bound to the app's completer) — to pull discrete stress labels from prose.
- Combine, dedupe case-insensitively (keep first casing), return the label list. Empty/missing profile → `[]`.
- `extract` is injected so the module is pure and unit-testable offline (no LLM call in tests).

Rationale for splitting discrete vs free-text: tags/conditions/etc. are already atomic labels (turning them into a sentence and re-extracting would lose fidelity), while challenges/goals/notes are prose that needs the LLM to itemize.

### Store — generalized merge-insert

Refactor B2's `add_voice_stress` so the merge-insert is shared:
- New `add_stress(cx, tid, label, *, source='voice', balance='required') -> bool` — the current `add_voice_stress` body, parameterized on `source`/`balance`. Stores `code=_norm(label)`, merges (no insert, returns False) when the normalized label already exists for the test in ANY source.
- `add_voice_stress(cx, tid, label)` becomes `return add_stress(cx, tid, label, source='voice', balance='required')` (behavior identical; B2 tests stay green).
- B3a inserts with `source='tag'`, `balance='required'`.

### Route + always-on hook

- `POST /author/<test_id>/mine-profile`: resolve the client email from `_report_for`; if none → `{"added": 0, "error": "No client selected yet"}`. Call `fetch_profile(email)`; if empty → `{"added": 0}`. Run `mine_profile_stresses(profile, extract)` where `extract = lambda t: interpret_stresses(t, interpret_complete)`; add each via `add_stress(..., source='tag')`; return `{"added": <count newly inserted>}`. The whole profile path is wrapped best-effort (try/except → `{"added": 0, "error": ...}`) so a network/profile failure never blocks intake.
- **Always-on:** the existing `_seed_stresses(cx, test_id)` header-save hook also calls profile mining best-effort (guarded so a failure is swallowed). Idempotent via the normalized-label merge, so re-runs add nothing new. Profile mining runs even when there is no fresh scan (a client may have profile data but no recent scan) — it does not depend on `scan_context`.

### Report weaving

Extend the narrative generator to optionally carry profile context, mirroring the existing scan pattern (`_system_with_scan`, which keeps the no-context prompt byte-identical):
- `generate_narrative(report, notes, complete, scan=None, profile=None)` and `build_narrative_prompt(report, notes, scan=None, profile=None)`.
- A `_PROFILE_GUIDANCE` block + the client's stated challenges/goals/conditions are appended to the system prompt ONLY when `profile` carries content, instructing the narrative to acknowledge the client's own stated concerns and goals in plain language. No profile → prompt unchanged (back-compat; existing narrative tests stay green).
- The `narrative_generate` route fetches the profile (best-effort) and passes it as `profile=` alongside the existing `scan=ctx`.

### Components / files

- `biofield_local_app.py` — `fetch_profile` (injectable) + `_mine_profile(cx, test_id)` helper; `POST /author/<id>/mine-profile`; call `_mine_profile` from `_seed_stresses`; pass `profile=` in `narrative_generate`.
- `dashboard/biofield_profile.py` (new) — `mine_profile_stresses(profile, extract)`.
- `dashboard/biofield_stress.py` — `add_stress(...)`; `add_voice_stress` delegates.
- `dashboard/biofield_narrative.py` — `profile` param + `_PROFILE_GUIDANCE` via a `_system_with_context` helper (keep `_system_with_scan` behavior intact).
- `dashboard/biofield_report_html.py` — "Mine profile → stresses" button + `mineProfile()` JS (POST → `loadStress()`).

### Testing (TDD, offline)

`fetch_profile`, `extract`, and `complete` injected; sqlite tmp DBs.
1. **mine_profile_stresses** — discrete fields (list AND comma-string forms) become labels; free-text fields go through the stubbed `extract`; combined + deduped; empty profile → `[]`.
2. **add_stress / add_voice_stress** — `add_stress(source='tag')` inserts a tag stress (required); cross-source normalized-label merge holds; `add_voice_stress` still behaves exactly as B2 (its tests unchanged).
3. **mine-profile route** — stubbed `fetch_profile` + `interpret_complete` → tag stresses added and visible in `/stresses`; no-email → error; empty profile → `{"added":0}`; profile failure → best-effort error, no crash; merge dedups against an existing scan stress of the same name.
4. **narrative weaving** — `build_narrative_prompt(..., profile=<content>)` includes the guidance + stated concerns; `profile=None` → byte-identical to the pre-B3a prompt (existing narrative tests green).
5. **UI** — the "Mine profile → stresses" button + `mineProfile()` render and post to the route.

## Rollout

Local-only tool on Glen's Mac. No feature flag, no prod deploy. The `/api/people` endpoint already exists in prod. Additive: one new module, one generalized store function (B2-compatible), one narrative param (back-compatible), one new route + a best-effort hook call, one button. Voice/scan stresses, Phase 2, and the no-profile narrative prompt are unchanged.
