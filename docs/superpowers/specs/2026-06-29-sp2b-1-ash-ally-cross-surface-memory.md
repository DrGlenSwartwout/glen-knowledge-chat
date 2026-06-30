# SP2b-1 — ASH ally cross-surface memory (helper + streaming surfaces)

**Date:** 2026-06-29
**Status:** Design / spec (pending review)
**Repo:** deploy-chat (illtowell.com)
**Parent:** the ASH voice-first health ally (foundation `2026-06-27-ash-health-ally-foundation.md`).
SP1 (voice doorway) + SP2a (`dashboard/ash_map.py`, the email-keyed coverage map + memory) shipped.

## Context

SP2b makes the ally **follow the person across every chat surface**: wherever a chat knows who it's
talking about, it reads that person's `ash_map` memory into its prompt and updates it after the reply.
The memory is the constant; the *presentation* (written replies now, voice/video "Glendalf" later) is a
separately-staged, cost-gated layer. This spec is **SP2b-1**, the first slice:

- **SP2b-1 (THIS spec):** the shared, fail-open, flag-gated helper `dashboard/ash_ally.py`, wired into
  the **4 streaming (SSE) chat surfaces**. Written replies, every tier. The foundation the rest reuse.
- **SP2b-2 (later):** the 3 `scoped_reply` surfaces (dispensary client, invoice) + the shared
  `scoped_reply` taking a `subject_email`.
- **SP2b-3 (later):** the practitioner surface with a **client name/email search** → the in-focus
  client becomes the subject (the practitioner builds a persistent picture of each client).
- **SP2b-4 (later):** `/member/scan-analysis/chat` (single-turn).
- **Later, separate:** the Glendalf voice/video presentation as the premium, path-gated experience
  (opt-in → $1 → member); dropping the path costs the voice/video, never the memory.

Out of scope for ALL of SP2b: the owner's-manual surfacing + always-reach trio (that's SP2c).

## The subject-email abstraction

The ally memory is keyed by the **subject** — the health-seeker being discussed — which is usually the
speaker, but is sometimes a contextual person (a portal client, an order's buyer, or — in SP2b-3 — a
searched client). Each surface computes its own `subject_email`; the helper is identical everywhere.
Where there is no subject email, the layer is **inert** (no overlay, no record, no extra cost). This
keeps the high-volume anonymous/pre-opt-in traffic completely untouched.

## Module — `dashboard/ash_ally.py`

New module. Imports `ash_map` (same package) + stdlib only; **no `app`/Flask import** (so it is
pure-module testable, like `ash_map`/`journal_store`). The app's `LOG_DB` path and `_db_lock` are
**injected** by the caller, never imported (avoids a circular import and keeps tests hermetic).

- `ENABLED() -> bool` — reads `os.environ.get("ASH_ALLY_ENABLED")` truthy (`"1"/"true"/"yes"`, case-
  insensitive). The single feature gate. Read at call time (not import) so the flag can flip without a
  code deploy.

- `ally_overlay(cx, subject_email) -> str` — the read side, for splicing into a surface's system
  prompt. Returns `""` when: `ENABLED()` is false, `subject_email` is empty, the memory is all-untouched
  (first contact, nothing to recall), **or anything raises** (fail-open — wrapped in try/except → `""`).
  Otherwise returns `ash_map.context_block(ash_map.get(cx, subject_email))` wrapped in the framing block
  below. `cx` is a caller-supplied sqlite3 connection (read-only use here).

- `record_turn(db_path, lock, subject_email, user_text, ally_text="") -> None` — the write side, called
  **after** the reply. No-ops when `ENABLED()` is false or `subject_email` is empty. **Fail-open:** the
  whole body is wrapped so it can never raise into the caller. **Lock discipline (required):** it must
  NOT hold `lock` across the Haiku extract. Sequence:
  1. *(locked)* `with lock, sqlite3.connect(db_path) as cx: memory = ash_map.get(cx, subject_email)`
  2. *(unlocked)* `extracted = ash_map._haiku_extract(memory, user_text, ally_text)` — the slow network
     call, no lock held.
  3. *(locked)* `with lock, sqlite3.connect(db_path) as cx:` re-read the memory, `merged =
     ash_map.merge_turn(fresh, extracted)`, persist it. Re-reading under the lock means two concurrent
     same-email turns converge correctly (`merge_turn` is forward-only).
  `record_turn` is designed to be dispatched **off the request path** (see wiring) so its latency never
  affects the user; the caller owns the background dispatch, the helper just does the work safely.

  To persist the merge under the lock without re-running the LLM, `ash_map` needs a public seam.
  **Add to `ash_map.py`:** `persist_extract(cx, email, extracted) -> dict` = `get` → `merge_turn` →
  `_upsert` → return merged (the locked tail of `update_from_turn`). Refactor `update_from_turn` to call
  it (`get` → `_haiku_extract` → `persist_extract`), so existing behavior/tests are unchanged. Then
  step 3 above is `ash_map.persist_extract(cx, subject_email, extracted)`.

**The overlay framing block** (what `ally_overlay` wraps `context_block` in) — tells the model to USE
the memory naturally, not recite it:

```
━━━ WHAT YOU ALREADY KNOW ABOUT THIS PERSON ━━━
<context_block output>

Greet them with continuity and pick up the threads they've opened. Don't re-ask what they've already
shared. Never read this back as a list, never mention "dimensions", a "map", or that you track anything
— just let it make you feel like someone who remembers them.
```

(Exact wording is a single module constant; the plan may refine it.)

## Wiring the 4 SSE surfaces

Each surface gets the **same two touches**. The subject-email rule differs per surface.

| Surface | app.py | subject_email | overlay injection point | record_turn fire point |
|---|---|---|---|---|
| `/chat` | 3197 | `email` (already resolved: body, or auth-cookie override) | prepend to the system/context where `_member_context_for_email` is applied (~3349) | after the `done` SSE event, beside `log_query` |
| `/begin/match/chat` | 3716 | `email`, **only if `for_whom != "someone-else"`** (mirrors the existing member-context guard); else `""` | same precedent as `/chat` | after `done`, beside `log_query` |
| `/begin/concierge/chat` | 7136 | `email` (the gated member email) | into its system-prompt assembly | after `done` (this surface already makes a post-stream extract call — fire beside it) |
| `/api/portal/<token>/chat` | 12091 | `portal.get("email")` (token→portal record) | into `dashboard/portal_concierge.system_prompt` assembly / the context prefix | after `done` (beside its post-stream suggestion call) |

**Injection:** prepend `ally_overlay(cx, subject_email)` to the surface's existing system prompt or
context string (an empty string is a no-op, so unconditional prepend is safe). Place it adjacent to any
existing `_member_context_for_email` overlay — same role, same spot.

**Fire:** after the surface yields its terminal `done` event, dispatch
`ash_ally.record_turn(LOG_DB, _db_lock, subject_email, query, answer)` in the background so it never
delays connection teardown or the user. Use the app's existing background-dispatch idiom; under gevent a
spawned worker is a cooperative greenlet, and `requests`/sqlite calls inside it yield correctly. If the
app has no existing helper, a `threading.Thread(daemon=True, target=...)` is acceptable (gevent patches
it to a greenlet). The dispatch wrapper itself is try/except so a spawn failure can't break the stream.

**Anonymous / first turn:** `subject_email == ""` → overlay `""`, `record_turn` no-ops. No behavior
change, no extra Haiku call for anonymous traffic.

## Reuse / consistency

- Mirrors the **read-only overlay** pattern of `_member_context_for_email` (app.py:8964) — the ally
  overlay sits beside it.
- Mirrors the **post-stream second-Haiku-call** pattern these surfaces already use (next-question /
  concierge-suggestion) — the extract is a third such call, but fire-and-forget and background.
- Uses `ash_map`'s existing public surface (`get`, `context_block`, `merge_turn`) plus the new
  `persist_extract` seam; `_haiku_extract` is imported within the package.
- `LOG_DB` + `_db_lock` idiom (`with _db_lock, sqlite3.connect(LOG_DB) as cx:`) is the app's standard;
  `record_turn` receives both rather than importing them.

## Testing — `tests/test_ash_ally.py` (pure-module, plain pytest, no app import)

The helper is fully testable without Flask by injecting a temp db path + a real `threading.Lock` and
monkeypatching `ash_map._haiku_extract` (the same seam SP2a's tests use).

- `ENABLED()` true/false by env.
- `ally_overlay`: returns `""` when flag off / empty email / all-untouched memory / when `ash_map.get`
  raises (monkeypatch it to raise → still `""`, no exception). Returns the framed block (containing the
  `context_block` text and the framing header) for a populated memory.
- `record_turn`: no-ops on flag off / empty email; on a real temp db with `_haiku_extract` monkeypatched
  to a fixed output, the memory is persisted (a follow-up `ash_map.get` shows the merged state); a
  second call accumulates; an exception from `_haiku_extract` or the db is swallowed (no raise) — assert
  `record_turn(...)` returns `None` and does not propagate.
- **Lock discipline test:** monkeypatch `_haiku_extract` to assert the lock is NOT held while it runs
  (e.g. it tries `lock.acquire(blocking=False)` and records the result) → confirm the extract ran
  unlocked, while `get`/persist ran locked. This pins the core performance requirement.

For `ash_map.persist_extract`: add a focused test (get → merge → upsert round-trip equals
`merge_turn(get(...), extracted)` then persisted) and confirm `update_from_turn` still behaves
identically (existing SP2a tests must stay green).

## Verification (go-live)

- `python3 -m pytest tests/test_ash_ally.py tests/test_ash_map.py -v` (pure-module; no doppler/network).
- Flag dark by default. To enable: set `ASH_ALLY_ENABLED=1` in Render, then **render-verify each of the
  4 surfaces** in a headless browser — load the surface as an identified user with prior memory, confirm
  the reply reflects continuity (no re-asking), assert zero console errors, and confirm a follow-up turn
  updated the stored map. Verify anonymous use of each surface is unchanged (no overlay, no error).
  (Per the render-verify lesson: actually render + assert behavior, never just confirm the code path.)

## Out of scope (SP2b-1)

- The 3 `scoped_reply` surfaces and the shared `scoped_reply` subject param (SP2b-2).
- The practitioner client-search + in-focus subject (SP2b-3).
- `/member/scan-analysis/chat` (SP2b-4).
- Voice/video "Glendalf" presentation and tier-gating (separate, later).
- Per-email (vs global) lock granularity — the lock-splitting here keeps hold times to fast sqlite ops;
  finer locking is a future optimization only if contention shows up.
- Backfilling memory from historical `query_log` (a later option).

## Go-live checklist (ordered)

1. Merge with the flag dark (`ASH_ALLY_ENABLED` unset). The layer is inert on prod — `ally_overlay` returns `""`, `record_turn` no-ops — so nothing changes for any user and no extra Haiku calls are made.
2. Set `ASH_ALLY_ENABLED=1` in Render; redeploy/restart.
3. Render-verify each of the 4 SSE surfaces in a headless browser as an identified user who already has `ash_map` memory:
   - `/chat`, `/begin/match/chat` (with `for_whom=me`), `/begin/concierge/chat` (as a gated member), a `/portal/<token>` page.
   - For each: the reply reflects continuity (no re-asking known facts), zero console errors; send a follow-up turn, then confirm via `ash_map.get` / a DB read that the stored map advanced.
4. Confirm anonymous use of each surface is unchanged (no overlay, no errors, no extra Haiku call).
5. Rollback if needed: unset `ASH_ALLY_ENABLED` — instant, no code change.
