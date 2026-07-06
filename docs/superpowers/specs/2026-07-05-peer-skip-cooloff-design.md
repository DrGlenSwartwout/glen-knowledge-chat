# Community — peer matching: skip cool-off (pool-dry fallback) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (pool-dry fallback only — a skipped person resurfaces ONLY when a proposal would otherwise be empty; 30-day cool-off, env `PEER_SKIP_COOLOFF_DAYS`; skips only — a non-mutual connect never resurfaces).
**Repo:** deploy-chat

**Relates to / reuses:**
- Peer matching v1-v3: `dashboard/peer_connect.py` (`eligible_candidates`, `interest_kind`, `record_interest`, `member_ref`, exclusion helpers), `app.py` `peer_proposal` (the co-ranked blend `_peer_blended_candidate` over `eligible_candidates`), `_is_paid_member`, `_peer_ident_paid`.
- [[project_pb_to_illtowell_evox]], the peer-matching privacy line.

## Context and boundary

Today a "Not now" writes a `peer_interest` row `kind='skip'`, and `eligible_candidates` excludes that candidate **permanently** (`interest_kind(me,n) is not None` lumps connect + skip). In a small paid pool a member can exhaust fresh candidates and then see an empty "no one right now" card, even though people they briefly passed on months ago are still in the pool.

This slice adds a **pool-dry fallback**: only when the normal proposal would come back empty, the system re-offers the best person the member skipped **more than 30 days ago**. A skip newer than the cool-off never resurfaces; a non-mutual connect never resurfaces (the member's standing "yes" already holds — if the other side ever connects, it becomes mutual regardless).

**Structural guarantee (load-bearing):** a stale skip can NEVER outrank or displace a fresh candidate, because resurfacing is a SECOND blend pass that runs only when the first (fresh) pass yields nothing. There is no single ranking that mixes fresh and stale-skipped candidates.

**Privacy (unchanged):** proposals stay anonymous (`{member_ref, shared_topics, semantic}`); every exclusion (matched, connect-toward, skipped-me, person-blocked, non-paid) still applies in BOTH passes — the fallback pass relaxes exactly one thing: a stale skip. A resurfaced candidate is still just an opted-in `member_ref`; the member is never told it is someone they passed before.

## Scope

**A second (fallback) blend pass in `/api/peer/proposal` over a pool that re-admits only stale (>cool-off) skips, run only when the fresh pass returns nothing.** A store change to `eligible_candidates` (an `include_stale_skips` mode + a skip-age helper), the route's two-pass wiring, one config constant, and tests. No frontend change (a resurfaced candidate renders identically to any other proposal).

**Non-goals / deferred:** pure time-based expiry (rejected — re-shows skips while fresh people exist); a "you passed on this person before" hint (breaks anonymity); a per-skip custom cool-off; resurfacing non-mutual connects; a UI to review/undo skips.

## Components

### 1. Store (`dashboard/peer_connect.py`)

- Add `_my_interest(cx, from_email, to_email) -> (kind, created_at)|(None, None)` — the caller's directional interest row (kind + timestamp), or `(None, None)`.
- Extend `eligible_candidates(cx, me, is_paid=None, *, include_stale_skips=False, cutoff_iso=None) -> [email]`:
  - Keep, unchanged and always applied: exclude self, non-paid (`is_paid`), existing `peer_matches`, `interest_kind(n,me)=='skip'` (they passed on me), person-blocked either direction.
  - Replace the single `interest_kind(me,n) is not None` exclusion with a kind-aware rule using `_my_interest(cx, me, n)`:
    - my `connect` → exclude always (standing yes; never re-propose).
    - my `skip` → exclude UNLESS `include_stale_skips` AND `cutoff_iso` AND the skip's `created_at < cutoff_iso` (a stale skip is re-admitted only in the fallback pass; a fresh skip stays excluded).
    - no interest → include.
  - Default (`include_stale_skips=False`) reproduces today's behavior exactly (all skips excluded) — the fresh pass is unchanged.

### 2. Two-pass proposal (`app.py`)

- `PEER_SKIP_COOLOFF_DAYS = int(os.environ.get("PEER_SKIP_COOLOFF_DAYS", "30"))`.
- In `peer_proposal`, after auth/eligibility, replace the single blend call with:
  ```
  pool = eligible_candidates(cx, email, is_paid=_is_paid_member)          # fresh pass
  cand = _peer_blended_candidate(cx, email, pool)
  if cand is None:                                                         # pool dry -> fallback
      cutoff = (datetime.now(timezone.utc) - timedelta(days=PEER_SKIP_COOLOFF_DAYS)).isoformat()
      fb = eligible_candidates(cx, email, is_paid=_is_paid_member,
                               include_stale_skips=True, cutoff_iso=cutoff)
      cand = _peer_blended_candidate(cx, email, fb)
  return {"candidate": cand}
  ```
  The fallback pool is fresh ∪ stale-skips; since the fresh pass already yielded nothing, the fallback effectively surfaces the best stale-skipped candidate (still subject to the blend's qualify rule + 0.80 cosine floor). `peer_interest`/`peer_state`/peer-thread routes unchanged. `cutoff_iso` and `_now()` are both full ISO-8601 UTC from the same clock, so the `created_at < cutoff_iso` string compare is valid.

### 3. Config

- New `PEER_SKIP_COOLOFF_DAYS` (default 30). No other new env.

## Data flow

1. `/api/peer/proposal` runs the blend over the fresh eligible pool (skips excluded, as today).
2. If a candidate is found → return it (a stale skip never competes).
3. If none, recompute the pool with stale (>30-day) skips re-admitted and blend again → return the best, or null.
4. Connect/skip/reveal/thread downstream unchanged; re-offering a resurfaced candidate and connecting works identically (a fresh `connect` interest upserts over the old `skip` row via `record_interest`).

## Error handling

- The fallback pass reuses the same best-effort `_peer_blended_candidate` (try/except → None); no new failure surface.
- If the cutoff computation or fallback yields nothing, `candidate` is null (an empty card is still correct when there is genuinely no one, fresh or stale).
- The stale-skip test is a string compare on same-format ISO timestamps; a malformed/absent `created_at` fails the `< cutoff` test and keeps the skip excluded (fail-safe — never over-resurfaces).

## Testing

- **Store:** `eligible_candidates` default (no flag) still excludes ALL skips (fresh pass unchanged) and connects; with `include_stale_skips=True` + a cutoff, a skip older than the cutoff is INCLUDED while a skip newer than the cutoff stays excluded; a `connect` is excluded in BOTH modes; matched/skipped-me/person-blocked/non-paid stay excluded in BOTH modes. `_my_interest` returns (kind, created_at) or (None, None).
- **Route (mock `_member_interest_vec` + seed signals/interests):** a fresh skip is not proposed even when the card is otherwise empty; a stale (>cool-off) skip is NOT proposed while a fresh qualifying candidate exists (pool-dry guarantee — assert the fresh candidate is returned, not the stale skip), and IS proposed when no fresh candidate exists; a non-mutual connect is never proposed; a resurfaced candidate is anonymous (no email/name). Seed skip `created_at` directly (old vs recent) to control staleness deterministically.
- **Privacy regression:** the fallback proposal payload carries no email/name; every non-stale-skip exclusion still holds in the fallback pass.
- **Go-live:** a member skips everyone (card goes empty), waits past the cool-off (simulate via a back-dated skip) — the earliest-skipped person resurfaces; a member with a fresh skip + a fresh candidate still sees only the fresh candidate.

## Deferred (later)

- Pure time-based expiry; a per-skip cool-off; a "review your passes" UI; resurfacing non-mutual connects; surfacing a subtle "you saw this person before" cue (privacy-guarded).
