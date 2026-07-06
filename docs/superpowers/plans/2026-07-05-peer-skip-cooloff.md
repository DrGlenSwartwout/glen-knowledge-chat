# Peer Skip Cool-off (pool-dry fallback) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a peer proposal would come back empty, re-offer the best candidate the member skipped more than 30 days ago (a second blend pass over a pool that re-admits only stale skips) — so a small pool doesn't dead-end, without ever re-showing a skipped person while fresh candidates exist.

**Architecture:** A store change adds an `include_stale_skips` mode to `peer_connect.eligible_candidates` (plus a skip-age helper), splitting the current "any prior interest excludes" rule into connect-always-excluded vs skip-excluded-unless-stale. `app.py` `peer_proposal` runs a second blend pass over the stale-skip pool only when the first (fresh) pass returns nothing. No frontend change.

**Tech Stack:** Python 3 / Flask, sqlite, reuses the peer-matching store + the co-ranked blend `_peer_blended_candidate`.

## Global Constraints

- **Pool-dry guarantee (structural):** the fallback is a SEPARATE second blend pass, run ONLY when the fresh pass returns `None`. A stale skip can never outrank or displace a fresh (qualifying) candidate. Never merge fresh + stale into one ranking.
- **Skips only:** a stale skip (>cool-off) can resurface; a `connect` (non-mutual, standing yes) NEVER resurfaces; a fresh skip (<cool-off) never resurfaces.
- **All other exclusions hold in BOTH passes:** self, non-paid (`is_paid`), existing `peer_matches`, `interest_kind(n,me)=='skip'` (they passed on me), person-blocked either direction. The fallback pass relaxes exactly one thing: my own stale skip.
- **Default behavior unchanged:** `eligible_candidates(cx, me, is_paid)` with no new kwargs must reproduce today's output exactly (all skips + connects excluded) — the fresh pass is byte-identical.
- **Privacy:** proposals stay anonymous; the fallback payload carries no email/name; the member is never told a candidate is one they skipped.
- **Fail-safe:** a malformed/absent skip `created_at` fails the `< cutoff` test → stays excluded (never over-resurfaces). `cutoff_iso` and `_now()` are both full ISO-8601 UTC from the same clock, so the string compare is valid.
- New `PEER_SKIP_COOLOFF_DAYS` (default 30). No other new env. DRY, YAGNI, TDD.

**Repo facts (verified anchors):**
- `dashboard/peer_connect.py`: current `eligible_candidates(cx, me, is_paid=None)` excludes self / non-paid / `_pair_has_match` / `interest_kind(cx, me, n) is not None` / `interest_kind(cx, n, me)=='skip'` / `_person_blocked` (both). `interest_kind(cx, from, to) -> kind|None`. `peer_interest(from_email, to_email, kind, created_at, UNIQUE(from_email,to_email))`. `record_interest`, `opted_in_members`, `_pair_has_match`, `_person_blocked`, `_lc`.
- `app.py`: `peer_proposal` currently does one blend: `pool = _pc.eligible_candidates(cx, email, is_paid=_is_paid_member); return {"candidate": _peer_blended_candidate(cx, email, pool)}`. `PEER_BLEND_WEIGHT`/`PEER_SEMANTIC_MIN_COSINE` defined near the peer helpers — add `PEER_SKIP_COOLOFF_DAYS` beside them. `datetime, timezone, timedelta` already imported at app.py:24 (module level) — no local import needed. `_peer_blended_candidate`, `_is_paid_member`, `_peer_ident_paid`.

**Testing note:** store test is pure (`python3 -m pytest <path> -q`). Route test imports app → override DATA_DIR, run the file ALONE in a fresh DATA_DIR; namespace test emails/topics per test (shared `chat_log.db` per pytest process).

---

### Task 1: Skip cool-off store + two-pass proposal

**Files:**
- Modify: `dashboard/peer_connect.py` (`_my_interest`, extend `eligible_candidates`)
- Modify: `app.py` (`PEER_SKIP_COOLOFF_DAYS`, two-pass `peer_proposal`)
- Test: `tests/test_peer_skip_cooloff_store.py`, `tests/test_peer_skip_cooloff_api.py`

**Interfaces:**
- Produces (store): `_my_interest(cx, from_email, to_email) -> (kind, created_at)|(None, None)`; `eligible_candidates(cx, me, is_paid=None, *, include_stale_skips=False, cutoff_iso=None)`.
- Produces (app): `PEER_SKIP_COOLOFF_DAYS`; two-pass `peer_proposal`.

- [ ] **Step 1: Write the failing store test**

```python
# tests/test_peer_skip_cooloff_store.py
import sqlite3
from dashboard import peer_connect as _pc, community_signals as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
    return cx


def _skip(cx, frm, to, created_at):
    _pc.record_interest(cx, frm, to, "skip")
    cx.execute("UPDATE peer_interest SET created_at=? WHERE from_email=? AND to_email=?",
               (created_at, frm, to)); cx.commit()


def test_default_excludes_all_skips_and_connects():
    cx = _cx()
    for e in ("me@x.com", "sk@x.com", "cn@x.com"):
        _pc.set_optin(cx, e, True)
    _skip(cx, "me@x.com", "sk@x.com", "2020-01-01T00:00:00+00:00")   # even an OLD skip
    _pc.record_interest(cx, "me@x.com", "cn@x.com", "connect")
    assert set(_pc.eligible_candidates(cx, "me@x.com")) == set()      # fresh pass: both excluded


def test_stale_skip_readmitted_only_in_fallback():
    cx = _cx()
    for e in ("me@x.com", "old@x.com", "new@x.com"):
        _pc.set_optin(cx, e, True)
    _skip(cx, "me@x.com", "old@x.com", "2020-01-01T00:00:00+00:00")   # stale
    _skip(cx, "me@x.com", "new@x.com", "2099-01-01T00:00:00+00:00")   # fresh (future = not stale)
    cutoff = "2026-01-01T00:00:00+00:00"
    fresh = set(_pc.eligible_candidates(cx, "me@x.com"))
    fb = set(_pc.eligible_candidates(cx, "me@x.com", include_stale_skips=True, cutoff_iso=cutoff))
    assert fresh == set()                       # both skips excluded in the fresh pass
    assert fb == {"old@x.com"}                  # only the stale skip re-admitted


def test_connect_never_readmitted_even_in_fallback():
    cx = _cx()
    for e in ("me@x.com", "cn@x.com"):
        _pc.set_optin(cx, e, True)
    _pc.record_interest(cx, "me@x.com", "cn@x.com", "connect")
    cx.execute("UPDATE peer_interest SET created_at='2020-01-01T00:00:00+00:00'"); cx.commit()
    assert set(_pc.eligible_candidates(cx, "me@x.com", include_stale_skips=True,
                                       cutoff_iso="2026-01-01T00:00:00+00:00")) == set()


def test_other_exclusions_hold_in_fallback():
    cx = _cx()
    for e in ("me@x.com", "sk@x.com", "theyskip@x.com"):
        _pc.set_optin(cx, e, True)
    _skip(cx, "me@x.com", "sk@x.com", "2020-01-01T00:00:00+00:00")    # stale skip by me
    _skip(cx, "theyskip@x.com", "me@x.com", "2020-01-01T00:00:00+00:00")  # they skipped me (stale)
    fb = set(_pc.eligible_candidates(cx, "me@x.com", include_stale_skips=True,
                                     cutoff_iso="2026-01-01T00:00:00+00:00", is_paid=lambda e: True))
    assert fb == {"sk@x.com"}      # my stale skip re-admitted; a stale skip-of-me stays excluded


def test_my_interest_shape():
    cx = _cx()
    assert _pc._my_interest(cx, "a@x.com", "b@x.com") == (None, None)
    _pc.record_interest(cx, "a@x.com", "b@x.com", "skip")
    kind, at = _pc._my_interest(cx, "a@x.com", "b@x.com")
    assert kind == "skip" and at
```

- [ ] **Step 2: Run the store test to verify it fails**

Run: `python3 -m pytest tests/test_peer_skip_cooloff_store.py -q`
Expected: FAIL — `_my_interest` missing, `include_stale_skips` kwarg unknown.

- [ ] **Step 3: Store implementation**

In `dashboard/peer_connect.py`, add `_my_interest` (near `interest_kind`):

```python
def _my_interest(cx, from_email, to_email):
    """(kind, created_at) of the caller's directional interest toward a member, or
    (None, None)."""
    row = cx.execute("SELECT kind, created_at FROM peer_interest WHERE from_email=? "
                     "AND to_email=?", (_lc(from_email), _lc(to_email))).fetchone()
    return (row["kind"], row["created_at"]) if row else (None, None)
```

Replace `eligible_candidates` with the kind-aware version:

```python
def eligible_candidates(cx, me, is_paid=None, *, include_stale_skips=False, cutoff_iso=None):
    """Opted-in members that pass every exclusion (self, non-paid, existing match,
    they-skipped-me, person-blocked) and my own prior interest. My `connect` (standing
    yes) is always excluded. My `skip` is excluded UNLESS include_stale_skips and the
    skip is older than cutoff_iso (the pool-dry fallback). The default (no kwargs)
    reproduces the fresh pass exactly."""
    me = _lc(me)
    out = []
    for n in opted_in_members(cx):
        if n == me:
            continue
        if is_paid is not None and not is_paid(n):
            continue
        if _pair_has_match(cx, me, n):
            continue
        if interest_kind(cx, n, me) == "skip":
            continue
        if _person_blocked(cx, me, n) or _person_blocked(cx, n, me):
            continue
        mine_kind, mine_at = _my_interest(cx, me, n)
        if mine_kind == "connect":
            continue                                     # standing yes; never re-propose
        if mine_kind == "skip":
            stale = bool(include_stale_skips and cutoff_iso and mine_at and mine_at < cutoff_iso)
            if not stale:
                continue                                 # fresh skip (or fallback off) -> excluded
        out.append(n)
    return out
```

- [ ] **Step 4: Run the store test to verify it passes**

Run: `python3 -m pytest tests/test_peer_skip_cooloff_store.py -q` → 5 passed.
Also run `python3 -m pytest tests/test_peer_eligible_pool.py tests/test_peer_connect_store.py -q` → still green (default behavior unchanged).

- [ ] **Step 5: Write the failing route test**

```python
# tests/test_peer_skip_cooloff_api.py
import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import peer_connect as _pc, community_signals as _cs, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _member(email, *topics):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _pc.init_peer_tables(cx); _cs.init_signal_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, email, email.split("@")[0].title())
        for t in topics:
            _cs.set_signal(cx, email, "topic", t, "like")
        tok = _ev.ensure_portal_token(cx, email, email.split("@")[0]); cx.commit()
    return tok


def _optin(*emails):
    for e in emails:
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, e, True); cx.commit()


def _skip(frm, to, created_at):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        _pc.record_interest(cx, frm, to, "skip")
        cx.execute("UPDATE peer_interest SET created_at=? WHERE from_email=? AND to_email=?",
                   (created_at, frm, to)); cx.commit()


def _proposal(tok, vecs):
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=lambda cx, e, l: vecs.get(e, [])):
        return _client().get(f"/api/peer/proposal?token={tok}").get_json()


def test_fresh_skip_not_proposed_even_when_empty():
    me = _member("sc1_me@x.com", "liver"); _member("sc1_sk@x.com", "liver")
    _optin("sc1_me@x.com", "sc1_sk@x.com")
    _skip("sc1_me@x.com", "sc1_sk@x.com", "2099-01-01T00:00:00+00:00")   # fresh (future)
    d = _proposal(me, {"sc1_me@x.com": [1.0, 0.0], "sc1_sk@x.com": [1.0, 0.0]})
    assert d["candidate"] is None                       # fresh skip stays excluded, card empty


def test_stale_skip_not_proposed_while_fresh_candidate_exists():
    me = _member("sc2_me@x.com", "liver")
    _member("sc2_stale@x.com", "liver"); _member("sc2_fresh@x.com", "liver")
    _optin("sc2_me@x.com", "sc2_stale@x.com", "sc2_fresh@x.com")
    _skip("sc2_me@x.com", "sc2_stale@x.com", "2020-01-01T00:00:00+00:00")   # stale
    d = _proposal(me, {"sc2_me@x.com": [1.0, 0.0], "sc2_stale@x.com": [1.0, 0.0],
                       "sc2_fresh@x.com": [1.0, 0.0]})
    assert d["candidate"]["member_ref"] == _pc.member_ref("sc2_fresh@x.com")   # fresh wins, not stale


def test_stale_skip_resurfaces_when_pool_dry():
    me = _member("sc3_me@x.com", "liver"); _member("sc3_stale@x.com", "liver")
    _optin("sc3_me@x.com", "sc3_stale@x.com")
    _skip("sc3_me@x.com", "sc3_stale@x.com", "2020-01-01T00:00:00+00:00")   # stale, only candidate
    d = _proposal(me, {"sc3_me@x.com": [1.0, 0.0], "sc3_stale@x.com": [1.0, 0.0]})
    assert d["candidate"]["member_ref"] == _pc.member_ref("sc3_stale@x.com")   # resurfaced
    assert "sc3_stale@x.com" not in json.dumps(d)       # still anonymous


def test_non_mutual_connect_never_resurfaces():
    me = _member("sc4_me@x.com", "liver"); _member("sc4_cn@x.com", "liver")
    _optin("sc4_me@x.com", "sc4_cn@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
        _pc.record_interest(cx, "sc4_me@x.com", "sc4_cn@x.com", "connect")
        cx.execute("UPDATE peer_interest SET created_at='2020-01-01T00:00:00+00:00'"); cx.commit()
    d = _proposal(me, {"sc4_me@x.com": [1.0, 0.0], "sc4_cn@x.com": [1.0, 0.0]})
    assert d["candidate"] is None                       # standing connect never re-proposed
```

- [ ] **Step 6: Run the route test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_skip_cooloff_api.py -q`
Expected: FAIL — `test_stale_skip_resurfaces_when_pool_dry` fails (no fallback pass yet).

- [ ] **Step 7: Route implementation**

In `app.py`, add beside `PEER_BLEND_WEIGHT`:

```python
PEER_SKIP_COOLOFF_DAYS = int(os.environ.get("PEER_SKIP_COOLOFF_DAYS", "30"))
```

Change `peer_proposal`'s final lines to the two-pass form:

```python
        pool = _pc.eligible_candidates(cx, email, is_paid=_is_paid_member)
        cand = _peer_blended_candidate(cx, email, pool)
        if cand is None:
            # pool dry -> re-admit skips older than the cool-off and try once more
            cutoff = (datetime.now(timezone.utc) - timedelta(days=PEER_SKIP_COOLOFF_DAYS)).isoformat()
            fb = _pc.eligible_candidates(cx, email, is_paid=_is_paid_member,
                                         include_stale_skips=True, cutoff_iso=cutoff)
            cand = _peer_blended_candidate(cx, email, fb)
        return jsonify({"candidate": cand})
```

(`datetime, timezone, timedelta` are already imported at app.py module level.)

- [ ] **Step 8: Run the route test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_skip_cooloff_api.py -q` → 4 passed.
Then confirm no regression (each in a fresh DATA_DIR): `tests/test_peer_blend_api.py`, `tests/test_peer_semantic_api.py`, `tests/test_peer_match_api.py`, `tests/test_peer_thread_api.py`.

- [ ] **Step 9: Commit**

```bash
git add dashboard/peer_connect.py app.py tests/test_peer_skip_cooloff_store.py tests/test_peer_skip_cooloff_api.py
git commit -m "feat(community): peer skip cool-off — pool-dry fallback re-offers a >30-day skip"
```

---

## Definition of Done

- A "Not now" excludes the candidate while fresh candidates exist; when a proposal would be empty, a stale (>`PEER_SKIP_COOLOFF_DAYS`) skip resurfaces via a second blend pass; a fresh skip and any non-mutual connect never resurface; all other exclusions hold in both passes.
- The fresh pass (default `eligible_candidates`) is byte-identical to before; peer_interest/peer_state/peer-thread and the blend are unchanged; privacy holds (anonymous, no email/name).
- All new tests pass; v1-v3 peer suites still green.

## Deferred (not in this plan)

- Pure time-based expiry; per-skip cool-off; a review-your-passes UI; resurfacing non-mutual connects; a subtle "seen before" cue.
