# Peer Matching v3 — Co-Ranked Blend (replace the gap-filler) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the semantic gap-filler with a co-ranked blend — `/api/peer/proposal` ranks the whole eligible pool by `len(shared_topics) + 1.75·cosine`, so a deeply-aligned semantic peer can outrank a lone shared topic while 2+ shared topics always win.

**Architecture:** One route-layer change in `app.py`: the gap-filler helper `_peer_semantic_candidate` becomes `_peer_blended_candidate` (unified blend over the eligible pool), and `/api/peer/proposal` calls it directly instead of "exact-first, else semantic." The pure store, the exact `next_candidate` (kept for `peer_state`'s hint), and the frontend are unchanged.

**Tech Stack:** Python 3 / Flask (`app.py`), reuses `peer_connect.eligible_candidates`/`liked_topics`/`blocked_topics`/`member_ref`, `_member_interest_vec` (cached vectors), `community_feed.cosine`.

## Global Constraints

- **Blend formula:** over `eligible_candidates`, a candidate qualifies if it shares ≥1 topic OR `cosine >= PEER_SEMANTIC_MIN_COSINE` (0.80). `score = len(shared) + PEER_BLEND_WEIGHT · cosine` (`PEER_BLEND_WEIGHT` default 1.75). Rank by `(score, member_ref lexicographic)`. Guarantee: 2+ shared topics (score ≥ 2.0) always beats any pure-semantic (max ≈ 1.75).
- **Why-line honesty:** the returned candidate carries its winner's real `shared_topics` (sorted); `semantic = (len(shared_topics) == 0)`. Frontend already branches on empty `shared_topics` — DO NOT change the frontend.
- **Privacy (unchanged):** payload is `{member_ref, shared_topics, semantic}` — NO email/name/vector/score. Every exclusion is applied by `eligible_candidates` BEFORE any embedding. Connect/reveal/thread unchanged (`resolve_ref`).
- **Best-effort:** the helper is fully try/except → None; a member with no vector and no shared topics yields no candidate; never 500s.
- `next_candidate` (exact) stays ONLY in `peer_state`'s `has_proposal` hint — do not remove it. `peer_interest`/`peer_state`/peer-thread routes untouched.
- No new required env. Copy: no em dashes, no ALL CAPS. DRY, YAGNI, TDD.

**Repo facts (verified anchors):**
- `app.py` current gap-filler `_peer_semantic_candidate(cx, me, pool)` returns `{member_ref, shared_topics: [], semantic: True}` — REPLACE it. `PEER_SEMANTIC_MIN_COSINE = float(os.environ.get("PEER_SEMANTIC_MIN_COSINE", "0.80"))` is defined near it — KEEP it, add `PEER_BLEND_WEIGHT` beside it.
- `peer_proposal` currently: `cand = next_candidate(...); if cand is None: cand = _peer_semantic_candidate(cx, email, eligible_candidates(...))` — REWIRE to a single blend call.
- `_member_interest_vec(cx, email, liked_topics) -> list` (cached, `[]` on cold-start/failure, never raises). `community_feed.cosine(a, b) -> float` (0.0 if either empty). `peer_connect.eligible_candidates(cx, me, is_paid)`, `liked_topics(cx, e)`, `blocked_topics(cx, e)`, `member_ref(e)`.
- v2 test file `tests/test_peer_semantic_api.py` has: `test_semantic_fills_when_no_exact_overlap` (keep), `test_exact_overlap_never_triggers_semantic` (REWRITE — the blend embeds everyone, so its raise-if-called mock is now invalid), `test_below_threshold_returns_no_candidate` (keep).

**Testing note:** route tests `import app` → override DATA_DIR, run EACH file ALONE in a fresh DATA_DIR (shared `chat_log.db` per pytest process). Namespace test emails/topics per test to avoid cross-test bleed (as the existing peer test files do):
```
export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <one path> -q
```

---

### Task 1: Co-ranked blend (`app.py`)

**Files:**
- Modify: `app.py` (`PEER_BLEND_WEIGHT`, `_peer_semantic_candidate` → `_peer_blended_candidate`, `peer_proposal` rewire)
- Modify: `tests/test_peer_semantic_api.py` (rewrite the one obsolete test)
- Create: `tests/test_peer_blend_api.py` (blend-ranking tests)

**Interfaces:**
- Consumes: `peer_connect.eligible_candidates`/`liked_topics`/`blocked_topics`/`member_ref`, `_member_interest_vec`, `community_feed.cosine`, `_is_paid_member`, `_peer_ident_paid`.
- Produces: `PEER_BLEND_WEIGHT`; `_peer_blended_candidate(cx, me, pool)`.

- [ ] **Step 1: Write the new blend tests + rewrite the obsolete one**

Create `tests/test_peer_blend_api.py`:

```python
# tests/test_peer_blend_api.py
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


def _run(me_tok, vecs):
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec",
                           side_effect=lambda cx, email, liked: vecs.get(email, [])):
        return _client().get(f"/api/peer/proposal?token={me_tok}").get_json()


def test_strong_semantic_beats_lone_shared_topic():
    # me shares 1 topic with "lone" (low vector match), 0 topics with "deep" (high match)
    me = _member("blend1_me@x.com", "liver")
    _member("blend1_lone@x.com", "liver")          # shares 1 topic
    _member("blend1_deep@x.com", "gallbladder")    # shares 0 topics
    _optin("blend1_me@x.com", "blend1_lone@x.com", "blend1_deep@x.com")
    vecs = {"blend1_me@x.com": [1.0, 0.0],
            "blend1_lone@x.com": [0.2, 0.98],       # cos ~0.20 -> lone score 1 + 1.75*0.20 = 1.35
            "blend1_deep@x.com": [0.99, 0.14]}      # cos ~0.99 -> deep score 0 + 1.75*0.99 = 1.73
    d = _run(me, vecs)
    assert d["candidate"]["member_ref"] == _pc.member_ref("blend1_deep@x.com")
    assert d["candidate"]["shared_topics"] == [] and d["candidate"]["semantic"] is True
    assert "blend1_deep@x.com" not in json.dumps(d)     # anonymous


def test_two_shared_topics_beats_any_semantic():
    me = _member("blend2_me@x.com", "liver", "sleep")
    _member("blend2_two@x.com", "liver", "sleep")   # shares 2 -> score >= 2.0
    _member("blend2_deep@x.com", "yoga")            # shares 0, near-perfect vector
    _optin("blend2_me@x.com", "blend2_two@x.com", "blend2_deep@x.com")
    vecs = {"blend2_me@x.com": [1.0, 0.0],
            "blend2_two@x.com": [0.0, 1.0],          # cos 0 -> score 2 + 0 = 2.0
            "blend2_deep@x.com": [1.0, 0.02]}        # cos ~1.0 -> score 1.75
    d = _run(me, vecs)
    assert d["candidate"]["member_ref"] == _pc.member_ref("blend2_two@x.com")
    assert d["candidate"]["shared_topics"] == ["liver", "sleep"] and d["candidate"]["semantic"] is False


def test_cosine_tiebreak_among_single_topic_matches():
    me = _member("blend3_me@x.com", "liver")
    _member("blend3_a@x.com", "liver")              # both share 1 topic
    _member("blend3_b@x.com", "liver")
    _optin("blend3_me@x.com", "blend3_a@x.com", "blend3_b@x.com")
    vecs = {"blend3_me@x.com": [1.0, 0.0],
            "blend3_a@x.com": [0.3, 0.95],           # cos ~0.30
            "blend3_b@x.com": [0.95, 0.31]}          # cos ~0.95 -> higher -> wins
    d = _run(me, vecs)
    assert d["candidate"]["member_ref"] == _pc.member_ref("blend3_b@x.com")
    assert d["candidate"]["shared_topics"] == ["liver"]


def test_zero_shared_below_floor_not_offered():
    me = _member("blend4_me@x.com", "liver")
    _member("blend4_far@x.com", "marathon")          # 0 shared, orthogonal vector
    _optin("blend4_me@x.com", "blend4_far@x.com")
    vecs = {"blend4_me@x.com": [1.0, 0.0], "blend4_far@x.com": [0.0, 1.0]}   # cos 0 < 0.80
    d = _run(me, vecs)
    assert d["candidate"] is None
```

In `tests/test_peer_semantic_api.py`, REWRITE `test_exact_overlap_never_triggers_semantic` (the blend embeds everyone, so the raise-if-called mock is invalid) to assert the exact-topic candidate still wins:

```python
def test_exact_topic_candidate_wins_in_blend():
    c = _client()
    me = _member("zzz_ex_me@x.com", "liver")
    _member("zzz_ex_share@x.com", "liver")
    for email in ("zzz_ex_me@x.com", "zzz_ex_share@x.com"):
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, email, True); cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec",
                           side_effect=lambda cx, email, liked: [1.0, 0.0]):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"]["shared_topics"] == ["liver"] and d["candidate"]["semantic"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_blend_api.py -q`
Expected: FAIL — the current gap-filler returns the lone-shared exact candidate (via `next_candidate`) instead of the deep-semantic one, so `test_strong_semantic_beats_lone_shared_topic` fails; `_peer_blended_candidate` does not exist yet.

- [ ] **Step 3: Write the blend implementation**

In `app.py`, add `PEER_BLEND_WEIGHT` beside `PEER_SEMANTIC_MIN_COSINE`:

```python
PEER_BLEND_WEIGHT = float(os.environ.get("PEER_BLEND_WEIGHT", "1.75"))
```

Replace `_peer_semantic_candidate` with `_peer_blended_candidate`:

```python
def _peer_blended_candidate(cx, me, pool):
    """Co-ranked blend over the eligible pool: score each candidate
    len(shared_topics) + PEER_BLEND_WEIGHT * cosine(interest vectors). Qualify with
    >=1 shared topic OR cosine >= PEER_SEMANTIC_MIN_COSINE (the zero-shared floor).
    Returns the top anonymized ({member_ref, shared_topics, semantic}); the winner's
    real shared_topics drive the why-line. Best-effort; never raises."""
    try:
        from dashboard import peer_connect as _pc, community_feed as _cf
        my_liked = _pc.liked_topics(cx, me) - _pc.blocked_topics(cx, me)
        my_vec = _member_interest_vec(cx, me, sorted(my_liked))
        best = None                                    # (score, member_ref, shared_sorted)
        for n in pool:
            shared = my_liked & (_pc.liked_topics(cx, n) - _pc.blocked_topics(cx, n))
            cos = _cf.cosine(my_vec, _member_interest_vec(cx, n, sorted(_pc.liked_topics(cx, n))))
            if not shared and cos < PEER_SEMANTIC_MIN_COSINE:
                continue
            score = len(shared) + PEER_BLEND_WEIGHT * cos
            ref = _pc.member_ref(n)
            if best is None or score > best[0] or (score == best[0] and ref < best[1]):
                best = (score, ref, sorted(shared))
        if best is None:
            return None
        return {"member_ref": best[1], "shared_topics": best[2],
                "semantic": len(best[2]) == 0}
    except Exception:
        app.logger.exception("peer blended candidate failed")
        return None
```

Rewire `peer_proposal` — replace the `next_candidate`/gap-filler two-step with a single blend call (keep the auth/eligibility lines above it unchanged):

```python
        if not (eligible and _pc.is_opted_in(cx, email)):
            return jsonify({"candidate": None})
        pool = _pc.eligible_candidates(cx, email, is_paid=_is_paid_member)
        return jsonify({"candidate": _peer_blended_candidate(cx, email, pool)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_blend_api.py -q` → 4 passed.
Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_semantic_api.py -q` → 3 passed (the two kept tests + the rewritten one).
Run each of `tests/test_peer_match_api.py`, `tests/test_peer_thread_api.py` in a fresh DATA_DIR, and `python3 -m pytest tests/test_peer_connect_store.py tests/test_peer_eligible_pool.py -q` → all green (store + v1 routes unchanged).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_peer_blend_api.py tests/test_peer_semantic_api.py
git commit -m "feat(community): co-ranked blend replaces the semantic gap-filler for peer proposals"
```

---

## Definition of Done

- `/api/peer/proposal` ranks the eligible pool by `len(shared) + 1.75·cosine`: a deeply-aligned semantic peer outranks a lone shared topic; 2+ shared topics always win; cosine breaks ties among equal-overlap candidates; a zero-shared candidate below 0.80 cosine is not offered.
- The why-line stays honest (winner's real shared topics; soft line only when semantic-only) with NO frontend change; the payload is anonymous (no email/name/vector/score).
- All privacy invariants hold; `peer_interest`/`peer_state`/peer-thread routes and the store are unchanged; every peer test file is green.

## Deferred (not in this plan)

- Skip-embed optimization when a 2+ exact match already tops; UI weight control; nearest-theme naming for semantic winners; per-member weight; weight/threshold tuning from live acceptance data.
