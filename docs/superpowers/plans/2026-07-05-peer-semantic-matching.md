# Peer Matching v2 — Semantic Interest-Vector Gap-Filler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the peer matcher has no exact shared-topic candidate for a member, fall back to the eligible member whose interest vector is closest (above a threshold), shown with a soft "similar path" why-line — reusing the community feed's cached member vectors.

**Architecture:** A no-behavior-change refactor exposes the eligible pool from the pure store (`peer_connect.eligible_candidates`). The embedding-dependent semantic ranking lives in the route layer (`app.py`), reusing the existing `_member_interest_vec` (get-or-build, cached in `member_interest`) + `community_feed.cosine`. `/api/peer/proposal` tries the exact matcher first, then the semantic gap-filler. One why-line branch in the portal card.

**Tech Stack:** Python 3 / Flask (`app.py` + `dashboard/*.py`), sqlite, reuses `community_feed`, `community.member_interest`, `embed()` (text-embedding-ada-002).

## Global Constraints

- **Gap-filler only:** the exact shared-topics matcher (`next_candidate`) is unchanged and always returned first. Semantic runs ONLY when `next_candidate` returns None. Never co-rank.
- **Privacy (unchanged from v1):** the semantic candidate is anonymous — `{member_ref, shared_topics: [], semantic: true}`, no email/name/vector. Every v1 exclusion (self, non-opted, matched, acted, skipped-me, person-blocked, non-paid) applies before any embedding. Downstream connect/reveal/thread is byte-identical (a semantic candidate resolves via the existing `resolve_ref`).
- **Best-effort:** an `embed()` failure or a member with no liked-topics (no vector, `[]`) yields no semantic candidate — the card is simply empty, never a 500.
- **Threshold:** `PEER_SEMANTIC_MIN_COSINE` (default 0.80); below it, return None.
- **Model:** reuse `COMMUNITY_FEED_MODEL` (`"text-embedding-ada-002"`) so peer vectors share the feed's `member_interest` cache.
- No new required env. Copy: no em dashes, no ALL CAPS; dynamic strings via `textContent`. DRY, YAGNI, TDD.

**Repo facts (verified anchors):**
- `dashboard/peer_connect.py` `next_candidate(cx, me, is_paid=None)` — the exclusion loop to refactor. Helpers: `opted_in_members`, `_pair_has_match`, `interest_kind`, `_person_blocked`, `liked_topics`, `blocked_topics`, `member_ref`, `_lc`.
- `app.py`: `_member_interest_vec(cx, email, liked_topics) -> list` (get-or-build member vector, cached in `member_interest` via `COMMUNITY_FEED_MODEL`, `[]` on cold-start/failure, never raises); `COMMUNITY_FEED_MODEL = "text-embedding-ada-002"`; `embed(text)`; `_evox_ident`; `_is_paid_member`; `_peer_ident_paid(cx, token) -> (email|None, eligible)`; `_db_lock`; `LOG_DB`. `community_feed.cosine(a, b) -> float` (0.0 if either empty).
- `peer_proposal` route (app.py) currently: auth → `if not (eligible and is_opted_in): {candidate:None}` → `{candidate: next_candidate(cx, email, is_paid=_is_paid_member)}`.
- Frontend `renderPeerProposal` (static/client-portal.html ~1891): builds the desc line `topics.length ? "A member who also resonates with " + topics.join(" and ") : "A member who also resonates with what you're working on"`.

**Testing note:** route tests `import app` → override DATA_DIR, run the file ALONE in a fresh DATA_DIR:
```
export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <one path> -q
```
Store test (Task 1) is pure: `python3 -m pytest <path> -q`.

---

### Task 1: Expose the eligible pool (`dashboard/peer_connect.py`)

**Files:**
- Modify: `dashboard/peer_connect.py` (extract `eligible_candidates`, refactor `next_candidate` to use it — no behavior change)
- Test: `tests/test_peer_eligible_pool.py`

**Interfaces:**
- Produces: `eligible_candidates(cx, me, is_paid=None) -> [email]` (all v1 exclusions, NO shared-topic filter). `next_candidate` output unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_eligible_pool.py
import sqlite3
from dashboard import peer_connect as _pc, community_signals as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
    return cx


def _like(cx, email, *topics):
    for t in topics:
        _cs.set_signal(cx, email, "topic", t, "like")


def test_eligible_pool_applies_exclusions_without_topic_filter():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver")
    _pc.set_optin(cx, "noshare@x.com", True); _like(cx, "noshare@x.com", "sleep")   # 0 shared, still eligible
    _pc.set_optin(cx, "acted@x.com", True); _like(cx, "acted@x.com", "detox")
    _pc.record_interest(cx, "me@x.com", "acted@x.com", "skip")                        # I acted -> excluded
    _pc.set_optin(cx, "off@x.com", False); _like(cx, "off@x.com", "keto")            # not opted -> excluded
    pool = set(_pc.eligible_candidates(cx, "me@x.com"))
    assert pool == {"noshare@x.com"}                       # exact-less but eligible; excludes self/acted/non-opted


def test_eligible_pool_respects_is_paid_and_next_candidate_unchanged():
    cx = _cx()
    _pc.set_optin(cx, "me@x.com", True); _like(cx, "me@x.com", "liver", "sleep")
    _pc.set_optin(cx, "free@x.com", True); _like(cx, "free@x.com", "yoga")            # eligible-but-free
    _pc.set_optin(cx, "share@x.com", True); _like(cx, "share@x.com", "liver")         # shares a topic
    paid = lambda e: e != "free@x.com"
    assert set(_pc.eligible_candidates(cx, "me@x.com", is_paid=paid)) == {"share@x.com"}
    # next_candidate still returns the exact-shared one, unchanged
    c = _pc.next_candidate(cx, "me@x.com", is_paid=paid)
    assert c["member_ref"] == _pc.member_ref("share@x.com") and c["shared_topics"] == ["liver"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_peer_eligible_pool.py -q`
Expected: FAIL — `eligible_candidates` missing.

- [ ] **Step 3: Write minimal implementation**

Refactor in `dashboard/peer_connect.py`. Add `eligible_candidates` and rewrite `next_candidate` to consume it (identical output):

```python
def eligible_candidates(cx, me, is_paid=None):
    """Opted-in members that pass every v1 exclusion (self, non-paid, existing match,
    already-acted, skipped-me, person-blocked) — WITHOUT the shared-topic requirement.
    The exact matcher ranks these by shared topics; the semantic gap-filler ranks the
    exact-less remainder by interest-vector cosine."""
    me = _lc(me)
    out = []
    for n in opted_in_members(cx):
        if n == me:
            continue
        if is_paid is not None and not is_paid(n):
            continue
        if _pair_has_match(cx, me, n):
            continue
        if interest_kind(cx, me, n) is not None:
            continue
        if interest_kind(cx, n, me) == "skip":
            continue
        if _person_blocked(cx, me, n) or _person_blocked(cx, n, me):
            continue
        out.append(n)
    return out


def next_candidate(cx, me, is_paid=None):
    me = _lc(me)
    mine = liked_topics(cx, me) - blocked_topics(cx, me)
    if not mine:
        return None
    best = None
    for n in eligible_candidates(cx, me, is_paid=is_paid):
        shared = mine & (liked_topics(cx, n) - blocked_topics(cx, n))
        if not shared:
            continue
        score = len(shared)
        if best is None or score > best[0] or (score == best[0] and member_ref(n) < best[1]):
            best = (score, member_ref(n), sorted(shared))
    if best is None:
        return None
    return {"member_ref": best[1], "shared_topics": best[2]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_peer_eligible_pool.py -q` and also the existing `python3 -m pytest tests/test_peer_connect_store.py -q` (unchanged behavior).
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/peer_connect.py tests/test_peer_eligible_pool.py
git commit -m "refactor(community): expose eligible_candidates pool for peer matching"
```

---

### Task 2: Semantic fallback + proposal wiring (`app.py`)

**Files:**
- Modify: `app.py` (`PEER_SEMANTIC_MIN_COSINE`, `_peer_semantic_candidate`, wire into `peer_proposal`)
- Test: `tests/test_peer_semantic_api.py`

**Interfaces:**
- Consumes: `peer_connect.eligible_candidates`/`next_candidate`/`liked_topics`/`member_ref`, `_member_interest_vec`, `community_feed.cosine`, `_is_paid_member`, `_peer_ident_paid`, `LOG_DB`.
- Produces: `PEER_SEMANTIC_MIN_COSINE`; `_peer_semantic_candidate(cx, me, pool) -> {member_ref, shared_topics:[], semantic:True}|None`.

**Contract:** `peer_proposal` returns the exact candidate when `next_candidate` gives one (semantic not consulted). Otherwise it ranks `eligible_candidates` by `cosine(my_vec, their_vec)` (vectors via `_member_interest_vec`, reusing the feed cache) and returns the max `>= PEER_SEMANTIC_MIN_COSINE`, anonymized with `shared_topics: []` + `semantic: True`; else `candidate: null`. A member with no liked-topics (vector `[]`) yields no semantic candidate. Never 500s.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_semantic_api.py
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


# controlled vectors: me close to "close", far from "far"
_VECS = {"me@x.com": [1.0, 0.0], "close@x.com": [0.99, 0.14], "far@x.com": [0.0, 1.0]}
def _fake_vec(cx, email, liked):
    return _VECS.get(email, [])


def test_semantic_fills_when_no_exact_overlap():
    c = _client()
    me = _member("me@x.com", "liver")          # no one else likes "liver"
    _member("close@x.com", "gallbladder")
    _member("far@x.com", "marathon")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        c.post(f"/api/peer/optin?token={me}", json={"active": True})
    for email in ("close@x.com", "far@x.com"):     # opt the candidates in
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, email, True); cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=_fake_vec):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"]["member_ref"] == _pc.member_ref("close@x.com")   # closest, above threshold
    assert d["candidate"]["shared_topics"] == [] and d["candidate"]["semantic"] is True
    assert "close@x.com" not in json.dumps(d)                              # anonymous


def test_exact_overlap_never_triggers_semantic():
    c = _client()
    me = _member("me@x.com", "liver"); _member("share@x.com", "liver")
    for email in ("me@x.com", "share@x.com"):
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, email, True); cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=AssertionError("semantic must not run")):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"]["shared_topics"] == ["liver"] and "semantic" not in d["candidate"]


def test_below_threshold_returns_no_candidate():
    c = _client()
    me = _member("me@x.com", "liver"); _member("far@x.com", "marathon")
    for email in ("me@x.com", "far@x.com"):
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, email, True); cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=_fake_vec):
        d = c.get(f"/api/peer/proposal?token={me}").get_json()
    assert d["candidate"] is None                       # cosine(me,far)=0 < 0.80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_semantic_api.py -q`
Expected: FAIL — semantic path missing.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the peer routes):

```python
PEER_SEMANTIC_MIN_COSINE = float(os.environ.get("PEER_SEMANTIC_MIN_COSINE", "0.80"))


def _peer_semantic_candidate(cx, me, pool):
    """Gap-filler: the eligible member whose interest vector is closest to `me`'s,
    above PEER_SEMANTIC_MIN_COSINE. Anonymous ({member_ref, shared_topics:[],
    semantic:True}); reuses the feed's cached member vectors. Best-effort: any
    failure or a member with no vector yields no candidate. Never raises."""
    try:
        from dashboard import peer_connect as _pc, community_feed as _cf
        my_vec = _member_interest_vec(cx, me, sorted(_pc.liked_topics(cx, me)))
        if not my_vec:
            return None
        best = None
        for n in pool:
            v = _member_interest_vec(cx, n, sorted(_pc.liked_topics(cx, n)))
            if not v:
                continue
            score = _cf.cosine(my_vec, v)
            if score < PEER_SEMANTIC_MIN_COSINE:
                continue
            ref = _pc.member_ref(n)
            if best is None or score > best[0] or (score == best[0] and ref < best[1]):
                best = (score, ref)
        if best is None:
            return None
        return {"member_ref": best[1], "shared_topics": [], "semantic": True}
    except Exception:
        app.logger.exception("peer semantic candidate failed")
        return None
```

Change `peer_proposal`'s return:

```python
        cand = _pc.next_candidate(cx, email, is_paid=_is_paid_member)
        if cand is None:
            pool = _pc.eligible_candidates(cx, email, is_paid=_is_paid_member)
            cand = _peer_semantic_candidate(cx, email, pool)
        return jsonify({"candidate": cand})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_semantic_api.py -q`
Expected: PASS (3 passed). Also re-run `tests/test_peer_match_api.py` in a fresh DATA_DIR to confirm v1 proposals unaffected.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_peer_semantic_api.py
git commit -m "feat(community): semantic interest-vector gap-filler for peer proposals"
```

---

### Task 3: Portal why-line (`static/client-portal.html`)

**Files:**
- Modify: `static/client-portal.html` (`renderPeerProposal` why-line branch)
- Test: JS parse check.

**Interfaces:**
- Consumes: the proposal payload's `shared_topics` (empty for a semantic candidate).

**Design note:** In `renderPeerProposal` (~line 1902-1910), the desc line currently is `topics.length ? "A member who also resonates with " + topics.join(" and ") : "A member who also resonates with what you're working on"`. Change the empty-topics branch to the approved semantic line: **"You seem to be walking a similar path."** Keep the non-empty branch exactly as-is (exact matches unchanged). Set via `textContent`. No em dashes, no ALL CAPS. Everything else (Connect / Not now / anonymity) unchanged.

- [ ] **Step 1: Update the why-line branch**

```javascript
  const topics = Array.isArray(candidate.shared_topics) ? candidate.shared_topics.filter(Boolean) : [];
  desc.textContent = topics.length
    ? ("A member who also resonates with " + topics.join(" and "))
    : "You seem to be walking a similar path.";
```

- [ ] **Step 2: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/client-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(community): soft why-line for semantic peer matches"
```

---

## Definition of Done

- A member with no exact shared-topic candidate is offered the eligible member whose interest vector is closest (above threshold), anonymously, with the "You seem to be walking a similar path." line; connect/reveal/thread work identically to v1. A member WITH an exact shared-topic candidate always sees that (with the topic line) and semantic never runs.
- Privacy holds: semantic proposal payloads carry no email/name; every v1 exclusion applies before embedding; below-threshold or no-vector → empty card (never a 500).
- All new tests pass; v1 peer matching, coaching, the feed, and the appointment loop are untouched (the `next_candidate` refactor is output-identical; the feed's `member_interest` cache is shared, not duplicated).

## Deferred (not in this plan)

- Co-ranked blended scoring; naming the nearest shared theme in the why-line; semantic-aware `has_proposal`; periodic member-vector rebuild on likes-change; threshold tuning from live acceptance data.
