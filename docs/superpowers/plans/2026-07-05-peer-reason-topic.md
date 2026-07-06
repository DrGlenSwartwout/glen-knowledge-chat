# Peer Anchored Semantic Why-Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For a semantic-only peer proposal (no shared topic), replace the generic "You seem to be walking a similar path." with a line anchored on the asker's OWN interest — "Another member exploring themes close to your interest in <topic>." — which is honest and leaks nothing about the winner.

**Architecture:** `_peer_blended_candidate` adds a `reason_topic` field (the asker's first liked topic) to a semantic winner's payload; the proposal card's why-line becomes three-way. No matcher/store change.

**Tech Stack:** Python 3 / Flask (`app.py`), `static/client-portal.html`, reuses `peer_connect.liked_topics`/`blocked_topics`.

## Global Constraints

- **Privacy (load-bearing):** `reason_topic` carries ONLY the asker's own liked topic (their own data). It NEVER carries the winner's topic, name, email, or signals. Only added when `shared_topics == []` (semantic winner). Exact winners are unchanged (they name the shared topic).
- **Deterministic:** `reason_topic = sorted(my_liked)[0]` — stable across refreshes.
- **Fail-safe:** if the asker has no liked topics, omit `reason_topic`; the frontend falls back to the generic line.
- No matcher/store change, no new env. Copy: no em dashes, no ALL CAPS; server-supplied topic via `textContent`. DRY, YAGNI, TDD.

**Repo facts (verified anchors):**
- `app.py` `_peer_blended_candidate(cx, me, pool)`: computes `my_liked = _pc.liked_topics(cx, me) - _pc.blocked_topics(cx, me)` near the top; ends with `return {"member_ref": best[1], "shared_topics": best[2], "semantic": len(best[2]) == 0}` inside a try/except → None. `best[2]` is `sorted(shared)` (empty for a semantic winner).
- `static/client-portal.html` `renderPeerProposal`: `desc.textContent = topics.length ? ("A member who also resonates with " + topics.join(" and ")) : "You seem to be walking a similar path.";` where `topics = candidate.shared_topics` filtered.

**Testing note:** route test `import app` → DATA_DIR override under Doppler, run ALONE in a fresh DATA_DIR; namespace test emails/topics per test. Frontend: `node --check` on extracted `<script>` blocks. Mock `_member_interest_vec` to control which candidate wins.

---

### Task 1: reason_topic payload + three-way why-line

**Files:**
- Modify: `app.py` (`_peer_blended_candidate` return)
- Modify: `static/client-portal.html` (`renderPeerProposal` why-line)
- Test: `tests/test_peer_reason_topic_api.py`

**Interfaces:**
- Produces: `reason_topic` on a semantic winner's proposal payload.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_reason_topic_api.py
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


def _proposal(tok, vecs):
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_member_interest_vec", side_effect=lambda cx, e, l: vecs.get(e, [])):
        return _client().get(f"/api/peer/proposal?token={tok}").get_json()


def test_semantic_winner_anchors_on_asker_own_topic():
    # me likes rt1_zeta + rt1_alpha (disjoint from the winner's topic) -> semantic match
    me = _member("rt1_me@x.com", "rt1_zeta", "rt1_alpha")
    _member("rt1_win@x.com", "rt1_other")
    _optin("rt1_me@x.com", "rt1_win@x.com")
    d = _proposal(me, {"rt1_me@x.com": [1.0, 0.0], "rt1_win@x.com": [0.99, 0.14]})   # cos ~0.99
    cand = d["candidate"]
    assert cand["member_ref"] == _pc.member_ref("rt1_win@x.com") and cand["shared_topics"] == []
    assert cand["reason_topic"] == "rt1_alpha"                    # asker's FIRST sorted liked topic
    assert "rt1_other" not in json.dumps(d)                       # never the winner's topic
    assert "rt1_win@x.com" not in json.dumps(d)                   # anonymous


def test_exact_winner_has_no_reason_topic():
    me = _member("rt2_me@x.com", "rt2_liver"); _member("rt2_win@x.com", "rt2_liver")
    _optin("rt2_me@x.com", "rt2_win@x.com")
    d = _proposal(me, {"rt2_me@x.com": [1.0, 0.0], "rt2_win@x.com": [0.0, 1.0]})     # cos 0, shared topic
    cand = d["candidate"]
    assert cand["shared_topics"] == ["rt2_liver"] and "reason_topic" not in cand
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_reason_topic_api.py -q`
Expected: FAIL — `reason_topic` not in the payload yet.

- [ ] **Step 3: Backend implementation**

In `app.py` `_peer_blended_candidate`, replace the return:

```python
        shared = best[2]
        out = {"member_ref": best[1], "shared_topics": shared, "semantic": len(shared) == 0}
        if not shared and my_liked:
            out["reason_topic"] = sorted(my_liked)[0]   # asker's own topic; never the winner's
        return out
```

(`my_liked` is already in scope from the top of the function. No other change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_reason_topic_api.py -q` → 2 passed.
Confirm no regression (fresh DATA_DIR each): `tests/test_peer_blend_api.py`, `tests/test_peer_semantic_api.py`, `tests/test_peer_skip_cooloff_api.py`.

- [ ] **Step 5: Frontend three-way why-line (`static/client-portal.html`)**

In `renderPeerProposal`, change the `desc.textContent` assignment to three-way:

```javascript
  if(topics.length){
    desc.textContent = "A member who also resonates with " + topics.join(" and ");
  } else if(candidate.reason_topic){
    desc.textContent = "Another member exploring themes close to your interest in " + candidate.reason_topic + ".";
  } else {
    desc.textContent = "You seem to be walking a similar path.";
  }
```

(Replaces the existing ternary. `candidate.reason_topic` is a server-supplied string set via `textContent`. No em dashes, no ALL CAPS.)

- [ ] **Step 6: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/client-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add app.py static/client-portal.html tests/test_peer_reason_topic_api.py
git commit -m "feat(community): anchor the semantic peer why-line on the asker's own interest"
```

---

## Definition of Done

- A semantic-only proposal names the asker's own first liked topic ("Another member exploring themes close to your interest in <topic>."); an exact proposal is unchanged (shared-topic line); the generic line remains a fallback. The payload's `reason_topic` is always one of the asker's own topics, never the winner's; no email/name/winner-topic leaks.
- Peer suites (blend/semantic/skip-cooloff) still green; JS parse clean.

## Deferred (not in this plan)

- Relevance-ranking the asker's topics (needs per-topic embeddings); an inferred shared-theme line; per-member copy variation.
