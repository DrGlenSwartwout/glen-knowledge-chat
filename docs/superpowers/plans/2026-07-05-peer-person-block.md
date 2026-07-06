# Peer Person-Block Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a permanent "Not a fit" action to the anonymous peer proposal that writes a person-block (keyed on the candidate's `member_ref`), permanently excluding them from proposals (superseding the skip cool-off) and, mutually, removing the blocker from that person's proposals.

**Architecture:** A `kind:"block"` branch on the existing `POST /api/peer/interest` route writes a `community_signals` person-block; the matcher's `_person_blocked` already honors it in both passes and both directions, so no matcher/store change. A third button on the proposal card. Everything else is reuse.

**Tech Stack:** Python 3 / Flask (`app.py`), sqlite, reuses `community_signals.set_signal`, `peer_connect.resolve_ref`/`member_ref`/`_person_blocked`, and the peer proposal frontend.

## Global Constraints

- **Block writes a person-block signal**, not a `peer_interest` row: `set_signal(cx, email, "person", member_ref(target), "block")`. `target_key` is the canonical `member_ref(target)` (the matcher checks exactly this). No email/name stored — anonymous.
- **Permanent + supersedes skip:** `_person_blocked` is checked before the kind-aware (skip cool-off) branch in `eligible_candidates`, so a block excludes the candidate in BOTH the fresh and stale-skip passes, regardless of any prior skip. (No matcher change needed — this already holds.)
- **Mutual:** `eligible_candidates` excludes on `_person_blocked(me,n) OR _person_blocked(n,me)`, so a block removes the blocker from the blocked person's proposals too. (Already holds — do not change.)
- **Auth/validation unchanged:** `_evox_ident`→404, non-paid/non-opted→403, bad kind→400, stale/forged ref→404.
- **No new env. No matcher change, no store change.** Copy: no em dashes, no ALL CAPS; frontend strings via `textContent`. DRY, YAGNI, TDD.

**Repo facts (verified anchors):**
- `app.py` `peer_interest`: currently `if kind not in ("connect", "skip"): kind = None`; after `target = _pc.resolve_ref(cx, email, ref)` (404 if None) it does `_pc.record_interest(cx, email, target, kind)` then connect/mutual logic. Response shape `{ok, matched}`.
- `dashboard/community_signals.py`: `set_signal(cx, email, target_type, target_key, signal)`, `init_signal_tables`, `TARGET_TYPES=['topic','person']`, `SIGNALS=['like','block']`.
- `dashboard/peer_connect.py`: `resolve_ref(cx, me, ref)->email|None`, `member_ref(email)`, `_person_blocked(cx, blocker, blocked)` (checks `community_signals` `target_type='person'`, `target_key=member_ref(blocked)`, `signal='block'`), `eligible_candidates` (person-block checked before the kind branch).
- Frontend `static/client-portal.html`: `renderPeerProposal` builds `connectBtn` ("Connect", `btn full`) + `skipBtn` ("Not now", `btn ghost`), each calling `peerInterest(host, candidate.member_ref, kind, connectBtn, skipBtn, msg, err)`. `peerInterest(host, memberRef, kind, connectBtn, skipBtn, msg, err)` disables connect+skip, POSTs `/api/peer/interest {member_ref, kind}`, and for non-connect kinds does `await loadPeerProposal(host)` (loads the next proposal); on error re-enables connect+skip.

**Testing note:** route/matcher tests `import app` → override DATA_DIR, run the file ALONE in a fresh DATA_DIR; namespace test emails/topics per test (shared `chat_log.db`). Frontend: `node --check` on extracted `<script>` blocks.

---

### Task 1: Person-block action (route + button)

**Files:**
- Modify: `app.py` (`peer_interest` block branch)
- Modify: `static/client-portal.html` ("Not a fit" button + `peerInterest` blockBtn param)
- Test: `tests/test_peer_block_api.py`

**Interfaces:**
- Consumes: `community_signals.set_signal`, `peer_connect.resolve_ref`/`member_ref`/`eligible_candidates`/`set_optin`.
- Produces: `kind:"block"` on `POST /api/peer/interest`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_block_api.py
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


def test_block_writes_person_signal():
    c = _client(); me = _member("pb1_me@x.com", "liver"); _member("pb1_n@x.com", "liver")
    _optin("pb1_me@x.com", "pb1_n@x.com")
    ref = _pc.member_ref("pb1_n@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        r = c.post(f"/api/peer/interest?token={me}", json={"member_ref": ref, "kind": "block"})
    d = r.get_json()
    assert d["ok"] is True and d["matched"] is False
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_signal_tables(cx)
        row = cx.execute("SELECT * FROM community_signals WHERE email='pb1_me@x.com' "
                         "AND target_type='person'").fetchone()
        assert row["target_key"] == ref and row["signal"] == "block"
        assert "pb1_n@x.com" not in json.dumps(dict(row))     # anonymous: no email stored


def test_block_excludes_both_directions():
    _member("pb2_me@x.com", "liver"); _member("pb2_n@x.com", "liver")
    _optin("pb2_me@x.com", "pb2_n@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
        _cs.set_signal(cx, "pb2_me@x.com", "person", _pc.member_ref("pb2_n@x.com"), "block")
        paid = lambda e: True
        assert _pc.eligible_candidates(cx, "pb2_me@x.com", is_paid=paid) == []     # N gone from mine
        assert _pc.eligible_candidates(cx, "pb2_n@x.com", is_paid=paid) == []       # I'm gone from N's (mutual)


def test_block_supersedes_stale_skip():
    _member("pb3_me@x.com", "liver"); _member("pb3_n@x.com", "liver")
    _optin("pb3_me@x.com", "pb3_n@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
        _pc.record_interest(cx, "pb3_me@x.com", "pb3_n@x.com", "skip")
        cx.execute("UPDATE peer_interest SET created_at='2020-01-01T00:00:00+00:00'"); cx.commit()
        _cs.set_signal(cx, "pb3_me@x.com", "person", _pc.member_ref("pb3_n@x.com"), "block")
        # even the stale-skip fallback pass excludes a blocked person
        fb = _pc.eligible_candidates(cx, "pb3_me@x.com", is_paid=lambda e: True,
                                     include_stale_skips=True, cutoff_iso="2026-01-01T00:00:00+00:00")
        assert fb == []


def test_block_stale_ref_404():
    c = _client(); me = _member("pb4_me@x.com", "liver"); _optin("pb4_me@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        r = c.post(f"/api/peer/interest?token={me}", json={"member_ref": "deadbeefdeadbeef",
                                                           "kind": "block"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_block_api.py -q`
Expected: FAIL — `kind:"block"` currently normalizes to None → 400 (not the 200 + signal the tests expect).

- [ ] **Step 3: Route implementation**

In `app.py` `peer_interest`, widen the kind set and add the block branch after `resolve_ref`:

```python
    if kind not in ("connect", "skip", "block"):
        kind = None
```
then, right after `target = _pc.resolve_ref(cx, email, ref)` / the `if target is None: return 404`:
```python
        if kind == "block":
            from dashboard import community_signals as _cs
            _cs.init_signal_tables(cx)
            _cs.set_signal(cx, email, "person", _pc.member_ref(target), "block")
            return jsonify({"ok": True, "matched": False})
        _pc.record_interest(cx, email, target, kind)     # connect / skip (unchanged below)
```
(Leave the connect/skip/mutual logic below unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_peer_block_api.py -q` → 4 passed.
Confirm no regression (fresh DATA_DIR each): `tests/test_peer_match_api.py`, `tests/test_peer_skip_cooloff_api.py`, `tests/test_peer_blend_api.py`.

- [ ] **Step 5: Add the "Not a fit" button (`static/client-portal.html`)**

In `renderPeerProposal`, after the `skipBtn` block, add a block button and wire it; extend `peerInterest` to also disable/re-enable it:

```javascript
  const blockBtn = document.createElement("button");
  blockBtn.type = "button";
  blockBtn.className = "btn ghost";
  blockBtn.style.marginTop = ".4rem";
  blockBtn.textContent = "Not a fit";
  blockBtn.addEventListener("click", function(){
    if(!window.confirm("You will not be shown this person again.")) return;
    peerInterest(host, candidate.member_ref, "block", connectBtn, skipBtn, msg, err, blockBtn);
  });
```
Append `row.appendChild(blockBtn);` alongside the other buttons. Update the `peerInterest` signature to `function peerInterest(host, memberRef, kind, connectBtn, skipBtn, msg, err, blockBtn){` and, wherever it does `connectBtn.disabled = true; skipBtn.disabled = true;` and the re-enable in `catch`, also toggle `if(blockBtn) blockBtn.disabled = ...`. The existing non-connect branch (`await loadPeerProposal(host)`) already handles `block` (loads the next proposal). All strings via `textContent`; no em dashes, no ALL CAPS.

- [ ] **Step 6: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re;h=open('static/client-portal.html').read();print('\n;\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S)))")`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add app.py static/client-portal.html tests/test_peer_block_api.py
git commit -m "feat(community): person-block action on the peer proposal (Not a fit)"
```

---

## Definition of Done

- "Not a fit" on an anonymous proposal writes a person-block keyed on the candidate's `member_ref`; the person is excluded from the member's proposals permanently (both passes, superseding any skip cool-off) and the member is mutually excluded from the blocked person's proposals. Connect/skip and the thread are unchanged.
- Privacy holds (no email/name stored or returned; keyed on the opaque ref); auth/validation unchanged; all peer suites green.

## Deferred (not in this plan)

- A review-your-blocks / unblock screen; block-with-reason or report; blocking from other surfaces; rate-limiting.
