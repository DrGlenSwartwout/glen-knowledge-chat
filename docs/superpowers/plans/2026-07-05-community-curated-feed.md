# Community Curated Feed (Layer C, slice C1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-member ranked "for you" feed on `/community` that surfaces the content a member can see, ordered by relevance from their likes/blocks and an interest vector embedded from their own journal + liked topics, with a transparency line and a cold-start fallback.

**Architecture:** A pure ranking module (`dashboard/community_feed.py`, injectable embedder) computes relevance; `dashboard/community.py` gains a content-embedding sidecar table and a member-interest cache; one feed route in `app.py` assembles the tier-visible candidates, lazily embeds them, builds/caches the member vector, ranks, and returns the top K with reasons; `static/community.html` shows a "For you" section. In-app cosine, no Pinecone.

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), OpenAI embeddings via the existing `embed(text)` helper (app.py:1763, model `text-embedding-ada-002`), vanilla JS/HTML.

## Global Constraints

- **Privacy (load-bearing):** the member interest vector is built ONLY from that member's own journal/likes (and best-effort own chat), keyed to their own email, used ONLY to rank their own feed, and NEVER returned to the client or any other member. No feed response contains another member's data. Blocked topics are hard-filtered out of the feed.
- **Tier depth + no leak:** free members get at most `FREE_K` (default 3) items, drawn from the free-visible set (out-take teasers) with **no full Rumble `video_ref`** in the payload (same allowlist as Layer A). Paid members get up to `PAID_K` (default 10) full-visible items.
- **Cold start:** a member with no interest signal (empty interest vector) still gets a feed — newest content, tie-broken by reaction count. The feed is never empty.
- **Embedding consistency:** content vectors and member vectors MUST use the SAME model. Reuse app.py's `embed(text)` helper (model string `text-embedding-ada-002`); store that model string with each vector so a future model change forces re-embed.
- **Copy:** client-facing copy (the "For you" header, transparency lines) has no em dashes and no ALL CAPS.
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; emails lowercased; embedding/network calls run OUTSIDE any DB lock.
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs:**
- `app.py` module-level `embed(text) -> [float]` (app.py:1763): `_oa.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding`. Reuse it as the injected embedder.
- Layer A: `dashboard/community.py:list_full(cx) -> [dict]` (published paid full items, each with `id, type, title, description, video_ref, interest_tags, published_at, outtakes[]`); `list_outtakes(cx) -> [dict]`; `get_content(cx, id)`. The Layer A `community_library` route (grep `@app.route("/api/community/library")`) shows the per-tier shaping to mirror: paid returns `list_full` items as-is; free returns per-item dicts WITHOUT `video_ref`, exposing `teaser_outtakes`.
- Layer B: `dashboard/community_signals.py:my_signals(cx, email) -> {"likes":[{target_type,target_key}], "blocks":[...]}`. Topic likes/blocks are entries with `target_type == "topic"`; `target_key` is the tag string.
- `dashboard/journal_store.py:select(cx, *, since_iso, order="desc", limit=None) -> [dict]` — the member's journal entries (dicts of text/JSON fields). Reads globally (journal is single-user-per-DB here); the route filters/uses recent entries as interest text.
- app.py helpers/constants: `_evox_ident(cx, token)` → identity with `.email` or None; `_is_paid_member(email)`; `LOG_DB`; `_db_lock`; `jsonify`; `request`; `sqlite3`.

**Testing note (READ FIRST):**
- Pure/store tests (Tasks 1-2) do NOT import `app` — run with plain `python3 -m pytest <path> -q`.
- Route tests (Task 3) `import app`; the prd Doppler config points `DATA_DIR` at a nonexistent prod path, so override it:
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```

---

### Task 1: Ranking module (`dashboard/community_feed.py`)

**Files:**
- Create: `dashboard/community_feed.py`
- Test: `tests/test_community_feed.py`

**Interfaces:**
- Consumes: an injected `embed(text)->[float]` callable (tests pass a fake).
- Produces:
  - `cosine(a, b) -> float`
  - `build_interest_text(journal_texts, liked_topics, chat_texts) -> str`
  - `rank(candidates, member_vec, content_vecs, liked_topics, blocked_topics, *, boost=0.15) -> [dict]` — each returned item is the candidate dict plus `score` (float) and `reason` (str); blocked-topic items removed; cold-start (empty `member_vec`) orders by newest `published_at` then `reaction_count`.
  - `reason_for(item, liked_topics, has_vec, cold_start) -> str`

Each `candidate` dict has at least: `id`, `interest_tags` (list), `published_at` (iso str), and `reaction_count` (int, may be 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_feed.py
from dashboard import community_feed as _f


def test_cosine_bounds():
    assert _f.cosine([1, 0], [1, 0]) == 1.0
    assert _f.cosine([1, 0], [0, 1]) == 0.0
    assert _f.cosine([], [1, 0]) == 0.0
    assert _f.cosine([0, 0], [1, 0]) == 0.0


def test_build_interest_text_empty_when_no_signal():
    assert _f.build_interest_text([], [], []) == ""


def test_build_interest_text_concatenates():
    t = _f.build_interest_text(["slept poorly"], ["sleep"], ["asked about melatonin"])
    assert "slept poorly" in t and "sleep" in t and "melatonin" in t


def _cand(id, tags, pub, rc=0):
    return {"id": id, "interest_tags": tags, "published_at": pub, "reaction_count": rc}


def test_rank_filters_blocked_topics():
    cands = [_cand(1, ["sleep"], "2026-01-01"), _cand(2, ["adrenals"], "2026-01-02")]
    vecs = {1: [1, 0], 2: [0, 1]}
    out = _f.rank(cands, [1, 0], vecs, liked_topics=[], blocked_topics=["adrenals"])
    assert [i["id"] for i in out] == [1]  # blocked item removed


def test_rank_liked_boost_changes_order():
    # item 2 is a weaker cosine match but carries a liked topic → boosted above item 1
    cands = [_cand(1, ["x"], "2026-01-01"), _cand(2, ["sleep"], "2026-01-02")]
    vecs = {1: [1, 0.0], 2: [0.9, 0.1]}
    member = [1, 0]
    no_boost = _f.rank(cands, member, vecs, liked_topics=[], blocked_topics=[])
    assert no_boost[0]["id"] == 1
    boosted = _f.rank(cands, member, vecs, liked_topics=["sleep"], blocked_topics=[])
    assert boosted[0]["id"] == 2
    assert "sleep" in boosted[0]["reason"]


def test_rank_cold_start_newest_then_reactions():
    cands = [_cand(1, [], "2026-01-01", rc=5), _cand(2, [], "2026-02-01", rc=0),
             _cand(3, [], "2026-01-01", rc=9)]
    out = _f.rank(cands, member_vec=[], content_vecs={}, liked_topics=[], blocked_topics=[])
    assert [i["id"] for i in out] == [2, 3, 1]  # newest first, then by reactions
    assert out[0]["reason"]  # non-empty cold-start reason


def test_reason_for_branches():
    item = {"interest_tags": ["sleep"]}
    assert "sleep" in _f.reason_for(item, ["sleep"], has_vec=True, cold_start=False)
    assert _f.reason_for({"interest_tags": ["x"]}, [], has_vec=True, cold_start=False) \
        == "Related to your recent reflections"
    assert _f.reason_for({"interest_tags": []}, [], has_vec=False, cold_start=True) \
        == "New in the community"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_feed.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/community_feed.py
"""Community curated-feed ranking (Layer C, slice C1). Pure logic, no sqlite and
no network: the embedder is injected by the caller. Relevance = cosine(member
interest vector, content vector) + a boost when a liked topic matches; blocked
topics are filtered out; cold start (no member vector) falls back to newest then
most-reacted. The member vector is built from the member's OWN data only."""

import math


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_interest_text(journal_texts, liked_topics, chat_texts):
    parts = list(journal_texts or []) + list(liked_topics or []) + list(chat_texts or [])
    return " ".join(p for p in parts if (p or "").strip()).strip()


def reason_for(item, liked_topics, has_vec, cold_start):
    tags = set(item.get("interest_tags") or [])
    liked = tags & set(liked_topics or [])
    if liked:
        return "Because you liked " + sorted(liked)[0]
    if cold_start or not has_vec:
        return "New in the community"
    return "Related to your recent reflections"


def rank(candidates, member_vec, content_vecs, liked_topics, blocked_topics, *, boost=0.15):
    blocked = set(blocked_topics or [])
    liked = set(liked_topics or [])
    kept = [c for c in candidates if not (set(c.get("interest_tags") or []) & blocked)]
    cold_start = not member_vec
    if cold_start:
        ordered = sorted(kept, key=lambda c: (c.get("published_at") or "",
                                              c.get("reaction_count") or 0), reverse=True)
        return [{**c, "score": 0.0,
                 "reason": reason_for(c, liked, has_vec=False, cold_start=True)}
                for c in ordered]
    scored = []
    for c in kept:
        sim = cosine(member_vec, content_vecs.get(c["id"], []))
        if set(c.get("interest_tags") or []) & liked:
            sim += boost
        scored.append({**c, "score": sim,
                       "reason": reason_for(c, liked, has_vec=True, cold_start=False)})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_feed.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/community_feed.py tests/test_community_feed.py
git commit -m "feat(community): feed ranking module (cosine + boost + cold-start)"
```

---

### Task 2: Embedding + interest-cache store (`dashboard/community.py` extension)

**Files:**
- Modify: `dashboard/community.py` (append new tables + accessors)
- Test: `tests/test_community_embeddings.py`

**Interfaces:**
- Consumes: nothing (pure sqlite).
- Produces:
  - `init_feed_tables(cx)` — creates `community_embeddings` and `member_interest`.
  - `set_embedding(cx, content_id, vec, model)`
  - `get_embeddings(cx, content_ids, model) -> {content_id: [float]}` — only rows whose `model` matches.
  - `get_member_interest(cx, email, model) -> {"vec":[float], "built_at":str} | None` — only if the row's `model` matches.
  - `set_member_interest(cx, email, vec, model)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_embeddings.py
import sqlite3
from dashboard import community as _c


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _c.init_feed_tables(cx)
    return cx


def test_embedding_roundtrip():
    cx = _cx()
    _c.set_embedding(cx, 5, [0.1, 0.2, 0.3], "ada")
    assert _c.get_embeddings(cx, [5], "ada") == {5: [0.1, 0.2, 0.3]}


def test_get_embeddings_skips_other_model():
    cx = _cx()
    _c.set_embedding(cx, 5, [0.1], "ada")
    assert _c.get_embeddings(cx, [5], "newmodel") == {}   # model mismatch → re-embed
    assert _c.get_embeddings(cx, [5], "ada") == {5: [0.1]}


def test_set_embedding_upserts():
    cx = _cx()
    _c.set_embedding(cx, 5, [0.1], "ada")
    _c.set_embedding(cx, 5, [0.9], "ada")
    assert _c.get_embeddings(cx, [5], "ada") == {5: [0.9]}  # replaced, not duplicated


def test_member_interest_roundtrip_and_model_guard():
    cx = _cx()
    assert _c.get_member_interest(cx, "A@B.com", "ada") is None
    _c.set_member_interest(cx, "A@B.com", [0.4, 0.5], "ada")
    got = _c.get_member_interest(cx, "a@b.com", "ada")
    assert got["vec"] == [0.4, 0.5] and got["built_at"]
    assert _c.get_member_interest(cx, "a@b.com", "othermodel") is None  # stale model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_embeddings.py -q`
Expected: FAIL — `init_feed_tables` does not exist.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/community.py` (it already imports `json` and defines `_now`):

```python
_FEED_DDL = """
CREATE TABLE IF NOT EXISTS community_embeddings (
    content_id INTEGER PRIMARY KEY,
    vec TEXT,
    model TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS member_interest (
    email TEXT PRIMARY KEY,
    vec TEXT,
    model TEXT,
    built_at TEXT
);
"""


def init_feed_tables(cx):
    cx.executescript(_FEED_DDL)
    cx.commit()


def set_embedding(cx, content_id, vec, model):
    cx.execute(
        "INSERT INTO community_embeddings (content_id,vec,model,updated_at) VALUES (?,?,?,?) "
        "ON CONFLICT(content_id) DO UPDATE SET vec=excluded.vec, model=excluded.model, "
        "updated_at=excluded.updated_at",
        (content_id, json.dumps(list(vec)), model, _now()))
    cx.commit()


def get_embeddings(cx, content_ids, model):
    if not content_ids:
        return {}
    qs = ",".join("?" * len(content_ids))
    rows = cx.execute(
        f"SELECT content_id, vec FROM community_embeddings "
        f"WHERE model=? AND content_id IN ({qs})",
        [model, *content_ids]).fetchall()
    return {r["content_id"]: json.loads(r["vec"]) for r in rows}


def set_member_interest(cx, email, vec, model):
    cx.execute(
        "INSERT INTO member_interest (email,vec,model,built_at) VALUES (?,?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET vec=excluded.vec, model=excluded.model, "
        "built_at=excluded.built_at",
        ((email or "").strip().lower(), json.dumps(list(vec)), model, _now()))
    cx.commit()


def get_member_interest(cx, email, model):
    row = cx.execute("SELECT vec, built_at FROM member_interest WHERE email=? AND model=?",
                     ((email or "").strip().lower(), model)).fetchone()
    if not row:
        return None
    return {"vec": json.loads(row["vec"]), "built_at": row["built_at"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_embeddings.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/community.py tests/test_community_embeddings.py
git commit -m "feat(community): content-embedding + member-interest store"
```

---

### Task 3: Feed route (`app.py`)

**Files:**
- Modify: `app.py` (add one route near `community_library`; grep `@app.route("/api/community/library")`)
- Test: `tests/test_community_feed_api.py`

**Interfaces:**
- Consumes: `dashboard/community.py` (`list_full`, `list_outtakes`, `init_community_tables`, `init_feed_tables`, `get_embeddings`, `set_embedding`, `get_member_interest`, `set_member_interest`), `dashboard/community_feed.py` (`rank`, `build_interest_text`), `dashboard/community_signals.py:my_signals`, `dashboard/community_signals.py:reaction_counts`, `dashboard/journal_store.py:select`, `embed`, `_evox_ident`, `_is_paid_member`, `_db_lock`, `LOG_DB`.
- Produces: `GET /api/community/feed`.

**Contract:** `GET /api/community/feed?token=…` → bad token → 404 `{"error":"not_found"}`. Else:
- Build candidates: if `_is_paid_member(email)` → `list_full` items (K=`PAID_K`); else the free teaser shaping (each full item stripped of `video_ref`, K=`FREE_K`). Attach `reaction_count` (sum of that item's `reaction_counts`) to each candidate.
- `my_signals` → `liked_topics` / `blocked_topics` (topic targets only).
- Lazily embed any candidate lacking a current-model vector: `embed(title + " " + " ".join(tags) + " " + (transcript or "")[:2000])`, `set_embedding`. Content vectors then loaded via `get_embeddings`.
- Member interest vector: `get_member_interest`; if absent/stale-model, build from recent journal (`journal_store.select` last 90 days, up to 20 entries → their text) + liked topics, `embed`, `set_member_interest`. If the interest text is empty, member_vec = [] (cold start). Any embedding failure → cold start (never 500).
- `rank(...)`, take top K, return `{"items": [ {tier-appropriate fields + "reason"} ], "cold_start": bool}`. **The free payload must not contain any full `video_ref`.**

**Model constant:** define `COMMUNITY_FEED_MODEL = "text-embedding-ada-002"` (matches the reused `embed` helper) and `FREE_K`/`PAID_K` from env with defaults 3/10.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_feed_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, *, tags, n=1):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, community_signals as _cs
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx); _c.init_feed_tables(cx); _cs.init_signal_tables(cx)
        ids = []
        for i in range(n):
            cid = _c.upsert_full(cx, type="coaching_replay", title=f"T{i}", description="",
                                 video_ref=f"https://rumble.com/v-{i}",
                                 interest_tags=tags[i] if isinstance(tags[0], list) else tags,
                                 transcript="body"); _c.publish(cx, cid); ids.append(cid)
        token = _ev.ensure_portal_token(cx, email, "")
        cx.commit()
    return token, ids


def test_feed_bad_token_404():
    assert _client().get("/api/community/feed?token=nope").status_code == 404


def test_feed_paid_returns_items_with_reason():
    c = _client(); tok, ids = _seed("p@x.com", tags=["sleep"], n=2)
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "embed", return_value=[0.1, 0.2, 0.3]):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert "items" in d and len(d["items"]) >= 1
    assert all("reason" in it for it in d["items"])


def test_feed_free_no_full_video_ref_and_capped():
    c = _client()
    tok, ids = _seed("f@x.com", tags=[["a"], ["b"], ["c"], ["d"]], n=4)
    with mock.patch.object(appmod, "_is_paid_member", return_value=False), \
         mock.patch.object(appmod, "embed", return_value=[0.1, 0.2, 0.3]):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert len(d["items"]) <= 3                       # FREE_K cap
    assert all("video_ref" not in it for it in d["items"])  # no Rumble link for free


def test_feed_cold_start_when_embed_unavailable():
    c = _client(); tok, ids = _seed("c@x.com", tags=["sleep"], n=1)
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "embed", side_effect=RuntimeError("no key")):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert d["cold_start"] is True
    assert len(d["items"]) >= 1                        # still not empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_feed_api.py -q`
Expected: FAIL — route 404.

- [ ] **Step 3: Write minimal implementation**

Add near the community routes in `app.py`:

```python
COMMUNITY_FEED_MODEL = "text-embedding-ada-002"
COMMUNITY_FEED_FREE_K = int(os.environ.get("COMMUNITY_FEED_FREE_K", "3"))
COMMUNITY_FEED_PAID_K = int(os.environ.get("COMMUNITY_FEED_PAID_K", "10"))


def _community_candidates(cx, is_paid):
    """Tier-visible full items as ranking candidates. Free strips video_ref."""
    from dashboard import community as _cm
    full = _cm.list_full(cx)
    if is_paid:
        return full, {f["id"]: f for f in full}
    teasers = []
    for it in full:
        teasers.append({"id": it["id"], "type": it["type"], "title": it["title"],
                        "description": it["description"], "interest_tags": it["interest_tags"],
                        "published_at": it["published_at"], "teaser_outtakes": it["outtakes"]})
    return teasers, {f["id"]: f for f in full}  # full lookup for embed text only


def _member_interest_vec(cx, email, liked_topics):
    """Return the member's interest vector ([] on cold start / failure). Cached in
    member_interest; built from recent journal text + liked topics. Never raises."""
    from dashboard import community as _cm, journal_store as _js, community_feed as _cf
    cached = _cm.get_member_interest(cx, email, COMMUNITY_FEED_MODEL)
    if cached is not None:
        return cached["vec"]
    try:
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        entries = _js.select(cx, since_iso=since, order="desc", limit=20)
        jtexts = []
        for e in entries:
            jtexts.append(" ".join(str(v) for v in e.values()
                                   if isinstance(v, str) and v.strip())[:2000])
        text = _cf.build_interest_text(jtexts, liked_topics, [])
        if not text:
            return []
        vec = embed(text)
        _cm.set_member_interest(cx, email, vec, COMMUNITY_FEED_MODEL)
        return vec
    except Exception:
        app.logger.exception("member interest build failed")
        return []


@app.route("/api/community/feed")
def community_feed():
    from dashboard import community as _cm, community_signals as _cs, community_feed as _cf
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cm.init_community_tables(cx); _cm.init_feed_tables(cx); _cs.init_signal_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        email = ident.email
        is_paid = _is_paid_member(email)
        cands, full_by_id = _community_candidates(cx, is_paid)
        sig = _cs.my_signals(cx, email)
        liked = [s["target_key"] for s in sig["likes"] if s["target_type"] == "topic"]
        blocked = [s["target_key"] for s in sig["blocks"] if s["target_type"] == "topic"]
        for c in cands:
            c["reaction_count"] = sum(_cs.reaction_counts(cx, c["id"]).values())
        # lazy-embed any candidate missing a current-model vector
        have = _cm.get_embeddings(cx, [c["id"] for c in cands], COMMUNITY_FEED_MODEL)
        for c in cands:
            if c["id"] in have:
                continue
            f = full_by_id.get(c["id"], c)
            # list_full omits transcript, so pull it via get_content for the embed text
            row = _cm.get_content(cx, c["id"]) or {}
            text = (f.get("title", "") + " " + " ".join(f.get("interest_tags") or []) +
                    " " + (row.get("transcript") or "")[:2000])
            try:
                v = embed(text); _cm.set_embedding(cx, c["id"], v, COMMUNITY_FEED_MODEL)
                have[c["id"]] = v
            except Exception:
                app.logger.exception("content embed failed for %s", c["id"])
        member_vec = _member_interest_vec(cx, email, liked)
        ranked = _cf.rank(cands, member_vec, have, liked, blocked)
        k = COMMUNITY_FEED_PAID_K if is_paid else COMMUNITY_FEED_FREE_K
        top = ranked[:k]
        return jsonify({"items": top, "cold_start": not member_vec})
```

Note: `list_full` returns `{id,type,title,description,video_ref,interest_tags,published_at,outtakes}` — it does NOT include `transcript`, so the embed loop above pulls the transcript via `get_content(cx, id)`. Everything else (title, tags) comes from the candidate dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_feed_api.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_community_feed_api.py
git commit -m "feat(community): curated feed route (rank + lazy embed + cold start)"
```

---

### Task 4: "For you" section (`static/community.html`)

**Files:**
- Modify: `static/community.html`
- Test: manual JS parse check.

**Interfaces:**
- Consumes: `GET /api/community/feed?token=…` → `{items:[{...card fields..., reason}], cold_start}`.

**Design note:** read `static/community.html` first. Add a "For you" section ABOVE the existing library render, wrapped in `<!-- BEGIN foryou script -->` / `<!-- END foryou script -->`. On load, fetch `GET /api/community/feed?token=<TOKEN>`; render each item with the EXISTING card renderer used for library items (they carry the same fields — paid items have `video_ref`, free items don't), and under each card show its `reason` via `textContent`. Header text: if `cold_start` show "New in the community", else "For you". If `items` is empty, hide the section. The existing library section stays below, unchanged. Copy: no em dashes, no ALL CAPS.

- [ ] **Step 1: Read the page and add the "For you" section**

Read `static/community.html`. Add a container above the library list and a `fetch('/api/community/feed?token='+TOKEN)` that renders items with the existing card builder plus a `reason` line (`textContent`). Reuse the reaction/signal affordances already wired via `window.SignalLayer` if present (feature-detect, same as the library cards do).

- [ ] **Step 2: Verify the page JS parses**

Run: `node --check <(python3 -c "import re; h=open('static/community.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
git add static/community.html
git commit -m "feat(community): For you section on the library page"
```

---

## Definition of Done

- `/community` shows a per-member "For you" feed ranked by likes/blocks + an interest vector from the member's own journal/likes, each item with a transparency line; cold-start members get newest-and-most-reacted; free members get ≤3 items with no full Rumble link, paid get the full ranked feed.
- Relevance is in-app cosine over lazily-embedded content vectors and a cached member vector; blocked topics are hard-filtered; the member vector is never exposed.
- All new tests pass; Layer A/B untouched (feed only reads them + writes the two new caches); no member's data appears in another member's feed.

## Deferred (not in this plan)

- C3 (community-aware AI chat), C2 (opt-in introductions).
- Pinecone-backed retrieval (swap behind the `rank`/embedding interface at scale).
- Chat-history as an interest input beyond the current best-effort (journal + likes drive C1); "show me less like this" per-item feedback; periodic proactive interest-vector refresh cron (C1 rebuilds lazily on cache miss / model change).
