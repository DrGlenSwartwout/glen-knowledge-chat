# Topic Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build public, SEO-indexed symptom/condition/function pages on the proven content-module spine, with a hard compliance gate that structurally blocks disease-claim language from ever publishing.

**Architecture:** A new `topic_pages` module triplet (`dashboard/topic_pages.py` data, `dashboard/topic_copy.py` AI + compliance, `dashboard/topic_page_actions.py` console actions on the dispatch spine) plus a pure `dashboard/topic_render.py` server-renderer. Public routes live under `/learn/...` in `app.py`, gated by `TOPIC_PAGES_ENABLED`. The lifecycle mirrors `ingredient_pages` exactly — `draft → console approve → notify requesters → serve approved-only → regenerate` — with one new structural rule: `topic_page.approve` refuses to publish unless the stored compliance scan passed.

**Tech Stack:** Python 3, Flask, sqlite3 (the `LOG_DB` file), the Anthropic SDK via the shared `_cl` haiku client (`claude-haiku-4-5-20251001`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-20-topic-pages-design.md`. Every task implicitly inherits its requirements.
- **Never `import app` in tests** — Pinecone is built at import and fails in the sandbox. Test `dashboard/*` helpers directly; verify `app.py` edits with `python3 -m py_compile app.py`. Use `python3`, not `python`.
- All AI helpers follow the `propose_curation` contract: one synchronous haiku call, **never raise**, return a safe default on any failure. Compliance scan is the one exception — it **fails closed** (`passed=False`) on error.
- **No em dashes in generated copy or email bodies; use commas.** (Matches the existing `COMPLIANCE` block and `_strip_dash`.)
- Public path serves **approved-only** content. A `draft`/`gated`/absent page must never emit section text on any public route.
- Ships **DARK behind `TOPIC_PAGES_ENABLED`** (default off). No public route is reachable while the flag is off.
- Model id constant everywhere: `claude-haiku-4-5-20251001`.
- Console routes reuse the existing `_sales_console_ok()` auth guard.
- Actions register with `risk_tier=LOW_WRITE, permission=(OWNER, OPS)` and guard against double-registration via `get_action(...)`.
- Three entity kinds share one table via a `kind` discriminator: `symptom` | `condition` | `function`.
- Three narrative sections per page: `overview`, `contributing_factors`, `what_people_explore`.

---

### Task 1: `topic_pages.py` — schema + page CRUD

**Files:**
- Create: `dashboard/topic_pages.py`
- Test: `tests/test_topic_pages_store.py`

**Interfaces:**
- Produces:
  - `init_table(cx) -> None`
  - `get_page(cx, slug) -> dict | None` — keys: `slug, kind, name, state, content (dict), links (dict), compliance (dict), seo (dict), model, generated_at, approved_at, approved_by, created_at, updated_at`
  - `get_section(cx, slug, section) -> str | None`
  - `upsert_section(cx, slug, section, text, model="") -> None` (forces `state='draft'` on insert)
  - `set_state(cx, slug, state, by="") -> None` (`approved` also stamps `approved_at`/`approved_by`)
  - `set_name(cx, slug, name) -> None`
  - `set_kind(cx, slug, kind) -> None`
  - `set_links(cx, slug, links: dict) -> None`
  - `set_compliance(cx, slug, result: dict) -> None`
  - `set_seo(cx, slug, seo: dict) -> None`
  - `list_pages(cx) -> list[dict]` — `slug, kind, name, state, sections (sorted keys), compliance_passed (bool|None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_pages_store.py
import sqlite3
import sys
from pathlib import Path

import pytest


def _repo():
    return Path(__file__).resolve().parent.parent


def _mod():
    r = str(_repo())
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _cx(tp):
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_upsert_and_get_roundtrip():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "low-energy", "overview", "People often notice tiredness.")
    tp.set_kind(cx, "low-energy", "symptom")
    tp.set_name(cx, "low-energy", "Low Energy")
    page = tp.get_page(cx, "low-energy")
    assert page["kind"] == "symptom"
    assert page["name"] == "Low Energy"
    assert page["state"] == "draft"
    assert page["content"]["overview"] == "People often notice tiredness."


def test_set_links_compliance_seo_roundtrip():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "methylation", "overview", "x")
    tp.set_links(cx, "methylation", {"ingredients": [{"slug": "folate", "name": "Folate"}],
                                     "products": [], "topics": []})
    tp.set_compliance(cx, "methylation", {"passed": True, "flags": [], "scanned_at": "t", "model": "m"})
    tp.set_seo(cx, "methylation", {"title": "Methylation", "meta_description": "About methylation.",
                                   "jsonld": {}})
    page = tp.get_page(cx, "methylation")
    assert page["links"]["ingredients"][0]["slug"] == "folate"
    assert page["compliance"]["passed"] is True
    assert page["seo"]["title"] == "Methylation"


def test_approve_stamps_approved_fields():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "detox", "overview", "x")
    tp.set_state(cx, "detox", "approved", by="glen")
    page = tp.get_page(cx, "detox")
    assert page["state"] == "approved"
    assert page["approved_by"] == "glen"
    assert page["approved_at"]


def test_get_missing_returns_none():
    tp = _mod()
    cx = _cx(tp)
    assert tp.get_page(cx, "nope") is None


def test_list_pages_reports_compliance_passed():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "detox", "overview", "x")
    tp.set_compliance(cx, "detox", {"passed": False, "flags": [{"phrase": "cures", "reason": "claim"}],
                                    "scanned_at": "t", "model": "m"})
    rows = tp.list_pages(cx)
    row = [r for r in rows if r["slug"] == "detox"][0]
    assert row["compliance_passed"] is False
    assert "overview" in row["sections"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_pages_store.py -q`
Expected: SKIP-then-fail or collection error — `topic_pages not importable` (skips) until the module exists. Treat skip as "not yet implemented"; proceed.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/topic_pages.py
"""Store and request/notify for public topic pages (symptom/condition/function).

Mirrors dashboard/ingredient_pages.py. One table, kind-discriminated. Public path
serves approved-only. Adds links_json, compliance_json, seo_json columns.
"""
import datetime
import json
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


VALID_KINDS = ("symptom", "condition", "function")


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS topic_pages ("
        "slug TEXT PRIMARY KEY, "
        "kind TEXT DEFAULT '', "
        "name TEXT DEFAULT '', "
        "state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', "
        "links_json TEXT DEFAULT '{}', "
        "compliance_json TEXT DEFAULT '{}', "
        "seo_json TEXT DEFAULT '{}', "
        "model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', "
        "approved_at TEXT DEFAULT '', "
        "approved_by TEXT DEFAULT '', "
        "created_at TEXT DEFAULT '', "
        "updated_at TEXT DEFAULT '')"
    )
    cx.execute(
        "CREATE TABLE IF NOT EXISTS topic_page_requests ("
        "slug TEXT, "
        "email TEXT, "
        "requested_at TEXT, "
        "emailed_at TEXT DEFAULT '', "
        "PRIMARY KEY(slug, email))"
    )
    cx.commit()


def get_page(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    row = cur.execute(
        "SELECT slug, kind, name, state, content_json, links_json, compliance_json, "
        "seo_json, model, generated_at, approved_at, approved_by, created_at, updated_at "
        "FROM topic_pages WHERE slug=?", (slug,)
    ).fetchone()
    if not row:
        return None
    return {
        "slug": row["slug"],
        "kind": row["kind"] or "",
        "name": row["name"] or "",
        "state": row["state"] or "draft",
        "content": json.loads(row["content_json"] or "{}"),
        "links": json.loads(row["links_json"] or "{}"),
        "compliance": json.loads(row["compliance_json"] or "{}"),
        "seo": json.loads(row["seo_json"] or "{}"),
        "model": row["model"] or "",
        "generated_at": row["generated_at"] or "",
        "approved_at": row["approved_at"] or "",
        "approved_by": row["approved_by"] or "",
        "created_at": row["created_at"] or "",
        "updated_at": row["updated_at"] or "",
    }


def get_section(cx, slug, section):
    page = get_page(cx, slug)
    if not page:
        return None
    return page["content"].get(section) or None


def _upsert_col(cx, slug, col, value):
    """Generic single-column upsert that also touches updated_at/created_at."""
    init_table(cx)
    now = _now()
    cx.execute(
        f"INSERT INTO topic_pages (slug, {col}, created_at, updated_at) "
        f"VALUES (?, ?, ?, ?) "
        f"ON CONFLICT(slug) DO UPDATE SET {col}=excluded.{col}, updated_at=excluded.updated_at",
        (slug, value, now, now),
    )
    cx.commit()


def upsert_section(cx, slug, section, text, model=""):
    init_table(cx)
    now = _now()
    row = cx.execute("SELECT content_json FROM topic_pages WHERE slug=?", (slug,)).fetchone()
    content = json.loads(row[0]) if row and row[0] else {}
    content[section] = text
    cx.execute(
        "INSERT INTO topic_pages (slug, state, content_json, model, generated_at, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?, ?, ?) "
        "ON CONFLICT(slug) DO UPDATE SET "
        "content_json=excluded.content_json, model=excluded.model, "
        "generated_at=excluded.generated_at, updated_at=excluded.updated_at",
        (slug, json.dumps(content), model, now, now, now),
    )
    cx.commit()


def set_state(cx, slug, state, by=""):
    init_table(cx)
    now = _now()
    if state == "approved":
        cx.execute(
            "UPDATE topic_pages SET state=?, approved_at=?, approved_by=?, updated_at=? WHERE slug=?",
            (state, now, by, now, slug),
        )
    else:
        cx.execute(
            "UPDATE topic_pages SET state=?, updated_at=? WHERE slug=?", (state, now, slug)
        )
    cx.commit()


def set_name(cx, slug, name):
    _upsert_col(cx, slug, "name", name or "")


def set_kind(cx, slug, kind):
    _upsert_col(cx, slug, "kind", kind or "")


def set_links(cx, slug, links):
    _upsert_col(cx, slug, "links_json", json.dumps(links or {}))


def set_compliance(cx, slug, result):
    _upsert_col(cx, slug, "compliance_json", json.dumps(result or {}))


def set_seo(cx, slug, seo):
    _upsert_col(cx, slug, "seo_json", json.dumps(seo or {}))


def list_pages(cx):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT slug, kind, name, state, content_json, compliance_json "
        "FROM topic_pages ORDER BY updated_at DESC"
    ).fetchall()
    out = []
    for r in rows:
        content = json.loads(r["content_json"] or "{}")
        comp = json.loads(r["compliance_json"] or "{}")
        out.append({
            "slug": r["slug"], "kind": r["kind"] or "", "name": r["name"] or r["slug"],
            "state": r["state"] or "draft", "sections": sorted(content.keys()),
            "compliance_passed": comp.get("passed"),
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_pages_store.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_pages.py tests/test_topic_pages_store.py
git commit -m "feat(topic-pages): data layer (schema + page CRUD)"
```

---

### Task 2: `topic_pages.py` — requests + notify

**Files:**
- Modify: `dashboard/topic_pages.py` (append request/notify functions)
- Test: `tests/test_topic_pages_requests.py`

**Interfaces:**
- Produces:
  - `record_request(cx, slug, email) -> None` (INSERT OR IGNORE; ignores empty email)
  - `requesters_to_email(cx, slug) -> list[dict]` (rows with empty `emailed_at`)
  - `mark_emailed(cx, slug, email) -> None`
  - `notify_on_approve(cx, slug, name, base_url, *, send, strip=None) -> None` (links to `{base_url}/learn/{slug}`; at-most-once; never raises)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_pages_requests.py
import sqlite3
import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _cx(tp):
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_record_and_list_request():
    tp = _mod()
    cx = _cx(tp)
    tp.record_request(cx, "low-energy", "A@Example.com")
    tp.record_request(cx, "low-energy", "")  # ignored
    rows = tp.requesters_to_email(cx, "low-energy")
    assert [r["email"] for r in rows] == ["a@example.com"]


def test_notify_links_to_learn_and_marks_once():
    tp = _mod()
    cx = _cx(tp)
    tp.record_request(cx, "low-energy", "a@example.com")
    sent = []
    tp.notify_on_approve(cx, "low-energy", "Low Energy", "https://x.test",
                         send=lambda to, subj, body: sent.append((to, subj, body)))
    assert len(sent) == 1
    assert "https://x.test/learn/low-energy" in sent[0][2]
    # second call sends nothing (already emailed)
    tp.notify_on_approve(cx, "low-energy", "Low Energy", "https://x.test",
                         send=lambda to, subj, body: sent.append((to, subj, body)))
    assert len(sent) == 1


def test_notify_one_bad_send_does_not_stop_others():
    tp = _mod()
    cx = _cx(tp)
    tp.record_request(cx, "detox", "bad@example.com")
    tp.record_request(cx, "detox", "good@example.com")
    ok = []

    def _send(to, subj, body):
        if to == "bad@example.com":
            raise RuntimeError("smtp down")
        ok.append(to)

    tp.notify_on_approve(cx, "detox", "Detox", "https://x.test", send=_send)
    assert ok == ["good@example.com"]
    # bad one is still un-emailed and retryable
    assert [r["email"] for r in tp.requesters_to_email(cx, "detox")] == ["bad@example.com"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_pages_requests.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.topic_pages' has no attribute 'record_request'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/topic_pages.py`:

```python
# ---------------------------------------------------------------------------
# Request / notify (mirrors ingredient_pages)
# ---------------------------------------------------------------------------

def record_request(cx, slug, email):
    init_table(cx)
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT OR IGNORE INTO topic_page_requests (slug, email, requested_at, emailed_at) "
        "VALUES (?, ?, ?, '')",
        (slug, e, _now()),
    )
    cx.commit()


def requesters_to_email(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT email, requested_at FROM topic_page_requests "
        "WHERE slug=? AND COALESCE(emailed_at,'')='' ORDER BY requested_at",
        (slug,),
    ).fetchall()
    return [{"email": r["email"], "requested_at": r["requested_at"]} for r in rows]


def mark_emailed(cx, slug, email):
    init_table(cx)
    cx.execute(
        "UPDATE topic_page_requests SET emailed_at=? WHERE slug=? AND email=?",
        (_now(), slug, _norm(email)),
    )
    cx.commit()


def notify_on_approve(cx, slug, name, base_url, *, send, strip=None):
    """Email each un-emailed requester once; mark each after send; never raises."""
    if strip is None:
        strip = lambda s: s  # noqa: E731
    requesters = requesters_to_email(cx, slug)
    link = f"{base_url}/learn/{slug}"
    subject = f"Your {name} guide is ready"
    for r in requesters:
        email = r["email"]
        body = strip(
            f"Aloha,\n\nThe guide you asked about, {name}, is ready:\n\n{link}\n\n"
            f"In wellness,\nDr. Glen & Rae"
        )
        try:
            send(email, subject, body)
            mark_emailed(cx, slug, email)
        except Exception as exc:  # noqa: BLE001 - one bad send must not stop the rest
            print(f"[topic-pages] send failed for {email}: {exc}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_pages_requests.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_pages.py tests/test_topic_pages_requests.py
git commit -m "feat(topic-pages): request capture + notify-on-approve (/learn link)"
```

---

### Task 3: `topic_copy.py` — AI drafting, link validation, compliance gate

**Files:**
- Create: `dashboard/topic_copy.py`
- Test: `tests/test_topic_copy.py`

**Interfaces:**
- Produces:
  - `NARRATIVE_SECTIONS = ("overview", "contributing_factors", "what_people_explore")`
  - `COMPLIANCE` (str)
  - `build_section_prompt(section, topic) -> (system, user)` where `topic = {"name", "kind"}`
  - `propose_curation(topic, client) -> {"title", "meta_description", "links": {"ingredients": [slug...], "products": [slug...], "topics": [slug...]}}` (raw slugs, safe-default on error, never raises)
  - `validate_links(links_raw, *, ingredient_slugs, product_slugs, topic_slugs) -> {"ingredients": [{slug,name}], "products": [...], "topics": [...]}` (pure; `*_slugs` are `{slug: name}` dicts; drops unknown slugs)
  - `local_claim_flags(content) -> list[{"phrase", "reason"}]` (pure regex denylist)
  - `compliance_scan(content, client) -> {"passed", "flags", "scanned_at", "model"}` (fails closed)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_copy.py
import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_copy
        return topic_copy
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_copy not importable: {e}")


class _FakeContent:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeMsg:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeClient:
    def __init__(self, text):
        self._text = text
        self.messages = self

    def create(self, **kw):
        return _FakeMsg(self._text)


def test_build_section_prompt_carries_compliance():
    tc = _mod()
    system, user = tc.build_section_prompt("overview", {"name": "Low Energy", "kind": "symptom"})
    assert "diagnose, treat, cure" in system.lower()
    assert "Low Energy" in user


def test_validate_links_drops_unknown_slugs():
    tc = _mod()
    cleaned = tc.validate_links(
        {"ingredients": ["folate", "made-up"], "products": ["nope"], "topics": ["detox"]},
        ingredient_slugs={"folate": "Folate"},
        product_slugs={"neuro-magnesium": "Neuro Magnesium"},
        topic_slugs={"detox": "Detox"},
    )
    assert cleaned["ingredients"] == [{"slug": "folate", "name": "Folate"}]
    assert cleaned["products"] == []
    assert cleaned["topics"] == [{"slug": "detox", "name": "Detox"}]


def test_local_claim_flags_catches_disease_claim():
    tc = _mod()
    flags = tc.local_claim_flags({"overview": "This protocol cures cancer and treats diabetes."})
    phrases = " ".join(f["phrase"] for f in flags).lower()
    assert "cure" in phrases or "treat" in phrases
    assert flags  # non-empty


def test_local_claim_flags_clean_copy_passes():
    tc = _mod()
    flags = tc.local_claim_flags({"overview": "People exploring low energy often look into sleep and minerals."})
    assert flags == []


def test_compliance_scan_blocks_planted_claim_without_calling_model():
    tc = _mod()
    # local denylist trips first; model must NOT be consulted
    client = _FakeClient('{"passed": true, "flags": []}')
    res = tc.compliance_scan({"overview": "It cures cancer."}, client)
    assert res["passed"] is False
    assert res["flags"]


def test_compliance_scan_passes_clean_copy():
    tc = _mod()
    client = _FakeClient('{"passed": true, "flags": []}')
    res = tc.compliance_scan({"overview": "Supports healthy energy. People often explore minerals."}, client)
    assert res["passed"] is True
    assert res["flags"] == []


def test_compliance_scan_fails_closed_on_client_error():
    tc = _mod()

    class _Boom:
        messages = None

        def create(self, **kw):
            raise RuntimeError("api down")

    boom = _Boom()
    boom.messages = boom
    res = tc.compliance_scan({"overview": "Supports healthy energy."}, boom)
    assert res["passed"] is False


def test_propose_curation_safe_default_on_bad_json():
    tc = _mod()
    client = _FakeClient("not json")
    out = tc.propose_curation({"name": "Detox", "kind": "function"}, client)
    assert out["links"] == {"ingredients": [], "products": [], "topics": []}
    assert "title" in out and "meta_description" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q`
Expected: SKIP/FAIL — `topic_copy not importable`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/topic_copy.py
"""AI drafting + compliance gate for public topic pages.

Mirrors dashboard/ingredient_copy.py for prompts. Adds a wellness-framed brief,
catalog link validation, and a fail-closed compliance scan.
"""
import json
import re

NARRATIVE_SECTIONS = ("overview", "contributing_factors", "what_people_explore")

_MODEL = "claude-haiku-4-5-20251001"

COMPLIANCE = (
    "Use educational, structure/function language only (supports, promotes, helps maintain). "
    "Do NOT claim to diagnose, treat, cure, reverse, or prevent any disease, and name no "
    "disease as something this addresses. Prefer framing like 'people exploring X often look "
    "into Y'. Describe observations, not medical outcomes or probabilities. This is "
    "educational and not a substitute for medical advice. Do not use em dashes; use commas."
)

SECTION_BRIEFS = {
    "overview": (
        "Write ONE warm, plain-language paragraph (2-4 sentences) describing what this "
        "{kind} is in everyday terms and why people pay attention to it. Educational tone."
    ),
    "contributing_factors": (
        "Write a short lay-language paragraph (2-4 sentences) on the lifestyle, nutritional, "
        "and environmental factors people commonly associate with this {kind}. "
        "Frame as common associations, not causation or diagnosis."
    ),
    "what_people_explore": (
        "Write a short paragraph (2-4 sentences) on the wellness directions people commonly "
        "explore around this {kind} (nutrition, daily habits, targeted support). "
        "Use 'people often explore' framing. Make no promises and name no disease."
    ),
}

# Hard denylist: any of these in draft copy fails the gate locally, no model needed.
_BANNED = [
    (r"\bcure[sd]?\b", "claims to cure"),
    (r"\btreat(s|ed|ing|ment)?\b", "claims to treat"),
    (r"\breverse[sd]?\b", "claims to reverse"),
    (r"\bheal[s]?\s+(your\s+)?(disease|cancer|diabetes)", "claims to heal a disease"),
    (r"\bprevent[s]?\b", "claims to prevent disease"),
    (r"\bdiagnos(e|es|is|ing)\b", "claims to diagnose"),
    (r"\bguarantee[sd]?\b", "outcome guarantee"),
]


def build_section_prompt(section, topic):
    """Return (system, user) for one section and a topic dict {name, kind}."""
    name = topic.get("name", "")
    kind = topic.get("kind", "topic") or "topic"
    brief = SECTION_BRIEFS[section].format(kind=kind)
    system = (
        "You are writing public, SEO-friendly educational copy about a health topic for "
        "Dr. Glen Swartwout's wellness site. Voice: warm, clear, specific, no fluff, no "
        "clichés. " + COMPLIANCE
    )
    user = (
        f"Topic: {name} (kind: {kind})\n\n"
        f"Task: {brief}\n\n"
        "Return only the copy itself, with no headings, labels, or preamble."
    )
    return system, user


def _text_of(msg):
    return "".join(getattr(b, "text", "") for b in msg.content
                   if getattr(b, "type", "") == "text").strip()


def propose_curation(topic, client):
    """Propose SEO title + meta + raw related slugs. Never raises."""
    safe = {"title": topic.get("name", ""), "meta_description": "",
            "links": {"ingredients": [], "products": [], "topics": []}}
    try:
        name = topic.get("name", "")
        kind = topic.get("kind", "topic") or "topic"
        system = (
            "You return ONLY valid JSON with keys:\n"
            "  title: a concise SEO page title (max 60 chars)\n"
            "  meta_description: one plain sentence (max 155 chars), no disease claims\n"
            "  links: {\"ingredients\": [kebab-case slug...], \"products\": [slug...], "
            "\"topics\": [slug...]} of clearly related items.\n"
            "Slugs are lowercase, hyphens only. Only propose links you are confident relate. "
            "No disease claims. No em dashes. Return ONLY the JSON object."
        )
        user = f"Topic: {name} (kind: {kind}). Propose the curation JSON now."
        msg = client.messages.create(model=_MODEL, max_tokens=600, system=system,
                                     messages=[{"role": "user", "content": user}])
        raw = _text_of(msg)
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
        data = json.loads(raw)
        links = data.get("links") or {}
        return {
            "title": (data.get("title") or name).strip()[:60],
            "meta_description": (data.get("meta_description") or "").strip()[:155],
            "links": {
                "ingredients": [str(s).strip() for s in (links.get("ingredients") or []) if str(s).strip()],
                "products": [str(s).strip() for s in (links.get("products") or []) if str(s).strip()],
                "topics": [str(s).strip() for s in (links.get("topics") or []) if str(s).strip()],
            },
        }
    except Exception as exc:  # noqa: BLE001 - never raises
        print(f"[topic-copy] propose_curation failed: {exc}", flush=True)
        return safe


def validate_links(links_raw, *, ingredient_slugs, product_slugs, topic_slugs):
    """Drop any proposed slug not present in the real catalog. Pure."""
    def _keep(slugs, catalog):
        out, seen = [], set()
        for s in (slugs or []):
            s = str(s).strip()
            if s and s in catalog and s not in seen:
                out.append({"slug": s, "name": catalog[s]})
                seen.add(s)
        return out
    links_raw = links_raw or {}
    return {
        "ingredients": _keep(links_raw.get("ingredients"), ingredient_slugs or {}),
        "products": _keep(links_raw.get("products"), product_slugs or {}),
        "topics": _keep(links_raw.get("topics"), topic_slugs or {}),
    }


def local_claim_flags(content):
    """Pure regex denylist over all section text. Returns [{phrase, reason}]."""
    text = " ".join(str(v) for v in (content or {}).values()).lower()
    flags = []
    for pattern, reason in _BANNED:
        m = re.search(pattern, text)
        if m:
            flags.append({"phrase": m.group(0), "reason": reason})
    return flags


def compliance_scan(content, client):
    """Fail-closed compliance gate. Local denylist first, then a model judgment."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    local = local_claim_flags(content)
    if local:
        return {"passed": False, "flags": local, "scanned_at": now, "model": "local"}
    try:
        text = "\n\n".join(f"[{k}] {v}" for k, v in (content or {}).items())
        system = (
            "You are an FDA/FTC compliance reviewer for supplement wellness copy. Return ONLY "
            "JSON: {\"passed\": bool, \"flags\": [{\"phrase\": str, \"reason\": str}]}. "
            "passed=false if the copy claims to diagnose, treat, cure, reverse, or prevent any "
            "disease, names a disease as something it addresses, or promises a medical outcome. "
            "Structure/function and 'people explore' framing is allowed. Return ONLY the JSON."
        )
        msg = client.messages.create(model=_MODEL, max_tokens=500, system=system,
                                     messages=[{"role": "user", "content": text}])
        raw = _text_of(msg)
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
        data = json.loads(raw)
        passed = bool(data.get("passed"))
        flags = data.get("flags") or []
        flags = [{"phrase": str(f.get("phrase", "")), "reason": str(f.get("reason", ""))}
                 for f in flags if isinstance(f, dict)]
        return {"passed": passed and not flags, "flags": flags, "scanned_at": now, "model": _MODEL}
    except Exception as exc:  # noqa: BLE001 - fail closed
        print(f"[topic-copy] compliance_scan failed: {exc}", flush=True)
        return {"passed": False, "flags": [{"phrase": "", "reason": "scan error (fail-closed)"}],
                "scanned_at": now, "model": "error"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_copy.py tests/test_topic_copy.py
git commit -m "feat(topic-pages): AI drafting + link validation + fail-closed compliance gate"
```

---

### Task 4: `topic_render.py` — server-rendered public HTML (pure)

**Files:**
- Create: `dashboard/topic_render.py`
- Test: `tests/test_topic_render.py`

**Interfaces:**
- Consumes: a `page` dict from `topic_pages.get_page`.
- Produces:
  - `render_page_html(page, *, base_url="") -> str` — full HTML for an APPROVED page: `<title>`, `<meta name="description">`, JSON-LD `<script type="application/ld+json">`, `<h1>`, section text, a Related block (ingredients → `/begin/ingredient/<slug>`, products → `/begin/product/<slug>`, topics → `/learn/<slug>`), and a footer CTA linking to `/begin`.
  - `render_pending_html(slug, name) -> str` — "being prepared" page with an email request form posting to `/learn/<slug>/request`. Emits **no** section text.
  - `is_public(page) -> bool` — True only when `page` exists and `state == "approved"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_render.py
import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_render
        return topic_render
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_render not importable: {e}")


def _approved_page():
    return {
        "slug": "low-energy", "kind": "symptom", "name": "Low Energy", "state": "approved",
        "content": {"overview": "People often notice tiredness.",
                    "contributing_factors": "Sleep and minerals.",
                    "what_people_explore": "Many explore nutrition."},
        "links": {"ingredients": [{"slug": "folate", "name": "Folate"}],
                  "products": [{"slug": "neuro-magnesium", "name": "Neuro Magnesium"}],
                  "topics": [{"slug": "detox", "name": "Detox"}]},
        "seo": {"title": "Low Energy — wellness overview",
                "meta_description": "An educational look at low energy."},
    }


def test_is_public_only_for_approved():
    tr = _mod()
    assert tr.is_public(_approved_page()) is True
    draft = dict(_approved_page(), state="draft")
    assert tr.is_public(draft) is False
    assert tr.is_public(None) is False


def test_approved_render_has_seo_and_sections_and_links():
    tr = _mod()
    html = tr.render_page_html(_approved_page(), base_url="https://x.test")
    assert "People often notice tiredness." in html
    assert '<meta name="description" content="An educational look at low energy.' in html
    assert "application/ld+json" in html
    assert "/begin/ingredient/folate" in html
    assert "/begin/product/neuro-magnesium" in html
    assert "/learn/detox" in html
    assert "/begin" in html  # CTA


def test_pending_render_has_no_section_text_and_a_request_form():
    tr = _mod()
    html = tr.render_pending_html("low-energy", "Low Energy")
    assert "People often notice tiredness." not in html
    assert 'action="/learn/low-energy/request"' in html


def test_render_escapes_section_text():
    tr = _mod()
    page = _approved_page()
    page["content"]["overview"] = "5 < 10 & <script>alert(1)</script>"
    html = tr.render_page_html(page, base_url="https://x.test")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_render.py -q`
Expected: SKIP/FAIL — `topic_render not importable`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/topic_render.py
"""Pure server-side renderer for public topic pages (SEO-first).

No Flask import: takes a page dict, returns an HTML string. Keeps rendering
unit-testable without booting the app.
"""
import html
import json

_SECTION_TITLES = {
    "overview": "Overview",
    "contributing_factors": "Commonly Associated Factors",
    "what_people_explore": "What People Often Explore",
}
_SECTION_ORDER = ("overview", "contributing_factors", "what_people_explore")


def is_public(page):
    return bool(page) and page.get("state") == "approved"


def _esc(s):
    return html.escape(str(s or ""), quote=True)


def _related_block(links):
    links = links or {}
    rows = []
    for slug_name in (links.get("ingredients") or []):
        rows.append(f'<li><a href="/begin/ingredient/{_esc(slug_name["slug"])}">{_esc(slug_name["name"])}</a></li>')
    for slug_name in (links.get("products") or []):
        rows.append(f'<li><a href="/begin/product/{_esc(slug_name["slug"])}">{_esc(slug_name["name"])}</a></li>')
    for slug_name in (links.get("topics") or []):
        rows.append(f'<li><a href="/learn/{_esc(slug_name["slug"])}">{_esc(slug_name["name"])}</a></li>')
    if not rows:
        return ""
    return "<section class=\"related\"><h2>Related</h2><ul>" + "".join(rows) + "</ul></section>"


def render_page_html(page, *, base_url=""):
    name = page.get("name") or page.get("slug")
    seo = page.get("seo") or {}
    title = seo.get("title") or f"{name} — wellness overview"
    meta = seo.get("meta_description") or ""
    content = page.get("content") or {}

    body_sections = []
    article_text = []
    for sec in _SECTION_ORDER:
        text = content.get(sec)
        if not text:
            continue
        article_text.append(str(text))
        body_sections.append(
            f"<section><h2>{_esc(_SECTION_TITLES.get(sec, sec))}</h2>"
            f"<p>{_esc(text)}</p></section>"
        )

    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": meta,
        "articleBody": " ".join(article_text),
    }
    jsonld_tag = ('<script type="application/ld+json">'
                  + json.dumps(jsonld, ensure_ascii=False) + "</script>")

    cta = ('<section class="cta"><p>Want guidance matched to you? '
           '<a href="/begin">Start your free assessment</a>.</p></section>')

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(title)}</title>"
        f'<meta name="description" content="{_esc(meta)}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{jsonld_tag}</head><body>"
        f"<main><h1>{_esc(name)}</h1>"
        + "".join(body_sections)
        + _related_block(page.get("links"))
        + cta
        + "</main></body></html>"
    )


def render_pending_html(slug, name):
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(name)}</title>"
        '<meta name="robots" content="noindex">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "</head><body><main>"
        f"<h1>{_esc(name)}</h1>"
        "<p>This guide is being prepared. Leave your email and we will send it when it is ready.</p>"
        f'<form method="post" action="/learn/{_esc(slug)}/request">'
        '<input type="email" name="email" required placeholder="you@example.com">'
        '<button type="submit">Notify me</button></form>'
        "</main></body></html>"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_render.py -q`
Expected: PASS (4 passed). If a SyntaxError appears, fix the `render_pending_html` closing quotes per the note.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_render.py tests/test_topic_render.py
git commit -m "feat(topic-pages): pure server-side SEO renderer + pending page"
```

---

### Task 5: `topic_page_actions.py` — console actions with the compliance hard gate

**Files:**
- Create: `dashboard/topic_page_actions.py`
- Test: `tests/test_topic_page_actions.py`

**Interfaces:**
- Consumes: `topic_pages` (Tasks 1-2), `topic_copy` (Task 3), `dashboard.actions`, `dashboard.rbac`.
- Produces:
  - `configure(**kw)` — injects `client, send, strip, base_url` into `_DEPS`.
  - `register()` — registers `topic_page.approve`, `topic_page.edit`, `topic_page.regenerate` (idempotent).
  - `_exec_approve(params, ctx)` — **refuses unless compliance passed**: returns `{"slug", "ok": False, "error": "compliance_failed", "flags": [...]}` and leaves state unchanged; on pass sets `approved` + notifies, returns `{"slug", "ok": True, "state": "approved"}`.
  - `_exec_edit(params, ctx)` — updates a section/name/kind; forces `state="draft"`; **clears stale compliance** (`set_compliance(cx, slug, {})`).
  - `_exec_regenerate(params, ctx)` — re-drafts sections via `topic_copy`, re-validates links against injected catalog, re-runs `compliance_scan`; sets state `gated` if scan failed else `draft`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_page_actions.py
import sqlite3
import sys
from pathlib import Path

import pytest


def _ensure_path():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)


def _tp():
    _ensure_path()
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _tpa():
    _ensure_path()
    try:
        from dashboard import topic_page_actions
        return topic_page_actions
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_page_actions not importable: {e}")


def _actor():
    _ensure_path()
    from dashboard.rbac import Actor, OWNER
    return Actor(role=OWNER, name="glen")


def _get_action(key):
    _ensure_path()
    from dashboard.actions import get_action
    return get_action(key)


@pytest.fixture(autouse=True)
def _reset_deps():
    tpa = _tpa()
    tpa._DEPS.clear()
    yield
    tpa._DEPS.clear()


def _cx():
    tp = _tp()
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_approve_refused_when_compliance_failed():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "detox", "overview", "It cures cancer.")
    tp.set_compliance(cx, "detox", {"passed": False, "flags": [{"phrase": "cures", "reason": "claim"}]})
    res = _get_action("topic_page.approve").executor({"slug": "detox"}, {"cx": cx, "actor": _actor()})
    assert res["ok"] is False
    assert res["error"] == "compliance_failed"
    assert tp.get_page(cx, "detox")["state"] != "approved"


def test_approve_refused_when_no_scan_present():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "detox", "overview", "Supports healthy energy.")
    res = _get_action("topic_page.approve").executor({"slug": "detox"}, {"cx": cx, "actor": _actor()})
    assert res["ok"] is False
    assert tp.get_page(cx, "detox")["state"] != "approved"


def test_approve_succeeds_when_compliance_passed_and_notifies():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "low-energy", "overview", "Supports healthy energy.")
    tp.set_name(cx, "low-energy", "Low Energy")
    tp.set_compliance(cx, "low-energy", {"passed": True, "flags": []})
    tp.record_request(cx, "low-energy", "a@example.com")
    sent = []
    tpa.configure(send=lambda to, s, b: sent.append((to, s, b)), strip=lambda s: s, base_url="https://x.test")
    res = _get_action("topic_page.approve").executor({"slug": "low-energy"}, {"cx": cx, "actor": _actor()})
    assert res["ok"] is True
    assert tp.get_page(cx, "low-energy")["state"] == "approved"
    assert len(sent) == 1 and "/learn/low-energy" in sent[0][2]


def test_edit_clears_compliance_and_resets_draft():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "low-energy", "overview", "old")
    tp.set_compliance(cx, "low-energy", {"passed": True, "flags": []})
    tp.set_state(cx, "low-energy", "approved", by="glen")
    _get_action("topic_page.edit").executor(
        {"slug": "low-energy", "section": "overview", "text": "new"}, {"cx": cx, "actor": _actor()})
    page = tp.get_page(cx, "low-energy")
    assert page["content"]["overview"] == "new"
    assert page["state"] == "draft"
    assert page["compliance"] == {}


def test_actions_registered_owner_ops():
    tpa = _tpa()
    tpa.register()
    from dashboard.rbac import OWNER, OPS
    for key in ("topic_page.approve", "topic_page.edit", "topic_page.regenerate"):
        act = _get_action(key)
        assert act is not None
        assert act.permission == (OWNER, OPS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_page_actions.py -q`
Expected: SKIP/FAIL — `topic_page_actions not importable`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/topic_page_actions.py
"""Console actions for topic-page review: edit / approve / regenerate.

approve is COMPLIANCE-GATED: it refuses to publish unless the stored compliance
scan passed. Mirrors dashboard/ingredient_page_actions.py otherwise.
"""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import topic_pages as _tp
from dashboard import topic_copy as _tc

_MODEL = "claude-haiku-4-5-20251001"
_DEPS = {}  # client, send, strip, base_url, ingredient_slugs, product_slugs, topic_slugs


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_edit(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    section = (params.get("section") or "").strip()
    if section and section in _tc.NARRATIVE_SECTIONS:
        _tp.upsert_section(cx, slug, section, params.get("text") or "")
    if "name" in params:
        _tp.set_name(cx, slug, params.get("name") or "")
    if "kind" in params:
        _tp.set_kind(cx, slug, params.get("kind") or "")
    # editing invalidates any prior compliance verdict
    _tp.set_compliance(cx, slug, {})
    _tp.set_state(cx, slug, "draft", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "ok": True, "state": "draft"}


def _exec_approve(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    page = _tp.get_page(cx, slug) or {}
    comp = page.get("compliance") or {}
    if not comp.get("passed"):
        return {"slug": slug, "ok": False, "error": "compliance_failed",
                "flags": comp.get("flags") or []}
    _tp.set_state(cx, slug, "approved", by=_actor_name(ctx.get("actor")))
    try:
        send = _DEPS.get("send")
        strip = _DEPS.get("strip") or (lambda s: s)
        base = _DEPS.get("base_url", "")
        name = page.get("name") or slug
        if send is not None:
            _tp.notify_on_approve(cx, slug, name, base, send=send, strip=strip)
    except Exception as exc:  # noqa: BLE001 - notify must never fail the approve
        print(f"[topic-pages] notify skipped: {exc}", flush=True)
    return {"slug": slug, "ok": True, "state": "approved"}


def _exec_regenerate(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    client = _DEPS.get("client")
    if client is None:
        raise RuntimeError("regeneration unavailable: no client configured")
    cx = ctx["cx"]
    page = _tp.get_page(cx, slug) or {}
    name = page.get("name") or slug
    kind = page.get("kind") or "symptom"
    topic = {"name": name, "kind": kind}
    strip = _DEPS.get("strip") or (lambda s: s)

    content = {}
    for section in _tc.NARRATIVE_SECTIONS:
        try:
            system, user = _tc.build_section_prompt(section, topic)
            msg = client.messages.create(model=_MODEL, max_tokens=600, system=system,
                                         messages=[{"role": "user", "content": user}])
            text = "".join(getattr(b, "text", "") for b in msg.content
                           if getattr(b, "type", "") == "text")
            content[section] = strip(text).strip()
        except Exception as _e:
            print(f"[topic-pages] regenerate section {section} failed: {_e}", flush=True)
    for section, text in content.items():
        if text:
            _tp.upsert_section(cx, slug, section, text, model=_MODEL)

    curation = _tc.propose_curation(topic, client)
    links = _tc.validate_links(
        curation.get("links") or {},
        ingredient_slugs=_DEPS.get("ingredient_slugs") or {},
        product_slugs=_DEPS.get("product_slugs") or {},
        topic_slugs=_DEPS.get("topic_slugs") or {},
    )
    _tp.set_links(cx, slug, links)
    _tp.set_seo(cx, slug, {"title": curation.get("title") or name,
                           "meta_description": curation.get("meta_description") or ""})

    full = _tp.get_page(cx, slug) or {}
    result = _tc.compliance_scan(full.get("content") or {}, client)
    _tp.set_compliance(cx, slug, result)
    _tp.set_state(cx, slug, "draft" if result.get("passed") else "gated")
    return {"slug": slug, "state": "draft" if result.get("passed") else "gated",
            "compliance": result}


def register():
    if get_action("topic_page.approve"):
        return
    register_action(Action(
        key="topic_page.approve", module="topic_pages", title="Approve topic page",
        description="Publish a topic page (refused unless the compliance scan passed); emails requesters.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="topic_page.edit", module="topic_pages", title="Edit topic page",
        description="Edit a section/name/kind (resets to draft and clears the compliance verdict).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="topic_page.regenerate", module="topic_pages", title="Regenerate topic page",
        description="Re-draft sections, re-validate links, and re-run the compliance scan.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_regenerate))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_page_actions.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_page_actions.py tests/test_topic_page_actions.py
git commit -m "feat(topic-pages): console actions with compliance hard-gate on approve"
```

---

### Task 6: `app.py` wiring — public `/learn` routes, console, flag, registration, search index

**Files:**
- Modify: `app.py` (flag near line 2869; kickoff + public routes near the ingredient routes ~line 3877; console routes near line 8250; module registration near line 8602/after the `_ipa` block at line 20602)
- Create: `static/console-topic-pages.html` (clone of `static/console-ingredient-pages.html`)
- Modify: `static/console-search-index.json` (add the new console page)
- Test: `tests/test_topic_pages_wiring.py` (py_compile + search-index validity; no `import app`)

**Interfaces:**
- Consumes: every module from Tasks 1-5.
- Produces public routes `GET /learn`, `GET /learn/<slug>`, `POST /learn/<slug>/request`, `GET /learn/sitemap.xml`; console routes `GET /console/topic-pages`, `GET /api/console/topic-pages`, `GET /api/console/topic-page/<slug>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_pages_wiring.py
import json
import subprocess
import sys
from pathlib import Path


def _repo():
    return Path(__file__).resolve().parent.parent


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", str(_repo() / "app.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_console_search_index_lists_topic_pages():
    data = json.loads((_repo() / "static" / "console-search-index.json").read_text())
    # accept either a list of entries or {"pages": [...]}
    blob = json.dumps(data)
    assert "/console/topic-pages" in blob


def test_topic_console_html_exists():
    assert (_repo() / "static" / "console-topic-pages.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_pages_wiring.py -q`
Expected: FAIL — search index lacks `/console/topic-pages` and the HTML file is missing.

- [ ] **Step 3a: Add the feature flag**

In `app.py`, immediately after the `INGREDIENT_PAGES_PAID_ONLY` definition (line 2869), add:

```python
TOPIC_PAGES_ENABLED = os.environ.get("TOPIC_PAGES_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 3b: Add the kickoff build + catalog helper + public routes**

In `app.py`, after the ingredient routes block (after line 3877, before `begin_ingredient_page_gen`), add:

```python
def _topic_catalog_slugs():
    """Return ({ingredient_slug: name}, {product_slug: name}, {topic_slug: name}) for link validation."""
    import json as _json
    from dashboard import ingredients as _ing
    ing = dict(_ing._name_index())  # {slug: name}
    prods = {}
    try:
        raw = _json.loads(_ing._PRODUCTS.read_text()).get("products", {})
        prods = {slug: (p.get("name") or slug) for slug, p in raw.items()}
    except Exception:
        prods = {}
    topics = {}
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            from dashboard import topic_pages as _tp
            for row in _tp.list_pages(cx):
                if row["state"] == "approved":
                    topics[row["slug"]] = row["name"]
    except Exception:
        topics = {}
    return ing, prods, topics


def _topic_kickoff_build(slug, kind, name):
    """Best-effort, non-blocking AI draft + link + compliance build for a topic. Never raises."""
    import threading as _threading
    from dashboard import topic_copy as _tc
    from dashboard import topic_pages as _tp

    def _build():
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                _tp.init_table(cx)
                page = _tp.get_page(cx, slug)
                if page:
                    c = page.get("content") or {}
                    if all(c.get(s) for s in _tc.NARRATIVE_SECTIONS):
                        return  # already built
            topic = {"name": name, "kind": kind}
            content = {}
            for section in _tc.NARRATIVE_SECTIONS:
                try:
                    system, user = _tc.build_section_prompt(section, topic)
                    msg = _cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=600,
                                              system=system, messages=[{"role": "user", "content": user}])
                    text = "".join(getattr(b, "text", "") for b in msg.content
                                   if getattr(b, "type", "") == "text").strip()
                    content[section] = _strip_dash(text)
                except Exception as _se:
                    print(f"[topic-build] section {section} failed: {_se}", flush=True)
            curation = _tc.propose_curation(topic, _cl)
            ing, prods, topics = _topic_catalog_slugs()
            links = _tc.validate_links(curation.get("links") or {},
                                       ingredient_slugs=ing, product_slugs=prods, topic_slugs=topics)
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                _tp.init_table(cx)
                for section, text in content.items():
                    if text:
                        _tp.upsert_section(cx, slug, section, text, model="claude-haiku-4-5-20251001")
                _tp.set_name(cx, slug, name)
                _tp.set_kind(cx, slug, kind)
                _tp.set_links(cx, slug, links)
                _tp.set_seo(cx, slug, {"title": curation.get("title") or name,
                                       "meta_description": curation.get("meta_description") or ""})
                full = _tp.get_page(cx, slug) or {}
                result = _tc.compliance_scan(full.get("content") or {}, _cl)
                _tp.set_compliance(cx, slug, result)
                _tp.set_state(cx, slug, "draft" if result.get("passed") else "gated")
        except Exception as exc:  # noqa: BLE001
            print(f"[topic-build] {slug}: {exc}", flush=True)

    _threading.Thread(target=_build, daemon=True).start()


@app.route("/learn/<slug>")
def learn_topic_page(slug):
    from dashboard import topic_pages as _tp, topic_render as _tr
    if not TOPIC_PAGES_ENABLED:
        return ("Not found", 404)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        page = _tp.get_page(cx, slug)
    if _tr.is_public(page):
        return Response(_tr.render_page_html(page, base_url=PUBLIC_BASE_URL), mimetype="text/html")
    name = (page or {}).get("name") or slug.replace("-", " ").title()
    return Response(_tr.render_pending_html(slug, name), mimetype="text/html", status=200)


@app.route("/learn/<slug>/request", methods=["POST"])
def learn_topic_request(slug):
    from dashboard import topic_pages as _tp
    if not TOPIC_PAGES_ENABLED:
        return ("Not found", 404)
    email = (request.form.get("email") or request.values.get("email") or "").strip()
    kind = (request.values.get("kind") or "symptom").strip()
    name = slug.replace("-", " ").title()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _tp.init_table(cx)
        if email:
            _tp.record_request(cx, slug, email)
        _tp.set_name(cx, slug, name)
        if not (_tp.get_page(cx, slug) or {}).get("kind"):
            _tp.set_kind(cx, slug, kind)
    _topic_kickoff_build(slug, kind, name)
    return jsonify({"ok": True, "state": "preparing"})


@app.route("/learn")
def learn_index():
    from dashboard import topic_pages as _tp
    if not TOPIC_PAGES_ENABLED:
        return ("Not found", 404)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        rows = [r for r in _tp.list_pages(cx) if r["state"] == "approved"]
    items = "".join(
        f'<li><a href="/learn/{r["slug"]}">{r["name"]}</a></li>' for r in rows
    )
    html_doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<title>Learn</title></head><body><main><h1>Wellness Topics</h1>"
                f"<ul>{items}</ul></main></body></html>")
    return Response(html_doc, mimetype="text/html")


@app.route("/learn/sitemap.xml")
def learn_sitemap():
    from dashboard import topic_pages as _tp
    if not TOPIC_PAGES_ENABLED:
        return ("Not found", 404)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        rows = [r for r in _tp.list_pages(cx) if r["state"] == "approved"]
    urls = "".join(f"<url><loc>{PUBLIC_BASE_URL}/learn/{r['slug']}</loc></url>" for r in rows)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + urls + "</urlset>")
    return Response(xml, mimetype="application/xml")
```

> Verify `PUBLIC_BASE_URL` is the symbol used elsewhere (it is — see the `_ipa.configure` call at line 20602). If the symbol differs in this file, use the same one that block uses.

- [ ] **Step 3c: Add the console routes**

In `app.py`, after the ingredient console block (after line 8250), add:

```python
@app.route("/console/topic-pages")
def console_topic_pages_page():
    bad = _sales_console_ok()
    if bad:
        return bad
    resp = send_from_directory(STATIC, "console-topic-pages.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/console/topic-pages", methods=["GET"])
def api_console_topic_pages_list():
    bad = _sales_console_ok()
    if bad:
        return bad
    from dashboard import topic_pages as _tp
    with sqlite3.connect(LOG_DB) as cx:
        pages = _tp.list_pages(cx)
    return jsonify({"ok": True, "pages": pages})


@app.route("/api/console/topic-page/<slug>", methods=["GET"])
def api_console_topic_page_load(slug):
    bad = _sales_console_ok()
    if bad:
        return bad
    from dashboard import topic_pages as _tp, topic_copy as _tc
    with sqlite3.connect(LOG_DB) as cx:
        page = _tp.get_page(cx, slug)
    content = (page or {}).get("content") or {}
    sections = [{"id": s, "text": content.get(s, "")} for s in _tc.NARRATIVE_SECTIONS]
    return jsonify({
        "ok": True, "slug": slug,
        "name": (page or {}).get("name", slug),
        "kind": (page or {}).get("kind", ""),
        "state": (page or {}).get("state", "none"),
        "sections": sections,
        "links": (page or {}).get("links") or {},
        "compliance": (page or {}).get("compliance") or {},
        "seo": (page or {}).get("seo") or {},
        "live_url": f"/learn/{slug}",
    })
```

- [ ] **Step 3d: Register the action module + inject the catalog**

In `app.py`, immediately after the `_ipa.configure(...)` line (line 20602), add:

```python
from dashboard import topic_page_actions as _tpa
_tpa.register()
_tpa.configure(client=_cl, send=_inbox.send_email, strip=_strip_dash, base_url=PUBLIC_BASE_URL)
try:
    _tpa_ing, _tpa_prods, _tpa_topics = _topic_catalog_slugs()
    _tpa.configure(ingredient_slugs=_tpa_ing, product_slugs=_tpa_prods, topic_slugs=_tpa_topics)
except Exception as _tpa_e:  # noqa: BLE001
    print(f"[topic-pages] catalog inject skipped: {_tpa_e}", flush=True)
```

- [ ] **Step 3e: Create the console HTML page**

```bash
cd /tmp/wt-deploy-chat-252dcf59
cp static/console-ingredient-pages.html static/console-topic-pages.html
```

Then edit `static/console-topic-pages.html`, changing only the endpoint URLs, the page title, and the action keys:
- Replace `/api/console/ingredient-pages` → `/api/console/topic-pages`
- Replace `/api/console/ingredient-page/` → `/api/console/topic-page/`
- Replace action keys `ingredient_page.approve|edit|regenerate` → `topic_page.approve|edit|regenerate`
- Replace the page `<title>`/heading text "Ingredient Pages" → "Topic Pages"
- Add a read-only "Compliance" line in the detail view that shows `data.compliance.passed` and, when false, lists `data.compliance.flags[].phrase`. (The approve button already calls the dispatch endpoint; when approve returns `ok:false, error:"compliance_failed"`, surface the returned `flags` to the operator instead of treating it as success.)

- [ ] **Step 3f: Add the console page to the search index**

Edit `static/console-search-index.json` and add an entry mirroring the existing ingredient-pages entry, with:
- title/label: `Topic Pages`
- url/href: `/console/topic-pages`
- a short description: `Review and approve public symptom/condition/function pages (compliance-gated).`

(Match the exact shape of the neighboring entries — if entries are objects with `title`/`url`/`keywords`, follow that; if they are `{name, path}`, follow that.)

- [ ] **Step 4: Run tests + compile**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m py_compile app.py && python3 -m json.tool static/console-search-index.json > /dev/null && python3 -m pytest tests/test_topic_pages_wiring.py -q`
Expected: `app.py` compiles, JSON is valid, 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add app.py static/console-topic-pages.html static/console-search-index.json tests/test_topic_pages_wiring.py
git commit -m "feat(topic-pages): /learn public routes, console, flag, action registration, search index"
```

---

### Task 7: Full-suite regression + spec coverage check

**Files:** none (verification only)

- [ ] **Step 1: Run the new module suite**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_pages_store.py tests/test_topic_pages_requests.py tests/test_topic_copy.py tests/test_topic_render.py tests/test_topic_page_actions.py tests/test_topic_pages_wiring.py -q`
Expected: all green.

- [ ] **Step 2: Run the broader suite to catch isolation regressions**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/ -q -k "topic or ingredient or sales or dispatch or actions"`
Expected: no NEW failures vs. baseline. (Per project memory, app-import tests that fail on Pinecone are environmental, not regressions; note any such skips.)

- [ ] **Step 3: Confirm dark-by-default**

Manually confirm `TOPIC_PAGES_ENABLED` defaults to off (Task 6 Step 3a) so `/learn/*` returns 404 until the flag is flipped. No code change; this is a read-check of the diff.

- [ ] **Step 4: Commit (if any fixups were needed)**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add -A && git commit -m "test(topic-pages): full-suite regression pass" || echo "nothing to commit"
```

---

## Rollout (post-merge, not a code task)

- Open a PR from `sess/252dcf59` → `main`; merge (direct-push to main is guarded).
- Go-live: in Doppler `remedy-match/prd`, set `TOPIC_PAGES_ENABLED=true` after (a) a CSS/visual pass on the public `/learn/<slug>` template + the console queue, and (b) one real seeded topic taken draft → regenerate (compliance pass) → approve → `/learn/<slug>` renders and is indexable. Reversible by flipping the flag back.

## Self-Review notes (author)

- **Spec coverage:** §3 architecture → Tasks 1-6; §4 data model → Task 1; §5 draft+cross-link → Tasks 3 (validate_links) + 6 (`_topic_kickoff_build`); §6 compliance hard gate → Task 3 (`compliance_scan`, fail-closed) + Task 5 (`_exec_approve` refusal) ; §7 public SEO render → Task 4 + Task 6 routes; §8 console → Task 6; §9 testing → Tasks 1-5 unit + Task 6 wiring; §10 rollout → flag in Task 6, go-live above. No gaps.
- **Type consistency:** `compliance` dict shape `{passed, flags, scanned_at, model}` is written by `set_compliance` (Task 1), produced by `compliance_scan` (Task 3), read by `_exec_approve` (Task 5) and the console loader (Task 6) — consistent. `links` shape `{ingredients/products/topics: [{slug,name}]}` produced by `validate_links` (Task 3), stored by `set_links` (Task 1), rendered by `topic_render` (Task 4) — consistent. `NARRATIVE_SECTIONS` defined once in Task 3 and imported everywhere.
- **Known watch-item:** Task 4's `render_pending_html` string has a documented closing-quote correction — the test will catch a SyntaxError if it is copied verbatim without the fix.
