# Lead-Magnet Quiz Funnel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a quiz-first, AMD-aware lead-magnet funnel that captures an email, personalizes a structure-function-compliant recommendation, and routes the lead into the already-merged Founding Protocol reserve for Neuro Magnesium.

**Architecture:** A new pure `quiz_engine.py` module (scoring + result logic + table init, mirroring `begin_funnel.py`), quiz content as editable `data/quizzes.json` config (no migration to change copy), one new `quiz_responses` sqlite table, seven new `/begin/quiz*` Flask routes in `app.py` that REUSE the existing email-capture (`record_unlock` → free_tier → GHL onboard), magic-link (`auth_tokens`), and R2-serve plumbing, plus three static HTML pages modeled on the existing `/begin/*` pages.

**Tech Stack:** Python 3.11+, Flask, sqlite3 (the `chat_log.db` at `app.LOG_DB`), Cloudflare R2 (boto3 s3 client via `app._r2()`), GHL v1 API, pytest.

## Global Constraints

- **Compliance — structure-function only.** No screen may imply the product treats / prevents / slows / reverses any disease. No disease nouns (AMD, macular degeneration, glaucoma, cataract, Alzheimer's, dementia) anywhere as the thing the product acts on. The hook may name the reader's *experience* ("been told it's just aging"), never a product disease claim.
- **DSHEA disclaimer** must appear on the result page: `These statements have not been evaluated by the Food and Drug Administration. This product is not intended to diagnose, treat, cure, or prevent any disease.`
- **Copy hygiene (Glen's standing rules):** real apostrophes (`'` U+2019 or plain `'`, never a backtick); **no em-dashes** (`—`) in any user-facing copy; no ALL-CAPS shouting in body copy.
- **Founder story is biography-only.** No "you will reverse too" / outcome promises.
- **Product slug is `neuro-magnesium`** (price_cents 8000) — this is BOTH the `data/products.json` `products` key AND the `data/founding_launches.json` key. The other `neuromagnesium` slug (6997) is a different, non-founding SKU; do not use it.
- **Founding card only shows when the launch is open** — gate every founding surface on `dashboard.founding.is_open(cx, "neuro-magnesium", now_iso=...)`.
- **Graceful when `LEAD_MAGNET_PDF_KEY` env is unset** — the guide route must degrade to a friendly "guide coming" page, never 500.
- **DB writes go through `app._db_lock`** (a non-reentrant lock); reads may open `sqlite3.connect(app.LOG_DB)` directly.
- **Pure modules take a caller-supplied connection** (the `begin_funnel.py` convention) so they are testable without the app.

---

## Reference: confirmed reuse points (recon 2026-06-24)

| Need | Where | Signature / note |
|---|---|---|
| Flask app object | `app.py:55` `STATIC = Path(__file__).parent / "static"`; `app.py:146` `LOG_DB` | routes use `@app.route(...)`, `send_from_directory(STATIC, "x.html")` |
| JSON-injection page | `app.py:1417` `begin_explore()` | reads html, injects `<script>window.__X__ = {json}</script>` before `</head>` |
| Static page serve + session cookie | `app.py:1320` `_serve_funnel_home()`, `app.py:3762` `begin_product_page` | sets `amg_session` cookie + no-cache headers |
| Email capture / free-tier | `app.py:2139` `begin_unlock()` + `begin_funnel.record_unlock` (`begin_funnel.py:218`) | reaching `free_tier` needs email + tos; fires `ghl_onboard_contact` in a daemon thread on the free_tier transition |
| Accepted triggers | `begin_funnel.py:81` `VALID_TRIGGERS` | includes `"email"`, `"tos"`, `"name"`, `"question"`, `"quiz"` |
| GHL onboard | `app.py:6514` `ghl_onboard_contact(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None)` | upsert → pipeline → workflow; tolerates missing API key |
| Founding config | `dashboard/founding.py` `get_launch(slug)`, `remaining(cx, slug)`, `is_open(cx, slug, *, now_iso=None)` | config at `data/founding_launches.json` |
| Founding status route (pattern) | `app.py:12418` `begin_founding_status` | |
| Magic-link mint/validate (pattern) | `app.py:6934` `_mint_membership_magic_link`, `app.py:6967` `_validate_membership_magic_link`; `_hash_token` `app.py:232` | tokens in `auth_tokens(token_hash,email,purpose,extra,created_at,expires_at,consumed_at)` |
| R2 client | `app.py:1199` `_r2()`; serve pattern `app.py:1212` `serve_clip`; bucket env `R2_BUCKET` (default `rm-clips`) | |
| Member check | `app.py:451` `is_member(session_id="", email="")` | True once ToS agreed |
| Startup table init | `app.py:6795` `_init_journey_tables()` (calls `begin_funnel.init_journey_tables(cx)`) | add `quiz_engine.init_quiz_tables(cx)` here |
| Helpers | `app.py:236` `_now_utc()`, `app.py` `PUBLIC_BASE_URL`, `import json`, `import uuid`, `import sqlite3` all already in scope | |
| Test pattern (routes) | `tests/test_begin_routes.py` | `_load_app()` via importlib, `monkeypatch.setattr(app_module,"LOG_DB",str(tmp_path/"chat_log.db"))`, `app.app.test_client()` |
| Test pattern (founding api) | `tests/test_founding_counter_api.py` | `monkeypatch.setattr(founding, "is_open", ...)` |
| Run tests | `cd ~/deploy-chat && python3 -m pytest tests/<file> -v` (worktree: run from the worktree root) | |

**Genuinely greenfield:** the quiz engine + `quiz_responses` table + `data/quizzes.json` + the three pages + the guide-download route (no existing R2 *file/PDF* download exists — only `serve_clip` for video; model the guide serve on it).

---

### Task 1: Quiz engine module + quiz content config

**Files:**
- Create: `quiz_engine.py`
- Create: `data/quizzes.json`
- Test: `tests/test_quiz_engine.py`

**Interfaces:**
- Produces:
  - `quiz_engine.load_config() -> dict` — parsed `data/quizzes.json`.
  - `quiz_engine.get_quiz(quiz_id, config=None) -> dict | None`.
  - `quiz_engine.segment_of(answers: dict) -> str` — value of `q1` (or `"general"`).
  - `quiz_engine.depletion_score(answers: dict) -> int` — 0..N tally of high-signal answers in q2-q8.
  - `quiz_engine.result_for(quiz: dict, answers: dict) -> dict` — `{"band","headline","reasoning","bullets","segment","depletion"}`.
  - `quiz_engine.init_quiz_tables(cx)` (added in Task 2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_engine.py
import quiz_engine


def test_config_loads_and_quiz_present():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    assert q is not None
    assert q["product_slug"] == "neuro-magnesium"
    assert len(q["questions"]) == 9
    # disclaimer present and DSHEA-correct
    assert "not been evaluated by the Food and Drug Administration" in q["disclaimer"]


def test_segment_of_reads_q1():
    assert quiz_engine.segment_of({"q1": "watch_wait", "q2": "restful"}) == "watch_wait"
    assert quiz_engine.segment_of({}) == "general"


def test_depletion_score_counts_high_signals():
    high = {"q2": "frequent", "q3": "often", "q4": "frequent_fog",
            "q5": "6plus", "q6": "avoid", "q7": "rarely", "q8": "none"}
    assert quiz_engine.depletion_score(high) == 7
    low = {"q2": "restful", "q3": "rarely", "q4": "sharp",
           "q5": "under2", "q6": "comfortable", "q7": "yes", "q8": "both"}
    assert quiz_engine.depletion_score(low) == 0


def test_result_barrier_band_for_watch_wait():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    r = quiz_engine.result_for(q, {"q1": "watch_wait", "q8": "eye_formula"})
    assert r["band"] == "barrier"
    assert "barrier" in r["reasoning"].lower()
    assert r["segment"] == "watch_wait"
    assert isinstance(r["bullets"], list) and r["bullets"]


def test_result_calm_band_for_stress():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    r = quiz_engine.result_for(q, {"q1": "general", "q2": "frequent", "q3": "often"})
    assert r["band"] == "calm"


def test_result_always_has_no_disease_nouns_or_emdash():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    banned = ["macular", "amd", "glaucoma", "cataract", "alzheimer", "dementia", "—"]
    for answers in ({"q1": "watch_wait"}, {"q1": "general", "q4": "frequent_fog"},
                    {"q1": "family", "q8": "both"}, {"q1": "supplement_gap", "q5": "6plus"}):
        r = quiz_engine.result_for(q, answers)
        blob = (r["headline"] + " " + r["reasoning"] + " " + " ".join(r["bullets"])).lower()
        for b in banned:
            assert b not in blob, f"banned token {b!r} in result for {answers}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quiz_engine'`.

- [ ] **Step 3: Create `data/quizzes.json`**

```json
{
  "version": 1,
  "quizzes": {
    "eye-brain": {
      "id": "eye-brain",
      "product_slug": "neuro-magnesium",
      "title": "Your Foundational Eye-and-Brain Self-Assessment",
      "hook": "Have you been told your vision changes are just aging, as if there's nothing you can do? There's now foundational support that actually reaches your eyes and brain. Take 60 seconds to see if it's for you.",
      "disclaimer": "These statements have not been evaluated by the Food and Drug Administration. This product is not intended to diagnose, treat, cure, or prevent any disease.",
      "questions": [
        {"id": "q1", "ordinal": 1, "prompt": "What brought you here today?", "type": "single", "options": [
          {"value": "watch_wait", "label": "My doctor told me to monitor and watch and wait"},
          {"value": "noticing", "label": "I'm noticing changes in my focus or clarity"},
          {"value": "family", "label": "Vision runs in my family and I want to be proactive"},
          {"value": "supplement_gap", "label": "I supplement, but I wonder if I'm missing something"},
          {"value": "general", "label": "General foundational health"}
        ]},
        {"id": "q2", "ordinal": 2, "prompt": "How restful is your sleep and how calm do you feel day to day?", "type": "single", "options": [
          {"value": "restful", "label": "Restful and calm"},
          {"value": "occasional", "label": "Occasional restlessness"},
          {"value": "frequent", "label": "Frequent tension or poor sleep"}
        ]},
        {"id": "q3", "ordinal": 3, "prompt": "Do you notice muscle tension, cramps, or twitches?", "type": "single", "options": [
          {"value": "rarely", "label": "Rarely"},
          {"value": "sometimes", "label": "Sometimes"},
          {"value": "often", "label": "Often"}
        ]},
        {"id": "q4", "ordinal": 4, "prompt": "How is your mental clarity and focus?", "type": "single", "options": [
          {"value": "sharp", "label": "Sharp"},
          {"value": "occasional_fog", "label": "Occasional fog"},
          {"value": "frequent_fog", "label": "Frequent fog"}
        ]},
        {"id": "q5", "ordinal": 5, "prompt": "How many hours a day are you on screens?", "type": "single", "options": [
          {"value": "under2", "label": "Under 2"},
          {"value": "2to6", "label": "2 to 6"},
          {"value": "6plus", "label": "6 or more"}
        ]},
        {"id": "q6", "ordinal": 6, "prompt": "How comfortable are you driving at night or in low light?", "type": "single", "options": [
          {"value": "comfortable", "label": "Comfortable"},
          {"value": "some_difficulty", "label": "Some difficulty"},
          {"value": "avoid", "label": "I avoid it"}
        ]},
        {"id": "q7", "ordinal": 7, "prompt": "Do you eat magnesium-rich foods daily, like greens, nuts, and seeds?", "type": "single", "options": [
          {"value": "yes", "label": "Yes, most days"},
          {"value": "sometimes", "label": "Sometimes"},
          {"value": "rarely", "label": "Rarely"}
        ]},
        {"id": "q8", "ordinal": 8, "prompt": "What are you currently taking?", "type": "single", "options": [
          {"value": "magnesium", "label": "A magnesium supplement"},
          {"value": "eye_formula", "label": "An eye formula (AREDS type)"},
          {"value": "both", "label": "Both"},
          {"value": "none", "label": "Neither"}
        ]},
        {"id": "q9", "ordinal": 9, "prompt": "How proactive do you want to be about your long-term eye and brain health?", "type": "single", "options": [
          {"value": "yes", "label": "Very proactive"},
          {"value": "somewhat", "label": "Somewhat"}
        ]}
      ],
      "bands": {
        "barrier": {
          "headline": "Your foundation may be missing a magnesium that can actually reach where it counts.",
          "reasoning": "Based on what you shared, ordinary magnesium and a standard eye formula may not be crossing the barrier into your eyes and brain. Neuro Magnesium is formulated to reach the blood-brain and blood-eye barrier, where ordinary magnesium can't go.",
          "bullets": [
            "Forms chosen to cross the blood-brain and blood-eye barrier",
            "Foundational support that complements a standard eye formula",
            "Made in violet Miron glass, with no non-nutritive fillers"
          ]
        },
        "calm": {
          "headline": "Your body may be asking for calm, without the fog.",
          "reasoning": "The tension, restlessness, and cramping you described are classic signs the body draws on. Neuro Magnesium is formulated for calm without the dullness, in forms designed to reach the brain.",
          "bullets": [
            "Calm, steady support without the daytime fog",
            "Forms designed to reach the brain, not just the gut",
            "Made in violet Miron glass, with no non-nutritive fillers"
          ]
        },
        "clarity": {
          "headline": "Your foundation may be asking for a clearer, steadier mind.",
          "reasoning": "The mental fog you described is something the body's foundation can influence. Neuro Magnesium is formulated to support a clear, steady mind, in forms designed to reach the brain.",
          "bullets": [
            "Support for a clear, steady mind",
            "Forms designed to reach the brain",
            "Made in violet Miron glass, with no non-nutritive fillers"
          ]
        },
        "hardworking": {
          "headline": "Your eyes work hard. Their foundation should keep up.",
          "reasoning": "Long screen hours and low-light strain ask a lot of your eyes. Neuro Magnesium gives foundational support for eyes and a brain that work hard, in forms designed to reach them.",
          "bullets": [
            "Foundational support for eyes that work hard",
            "Forms designed to reach the eyes and brain",
            "Made in violet Miron glass, with no non-nutritive fillers"
          ]
        },
        "foundational": {
          "headline": "A strong foundation is the most proactive place to start.",
          "reasoning": "You're being proactive, and the foundation is the right place to begin. Neuro Magnesium offers foundational eye-and-brain support in forms designed to reach where ordinary magnesium can't.",
          "bullets": [
            "Foundational eye-and-brain support",
            "Forms designed to reach the eyes and brain",
            "Made in violet Miron glass, with no non-nutritive fillers"
          ]
        }
      }
    }
  }
}
```

- [ ] **Step 4: Write minimal implementation**

```python
# quiz_engine.py
"""Quiz engine for the /begin/quiz lead-magnet funnel.

Pure functions over the quiz config + an answers dict, mirroring begin_funnel.py.
Quiz content lives in data/quizzes.json (editable without a migration). The only
mutating I/O is the quiz_responses table (init/store/get), all via a
caller-supplied sqlite3 connection.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "quizzes.json")

# High-signal (magnesium-depletion / foundational-gap) answer values per question.
_HIGH_SIGNALS = {
    "q2": {"frequent"},
    "q3": {"often"},
    "q4": {"frequent_fog"},
    "q5": {"6plus"},
    "q6": {"avoid"},
    "q7": {"rarely"},
    "q8": {"none"},
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": 0, "quizzes": {}}


def get_quiz(quiz_id: str, config: dict | None = None) -> dict | None:
    cfg = config if config is not None else load_config()
    return (cfg.get("quizzes") or {}).get(quiz_id)


def segment_of(answers: dict) -> str:
    return (answers or {}).get("q1") or "general"


def depletion_score(answers: dict) -> int:
    answers = answers or {}
    return sum(1 for q, highs in _HIGH_SIGNALS.items() if answers.get(q) in highs)


def _band_key(answers: dict) -> str:
    a = answers or {}
    q1 = a.get("q1")
    q8 = a.get("q8")
    if q1 in ("watch_wait", "family") or q8 in ("eye_formula", "both"):
        return "barrier"
    if a.get("q2") == "frequent" or a.get("q3") == "often":
        return "calm"
    if a.get("q4") in ("frequent_fog", "occasional_fog"):
        return "clarity"
    if a.get("q5") == "6plus" or a.get("q6") in ("avoid", "some_difficulty"):
        return "hardworking"
    return "foundational"


def result_for(quiz: dict, answers: dict) -> dict:
    band = _band_key(answers)
    spec = (quiz.get("bands") or {}).get(band) or (quiz.get("bands") or {}).get("foundational") or {}
    return {
        "band": band,
        "headline": spec.get("headline", ""),
        "reasoning": spec.get("reasoning", ""),
        "bullets": list(spec.get("bullets", [])),
        "segment": segment_of(answers),
        "depletion": depletion_score(answers),
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quiz_engine.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add quiz_engine.py data/quizzes.json tests/test_quiz_engine.py
git commit -m "feat(quiz): quiz engine + eye-brain quiz config (compliant copy)"
```

---

### Task 2: `quiz_responses` persistence + startup init

**Files:**
- Modify: `quiz_engine.py` (add table init + store/get)
- Modify: `app.py:6795-6799` (`_init_journey_tables`) — add `quiz_engine.init_quiz_tables(cx)`
- Test: `tests/test_quiz_responses.py`

**Interfaces:**
- Consumes: `quiz_engine.segment_of` (Task 1).
- Produces:
  - `quiz_engine.init_quiz_tables(cx)` — idempotent `CREATE TABLE IF NOT EXISTS quiz_responses`.
  - `quiz_engine.store_response(cx, *, session_id, quiz_id, answers: dict, email="") -> int` — upsert by `(session_id, quiz_id)`; stamps `segment`; returns row id.
  - `quiz_engine.get_response(cx, *, session_id, quiz_id) -> dict | None` — `{session_id,email,quiz_id,answers,segment,created_at,updated_at}` with `answers` already JSON-decoded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_responses.py
import sqlite3
import quiz_engine


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    quiz_engine.init_quiz_tables(cx)
    return cx


def test_init_is_idempotent(tmp_path):
    cx = _cx(tmp_path)
    quiz_engine.init_quiz_tables(cx)  # second call must not raise
    cols = [r[1] for r in cx.execute("PRAGMA table_info(quiz_responses)")]
    assert {"session_id", "email", "quiz_id", "answers_json", "segment"} <= set(cols)


def test_store_then_get_roundtrip(tmp_path):
    cx = _cx(tmp_path)
    rid = quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain",
                                     answers={"q1": "watch_wait", "q2": "frequent"})
    assert rid > 0
    got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["answers"] == {"q1": "watch_wait", "q2": "frequent"}
    assert got["segment"] == "watch_wait"
    assert got["email"] == ""


def test_store_upserts_same_session_quiz(tmp_path):
    cx = _cx(tmp_path)
    quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain", answers={"q1": "general"})
    quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain",
                               answers={"q1": "family"}, email="A@B.com")
    rows = cx.execute("SELECT COUNT(*) FROM quiz_responses WHERE session_id='s1'").fetchone()[0]
    assert rows == 1
    got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["segment"] == "family"
    assert got["email"] == "a@b.com"


def test_get_missing_returns_none(tmp_path):
    cx = _cx(tmp_path)
    assert quiz_engine.get_response(cx, session_id="nope", quiz_id="eye-brain") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_responses.py -v`
Expected: FAIL with `AttributeError: module 'quiz_engine' has no attribute 'init_quiz_tables'`.

- [ ] **Step 3: Add persistence to `quiz_engine.py`**

Append to `quiz_engine.py`:

```python
def init_quiz_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS quiz_responses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            email       TEXT DEFAULT '',
            quiz_id     TEXT NOT NULL,
            answers_json TEXT NOT NULL DEFAULT '{}',
            segment     TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_quiz_resp_session_quiz "
               "ON quiz_responses(session_id, quiz_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_quiz_resp_email ON quiz_responses(email)")
    cx.commit()


def store_response(cx, *, session_id, quiz_id, answers: dict, email="") -> int:
    cx.row_factory = sqlite3.Row
    now = _now()
    answers = answers or {}
    seg = segment_of(answers)
    email = (email or "").strip().lower()
    row = cx.execute(
        "SELECT id, email FROM quiz_responses WHERE session_id=? AND quiz_id=?",
        (session_id, quiz_id)).fetchone()
    if row is None:
        cur = cx.execute(
            "INSERT INTO quiz_responses (session_id, email, quiz_id, answers_json, "
            "segment, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (session_id, email, quiz_id, json.dumps(answers), seg, now, now))
        cx.commit()
        return cur.lastrowid
    keep_email = email or (row["email"] or "")
    cx.execute(
        "UPDATE quiz_responses SET email=?, answers_json=?, segment=?, updated_at=? WHERE id=?",
        (keep_email, json.dumps(answers), seg, now, row["id"]))
    cx.commit()
    return row["id"]


def get_response(cx, *, session_id, quiz_id) -> dict | None:
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT * FROM quiz_responses WHERE session_id=? AND quiz_id=?",
        (session_id, quiz_id)).fetchone()
    if row is None:
        return None
    return {
        "session_id": row["session_id"], "email": row["email"] or "",
        "quiz_id": row["quiz_id"], "answers": json.loads(row["answers_json"] or "{}"),
        "segment": row["segment"] or "", "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
```

- [ ] **Step 4: Wire startup init in `app.py`**

Find `app.py:6795`:

```python
def _init_journey_tables():
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)
```

Replace its body to add the quiz tables (import `quiz_engine` at the top of `app.py` alongside `import begin_funnel` if not already imported):

```python
def _init_journey_tables():
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)
        quiz_engine.init_quiz_tables(cx)
```

Add near the other top-level imports in `app.py` (where `import begin_funnel` appears):

```python
import quiz_engine
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quiz_responses.py -v`
Expected: PASS (4 tests).

Run: `python3 -c "import app"` — Expected: no ImportError (confirms `import quiz_engine` resolves and startup init runs).

- [ ] **Step 6: Commit**

```bash
git add quiz_engine.py app.py tests/test_quiz_responses.py
git commit -m "feat(quiz): quiz_responses persistence + startup table init"
```

---

### Task 3: Quiz page + quiz-data endpoint + quiz HTML

**Files:**
- Modify: `app.py` (add `GET /begin/quiz` and `GET /begin/quiz-data` near the other `/begin/*` routes, e.g. after `begin_tools` ~`app.py:1469`)
- Create: `static/begin-quiz.html`
- Test: `tests/test_quiz_page_routes.py`

**Interfaces:**
- Consumes: `quiz_engine.load_config`, `quiz_engine.get_quiz` (Task 1).
- Produces: `GET /begin/quiz` (serves page + mints `amg_session`), `GET /begin/quiz-data` (returns the active quiz config minus internal `bands`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_page_routes.py
import importlib, sys
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_quiz_page_served_and_mints_session():
    app_module = _load_app()
    c = app_module.app.test_client()
    r = c.get("/begin/quiz")
    assert r.status_code == 200
    assert "amg_session=" in r.headers.get("Set-Cookie", "")


def test_quiz_data_returns_questions_without_bands():
    app_module = _load_app()
    c = app_module.app.test_client()
    r = c.get("/begin/quiz-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["id"] == "eye-brain"
    assert len(body["questions"]) == 9
    assert "hook" in body and "disclaimer" in body
    assert "bands" not in body  # result logic stays server-side
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_page_routes.py -v`
Expected: FAIL (404 on `/begin/quiz`).

- [ ] **Step 3: Add the routes to `app.py`** (after `begin_tools`, ~line 1469)

```python
_ACTIVE_QUIZ_ID = "eye-brain"


@app.route("/begin/quiz")
def begin_quiz():
    resp = send_from_directory(STATIC, "begin-quiz.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/begin/quiz-data")
def begin_quiz_data():
    q = quiz_engine.get_quiz(_ACTIVE_QUIZ_ID)
    if not q:
        return jsonify({"error": "no_quiz"}), 404
    public = {k: v for k, v in q.items() if k != "bands"}
    return jsonify(public)
```

- [ ] **Step 4: Create `static/begin-quiz.html`**

A self-contained multi-step quiz page. It fetches `/begin/quiz-data`, renders one question at a time with a progress bar, then an email gate (name + email + ToS checkbox), POSTs answers to `/begin/quiz/answer` (Task 4) and the gate to `/begin/quiz/opt-in` (Task 5), then redirects to `/begin/quiz/result`. Model the head/style on `static/begin-explore.html`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your Foundational Eye-and-Brain Self-Assessment</title>
<style>
  :root { --ink:#1f2a37; --bg:#faf7f2; --accent:#4b6b57; --muted:#6b7280; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:Georgia,'Times New Roman',serif; color:var(--ink);
         background:var(--bg); line-height:1.5; }
  .wrap { max-width:640px; margin:0 auto; padding:32px 20px 64px; }
  .bar { height:6px; background:#e7e1d6; border-radius:6px; overflow:hidden; margin:18px 0 28px; }
  .bar > i { display:block; height:100%; width:0; background:var(--accent); transition:width .3s; }
  h1 { font-size:1.5rem; line-height:1.3; }
  .hook { color:var(--muted); font-size:1.02rem; }
  .q { font-size:1.2rem; margin:8px 0 18px; }
  button.opt { display:block; width:100%; text-align:left; background:#fff; border:1px solid #ddd;
    border-radius:10px; padding:14px 16px; margin:8px 0; font:inherit; cursor:pointer; }
  button.opt:hover { border-color:var(--accent); }
  .gate label { display:block; margin:12px 0 6px; }
  .gate input[type=text], .gate input[type=email] { width:100%; padding:12px; font:inherit;
    border:1px solid #ccc; border-radius:8px; }
  .gate .tos { font-size:.9rem; color:var(--muted); margin:14px 0; }
  .cta { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:14px 20px;
    font:inherit; cursor:pointer; width:100%; }
  .cta[disabled] { opacity:.5; cursor:not-allowed; }
  .disc { font-size:.78rem; color:var(--muted); margin-top:28px; }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <div id="intro">
    <h1 id="title"></h1>
    <p class="hook" id="hook"></p>
    <button class="cta" id="start">Start the 60-second assessment</button>
  </div>

  <div id="quiz" class="hidden">
    <div class="bar"><i id="fill"></i></div>
    <div class="q" id="prompt"></div>
    <div id="options"></div>
  </div>

  <div id="gate" class="hidden gate">
    <h1>See your result</h1>
    <p class="hook">Enter your details to see your personalized result and unlock your free guide.</p>
    <label for="name">First name</label>
    <input type="text" id="name" autocomplete="given-name">
    <label for="email">Email</label>
    <input type="email" id="email" autocomplete="email">
    <div class="tos"><label><input type="checkbox" id="tos">
      I agree to the Terms of Service and to receive related emails.</label></div>
    <button class="cta" id="reveal" disabled>Reveal my result</button>
  </div>

  <p class="disc" id="disclaimer"></p>
</div>
<script>
(function () {
  var quiz = null, idx = 0, answers = {};
  var el = function (id) { return document.getElementById(id); };

  fetch('/begin/quiz-data').then(function (r) { return r.json(); }).then(function (q) {
    quiz = q;
    el('title').textContent = q.title;
    el('hook').textContent = q.hook;
    el('disclaimer').textContent = q.disclaimer;
  });

  el('start').onclick = function () {
    el('intro').classList.add('hidden');
    el('quiz').classList.remove('hidden');
    renderQuestion();
  };

  function renderQuestion() {
    var qn = quiz.questions[idx];
    el('fill').style.width = Math.round((idx / quiz.questions.length) * 100) + '%';
    el('prompt').textContent = qn.prompt;
    var box = el('options'); box.innerHTML = '';
    qn.options.forEach(function (opt) {
      var b = document.createElement('button');
      b.className = 'opt'; b.textContent = opt.label;
      b.onclick = function () { answers[qn.id] = opt.value; next(); };
      box.appendChild(b);
    });
  }

  function next() {
    idx += 1;
    if (idx < quiz.questions.length) { renderQuestion(); return; }
    // persist answers, then show the email gate
    fetch('/begin/quiz/answer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quiz_id: quiz.id, answers: answers })
    }).finally(function () {
      el('quiz').classList.add('hidden');
      el('gate').classList.remove('hidden');
    });
  }

  function gateValid() {
    return el('name').value.trim() && /\S+@\S+\.\S+/.test(el('email').value) && el('tos').checked;
  }
  ['name', 'email', 'tos'].forEach(function (id) {
    el(id).addEventListener('input', function () { el('reveal').disabled = !gateValid(); });
    el(id).addEventListener('change', function () { el('reveal').disabled = !gateValid(); });
  });

  el('reveal').onclick = function () {
    el('reveal').disabled = true;
    fetch('/begin/quiz/opt-in', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        quiz_id: quiz.id, name: el('name').value.trim(),
        email: el('email').value.trim(), tos: el('tos').checked
      })
    }).then(function (r) { return r.json(); }).then(function () {
      window.location.href = '/begin/quiz/result';
    }).catch(function () { el('reveal').disabled = false; });
  };
})();
</script>
</body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quiz_page_routes.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add app.py static/begin-quiz.html tests/test_quiz_page_routes.py
git commit -m "feat(quiz): /begin/quiz page + quiz-data endpoint + quiz UI"
```

---

### Task 4: Answer-capture endpoint

**Files:**
- Modify: `app.py` (add `POST /begin/quiz/answer` after `begin_quiz_data`)
- Test: `tests/test_quiz_answer_route.py`

**Interfaces:**
- Consumes: `quiz_engine.store_response`, `quiz_engine.get_quiz`, `app._db_lock`, `app.LOG_DB`.
- Produces: `POST /begin/quiz/answer` — body `{quiz_id, answers}` → stores session-scoped → `{"ok": true, "segment": "..."}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_answer_route.py
import importlib, sys, sqlite3
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_answer_stores_and_returns_segment(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import quiz_engine
    with sqlite3.connect(db) as cx:
        quiz_engine.init_quiz_tables(cx)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.post("/begin/quiz/answer",
               json={"quiz_id": "eye-brain", "answers": {"q1": "watch_wait", "q2": "frequent"}})
    assert r.status_code == 200
    assert r.get_json()["segment"] == "watch_wait"
    with sqlite3.connect(db) as cx:
        got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["answers"]["q1"] == "watch_wait"


def test_answer_unknown_quiz_404(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.post("/begin/quiz/answer", json={"quiz_id": "nope", "answers": {}})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_answer_route.py -v`
Expected: FAIL (404 / route missing).

- [ ] **Step 3: Add the route to `app.py`**

```python
@app.route("/begin/quiz/answer", methods=["POST", "OPTIONS"])
def begin_quiz_answer():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    quiz_id = (data.get("quiz_id") or "").strip()
    answers = data.get("answers") or {}
    if not quiz_engine.get_quiz(quiz_id):
        return jsonify({"error": "unknown_quiz"}), 404
    session_id = (request.cookies.get("amg_session")
                  or (data.get("session_id") or "").strip() or uuid.uuid4().hex)
    if not isinstance(answers, dict):
        return jsonify({"error": "bad_answers"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        quiz_engine.store_response(cx, session_id=session_id, quiz_id=quiz_id, answers=answers)
    resp = jsonify({"ok": True, "segment": quiz_engine.segment_of(answers)})
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quiz_answer_route.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_quiz_answer_route.py
git commit -m "feat(quiz): POST /begin/quiz/answer session-scoped capture"
```

---

### Task 5: Opt-in endpoint (email capture + GHL + guide token)

**Files:**
- Modify: `app.py` (add `POST /begin/quiz/opt-in` after `begin_quiz_answer`; add a `_mint_lead_magnet_guide_link` helper near `_mint_membership_magic_link` ~`app.py:6934`)
- Test: `tests/test_quiz_optin_route.py`

**Interfaces:**
- Consumes: `begin_funnel.record_unlock` (email+tos→free_tier), `app.ghl_onboard_contact`, `quiz_engine.store_response`/`get_response`, `app._hash_token`, `auth_tokens` table.
- Produces:
  - `POST /begin/quiz/opt-in` — body `{quiz_id, name, email, tos}` → records email+tos (free_tier) and the `quiz` gate, attaches email to the quiz_response, fires GHL onboard with lead-magnet + segment tags (background, non-blocking), mints a guide token → `{"ok": true, "current_rung": "...", "guide_token": "...", "redirect": "/begin/quiz/result"}`.
  - `app._mint_lead_magnet_guide_link(email, ttl_min=43200) -> str` — plaintext token, purpose `lead_magnet_guide`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_optin_route.py
import importlib, sys, sqlite3, threading, time
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _setup_db(app_module, tmp_path, monkeypatch):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel, quiz_engine
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        quiz_engine.init_quiz_tables(cx)
    return db


def test_optin_reaches_free_tier_tags_and_mints_guide(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _setup_db(app_module, tmp_path, monkeypatch)
    captured = {}
    done = threading.Event()

    def fake_onboard(email, first="", last="", **kw):
        captured["email"] = email
        captured["tags"] = set(kw.get("extra_tags") or [])
        captured["source_tag"] = kw.get("source_tag")
        done.set()
        return {"contact_id": "x"}

    monkeypatch.setattr(app_module, "ghl_onboard_contact", fake_onboard)
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)

    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    # answer first so segment is known
    c.post("/begin/quiz/answer",
           json={"quiz_id": "eye-brain", "answers": {"q1": "watch_wait"}})
    r = c.post("/begin/quiz/opt-in",
               json={"quiz_id": "eye-brain", "name": "Ada",
                     "email": "ada@example.com", "tos": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["current_rung"] == "free_tier"
    assert body["redirect"] == "/begin/quiz/result"
    assert body["guide_token"]
    # email now attached to the quiz response
    import quiz_engine
    with sqlite3.connect(db) as cx:
        got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["email"] == "ada@example.com"
    # GHL onboard fired with the lead-magnet + segment tags
    assert done.wait(2.0)
    assert captured["email"] == "ada@example.com"
    assert "lead-magnet" in captured["tags"]
    assert "quiz-completed" in captured["tags"]
    assert "awareness:watch_wait" in captured["tags"]
    # guide token validates
    assert app_module._validate_lead_magnet_guide_link(body["guide_token"]) == "ada@example.com"


def test_optin_requires_email_and_tos(monkeypatch, tmp_path):
    app_module = _load_app()
    _setup_db(app_module, tmp_path, monkeypatch)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s2")
    r = c.post("/begin/quiz/opt-in",
               json={"quiz_id": "eye-brain", "name": "Bo", "email": "", "tos": True})
    assert r.status_code == 400
    r2 = c.post("/begin/quiz/opt-in",
                json={"quiz_id": "eye-brain", "name": "Bo", "email": "bo@x.com", "tos": False})
    assert r2.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_optin_route.py -v`
Expected: FAIL (route + helpers missing).

- [ ] **Step 3: Add the mint/validate helpers to `app.py`** (after `_validate_membership_magic_link`, ~line 6990)

```python
def _mint_lead_magnet_guide_link(email, ttl_min=60 * 24 * 30):
    """Single-use token (purpose lead_magnet_guide, 30-day TTL) for the free-guide
    download. Returns plaintext token; caller emails/returns it."""
    import secrets, json as _json
    plain = secrets.token_urlsafe(32)
    th = _hash_token(plain)
    now_iso = datetime.utcnow().isoformat() + "Z"
    exp_iso = (datetime.utcnow() + timedelta(minutes=int(ttl_min))).isoformat() + "Z"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (th, email, "lead_magnet_guide", _json.dumps({}), now_iso, exp_iso))
    return plain


def _validate_lead_magnet_guide_link(token):
    """Return the email for a valid (not consumed, not expired) lead_magnet_guide
    token, else None. Does not mark consumed (the guide is re-downloadable)."""
    if not token:
        return None
    th = _hash_token(token)
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='lead_magnet_guide'", (th,)).fetchone()
    if not row:
        return None
    email, expires_at, consumed_at = row
    if consumed_at:
        return None
    try:
        if datetime.fromisoformat(expires_at.rstrip("Z")) < datetime.utcnow():
            return None
    except Exception:
        return None
    return email
```

- [ ] **Step 4: Add the opt-in route to `app.py`** (after `begin_quiz_answer`)

```python
@app.route("/begin/quiz/opt-in", methods=["POST", "OPTIONS"])
def begin_quiz_optin():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    quiz_id = (data.get("quiz_id") or "").strip()
    if not quiz_engine.get_quiz(quiz_id):
        return jsonify({"error": "unknown_quiz"}), 404
    name = (data.get("name") or "").strip()
    first_name = name.split(None, 1)[0] if name else ""
    email = (data.get("email") or "").strip().lower()
    tos = bool(data.get("tos"))
    if not email or "@" not in email:
        return jsonify({"error": "valid email required"}), 400
    if not tos:
        return jsonify({"error": "tos required"}), 400
    session_id = (request.cookies.get("amg_session")
                  or (data.get("session_id") or "").strip() or uuid.uuid4().hex)
    ref_slug = (request.cookies.get("rm_ref") or (data.get("ref") or "")).strip()

    # capture email + ToS (free_tier) and mark the assessment gate
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        begin_funnel.record_unlock(
            cx, session_id=session_id, trigger="tos", email=email,
            first_name=first_name, tos=True, ref_slug=ref_slug,
            tos_version=BEGIN_TOS_VERSION)
        state = begin_funnel.record_unlock(
            cx, session_id=session_id, trigger="quiz", email=email,
            detail=f"quiz:{quiz_id}", ref_slug=ref_slug)
        seg = ""
        existing = quiz_engine.get_response(cx, session_id=session_id, quiz_id=quiz_id)
        if existing:
            seg = existing["segment"]
            quiz_engine.store_response(cx, session_id=session_id, quiz_id=quiz_id,
                                       answers=existing["answers"], email=email)

    # GHL onboarding + lead-magnet/segment tags, non-blocking (same pattern as /begin/unlock)
    import threading as _threading

    def _onboard():
        try:
            tags = ["begin", "lead-magnet", "quiz-completed"]
            if seg:
                tags.append(f"awareness:{seg}")
            if ref_slug:
                tags.append(f"ref:{ref_slug}")
                _capture_concierge_referral(email, first_name, "", ref_slug)
            ghl_onboard_contact(email, first_name, "", source_tag="lead-magnet", extra_tags=tags)
        except Exception as e:
            print(f"[quiz-optin] {e!r}", flush=True)

    _threading.Thread(target=_onboard, daemon=True).start()

    guide_token = _mint_lead_magnet_guide_link(email)
    resp = jsonify({"ok": True, "current_rung": state["current_rung"],
                    "guide_token": guide_token, "redirect": "/begin/quiz/result"})
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quiz_optin_route.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_quiz_optin_route.py
git commit -m "feat(quiz): opt-in captures email+ToS, tags GHL, mints guide token"
```

---

### Task 6: Result page + result-data endpoint (founding card gated)

**Files:**
- Modify: `app.py` (add `GET /begin/quiz/result` + `GET /begin/quiz/result-data` after the opt-in route)
- Create: `static/begin-quiz-result.html`
- Test: `tests/test_quiz_result_route.py`

**Interfaces:**
- Consumes: `quiz_engine.get_quiz`/`get_response`/`result_for`, `dashboard.founding.is_open`/`get_launch`/`remaining`, `app.LOG_DB`, `app._now_utc`.
- Produces:
  - `GET /begin/quiz/result` — serves `static/begin-quiz-result.html`.
  - `GET /begin/quiz/result-data` — `{profile:{band,headline,reasoning,bullets,segment}, disclaimer, founding:{batch_label,cap,remaining,personal_line}|null, product_url, taken:bool}`. `founding` is non-null only when `founding.is_open` for the product slug.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_result_route.py
import importlib, sys, sqlite3
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _seed(app_module, tmp_path, monkeypatch, answers):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import quiz_engine
    with sqlite3.connect(db) as cx:
        quiz_engine.init_quiz_tables(cx)
        quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain", answers=answers)
    return db


def test_result_page_served():
    app_module = _load_app()
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/result")
    assert r.status_code == 200


def test_result_data_open_founding_includes_card(monkeypatch, tmp_path):
    app_module = _load_app()
    _seed(app_module, tmp_path, monkeypatch, {"q1": "watch_wait", "q8": "eye_formula"})
    import dashboard.founding as founding
    monkeypatch.setattr(founding, "get_launch",
                        lambda s: {"cap": 2500, "batch_label": "Founding Batch No. 1", "closes_at": ""})
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: True)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 1900)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.get("/begin/quiz/result-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["taken"] is True
    assert body["profile"]["band"] == "barrier"
    assert "not been evaluated" in body["disclaimer"]
    assert body["founding"]["remaining"] == 1900
    assert body["product_url"] == "/begin/product/neuro-magnesium"
    assert body["founding"]["personal_line"]


def test_result_data_closed_founding_is_null(monkeypatch, tmp_path):
    app_module = _load_app()
    _seed(app_module, tmp_path, monkeypatch, {"q1": "general"})
    import dashboard.founding as founding
    monkeypatch.setattr(founding, "get_launch",
                        lambda s: {"cap": 2500, "batch_label": "x", "closes_at": ""})
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: False)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 0)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.get("/begin/quiz/result-data")
    body = r.get_json()
    assert body["founding"] is None


def test_result_data_no_answers_taken_false(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import quiz_engine
    with sqlite3.connect(db) as cx:
        quiz_engine.init_quiz_tables(cx)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "nobody")
    r = c.get("/begin/quiz/result-data")
    assert r.status_code == 200
    assert r.get_json()["taken"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_result_route.py -v`
Expected: FAIL (routes missing).

- [ ] **Step 3: Add the routes to `app.py`**

```python
# personalized founding one-liner per result band (structure-function only)
_QUIZ_FOUNDING_LINES = {
    "barrier": "You're a strong fit for the founding batch: a magnesium formulated to reach where ordinary magnesium can't.",
    "calm": "You're a strong fit for the founding batch: calm, steady support without the fog.",
    "clarity": "You're a strong fit for the founding batch: foundational support for a clear, steady mind.",
    "hardworking": "You're a strong fit for the founding batch: foundational support for eyes that work hard.",
    "foundational": "You're a strong fit for the founding batch: foundational eye-and-brain support.",
}


@app.route("/begin/quiz/result")
def begin_quiz_result():
    resp = send_from_directory(STATIC, "begin-quiz-result.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/begin/quiz/result-data")
def begin_quiz_result_data():
    quiz_id = (request.args.get("quiz_id") or "eye-brain").strip()
    quiz = quiz_engine.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "unknown_quiz"}), 404
    session_id = (request.cookies.get("amg_session") or "").strip()
    resp_row = None
    if session_id:
        with sqlite3.connect(LOG_DB) as cx:
            resp_row = quiz_engine.get_response(cx, session_id=session_id, quiz_id=quiz_id)
    if not resp_row:
        return jsonify({"taken": False, "disclaimer": quiz.get("disclaimer", "")})
    profile = quiz_engine.result_for(quiz, resp_row["answers"])
    slug = quiz["product_slug"]
    founding = None
    try:
        from dashboard import founding as _founding
        launch = _founding.get_launch(slug)
        if launch:
            today = _now_utc().strftime("%Y-%m-%d")
            with sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                if _founding.is_open(cx, slug, now_iso=today):
                    founding = {
                        "batch_label": launch.get("batch_label", ""),
                        "cap": int(launch.get("cap", 0)),
                        "remaining": _founding.remaining(cx, slug),
                        "personal_line": _QUIZ_FOUNDING_LINES.get(
                            profile["band"], _QUIZ_FOUNDING_LINES["foundational"]),
                    }
    except Exception as e:
        print(f"[quiz-result] founding enrich failed: {e!r}", flush=True)
    return jsonify({
        "taken": True, "profile": profile, "disclaimer": quiz.get("disclaimer", ""),
        "founding": founding, "product_url": f"/begin/product/{slug}",
    })
```

- [ ] **Step 4: Create `static/begin-quiz-result.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your Result</title>
<style>
  :root { --ink:#1f2a37; --bg:#faf7f2; --accent:#4b6b57; --muted:#6b7280; }
  body { margin:0; font-family:Georgia,'Times New Roman',serif; color:var(--ink);
         background:var(--bg); line-height:1.55; }
  .wrap { max-width:640px; margin:0 auto; padding:36px 20px 72px; }
  h1 { font-size:1.6rem; line-height:1.3; }
  .reason { font-size:1.08rem; }
  ul { padding-left:1.1rem; } li { margin:8px 0; }
  .card { background:#fff; border:1px solid #e2dccf; border-radius:14px; padding:22px; margin:26px 0; }
  .count { color:var(--accent); font-weight:bold; }
  .cta { display:inline-block; background:var(--accent); color:#fff; text-decoration:none;
    border-radius:10px; padding:14px 22px; margin-top:10px; }
  .guide a { color:var(--accent); }
  .disc { font-size:.78rem; color:var(--muted); margin-top:30px; }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap" id="root">
  <div id="empty" class="hidden">
    <h1>Let's find your result</h1>
    <p class="reason">It looks like you haven't taken the assessment yet.</p>
    <a class="cta" href="/begin/quiz">Take the 60-second assessment</a>
  </div>
  <div id="content" class="hidden">
    <h1 id="headline"></h1>
    <p class="reason" id="reasoning"></p>
    <ul id="bullets"></ul>
    <div class="card hidden" id="founding">
      <div id="personal"></div>
      <p><span class="count" id="count"></span> <span id="batch"></span></p>
      <a class="cta" id="reserve" href="#">Reserve your founding bottle</a>
    </div>
    <p class="guide" id="guide"></p>
  </div>
  <p class="disc" id="disclaimer"></p>
</div>
<script>
(function () {
  var el = function (id) { return document.getElementById(id); };
  fetch('/begin/quiz/result-data').then(function (r) { return r.json(); }).then(function (d) {
    el('disclaimer').textContent = d.disclaimer || '';
    if (!d.taken) { el('empty').classList.remove('hidden'); return; }
    el('content').classList.remove('hidden');
    el('headline').textContent = d.profile.headline;
    el('reasoning').textContent = d.profile.reasoning;
    (d.profile.bullets || []).forEach(function (b) {
      var li = document.createElement('li'); li.textContent = b; el('bullets').appendChild(li);
    });
    if (d.founding) {
      el('founding').classList.remove('hidden');
      el('personal').textContent = d.founding.personal_line;
      el('count').textContent = d.founding.remaining + ' of ' + d.founding.cap + ' remaining';
      el('batch').textContent = d.founding.batch_label;
      el('reserve').href = d.product_url;
    }
    // free guide download (token from the opt-in is held in sessionStorage by the quiz page)
    var t = window.sessionStorage.getItem('lm_guide_token');
    if (t) {
      el('guide').innerHTML = 'Your free guide is ready. ' +
        '<a href="/begin/quiz/guide?token=' + encodeURIComponent(t) + '">Download it here.</a>';
    }
  });
})();
</script>
</body>
</html>
```

Also update `static/begin-quiz.html` Step-5 reveal handler to stash the returned token so the result page can offer the download. In the `.then(function () { window.location.href = ... })` block, replace with:

```javascript
    }).then(function (r) { return r.json(); }).then(function (resp) {
      if (resp && resp.guide_token) {
        window.sessionStorage.setItem('lm_guide_token', resp.guide_token);
      }
      window.location.href = '/begin/quiz/result';
    }).catch(function () { el('reveal').disabled = false; });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quiz_result_route.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add app.py static/begin-quiz-result.html static/begin-quiz.html tests/test_quiz_result_route.py
git commit -m "feat(quiz): result page + result-data with gated founding card"
```

---

### Task 7: Free-guide delivery (R2, graceful degrade)

**Files:**
- Modify: `app.py` (add `GET /begin/quiz/guide` after the result routes)
- Test: `tests/test_quiz_guide_route.py`

**Interfaces:**
- Consumes: `app._validate_lead_magnet_guide_link` (Task 5), `app._r2()`, env `LEAD_MAGNET_PDF_KEY`, `R2_BUCKET`.
- Produces: `GET /begin/quiz/guide?token=...` — streams the PDF from R2 when both the token is valid and `LEAD_MAGNET_PDF_KEY` is set; otherwise returns a friendly 200 "guide coming" HTML (never 500). Invalid/expired token → 200 friendly "link expired" HTML (no enumeration).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quiz_guide_route.py
import importlib, sys
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_guide_no_pdf_key_degrades_gracefully(monkeypatch):
    app_module = _load_app()
    monkeypatch.delenv("LEAD_MAGNET_PDF_KEY", raising=False)
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: "a@b.com")
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/guide?token=valid")
    assert r.status_code == 200
    assert b"coming" in r.data.lower() or b"on its way" in r.data.lower()


def test_guide_invalid_token_friendly(monkeypatch):
    app_module = _load_app()
    monkeypatch.setenv("LEAD_MAGNET_PDF_KEY", "guides/eye-brain.pdf")
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: None)
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/guide?token=bogus")
    assert r.status_code == 200
    assert b"expired" in r.data.lower() or b"fresh" in r.data.lower()


def test_guide_valid_streams_pdf(monkeypatch):
    app_module = _load_app()
    monkeypatch.setenv("LEAD_MAGNET_PDF_KEY", "guides/eye-brain.pdf")
    monkeypatch.setenv("R2_BUCKET", "rm-clips")
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: "a@b.com")

    class _Body:
        def iter_chunks(self, chunk_size=65536):
            yield b"%PDF-1.4 fake"

    class _R2:
        def get_object(self, **kw):
            assert kw["Key"] == "guides/eye-brain.pdf"
            return {"Body": _Body(), "ContentType": "application/pdf", "ContentLength": 13}

    monkeypatch.setattr(app_module, "_r2", lambda: _R2())
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/guide?token=valid")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "application/pdf"
    assert b"%PDF" in r.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quiz_guide_route.py -v`
Expected: FAIL (route missing).

- [ ] **Step 3: Add the route to `app.py`**

```python
_GUIDE_PENDING_HTML = (
    "<!doctype html><meta charset=utf-8><title>Your guide</title>"
    "<div style='font-family:Georgia,serif;max-width:560px;margin:60px auto;padding:0 20px;color:#1f2a37'>"
    "<h1>Your free guide is on its way</h1>"
    "<p>Thank you. Your guide is being finalized and is coming shortly. "
    "We'll email it to you as soon as it's ready.</p>"
    "<p><a href='/begin/quiz/result' style='color:#4b6b57'>Back to your result</a></p></div>")

_GUIDE_EXPIRED_HTML = (
    "<!doctype html><meta charset=utf-8><title>Link expired</title>"
    "<div style='font-family:Georgia,serif;max-width:560px;margin:60px auto;padding:0 20px;color:#1f2a37'>"
    "<h1>That link has expired</h1>"
    "<p>For your security, guide links expire. "
    "<a href='/begin/quiz' style='color:#4b6b57'>Take the assessment again</a> to get a fresh link.</p></div>")


@app.route("/begin/quiz/guide")
def begin_quiz_guide():
    token = (request.args.get("token") or "").strip()
    email = _validate_lead_magnet_guide_link(token)
    if not email:
        return Response(_GUIDE_EXPIRED_HTML, mimetype="text/html")
    key = (os.environ.get("LEAD_MAGNET_PDF_KEY") or "").strip()
    if not key:
        return Response(_GUIDE_PENDING_HTML, mimetype="text/html")
    try:
        obj = _r2().get_object(Bucket=os.environ.get("R2_BUCKET", "rm-clips"), Key=key)
    except Exception as e:
        print(f"[quiz-guide] r2 fetch failed key={key}: {e!r}", flush=True)
        return Response(_GUIDE_PENDING_HTML, mimetype="text/html")
    headers = {
        "Content-Type": obj.get("ContentType", "application/pdf"),
        "Content-Disposition": 'inline; filename="foundational-eye-and-brain-guide.pdf"',
        "Cache-Control": "private, max-age=0, no-store",
    }
    if obj.get("ContentLength") is not None:
        headers["Content-Length"] = str(obj["ContentLength"])
    return Response(obj["Body"].iter_chunks(chunk_size=65536), status=200, headers=headers)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quiz_guide_route.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_quiz_guide_route.py
git commit -m "feat(quiz): R2 guide download with graceful degrade + token gate"
```

---

### Task 8: Founding-offer card in `surface()` (gated)

**Files:**
- Modify: `begin_funnel.py` (add a `founding_offer` entry to `CARD_CATALOG`; add a `surface_with_founding()` wrapper that prepends the founding card when open)
- Modify: `app.py` (in `begin_unlock`, use the founding-aware surface so a logged signal that's product-aware shows the founding card when the launch is open)
- Test: `tests/test_begin_founding_card.py`

**Interfaces:**
- Consumes: `begin_funnel.CARD_CATALOG`, `begin_funnel.surface`.
- Produces:
  - `begin_funnel.CARD_CATALOG["founding_offer"]` — `{title, sub, base_url: "/begin/product/neuro-magnesium", internal: True}`.
  - `begin_funnel.surface_with_founding(state, query_texts, ref="", founding_open=False) -> list` — same as `surface()` but, when `founding_open`, the founding card is prepended (deduped, list still capped at 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_begin_founding_card.py
import begin_funnel


def test_founding_card_in_catalog():
    c = begin_funnel.CARD_CATALOG["founding_offer"]
    assert c["base_url"] == "/begin/product/neuro-magnesium"
    assert c["internal"] is True


def test_surface_with_founding_prepends_when_open():
    state = {"awareness_stage": "product", "current_rung": "assess", "unlocked_gates": ["quiz"]}
    cards = begin_funnel.surface_with_founding(state, ["neuro magnesium"], ref="", founding_open=True)
    assert cards[0]["key"] == "founding_offer"
    assert len(cards) <= 3


def test_surface_with_founding_absent_when_closed():
    state = {"awareness_stage": "product", "current_rung": "assess", "unlocked_gates": ["quiz"]}
    cards = begin_funnel.surface_with_founding(state, ["neuro magnesium"], ref="", founding_open=False)
    assert all(c["key"] != "founding_offer" for c in cards)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_begin_founding_card.py -v`
Expected: FAIL (`KeyError: 'founding_offer'`).

- [ ] **Step 3: Edit `begin_funnel.py`**

Add to `CARD_CATALOG` (after the `"product"` entry, ~line 403):

```python
    "founding_offer":     {"title": "Neuro Magnesium — Founding Batch",
                           "sub": "Reserve your bottle from the first founding batch",
                           "base_url": "/begin/product/neuro-magnesium", "internal": True},
```

Add after `surface()` (~line 723):

```python
def surface_with_founding(state, query_texts, ref="", founding_open=False):
    """surface() plus the founding-offer card prepended when a launch is open.
    Deduped and capped at 3. When founding_open is False this equals surface()."""
    cards = surface(state, query_texts, ref)
    if not founding_open:
        return cards
    cards = [c for c in cards if c["key"] != "founding_offer"]
    return ([_card("founding_offer", ref)] + cards)[:3]
```

- [ ] **Step 4: Wire it in `app.py` `begin_unlock`** (replace the `payload["surfaced_cards"] = ...` line, ~`app.py:2224`)

```python
    _founding_open = False
    try:
        from dashboard import founding as _founding
        if _founding.get_launch("neuro-magnesium"):
            with sqlite3.connect(LOG_DB) as _fcx:
                _fcx.row_factory = sqlite3.Row
                _founding_open = _founding.is_open(
                    _fcx, "neuro-magnesium", now_iso=_now_utc().strftime("%Y-%m-%d"))
    except Exception:
        _founding_open = False
    payload["surfaced_cards"] = begin_funnel.surface_with_founding(
        state, query_texts, ref_slug, founding_open=_founding_open)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_begin_founding_card.py tests/test_begin_funnel.py -v`
Expected: PASS (new 3 + existing surface tests still green).

- [ ] **Step 6: Commit**

```bash
git add begin_funnel.py app.py tests/test_begin_founding_card.py
git commit -m "feat(quiz): founding-offer card surfaces in funnel when launch open"
```

---

### Task 9: Compliance copy audit test + full-suite green

**Files:**
- Create: `tests/test_quiz_compliance.py`
- (No code changes expected; if the test fails, fix the offending copy in `data/quizzes.json` / `app.py` result lines / static HTML.)

**Interfaces:**
- Consumes: `quiz_engine` config + the result band copy.

- [ ] **Step 1: Write the compliance test**

```python
# tests/test_quiz_compliance.py
import json
from pathlib import Path
import quiz_engine

_BANNED_DISEASE = ["macular", "amd", "glaucoma", "cataract", "alzheimer", "dementia",
                   "cure", "treat ", "prevent ", "reverse "]
_EMDASH = "—"


def _all_quiz_text():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    parts = [q["title"], q["hook"], q["disclaimer"]]
    for qq in q["questions"]:
        parts.append(qq["prompt"])
        parts += [o["label"] for o in qq["options"]]
    for band in q["bands"].values():
        parts += [band["headline"], band["reasoning"]] + band["bullets"]
    return parts


def test_no_disease_claims_in_quiz_copy():
    for s in _all_quiz_text():
        low = s.lower()
        for b in _BANNED_DISEASE:
            assert b not in low, f"banned term {b!r} in: {s!r}"


def test_no_emdash_in_quiz_copy():
    for s in _all_quiz_text():
        assert _EMDASH not in s, f"em-dash in: {s!r}"


def test_disclaimer_is_dshea():
    q = quiz_engine.get_quiz("eye-brain")
    assert "not been evaluated by the Food and Drug Administration" in q["disclaimer"]
    assert "not intended to diagnose, treat, cure, or prevent any disease" in q["disclaimer"]


def test_static_result_and_quiz_pages_no_emdash():
    root = Path(__file__).resolve().parent.parent
    for name in ("begin-quiz.html", "begin-quiz-result.html"):
        txt = (root / "static" / name).read_text()
        assert _EMDASH not in txt, f"em-dash in static/{name}"
```

- [ ] **Step 2: Run the compliance test**

Run: `python3 -m pytest tests/test_quiz_compliance.py -v`
Expected: PASS (4 tests). If any fail, fix the copy (not the test) and re-run.

- [ ] **Step 3: Run the full quiz + funnel + founding suites**

Run: `python3 -m pytest tests/test_quiz_engine.py tests/test_quiz_responses.py tests/test_quiz_page_routes.py tests/test_quiz_answer_route.py tests/test_quiz_optin_route.py tests/test_quiz_result_route.py tests/test_quiz_guide_route.py tests/test_quiz_compliance.py tests/test_begin_funnel.py tests/test_begin_routes.py tests/test_founding_counter_api.py -v`
Expected: all PASS (no regressions in begin_funnel / begin routes / founding).

- [ ] **Step 4: Commit**

```bash
git add tests/test_quiz_compliance.py
git commit -m "test(quiz): compliance audit — no disease claims, no em-dash, DSHEA disclaimer"
```

---

## Post-implementation (not code — for the launch checklist)

- **Build the guide PDF** (parallel BNSN task): combine Glen's existing "bigger picture / what comes after the remedy" book with a new product/ingredient front section (Part 1), upload to R2, then set `LEAD_MAGNET_PDF_KEY` (and confirm `R2_ENDPOINT` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` in Doppler `remedy-match/prd`). Until then the guide route degrades gracefully.
- **Confirm `FOUNDING_LAUNCH_ENABLED`** is set for go-live so the reserve flow accepts the quiz traffic.
- **GHL nurture**: build the lead-magnet nurture workflow in GHL UI keyed off the `lead-magnet` / `awareness:*` tags this funnel writes.
- Pieces 2–5 of the acquisition engine (host-beneficiary, webinar, community, email) are separate specs that drive traffic TO `/begin/quiz`.

## Self-Review notes
- **Spec coverage:** quiz engine (T1), data model `quiz_responses` + `data/quizzes.json` (T1/T2), all 7 endpoints (T3–T7), founding card in surface() (T8), compliance (baked into copy + asserted T1/T9), guide graceful-degrade (T7), reuse of `/begin/unlock`+GHL+magic-link+R2 (T2/T5/T7), repeatability via config + `is_open` gate (T1/T6/T8). Success criteria in §9 all map to T5/T6/T7 tests.
- **Deferred decisions resolved:** opt-in sets ToS/free-tier (full member) via `record_unlock(trigger="tos")` so the existing GHL onboard pattern is reused; guide delivery is magic-link-token-gated download (re-downloadable, not consumed); quiz is single-page client-stepped (one `/answer` write at the gate).
- **Type consistency:** `result_for` returns `{band,headline,reasoning,bullets,segment,depletion}` consumed identically in T6; `get_response` returns decoded `answers` used in T5/T6; founding dict shape `{batch_label,cap,remaining,personal_line}` matches T6 test + result HTML.
