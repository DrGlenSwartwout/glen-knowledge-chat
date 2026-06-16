# Certification Work-Product Submission Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let certification students submit published work through a portal we control, review it behind two gates (approve → publish), track each student against the certification completion rules, and feed published work into the `case-studies` proof library.

**Architecture:** Two pure modules (`cert_rules.py` = the rules; `cert_submissions.py` = a sqlite store mirroring `dashboard/cert_bonus.py`). Flask routes in `app.py` add a magic-link student portal (mirror the `/reorder` pattern), console-gated review endpoints (mirror `/api/cert/student`), and a publish step that upserts into the Pinecone `case-studies` namespace (read today by `surface_case_study_cards`). Two static HTML pages render the student form and the console review board. All student-facing routes gate behind a new `CERT_PORTAL_ENABLED` env flag (ships dark).

**Tech Stack:** Python 3.11, Flask, sqlite3 (`chat_log.db` at `LOG_DB`), Supabase (practitioners table via `db_supabase.supabase_cursor`), Pinecone (`_idx`), OpenAI embeddings (`embed()`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-cert-work-product-pipeline-design.md`
**Program rules source:** `~/AI-Training/00 Projects/certification/cert-work-product-framework.md`

**Branch:** `sess/b76661d9-cert-submissions` (worktree `/tmp/wt-deploy-chat-b76661d9`).

**Test invocation:**
- Pure modules (Tasks 1-2): `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_rules.py tests/test_cert_submissions.py -v`
- Route tests (Tasks 3-7): `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -v`
- Full new suite: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_rules.py tests/test_cert_submissions.py tests/test_cert_portal_routes.py -v`
- Ignore the 2 known pre-existing failures (`test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`).

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard/cert_rules.py` (new) | Pure: the 12 modules, the format catalog, `evaluate(submissions)` → progress vs the completion rules. The single place the rules live. |
| `dashboard/cert_submissions.py` (new) | Pure sqlite store: CRUD + status transitions for `cert_submissions`. |
| `app.py` (modify) | Flag helper, student magic-link auth + portal/submit/upload/mine routes, console review routes, publish-to-case-studies helper, `modules_completed` sync, page routes. |
| `static/cert-portal.html` (new) | Student UI: login, submission form (12-module / format / "other" checkboxes + permission), my-submissions + progress readout. |
| `static/console-cert.html` (new) | Console review board: list submissions, approve/return/publish, per-student progress. |
| `tests/test_cert_rules.py` (new) | Unit tests for `evaluate`. |
| `tests/test_cert_submissions.py` (new) | Unit tests for the store. |
| `tests/test_cert_portal_routes.py` (new) | Route tests (auth, submit, review, publish). |

---

## Task 1: `cert_rules.py` — the rules (pure)

**Files:**
- Create: `dashboard/cert_rules.py`
- Test: `tests/test_cert_rules.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cert_rules.py
from dashboard import cert_rules as cr


def _sub(modules, formats):
    return {"credited_modules": modules, "formats": formats}


def test_catalog_shape():
    assert len(cr.MODULES) == 12
    assert [m["id"] for m in cr.MODULES] == list(range(1, 13))
    assert cr.MODULES[0]["label"] == "Body"
    # every format has a kind we can test the written+video rule on
    kinds = {f["kind"] for f in cr.FORMATS}
    assert {"written", "video"} <= kinds
    assert cr.MIN_SUBMISSIONS == 12


def test_empty_is_incomplete():
    r = cr.evaluate([])
    assert r["complete"] is False
    assert r["approved_count"] == 0
    assert r["modules_covered"] == set()
    assert len(r["modules_missing"]) == 12
    assert r["has_written"] is False and r["has_video"] is False


def test_one_submission_can_cover_multiple_modules():
    r = cr.evaluate([_sub([1, 2, 3], ["case_report"])])
    assert r["modules_covered"] == {1, 2, 3}
    assert r["has_written"] is True and r["has_video"] is False
    assert r["complete"] is False  # <12 submissions, missing modules, no video


def test_written_and_video_detected_by_kind():
    subs = [_sub([1], ["article"]), _sub([2], ["talking_head_scripted"])]
    r = cr.evaluate(subs)
    assert r["has_written"] is True
    assert r["has_video"] is True
    assert r["multi_modality"] is True


def test_complete_when_all_rules_met():
    # 12 submissions, all 12 modules covered, both written + video present
    subs = [_sub([i], ["article"]) for i in range(1, 13)]
    subs[0]["formats"] = ["article", "talking_head_unscripted"]  # add a video
    r = cr.evaluate(subs)
    assert r["approved_count"] == 12
    assert r["modules_missing"] == []
    assert r["has_written"] and r["has_video"]
    assert r["complete"] is True
    assert r["reasons"] == []


def test_reasons_list_unmet_rules():
    r = cr.evaluate([_sub([1], ["article"])])
    joined = " ".join(r["reasons"]).lower()
    assert "12" in joined          # needs >=12
    assert "module" in joined      # missing modules
    assert "video" in joined       # no video yet
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.cert_rules'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/cert_rules.py
"""Certification work-product rules (pure: no I/O, no Flask).

The single place the certification completion rules live, so "rules subject to
change as the program evolves" stays a one-file edit.

Source of truth: ~/AI-Training/00 Projects/certification/cert-work-product-framework.md
"""

# The 12 module concepts (verbatim labels), id 1..12.
MODULES = [
    {"id": 1,  "label": "Body"},
    {"id": 2,  "label": "Mind"},
    {"id": 3,  "label": "Spirit"},
    {"id": 4,  "label": "Family inheritance / EVOX transgenerational perception reframing"},
    {"id": 5,  "label": "Personal history / retracing"},
    {"id": 6,  "label": "Epigenetics / EVOX / Infoceuticals"},
    {"id": 7,  "label": "Symptoms / embryological tissue layers"},
    {"id": 8,  "label": "Terrain phases"},
    {"id": 9,  "label": "Diagnostic category"},
    {"id": 10, "label": "Therapeutic hierarchy"},
    {"id": 11, "label": "Regulatory response"},
    {"id": 12, "label": "Prognosis / self-fulfilling prophecy / faith / belief"},
]

# Format catalog. `kind` groups formats so the written+video rule is testable.
# kind in {"written","video","audio","visual"}.
FORMATS = [
    {"key": "talking_head_scripted",   "label": "Talking-head video (scripted)",        "kind": "video"},
    {"key": "talking_head_unscripted", "label": "Talking-head video (unscripted)",      "kind": "video"},
    {"key": "slideshow_video",         "label": "Slideshow / screen-share video",       "kind": "video"},
    {"key": "interview_guest",         "label": "Video interview (being interviewed)",  "kind": "video"},
    {"key": "interview_host",          "label": "Video interview (interviewing someone)", "kind": "video"},
    {"key": "demo_video",              "label": "Demonstration / walkthrough video",    "kind": "video"},
    {"key": "webinar",                 "label": "Webinar / workshop recording",         "kind": "video"},
    {"key": "short_form_video",        "label": "Short-form social video",              "kind": "video"},
    {"key": "written_post",            "label": "Written post (social / blog)",         "kind": "written"},
    {"key": "article",                 "label": "Article / feature piece",              "kind": "written"},
    {"key": "white_paper",             "label": "White paper / longer paper",           "kind": "written"},
    {"key": "case_report",             "label": "Case report (single case)",            "kind": "written"},
    {"key": "study_case_control",      "label": "Study — case-control",                 "kind": "written"},
    {"key": "study_observational",     "label": "Study — group observational",          "kind": "written"},
    {"key": "study_controlled",        "label": "Study — controlled group",             "kind": "written"},
    {"key": "literature_review",       "label": "Literature review / synthesis",        "kind": "written"},
    {"key": "protocol_writeup",        "label": "Protocol / program design write-up",   "kind": "written"},
    {"key": "book_chapter",            "label": "Book chapter / ebook contribution",    "kind": "written"},
    {"key": "podcast",                 "label": "Podcast (host or guest)",              "kind": "audio"},
    {"key": "audio_testimonial",       "label": "Audio testimonial / narrated story",   "kind": "audio"},
    {"key": "infographic",             "label": "Infographic / carousel / one-sheet",   "kind": "visual"},
    {"key": "before_after",            "label": "Before/after photo essay (consent)",   "kind": "visual"},
    {"key": "annotated_scan",          "label": "Annotated scan / reading (de-identified)", "kind": "visual"},
    {"key": "conference",              "label": "Conference talk or poster",            "kind": "visual"},
    {"key": "qa_explainer",            "label": "Q&A / FAQ explainer",                  "kind": "written"},
    {"key": "social_thread",           "label": "Social thread / series",               "kind": "written"},
    {"key": "client_testimonial",      "label": "Client testimonial capture (consented)", "kind": "video"},
]

_KIND_BY_KEY = {f["key"]: f["kind"] for f in FORMATS}

MIN_SUBMISSIONS = 12


def kinds_for(format_keys):
    """The set of `kind` values for a list of format keys (unknown keys ignored)."""
    return {_KIND_BY_KEY[k] for k in (format_keys or []) if k in _KIND_BY_KEY}


def evaluate(submissions):
    """Given the student's approved+published submissions, return progress vs the
    completion rules.

    Each submission is a dict with:
      - "credited_modules": list[int]  (the module ids credited on approval)
      - "formats": list[str]           (format keys from FORMATS)

    Returns a dict; `complete` is the AND of every rule.
    """
    subs = list(submissions or [])
    approved_count = len(subs)

    covered = set()
    all_kinds = set()
    for s in subs:
        covered |= {int(m) for m in (s.get("credited_modules") or [])}
        all_kinds |= kinds_for(s.get("formats"))

    all_ids = [m["id"] for m in MODULES]
    modules_missing = [i for i in all_ids if i not in covered]
    has_written = "written" in all_kinds
    has_video = "video" in all_kinds
    multi_modality = len(all_kinds) >= 2

    reasons = []
    if approved_count < MIN_SUBMISSIONS:
        reasons.append(f"Needs at least {MIN_SUBMISSIONS} approved submissions "
                       f"(has {approved_count}).")
    if modules_missing:
        labels = ", ".join(str(i) for i in modules_missing)
        reasons.append(f"{len(modules_missing)} module(s) not yet covered: {labels}.")
    if not has_written:
        reasons.append("No written-format submission yet.")
    if not has_video:
        reasons.append("No video-format submission yet.")

    return {
        "approved_count": approved_count,
        "modules_covered": covered,
        "modules_missing": modules_missing,
        "has_written": has_written,
        "has_video": has_video,
        "multi_modality": multi_modality,
        "complete": not reasons,
        "reasons": reasons,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_rules.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/cert_rules.py tests/test_cert_rules.py
git commit -m "feat(cert): cert_rules — 12 modules, format catalog, completion evaluator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `cert_submissions.py` — the sqlite store (pure)

**Files:**
- Create: `dashboard/cert_submissions.py`
- Test: `tests/test_cert_submissions.py`

Mirrors `dashboard/cert_bonus.py` conventions: `_now()` ISO-UTC, `init_tables(cx)`, `cx.commit()` after writes, `dict(row)` reads. JSON list columns (`formats`, `modules`, `credited_modules`) are stored as JSON text and parsed on read.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cert_submissions.py
import sqlite3
from dashboard import cert_submissions as cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cs.init_tables(cx)
    return cx


def _make(cx, email="doc@x.com", modules=(1, 2), formats=("article",)):
    return cs.create(
        cx, sid="id-" + email + "-" + str(modules),
        email=email, practitioner_id="p1", title="My case",
        description="What happened", url="https://ex.com/post", file_path="",
        formats=list(formats), format_other="", modules=list(modules),
        module_other="", topic_angle="transformation", permission=1,
    )


def test_create_and_get_roundtrips_json():
    cx = _cx()
    sid = _make(cx, modules=(1, 2, 3), formats=("article", "demo_video"))
    row = cs.get(cx, sid)
    assert row["email"] == "doc@x.com"
    assert row["status"] == "submitted"
    assert row["modules"] == [1, 2, 3]
    assert row["formats"] == ["article", "demo_video"]
    assert row["credited_modules"] == []   # empty until approved
    assert row["permission"] == 1


def test_list_for_email_and_by_status():
    cx = _cx()
    _make(cx, email="a@x.com", modules=(1,))
    _make(cx, email="a@x.com", modules=(2,))
    _make(cx, email="b@x.com", modules=(3,))
    assert len(cs.list_for_email(cx, "a@x.com")) == 2
    assert len(cs.list_by_status(cx, "submitted")) == 3
    assert len(cs.list_by_status(cx, None)) == 3
    assert cs.list_by_status(cx, "approved") == []


def test_set_status_approve_sets_credited_modules():
    cx = _cx()
    sid = _make(cx, modules=(1, 2))
    cs.set_status(cx, sid, "approved", credited_modules=[1, 2], review_note="great")
    row = cs.get(cx, sid)
    assert row["status"] == "approved"
    assert row["credited_modules"] == [1, 2]
    assert row["review_note"] == "great"


def test_set_status_publish_records_case_study_id():
    cx = _cx()
    sid = _make(cx)
    cs.set_status(cx, sid, "approved", credited_modules=[1])
    cs.set_status(cx, sid, "published", case_study_id="cert-id-1")
    row = cs.get(cx, sid)
    assert row["status"] == "published"
    assert row["case_study_id"] == "cert-id-1"
    assert row["credited_modules"] == [1]   # preserved across the publish update


def test_get_missing_returns_none():
    cx = _cx()
    assert cs.get(cx, "nope") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_submissions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.cert_submissions'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/cert_submissions.py
"""Certification work-product submission store (pure: cx + args, no Flask).

Mirrors dashboard/cert_bonus.py conventions. JSON list columns (formats, modules,
credited_modules) are stored as JSON text and parsed on read.
"""
import json
from datetime import datetime, timezone

_JSON_COLS = ("formats", "modules", "credited_modules")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_tables(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS cert_submissions (
          id TEXT PRIMARY KEY,
          email TEXT NOT NULL,
          practitioner_id TEXT,
          title TEXT,
          description TEXT,
          url TEXT,
          file_path TEXT,
          formats TEXT,            -- JSON list of format keys
          format_other TEXT,
          modules TEXT,            -- JSON list of module ids (student-claimed)
          module_other TEXT,
          topic_angle TEXT,
          permission INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'submitted',
          credited_modules TEXT,   -- JSON list of module ids credited on approve
          review_note TEXT,
          case_study_id TEXT,
          created_at TEXT,
          updated_at TEXT
        )
        """
    )
    cx.execute("CREATE INDEX IF NOT EXISTS idx_cert_sub_email ON cert_submissions(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_cert_sub_status ON cert_submissions(status)")
    cx.commit()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for c in _JSON_COLS:
        try:
            d[c] = json.loads(d.get(c) or "[]")
        except Exception:
            d[c] = []
    return d


def create(cx, *, sid, email, practitioner_id, title, description, url,
           file_path, formats, format_other, modules, module_other,
           topic_angle, permission):
    now = _now()
    cx.execute(
        """
        INSERT INTO cert_submissions
          (id, email, practitioner_id, title, description, url, file_path,
           formats, format_other, modules, module_other, topic_angle,
           permission, status, credited_modules, review_note, case_study_id,
           created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'submitted', '[]', '', '', ?, ?)
        """,
        (sid, email, practitioner_id, title, description, url, file_path,
         json.dumps(list(formats or [])), format_other,
         json.dumps([int(m) for m in (modules or [])]), module_other,
         topic_angle, int(bool(permission)), now, now),
    )
    cx.commit()
    return sid


def get(cx, sid):
    r = cx.execute("SELECT * FROM cert_submissions WHERE id = ?", (sid,)).fetchone()
    return _row(r)


def list_for_email(cx, email):
    rows = cx.execute(
        "SELECT * FROM cert_submissions WHERE lower(email) = lower(?) "
        "ORDER BY created_at DESC", (email,)
    ).fetchall()
    return [_row(r) for r in rows]


def list_by_status(cx, status=None):
    if status:
        rows = cx.execute(
            "SELECT * FROM cert_submissions WHERE status = ? ORDER BY created_at DESC",
            (status,)
        ).fetchall()
    else:
        rows = cx.execute(
            "SELECT * FROM cert_submissions ORDER BY created_at DESC"
        ).fetchall()
    return [_row(r) for r in rows]


def set_status(cx, sid, status, *, credited_modules=None, review_note=None,
               case_study_id=None):
    sets = ["status = ?", "updated_at = ?"]
    vals = [status, _now()]
    if credited_modules is not None:
        sets.append("credited_modules = ?")
        vals.append(json.dumps([int(m) for m in credited_modules]))
    if review_note is not None:
        sets.append("review_note = ?")
        vals.append(review_note)
    if case_study_id is not None:
        sets.append("case_study_id = ?")
        vals.append(case_study_id)
    vals.append(sid)
    cx.execute(f"UPDATE cert_submissions SET {', '.join(sets)} WHERE id = ?", vals)
    cx.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_submissions.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/cert_submissions.py tests/test_cert_submissions.py
git commit -m "feat(cert): cert_submissions sqlite store (CRUD + status transitions)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Student magic-link auth + flag + portal page route

**Files:**
- Modify: `app.py` — add after the `/reorder/auth/<token>` block (near app.py:6940).
- Test: `tests/test_cert_portal_routes.py`

This mirrors the `/reorder` magic-link pattern exactly (`auth_tokens` with a new `purpose='cert_portal'`, a `rm_cert_email` cookie). The flag gates the student-facing surface.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cert_portal_routes.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CERT_PORTAL_ENABLED", "true")
    import app as appmod
    # Hermetic sqlite: point LOG_DB at a tmp file so tests never touch the dev db.
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_cert_token(appmod, email):
    """Insert a cert_portal auth token directly and return the raw token."""
    import secrets
    from datetime import timedelta
    tok = secrets.token_urlsafe(16)
    now = appmod._now_utc()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
            "VALUES (?,?,?,?,?)",
            (appmod._hash_token(tok), email, "cert_portal", now.isoformat(),
             (now + timedelta(minutes=appmod.AUTH_TOKEN_TTL_MIN)).isoformat()))
        cx.commit()
    return tok


def test_login_always_200(client):
    c, _ = client
    r = c.post("/cert/login", json={"email": "doc@x.com"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_auth_sets_cookie_and_redirects(client):
    c, appmod = client
    tok = _mint_cert_token(appmod, "doc@x.com")
    r = c.get(f"/cert/auth/{tok}")
    assert r.status_code == 302
    assert "rm_cert_email" in r.headers.get("Set-Cookie", "")


def test_auth_rejects_bad_token(client):
    c, _ = client
    r = c.get("/cert/auth/not-a-real-token")
    assert r.status_code == 400


def test_portal_page_served_when_enabled(client):
    c, _ = client
    r = c.get("/cert")
    assert r.status_code == 200


def test_portal_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    # The flag is read live (not at import), so force it off via the helper.
    monkeypatch.setattr(appmod, "_cert_portal_enabled", lambda: False)
    r = c.get("/cert")
    assert r.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -v`
Expected: FAIL (routes 404 / `_cert_portal_enabled` not defined)

- [ ] **Step 3: Write the implementation**

Add this block in `app.py` immediately after the `reorder_auth` function (after app.py:6940):

```python
# ── Certification work-product portal ───────────────────────────────────────
# Cert students submit published work here; Glen reviews behind two gates
# (approve -> publish). Student-facing surface gates behind CERT_PORTAL_ENABLED.

def _cert_portal_enabled() -> bool:
    return os.environ.get("CERT_PORTAL_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _cert_data_dir():
    """Runtime data root (same root LOG_DB uses). Uploaded files live here,
    never under the static/web-served dir."""
    return Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))


def _cert_email_from_cookie():
    return (request.cookies.get("rm_cert_email", "") or "").strip().lower()


@app.route("/cert")
def cert_portal_page():
    if not _cert_portal_enabled():
        return ("Not found", 404)
    resp = send_from_directory(STATIC, "cert-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/cert/login", methods=["POST"])
def cert_login():
    """Email a magic link to the cert portal. Always 200 (no enumeration)."""
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if email and "@" in email:
        token = secrets.token_urlsafe(32)
        now = _now_utc()
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                "VALUES (?,?,?,?,?)",
                (_hash_token(token), email, "cert_portal", now.isoformat(),
                 (now + timedelta(minutes=AUTH_TOKEN_TTL_MIN)).isoformat()))
            cx.commit()
        try:
            send_magic_link_email(email, "", f"{PUBLIC_BASE_URL}/cert/auth/{token}")
        except Exception as e:
            print(f"[cert] magic link send failed: {e!r}", flush=True)
    return jsonify({"ok": True})


@app.route("/cert/auth/<token>", methods=["GET"])
def cert_auth(token):
    """Validate the cert magic link, set rm_cert_email cookie, -> /cert."""
    from flask import redirect as _redirect
    if not _cert_portal_enabled():
        return ("Not found", 404)
    th = _hash_token((token or "").strip())
    email = None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='cert_portal'", (th,)).fetchone()
        if row and not row["consumed_at"]:
            try:
                if datetime.fromisoformat(row["expires_at"]) >= _now_utc():
                    email = row["email"]
            except Exception:
                email = None
        if email:
            cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
                       (_now_utc().isoformat(), th))
            cx.commit()
    if not email:
        return ("<p style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
                "This certification portal link is invalid or has expired. "
                "Please request a new one.</p>"), 400
    resp = _redirect("/cert", code=302)
    resp.set_cookie("rm_cert_email", email, max_age=60 * 60 * 24 * 30,
                    httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_cert_portal_routes.py
git commit -m "feat(cert): student magic-link auth + portal page route (flag-gated)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Student submit / upload / mine APIs

**Files:**
- Modify: `app.py` — add after the `cert_auth` block from Task 3.
- Test: `tests/test_cert_portal_routes.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cert_portal_routes.py`)

```python
def _auth_client(client):
    """Return a test client that already holds the rm_cert_email cookie."""
    c, appmod = client
    tok = _mint_cert_token(appmod, "doc@x.com")
    c.get(f"/cert/auth/{tok}")
    return c, appmod


def test_submit_requires_cookie(client):
    c, _ = client
    r = c.post("/api/cert/submit", json={"title": "x"})
    assert r.status_code == 401


def test_submit_requires_permission(client):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod)
    r = c.post("/api/cert/submit", json={
        "title": "My case", "description": "d", "url": "https://e.com/p",
        "formats": ["article"], "modules": [1], "permission": False})
    assert r.status_code == 400
    assert "permission" in r.get_json()["error"].lower()


def test_submit_requires_module_and_link(client):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod)
    # no module
    r = c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [], "permission": True})
    assert r.status_code == 400
    # no url and no file
    r = c.post("/api/cert/submit", json={
        "title": "t", "url": "", "formats": ["article"],
        "modules": [1], "permission": True})
    assert r.status_code == 400


def test_submit_creates_row_and_mine_lists_it(client):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod)
    r = c.post("/api/cert/submit", json={
        "title": "My case", "description": "what happened",
        "url": "https://e.com/p", "formats": ["article", "demo_video"],
        "modules": [1, 2], "topic_angle": "transformation", "permission": True})
    assert r.status_code == 200
    sid = r.get_json()["submission"]["id"]
    assert sid
    r2 = c.get("/api/cert/mine")
    body = r2.get_json()
    assert any(s["id"] == sid for s in body["submissions"])
    # progress rollup present; nothing approved yet so 0 covered
    assert body["progress"]["approved_count"] == 0


# Helper: monkeypatch the practitioner lookups used by submit/approve/publish.
def monkeypatch_pp(appmod):
    from dashboard import practitioner_portal as pp
    appmod._pp = pp  # ensure reference
    import types
    appmod.__dict__.setdefault("_orig_pp", pp)
    pp.id_for_email = lambda email: "p-test"  # added in Task 4 impl (best-effort)
```

> Note: `pp.id_for_email` is referenced best-effort; if you implement practitioner-id resolution inline instead, adjust the test to monkeypatch that path. Keep the resolution wrapped so a missing practitioner never 500s.

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -k "submit or mine" -v`
Expected: FAIL (`/api/cert/submit` 404)

- [ ] **Step 3: Write the implementation**

First add a best-effort practitioner-id resolver to `dashboard/practitioner_portal.py` (next to `modules_completed_for_email`, ~line 441):

```python
def id_for_email(email) -> Optional[str]:
    """practitioner_id for this email, or None. Best-effort (never raises)."""
    try:
        from db_supabase import supabase_cursor
        with supabase_cursor() as cur:
            cur.execute("SELECT id FROM practitioners WHERE lower(email)=lower(%s) "
                        "LIMIT 1", (str(email or "").strip(),))
            row = cur.fetchone()
            return str(row["id"]) if row else None
    except Exception:
        return None


def name_for_email(email) -> str:
    """practitioner display name for this email, or '' (best-effort)."""
    try:
        from db_supabase import supabase_cursor
        with supabase_cursor() as cur:
            cur.execute("SELECT name FROM practitioners WHERE lower(email)=lower(%s) "
                        "LIMIT 1", (str(email or "").strip(),))
            row = cur.fetchone()
            return str((row or {}).get("name") or "").strip()
    except Exception:
        return ""
```

Then add the routes in `app.py` after the `cert_auth` block:

```python
@app.route("/api/cert/submit", methods=["POST"])
def api_cert_submit():
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = _cert_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    url = (body.get("url") or "").strip()
    file_token = (body.get("file_token") or "").strip()
    formats = [str(x) for x in (body.get("formats") or [])]
    modules = []
    for m in (body.get("modules") or []):
        try:
            modules.append(int(m))
        except (TypeError, ValueError):
            pass
    permission = bool(body.get("permission"))

    if not title:
        return jsonify({"ok": False, "error": "title required"}), 400
    if not permission:
        return jsonify({"ok": False, "error":
                        "permission to publish is required to submit"}), 400
    if not modules:
        return jsonify({"ok": False, "error":
                        "select at least one module topic"}), 400
    # resolve a previously-uploaded file (path-guarded inside the cert dir)
    file_path = ""
    if file_token:
        cand = (_cert_data_dir() / "cert-files" / file_token).resolve()
        root = (_cert_data_dir() / "cert-files").resolve()
        if str(cand).startswith(str(root)) and cand.is_file():
            file_path = str(cand)
    if not url and not file_path:
        return jsonify({"ok": False, "error":
                        "provide a link to the published work or upload a file"}), 400

    from dashboard import practitioner_portal as _pp, cert_submissions as _cs
    try:
        pid = _pp.id_for_email(email) or ""
    except Exception:
        pid = ""
    sid = uuid.uuid4().hex
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        _cs.create(
            cx, sid=sid, email=email, practitioner_id=pid, title=title,
            description=(body.get("description") or "").strip(), url=url,
            file_path=file_path, formats=formats,
            format_other=(body.get("format_other") or "").strip(),
            modules=modules, module_other=(body.get("module_other") or "").strip(),
            topic_angle=(body.get("topic_angle") or "").strip(),
            permission=permission)
        sub = _cs.get(cx, sid)
    # best-effort notifications (never block)
    try:
        _send_inquiry_email(email, "We received your certification submission",
                            f"Thanks — '{title}' is in the review queue.")
        notify = os.environ.get("CERT_NOTIFY_EMAIL", "drglenswartwout@gmail.com")
        _send_inquiry_email(notify, "New certification submission",
                            f"{email} submitted '{title}'. Review in /console/cert.")
    except Exception as e:
        print(f"[cert] submit notify failed: {e!r}", flush=True)
    return jsonify({"ok": True, "submission": sub})


@app.route("/api/cert/upload", methods=["POST"])
def api_cert_upload():
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = _cert_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    f = request.files.get("file")
    if not f or not (f.filename or "").strip():
        return jsonify({"ok": False, "error": "file required"}), 400
    ctype = (f.mimetype or "").lower()
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
           "image/webp": "webp", "image/gif": "gif",
           "application/pdf": "pdf"}.get(ctype)
    if not ext:
        return jsonify({"ok": False, "error": "image or PDF only"}), 400
    blob = f.read()
    if len(blob) > 10 * 1024 * 1024:
        return jsonify({"ok": False, "error": "file too large (max 10MB)"}), 400
    cert_dir = _cert_data_dir() / "cert-files"
    cert_dir.mkdir(parents=True, exist_ok=True)
    token = f"{hashlib.sha256((email + f.filename).encode()).hexdigest()[:24]}.{ext}"
    (cert_dir / token).write_bytes(blob)
    return jsonify({"ok": True, "file_token": token})


@app.route("/api/cert/mine", methods=["GET"])
def api_cert_mine():
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = _cert_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    from dashboard import cert_submissions as _cs, cert_rules as _cr
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        subs = _cs.list_for_email(cx, email)
    counted = [s for s in subs if s["status"] in ("approved", "published")]
    prog = _cr.evaluate(counted)
    prog["modules_covered"] = sorted(prog["modules_covered"])  # JSON-safe
    return jsonify({"ok": True, "submissions": subs, "progress": prog,
                    "modules": _cr.MODULES, "formats": _cr.FORMATS})
```

Ensure `import uuid` exists at the top of `app.py` (it is already imported; if not, add it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -v`
Expected: PASS (all submit/mine/upload tests pass)

- [ ] **Step 5: Commit**

```bash
git add app.py dashboard/practitioner_portal.py tests/test_cert_portal_routes.py
git commit -m "feat(cert): student submit / upload / mine endpoints + progress rollup

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Console review APIs (approve / return / publish) + modules sync

**Files:**
- Modify: `app.py` — add after the student API block from Task 4.
- Test: `tests/test_cert_portal_routes.py` (append)

Console-gated by `CONSOLE_SECRET` (mirror `/api/cert/student`). On approve, recompute the student's covered-module count and sync `modules_completed`. Publish upserts into the `case-studies` namespace.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _console_key(appmod):
    return appmod.CONSOLE_SECRET or ""


def test_review_list_console_gated(client):
    c, appmod = client
    r = c.get("/api/cert/review/list")  # no key
    if appmod.CONSOLE_SECRET:
        assert r.status_code == 401
    else:
        assert r.status_code == 200


def test_approve_syncs_modules_completed(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod)
    # create a submission covering modules 1,2
    c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [1, 2], "permission": True})
    # capture upsert sync call
    calls = {}
    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "upsert_cert_student",
                        lambda email, **kw: calls.update(kw) or ("pid", kw.get("modules_completed", 0)))
    # find the submission id
    sid = c.get("/api/cert/mine").get_json()["submissions"][0]["id"]
    key = _console_key(appmod)
    r = c.post("/api/cert/review/approve?key=" + key,
               json={"id": sid, "credited_modules": [1, 2]})
    assert r.status_code == 200
    assert calls.get("modules_completed") == 2  # 2 distinct modules covered


def test_publish_requires_approved_and_permission(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod)
    c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [1], "permission": True})
    sid = c.get("/api/cert/mine").get_json()["submissions"][0]["id"]
    key = _console_key(appmod)
    # publish before approve → 400
    r = c.post("/api/cert/review/publish?key=" + key, json={"id": sid})
    assert r.status_code == 400
    # approve, then stub embed + pinecone, then publish
    c.post("/api/cert/review/approve?key=" + key,
           json={"id": sid, "credited_modules": [1]})
    monkeypatch.setattr(appmod, "embed", lambda text: [0.0] * 1536)
    captured = {}
    monkeypatch.setattr(appmod._idx, "upsert",
                        lambda **kw: captured.update(kw))
    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "name_for_email", lambda email: "Dr Test")
    r = c.post("/api/cert/review/publish?key=" + key, json={"id": sid})
    assert r.status_code == 200
    assert captured.get("namespace") == "case-studies"
    assert captured["vectors"][0]["id"] == "cert-" + sid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -k "review or approve or publish" -v`
Expected: FAIL (review routes 404)

- [ ] **Step 3: Write the implementation** (append in `app.py` after the student API block)

```python
def _cert_console_ok():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return False
    return True


def _cert_sync_modules(cx, email):
    """Recompute covered modules from this email's approved+published submissions
    and sync modules_completed on the practitioner record. Best-effort."""
    from dashboard import cert_submissions as _cs, cert_rules as _cr
    subs = [s for s in _cs.list_for_email(cx, email)
            if s["status"] in ("approved", "published")]
    covered = _cr.evaluate(subs)["modules_covered"]
    try:
        from dashboard import practitioner_portal as _pp
        _pp.upsert_cert_student(email, modules_completed=len(covered))
    except Exception as e:
        print(f"[cert] modules sync failed for {email}: {e!r}", flush=True)
    return len(covered)


def _cert_publish_to_proof(sub, name):
    """Embed + upsert one vector into the case-studies namespace. Returns the id."""
    title = sub.get("title") or "Practitioner case"
    desc = (sub.get("description") or "").strip()
    text = (title + ". " + desc).strip()
    vec = embed(text)
    vid = "cert-" + sub["id"]
    cond = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or ("cert-" + sub["id"][:8])
    excerpt = desc[:240]
    _idx.upsert(vectors=[{
        "id": vid,
        "values": vec,
        "metadata": {
            "condition": cond,
            "text": excerpt or title,
            "title": title,
            "url": sub.get("url") or "",
            "source": "cert-submission",
            "name": name or "",
        },
    }], namespace="case-studies")
    return vid


@app.route("/api/cert/review/list", methods=["GET"])
def api_cert_review_list():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    from dashboard import cert_submissions as _cs, cert_rules as _cr
    status = (request.args.get("status") or "").strip() or None
    email = (request.args.get("email") or "").strip() or None
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        if email:
            subs = _cs.list_for_email(cx, email)
        else:
            subs = _cs.list_by_status(cx, status)
        # per-student rollups
        emails = sorted({s["email"] for s in subs})
        rollups = {}
        for em in emails:
            counted = [s for s in _cs.list_for_email(cx, em)
                       if s["status"] in ("approved", "published")]
            p = _cr.evaluate(counted)
            p["modules_covered"] = sorted(p["modules_covered"])
            rollups[em] = p
    return jsonify({"ok": True, "submissions": subs, "rollups": rollups,
                    "modules": _cr.MODULES, "formats": _cr.FORMATS})


@app.route("/api/cert/review/approve", methods=["POST"])
def api_cert_review_approve():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("id") or "").strip()
    credited = [int(m) for m in (body.get("credited_modules") or []) if str(m).strip()]
    note = (body.get("note") or "").strip() or None
    from dashboard import cert_submissions as _cs
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        sub = _cs.get(cx, sid)
        if not sub:
            return jsonify({"ok": False, "error": "not found"}), 404
        if not credited:
            credited = list(sub.get("modules") or [])
        _cs.set_status(cx, sid, "approved", credited_modules=credited,
                       review_note=note)
        covered = _cert_sync_modules(cx, sub["email"])
    try:
        _send_inquiry_email(sub["email"], "Your certification submission was approved",
                            f"'{sub.get('title')}' is approved. Modules covered so far: {covered}/12.")
    except Exception as e:
        print(f"[cert] approve notify failed: {e!r}", flush=True)
    return jsonify({"ok": True, "id": sid, "modules_covered": covered})


@app.route("/api/cert/review/return", methods=["POST"])
def api_cert_review_return():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("id") or "").strip()
    note = (body.get("note") or "").strip() or "Please revise and resubmit."
    from dashboard import cert_submissions as _cs
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        if not _cs.get(cx, sid):
            return jsonify({"ok": False, "error": "not found"}), 404
        _cs.set_status(cx, sid, "returned", review_note=note)
    return jsonify({"ok": True, "id": sid})


@app.route("/api/cert/review/publish", methods=["POST"])
def api_cert_review_publish():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("id") or "").strip()
    from dashboard import cert_submissions as _cs
    from dashboard import practitioner_portal as _pp
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        sub = _cs.get(cx, sid)
        if not sub:
            return jsonify({"ok": False, "error": "not found"}), 404
        if sub["status"] not in ("approved", "published"):
            return jsonify({"ok": False, "error": "approve before publishing"}), 400
        if not sub.get("permission"):
            return jsonify({"ok": False, "error": "no publish permission on file"}), 400
        try:
            name = _pp.name_for_email(sub["email"])
        except Exception:
            name = ""
        try:
            vid = _cert_publish_to_proof(sub, name)
        except Exception as e:
            print(f"[cert] publish upsert failed: {e!r}", flush=True)
            return jsonify({"ok": False, "error": "publish failed; retry"}), 502
        _cs.set_status(cx, sid, "published", case_study_id=vid)
    return jsonify({"ok": True, "id": sid, "case_study_id": vid})
```

Confirm `import re` exists at the top of `app.py` (it does; if not, add it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_cert_portal_routes.py
git commit -m "feat(cert): console review (approve/return/publish) + modules sync + proof publish

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Student portal UI (`static/cert-portal.html`)

**Files:**
- Create: `static/cert-portal.html`

A single self-contained page (vanilla JS, inline CSS) matching the simple style of the other `static/*.html` pages. Two states: logged-out (email entry → `/cert/login`) and logged-in (renders the form from `/api/cert/mine`'s `modules` + `formats`, plus the submission list and the progress readout).

- [ ] **Step 1: Create the page**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Certification Portal</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;color:#1c2b2b;line-height:1.5}
  h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2rem}
  fieldset{border:1px solid #cdd;border-radius:.5rem;margin:.75rem 0;padding:.75rem}
  legend{font-weight:600;padding:0 .35rem}
  label.chk{display:block;font-size:.92rem;margin:.15rem 0}
  input[type=text],input[type=email],textarea{width:100%;padding:.5rem;border:1px solid #bcc;border-radius:.4rem;font:inherit}
  .row{margin:.6rem 0}
  button{background:#1b6b5e;color:#fff;border:0;border-radius:.4rem;padding:.55rem 1rem;font:inherit;cursor:pointer}
  button.secondary{background:#eef3f2;color:#1b6b5e}
  .pill{display:inline-block;padding:.1rem .5rem;border-radius:1rem;font-size:.8rem;background:#eef3f2}
  .ok{color:#1b6b5e;font-weight:600}.bad{color:#a23}
  .sub{border:1px solid #dde;border-radius:.5rem;padding:.6rem;margin:.5rem 0}
  .hint{color:#667;font-size:.85rem}
</style>
</head>
<body>
<h1>Certification Work-Product Portal</h1>
<div id="login">
  <p>Enter your email and we'll send you a sign-in link.</p>
  <div class="row"><input type="email" id="email" placeholder="you@example.com"/></div>
  <button onclick="requestLink()">Email me a sign-in link</button>
  <p id="login-msg" class="hint"></p>
</div>

<div id="app" style="display:none">
  <p>Progress: <span id="progress"></span></p>

  <h2>Submit a project</h2>
  <div class="row"><input type="text" id="title" placeholder="Title"/></div>
  <div class="row"><textarea id="description" rows="3" placeholder="What is this project about? (context, result)"></textarea></div>
  <div class="row"><input type="text" id="url" placeholder="Link to the published work (social, YouTube, article…)"/></div>
  <div class="row hint">A public post, video, or article counts as published. Or upload a written doc / image:</div>
  <div class="row"><input type="file" id="file" accept="image/*,application/pdf"/> <span id="file-msg" class="hint"></span></div>

  <fieldset><legend>Module topics covered (check all that apply)</legend>
    <div id="modules"></div>
    <div class="row"><input type="text" id="module_other" placeholder="Other topic (optional)"/></div>
  </fieldset>

  <fieldset><legend>Format(s) used (check all that apply)</legend>
    <div id="formats"></div>
    <div class="row"><input type="text" id="format_other" placeholder="Other format (optional)"/></div>
  </fieldset>

  <div class="row"><input type="text" id="topic_angle" placeholder="Topic angle (optional): goal / transformation / modality…"/></div>
  <label class="chk"><input type="checkbox" id="permission"/> I grant permission to publish and reuse this work in whole or part.</label>
  <div class="row"><button onclick="submitProject()">Submit project</button> <span id="submit-msg" class="hint"></span></div>

  <h2>My submissions</h2>
  <div id="list"></div>
</div>

<script>
let FILE_TOKEN = "";
async function requestLink(){
  const email=document.getElementById('email').value.trim();
  await fetch('/cert/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email})});
  document.getElementById('login-msg').textContent="If that email is registered, a sign-in link is on its way.";
}
async function load(){
  const r=await fetch('/api/cert/mine');
  if(r.status===401){return;} // stay on login
  const d=await r.json();
  document.getElementById('login').style.display='none';
  document.getElementById('app').style.display='block';
  // checkboxes
  document.getElementById('modules').innerHTML=d.modules.map(m=>
    `<label class="chk"><input type="checkbox" class="mod" value="${m.id}"/> ${m.id}. ${m.label}</label>`).join('');
  document.getElementById('formats').innerHTML=d.formats.map(f=>
    `<label class="chk"><input type="checkbox" class="fmt" value="${f.key}"/> ${f.label}</label>`).join('');
  renderProgress(d.progress);
  renderList(d.submissions);
}
function renderProgress(p){
  const w=p.has_written?'✓':'✗', v=p.has_video?'✓':'✗';
  document.getElementById('progress').innerHTML=
    `<span class="pill">${p.approved_count}/12 approved</span> `+
    `<span class="pill">${p.modules_covered.length}/12 modules</span> `+
    `<span class="pill">written ${w}</span> <span class="pill">video ${v}</span> `+
    (p.complete?'<span class="ok">— complete!</span>':'');
}
function renderList(subs){
  document.getElementById('list').innerHTML = subs.length? subs.map(s=>
    `<div class="sub"><b>${s.title||'(untitled)'}</b> <span class="pill">${s.status}</span><br>`+
    `<span class="hint">${(s.modules||[]).join(', ')||'no modules'} · ${(s.formats||[]).join(', ')||'no format'}</span>`+
    (s.review_note?`<br><span class="hint">Note: ${s.review_note}</span>`:'')+`</div>`).join('')
    : '<p class="hint">No submissions yet.</p>';
}
async function uploadFile(){
  const f=document.getElementById('file').files[0];
  if(!f){return "";}
  const fd=new FormData(); fd.append('file', f);
  const r=await fetch('/api/cert/upload',{method:'POST',body:fd});
  const d=await r.json();
  if(!d.ok){document.getElementById('file-msg').textContent=d.error||'upload failed';return "";}
  document.getElementById('file-msg').textContent='uploaded';
  return d.file_token||"";
}
async function submitProject(){
  const msg=document.getElementById('submit-msg'); msg.textContent='…';
  const file_token = await uploadFile();
  const modules=[...document.querySelectorAll('.mod:checked')].map(x=>parseInt(x.value));
  const formats=[...document.querySelectorAll('.fmt:checked')].map(x=>x.value);
  const payload={
    title:title.value, description:description.value, url:url.value,
    file_token, modules, formats,
    module_other:module_other.value, format_other:format_other.value,
    topic_angle:topic_angle.value, permission:permission.checked};
  const r=await fetch('/api/cert/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();
  if(!d.ok){msg.textContent=d.error||'error';return;}
  msg.textContent='submitted ✓';
  ['title','description','url','module_other','format_other','topic_angle'].forEach(id=>document.getElementById(id).value='');
  document.querySelectorAll('.mod:checked,.fmt:checked').forEach(x=>x.checked=false);
  document.getElementById('permission').checked=false;
  load();
}
load();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the page is served**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" CERT_PORTAL_ENABLED=true ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py::test_portal_page_served_when_enabled -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add static/cert-portal.html
git commit -m "feat(cert): student portal UI (submission form + progress)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Console review UI + page route + final review

**Files:**
- Create: `static/console-cert.html`
- Modify: `app.py` — add the `/console/cert` page route near the other `/console/*` routes (e.g. near app.py:17080).
- Test: `tests/test_cert_portal_routes.py` (append one route test)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_console_cert_page_served(client):
    c, _ = client
    r = c.get("/console/cert")
    assert r.status_code == 200
```

- [ ] **Step 2: Run it to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_portal_routes.py::test_console_cert_page_served -v`
Expected: FAIL (404)

- [ ] **Step 3: Add the page route** in `app.py` (near the other `/console/*` page routes, ~app.py:17080):

```python
@app.route("/console/cert")
def bos_cert_page():
    resp = send_from_directory(STATIC, "console-cert.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 4: Create `static/console-cert.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Cert Review</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:60rem;margin:1.5rem auto;padding:0 1rem;color:#1c2b2b}
  .sub{border:1px solid #dde;border-radius:.5rem;padding:.7rem;margin:.6rem 0}
  .pill{display:inline-block;padding:.1rem .5rem;border-radius:1rem;font-size:.78rem;background:#eef3f2;margin-right:.3rem}
  button{background:#1b6b5e;color:#fff;border:0;border-radius:.4rem;padding:.4rem .8rem;font:inherit;cursor:pointer;margin-right:.4rem}
  button.alt{background:#eef3f2;color:#1b6b5e}
  a{color:#1b6b5e} .hint{color:#667;font-size:.85rem}
  input.note{width:60%;padding:.3rem;border:1px solid #bcc;border-radius:.3rem}
  label.cm{font-size:.82rem;margin-right:.5rem;white-space:nowrap}
</style>
</head>
<body>
<h1>Certification Review</h1>
<p class="hint">Approve credits the modules (private). Publish pushes it to the proof library.</p>
<div>Filter:
  <select id="status" onchange="load()">
    <option value="">all</option><option>submitted</option><option>approved</option>
    <option>published</option><option>returned</option>
  </select>
</div>
<div id="list"></div>
<script>
const KEY = new URLSearchParams(location.search).get('key') || '';
let MODULES=[];
function q(p){return p+(p.includes('?')?'&':'?')+'key='+encodeURIComponent(KEY);}
async function load(){
  const st=document.getElementById('status').value;
  const r=await fetch(q('/api/cert/review/list'+(st?('?status='+st):'')));
  if(r.status===401){document.getElementById('list').innerHTML='<p>Add ?key=CONSOLE_SECRET to the URL.</p>';return;}
  const d=await r.json(); MODULES=d.modules;
  document.getElementById('list').innerHTML=d.submissions.map(s=>card(s,d.rollups[s.email])).join('')||'<p class="hint">Nothing here.</p>';
}
function card(s,roll){
  const checks=MODULES.map(m=>`<label class="cm"><input type="checkbox" class="cm-${s.id}" value="${m.id}" ${(s.credited_modules||s.modules||[]).includes(m.id)?'checked':''}/> ${m.id}</label>`).join('');
  const link=s.url?`<a href="${s.url}" target="_blank">link</a>`:(s.file_path?'(file on server)':'(no link)');
  const r=roll?`<span class="hint">${s.email}: ${roll.approved_count}/12 approved · ${roll.modules_covered.length}/12 modules · W${roll.has_written?'✓':'✗'} V${roll.has_video?'✓':'✗'}${roll.complete?' · COMPLETE':''}</span>`:'';
  return `<div class="sub"><b>${s.title||'(untitled)'}</b> <span class="pill">${s.status}</span> ${link}<br>
    <span class="hint">${s.description||''}</span><br>${r}<br>
    <div style="margin:.4rem 0">${checks}</div>
    <input class="note" id="note-${s.id}" placeholder="reviewer note (optional)"/><br>
    <div style="margin-top:.4rem">
      <button onclick="act('approve','${s.id}')">Approve</button>
      <button class="alt" onclick="act('return','${s.id}')">Return</button>
      <button onclick="act('publish','${s.id}')">Publish</button>
    </div></div>`;
}
async function act(kind,id){
  const note=document.getElementById('note-'+id).value;
  const body={id,note};
  if(kind==='approve'){body.credited_modules=[...document.querySelectorAll('.cm-'+id+':checked')].map(x=>parseInt(x.value));}
  const r=await fetch(q('/api/cert/review/'+kind),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(!d.ok){alert(d.error||'error');return;}
  load();
}
load();
</script>
</body>
</html>
```

- [ ] **Step 5: Run the full new suite**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_rules.py tests/test_cert_submissions.py tests/test_cert_portal_routes.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-cert.html tests/test_cert_portal_routes.py
git commit -m "feat(cert): console review UI + /console/cert page route

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final integration review (after all tasks)

- [ ] Run the full new suite once more (command above) — all green.
- [ ] Run a broad import smoke: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -c "import app"` — no import errors.
- [ ] Confirm `CERT_PORTAL_ENABLED` is documented as the dark-launch flag (off by default), and that console review endpoints work WITHOUT the flag (they gate on `CONSOLE_SECRET` only) so Glen can review pre-launch.
- [ ] Open a PR (Glen merges). PR body ends with:
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- [ ] Then: superpowers:finishing-a-development-branch.

## Notes for the implementer

- `app.py` is large; add each block where the task says (after the named anchor function). Do not reorder existing code.
- `embed`, `_idx`, `_now_utc`, `_hash_token`, `_db_lock`, `LOG_DB`, `STATIC`, `PUBLIC_BASE_URL`, `AUTH_TOKEN_TTL_MIN`, `CONSOLE_SECRET`, `send_magic_link_email`, `_send_inquiry_email` all already exist in `app.py` — reuse them, do not redefine.
- Stores call `init_tables(cx)` per-request (idempotent `CREATE TABLE IF NOT EXISTS`), matching the biofield/cert_bonus pattern. No startup wiring needed.
- `_send_inquiry_email` returns a bool (True/False), never raises. Wrap notify calls in try/except anyway; they must never block the user action.
- Never web-serve uploaded files; they live under `DATA_DIR/cert-files` and are only referenced by path on the submission row.
- Keep `dashboard/cert_rules.py` and `dashboard/cert_submissions.py` pure (no Flask import) so they test under the bare venv.
```
