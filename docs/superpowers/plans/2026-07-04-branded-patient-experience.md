# Branded Patient Experience (D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the patient's attributed doctor's identity (photo, name, practice, accent) as a "Your practitioner" band at the top of the patient portal — co-brand, not white-label.

**Architecture:** A best-effort `_patient_practitioner_brand(email)` helper (patient → `_last_attributed_practitioner` → pid → `practitioner_settings` branding + the doctor's name) adds a `practitioner_brand` key to the `api_client_portal` payload; `client-portal.html` renders a band from it. Reuses existing pieces; no schema changes.

**Tech Stack:** Python 3, Flask, SQLite (`practitioner_settings` in LOG_DB), Supabase (`practitioners` name), pytest, vanilla JS.

## Global Constraints

- Co-brand only: Remedy Match stays the platform chrome; the doctor's accent color applies ONLY to the band, never a global re-theme.
- **Attribution-only, NO consent gate:** the band is the doctor's public identity (photo/name/practice), independent of `practitioner_share_consent`. Do NOT gate it on C's consent. Do NOT touch C's data-access gate.
- Best-effort: any lookup/read failure omits the band and NEVER crashes the portal.
- The band appears only when the patient has an attributed doctor AND that doctor has non-empty branding.
- No patient health data in the band — only the doctor's public identity.
- Tests import `app` → run via `doppler run -p remedy-match -c dev -- python -m pytest ...`.

---

## File Structure

- **Modify** `app.py` — new `_patient_practitioner_brand(email, *, db_path=None)` helper; `api_client_portal` adds `payload["practitioner_brand"]`.
- **Modify** `static/client-portal.html` — the "Your practitioner" band.
- **Test** `tests/test_patient_brand.py`.

---

### Task 1: `_patient_practitioner_brand` helper + payload wiring

**Files:** Modify `app.py`; Test `tests/test_patient_brand.py`

**Interfaces:**
- Consumes: `_last_attributed_practitioner(email, *, db_path=None)` (#576); `dashboard/practitioner_settings.get_settings(cx, pid) -> {"branding": {...}}`.
- Produces: `_patient_practitioner_brand(email, *, db_path=None) -> {"name","practice_name","photo_url","logo_url","accent"} | None`; `api_client_portal` payload gains `practitioner_brand` (the dict or None).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patient_brand.py
import importlib, sys
from pathlib import Path

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    return importlib.import_module("app")

def test_brand_for_attributed_patient_with_branding(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner", lambda email, **k: {"pid": "prac-42", "consent": 0})
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {"practice_name": "Vital Roots", "photo_url": "http://x/p.jpg", "logo_url": "", "primary_color": "#123456"}})
    monkeypatch.setattr(app, "_practitioner_display_name", lambda pid: "Dr. Jane Ríos")
    b = app._patient_practitioner_brand("pat@x.com")
    assert b["name"] == "Dr. Jane Ríos" and b["practice_name"] == "Vital Roots"
    assert b["photo_url"] == "http://x/p.jpg" and b["accent"] == "#123456"

def test_none_when_no_attributed_doctor(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner", lambda email, **k: None)
    assert app._patient_practitioner_brand("pat@x.com") is None

def test_none_when_branding_empty(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner", lambda email, **k: {"pid": "prac-42", "consent": 1})
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {}})
    monkeypatch.setattr(app, "_practitioner_display_name", lambda pid: "Dr. Jane")
    assert app._patient_practitioner_brand("pat@x.com") is None   # no brand to show

def test_consent_independent(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_last_attributed_practitioner", lambda email, **k: {"pid": "prac-42", "consent": 0})  # NOT consented
    from dashboard import practitioner_settings as ps
    monkeypatch.setattr(ps, "get_settings", lambda cx, pid: {"branding": {"practice_name": "Vital Roots"}})
    monkeypatch.setattr(app, "_practitioner_display_name", lambda pid: "Dr. Jane")
    assert app._patient_practitioner_brand("pat@x.com") is not None   # branding shows regardless of consent

def test_never_raises_on_failure(monkeypatch):
    app = _app()
    def boom(*a, **k): raise RuntimeError("db down")
    monkeypatch.setattr(app, "_last_attributed_practitioner", boom)
    assert app._patient_practitioner_brand("pat@x.com") is None   # swallowed → None
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_patient_brand.py -q`
Expected: FAIL (`AttributeError: _patient_practitioner_brand`)

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the other patient-portal / practitioner helpers):

```python
def _practitioner_display_name(pid):
    """Best-effort practitioner display name from the practitioners record. None on any failure."""
    try:
        from db_supabase import supabase_cursor
        with supabase_cursor() as cur:
            cur.execute("SELECT name FROM practitioners WHERE id=%s", (str(pid),))
            row = cur.fetchone()
        return (row["name"] or "").strip() or None if row else None
    except Exception:
        return None


def _patient_practitioner_brand(email, *, db_path=None):
    """The patient's attributed doctor's PUBLIC identity for co-branding the patient
    portal: {name, practice_name, photo_url, logo_url, accent} or None. Attribution-only
    (NOT gated on results-sharing consent). Best-effort — returns None on any failure or
    when there's no attributed doctor / no branding set."""
    try:
        inh = _last_attributed_practitioner(email, db_path=db_path)
        if not inh:
            return None
        pid = inh["pid"]
        with sqlite3.connect(db_path or LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            from dashboard import practitioner_settings as _ps
            _ps.init_settings_table(cx)
            branding = (_ps.get_settings(cx, pid) or {}).get("branding") or {}
        name = _practitioner_display_name(pid)
        practice_name = (branding.get("practice_name") or "").strip()
        photo_url = (branding.get("photo_url") or "").strip()
        logo_url = (branding.get("logo_url") or "").strip()
        accent = (branding.get("primary_color") or branding.get("accent") or "").strip()
        # Require SOME brand to show (name alone isn't "branding set").
        if not (practice_name or photo_url or logo_url):
            return None
        return {"name": name or practice_name, "practice_name": practice_name,
                "photo_url": photo_url, "logo_url": logo_url, "accent": accent}
    except Exception:
        return None
```

Note: confirm the exact branding key for the accent color by reading `static/practitioner-settings.html` (it saves `primary_color`/`accent`/similar) — use the real key(s); the helper already tries `primary_color` then `accent`.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Wire into the payload**

In `api_client_portal(token)` (app.py ~13773), where the response payload dict is assembled (it already computes `email_for_reports`), add before the `return jsonify(...)`:

```python
    payload["practitioner_brand"] = _patient_practitioner_brand(email_for_reports)
```

Use the actual payload variable name the function returns (read the function — it may be `content` or a dedicated dict). Add a route test: seed/patch so an attributed patient's portal GET returns `practitioner_brand` in the JSON, and a non-attributed patient's does not. (Reuse the client-portal route test setup — grep `api_client_portal` / `/api/portal/` in tests/.)

- [ ] **Step 6: Run + commit**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_patient_brand.py -q`
```bash
git add app.py tests/test_patient_brand.py
git commit -m "feat(brand): patient portal payload carries the attributed doctor's brand (attribution-only)"
```

---

### Task 2: "Your practitioner" band in the patient portal

**Files:** Modify `static/client-portal.html`

**Interfaces:**
- Consumes: the portal payload's `d.practitioner_brand` (`{name, practice_name, photo_url, logo_url, accent}` or absent).

This is UI-only (no pytest cycle).

- [ ] **Step 1: Add the band markup + render**

In `static/client-portal.html`'s main render (`render(d, v)` — grep `function render`), when `d.practitioner_brand` is present, render a "Your practitioner" band at the TOP of the content: the doctor's `photo_url` (as a small round avatar; omit the img if empty), a heading like "Your practitioner" and a line "Your continuity care is guided by **{name}**{practice_name ? ' — ' + practice_name : ''}". Apply `d.practitioner_brand.accent` (if a valid color string) ONLY to the band (e.g. a left border or the heading color) — do NOT touch the global `--brand` theme vars. If `practitioner_brand` is absent, render nothing (portal unchanged). Match the file's existing card/vanilla-JS style; escape all injected strings (reuse the file's `esc()` helper).

- [ ] **Step 2: Verify (static)**

Extract the `<script>` and `node --check`; confirm the band reads `d.practitioner_brand` and only the band uses the accent (grep to confirm no assignment to `--brand`/theme vars from `practitioner_brand`). Report that live browser render is pending (controller will render-verify).

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(brand): patient portal 'Your practitioner' co-brand band"
```

---

## Self-Review

**Spec coverage:**
- Payload resolver (patient → doctor branding + name) → Task 1. ✓
- `practitioner_brand` only when attributed + non-empty branding → Task 1 (`if not (practice_name or photo_url or logo_url): return None`). ✓
- Attribution-only / consent-independent → Task 1 (`test_consent_independent`; no consent check in the helper). ✓
- Best-effort no-crash → Task 1 (`test_never_raises_on_failure`, outer try/except). ✓
- Render band, accent band-only, absent→unchanged → Task 2. ✓
- No patient-data leak → the helper only returns the doctor's public identity fields. ✓

**Placeholder scan:** Task 1 has complete helper + 5 tests; the payload-wire step names the exact insertion and instructs reading the real payload var + the real accent key (rather than guessing). Task 2 is UI with a static check + a deferred browser render-verify (appropriate for a render band).

**Type consistency:** `_patient_practitioner_brand(email, *, db_path=None) -> {name,practice_name,photo_url,logo_url,accent}|None` (Task 1) is exactly the shape Task 2's render consumes as `d.practitioner_brand`. `_last_attributed_practitioner(email, *, db_path=None) -> {pid,consent}|None` matches #576.

## Notes / open confirmations
- **Accent key:** confirm the real `branding_json` accent key from `static/practitioner-settings.html` (the helper tries `primary_color` then `accent`; adjust if the saved key differs).
- **Payload variable:** the wire step must use `api_client_portal`'s actual returned dict name (read it) — do not assume `payload`.
- **The doctor's name is Supabase** (best-effort, patchable via `_practitioner_display_name`); if unavailable the band uses `practice_name`.
