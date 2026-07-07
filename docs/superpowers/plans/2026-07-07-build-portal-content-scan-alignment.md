# build_portal_content Scan Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `build_portal_content` bake the findings of the *published* scan, not always the latest, by swapping `scan_context` for `findings_for_scan_date`.

**Architecture:** One-function change in `dashboard/biofield_portal_publish.py`; the injectable param `scan_context` becomes `findings_provider` (default `findings_for_scan_date`, added in #664). Two existing #661 tests are updated to inject a findings-list provider. No caller changes (`biofield_local_app._do_publish` never passes the injectable).

**Tech Stack:** Python; pytest. The test file imports only `dashboard` modules — plain `python3 -m pytest`, no Doppler.

## Global Constraints

- **One module changed:** `dashboard/biofield_portal_publish.py` + its test file. No `app.py`, no schema, no new endpoint. No change to `scan_context`/`_latest_scan`.
- **Strict, no fallback:** no exact-date scan → `findings == []` (never a different scan's findings).
- **Injectable provider:** keyword param `findings_provider=None`; when `None`, lazily import `dashboard.biofield_e4l.findings_for_scan_date`. Provider signature `(email, scan_date) -> list[finding dicts]` (returns the list directly, not a `{"findings": [...]}` dict).
- **Preserve:** the `if email and scan_date:` guard, the `try/except → []` none-raising contract, and the trim to exactly `{code, name, description, rank}`.

---

### Task 1: Swap `scan_context` → `findings_for_scan_date` in `build_portal_content`

**Files:**
- Modify: `dashboard/biofield_portal_publish.py` (`build_portal_content`, signature line 97 + findings block lines 140-158)
- Test: `tests/test_biofield_portal_publish_build.py` (update the two #661 findings tests, lines 74-108)

**Interfaces:**
- Consumes: `dashboard.biofield_e4l.findings_for_scan_date(email, scan_date) -> list[{code,name,description,rank,category,group}]` (exists from #664).
- Produces: `build_portal_content(..., findings_provider=None)` — `content["findings"]` now reflects the exact `scan_date`'s scan; `[]` on no match / provider error. The old `scan_context=` keyword is gone.

- [ ] **Step 1: Update the two existing tests to inject a findings-list provider**

In `tests/test_biofield_portal_publish_build.py`, replace `test_build_populates_findings_from_scan` (lines 74-98) and `test_build_findings_empty_when_scan_context_raises` (lines 101-108) with:

```python
def test_build_populates_findings_from_scan(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    captured = {}
    def fake_provider(email, scan_date):
        captured["args"] = (email, scan_date)
        return [
            {"code": "ED3", "name": "Cell Driver", "rank": 1,
             "description": "The Cell Driver supports cellular energy.",
             "category": "ED", "group": "infoceutical"},
            {"code": "ER9", "name": "Environmental Load", "rank": 2,
             "description": "", "category": "ER", "group": "stress"},
        ]
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000,
                                   catalog=CATALOG, findings_provider=fake_provider)
    f = out["content"]["findings"]
    assert len(f) == 2
    # trimmed to exactly the four portal-consumed fields; description preserved
    assert f[0] == {"code": "ED3", "name": "Cell Driver",
                    "description": "The Cell Driver supports cellular energy.", "rank": 1}
    # blank-description finding kept, description == ""
    assert f[1] == {"code": "ER9", "name": "Environmental Load",
                    "description": "", "rank": 2}
    # aligned to the published scan_date
    assert captured["args"] == ("permanentlyyours777@hawaiiantel.net", "2026-06-25")


def test_build_findings_empty_when_provider_raises(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    def boom(email, scan_date):
        raise RuntimeError("e4l.db unreadable")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000,
                                   catalog=CATALOG, findings_provider=boom)
    assert out["content"]["findings"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_biofield_portal_publish_build.py -q`
Expected: FAIL — `build_portal_content` has no `findings_provider` keyword (TypeError: unexpected keyword argument 'findings_provider').

- [ ] **Step 3: Update the signature**

In `dashboard/biofield_portal_publish.py`, change the signature (line 96-97):

```python
def build_portal_content(cx, test_id, *, special_price_cents, catalog=None,
                         audio_url=None, report_pdf_url=None, findings_provider=None):
```

- [ ] **Step 4: Swap the findings block**

Replace the current comment + findings block (lines 140-158) with:

```python
    # Bake the scan's findings (name + e4l_description) into the portal content so
    # the client-portal stress-pattern chips render. findings_for_scan_date reads the
    # local e4l.db and returns the findings for the EXACT scan_date being published
    # (scan_context would always return the latest scan). Injectable for tests; never
    # raises (portal must publish even when e4l.db is missing/unreadable). Trimmed to
    # the fields the portal uses. Empty when no scan matches that date.
    email = (client.get("email") or "").strip().lower()
    scan_date = rep.get("date") or ""
    findings = []
    if email and scan_date:
        try:
            _fp = findings_provider
            if _fp is None:
                from dashboard.biofield_e4l import findings_for_scan_date as _fp
            raw = _fp(email, scan_date) or []
            findings = [{"code": f.get("code", ""), "name": f.get("name", ""),
                         "description": f.get("description", ""), "rank": f.get("rank")}
                        for f in raw]
        except Exception:
            findings = []
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_biofield_portal_publish_build.py -q`
Expected: PASS — the two updated findings tests plus the three pre-existing `test_build_*` tests (which don't assert on findings; the real `findings_for_scan_date` returns `[]` when `e4l.db` is absent, so they stay deterministic).

- [ ] **Step 6: Commit**

Stage ONLY these two paths (never `git add -A`/`.`):

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_build.py
git commit -m "fix(portal): build_portal_content bakes the published scan's findings, not latest"
```

---

## Notes for the reviewer / executor

- No caller passes the old `scan_context=` keyword: `biofield_local_app._do_publish` calls `build_portal_content(cx, test_id, special_price_cents=…, audio_url=…, report_pdf_url=…)` — the injectable always defaulted. The rename is safe.
- Strict-by-design: if a report's `scan_date` has no exactly-matching E4L scan, findings are `[]` (better empty than another scan's findings). In practice the authored date == the E4L scan date.
- `findings_for_scan_date`'s own date-specific behavior is already covered by `tests/test_findings_for_scan_date.py` (#664); this task only needs to prove `build_portal_content` calls the provider with `(email, scan_date)` and trims/guards correctly.
