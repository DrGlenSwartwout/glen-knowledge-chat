# Biofield Report Formats — Phase 1 (Shared Renderer + Print/PDF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the existing 3-section Biofield report (Causal Chain · Remedy Schedule · Narrative) as a clean, client-facing **print/PDF** rendition — no editing chrome — generated as a real PDF file that is saved locally (to print + ship with an order).

**Architecture:** One pure presentation renderer (`dashboard/biofield_report_present.py`) turns the existing report dict + narrative text into clean, print-styled HTML in **schedule-forward** order with the confirmed branding. A small PDF helper (`dashboard/biofield_report_pdf.py`) renders that HTML to PDF bytes via headless Chromium (Playwright). Two new routes in the LOCAL app (`biofield_local_app.py`) serve the HTML view and the downloadable/saved PDF. All local — no server or schema changes.

**Tech Stack:** Python 3, Flask (existing local app), Playwright (already installed in the app's `python3`), sqlite (`chat_log.db`), pytest.

## Global Constraints

- **Local-only:** Phase 1 touches only `biofield_local_app.py` + `dashboard/biofield_*`; do NOT modify `app.py`, server routes, or any DB schema. PHI stays on the Mac.
- **Reuse the existing report dict:** `_report_for(cx, test_id)` already returns `{test_id, client:{name,email}, date, layers:[...], schedule:{slots,entries}}` (authored vs FMP). Do NOT re-query or change it.
- **Section order (print):** masthead → **Remedy Schedule** → **Narrative** → **Causal Chain table** → footer. Schedule leads because the sheet ships with the bottles.
- **Branding (verbatim):** masthead wordmark = `Accelerated Self Healing™`; footer = `In wellness, Dr. Glen & Rae · illtowell.com`. Text wordmark (no logo image yet).
- **No edit chrome:** the presentation renderer emits NO buttons, textareas, inputs, nav bars, or E4L picker — output only.
- **Escaping:** all dynamic text is HTML-escaped (reuse the `_e` pattern from `biofield_report_html.py`).
- **Schedule rendering parity:** mirror the existing slot/`as_directed`/`food` logic in `biofield_report_html.py` (entries have `name, dosage, frequency, timing, slots, food, as_directed`; `schedule.slots` is the ordered time-of-day list).

---

### Task 1: Presentation renderer — clean print-ready HTML

**Files:**
- Create: `dashboard/biofield_report_present.py`
- Test: `tests/test_biofield_report_present.py`

**Interfaces:**
- Consumes: a report dict `{test_id, client:{name,email}, date, layers:[{layer,head,most_affected,remedy,dosage,frequency,timing}], schedule:{slots:[str], entries:[{name,dosage,frequency,timing,slots:[str],food:str,as_directed:bool}]}}` and a `narrative` string.
- Produces: `render_present(report: dict, narrative: str = "") -> str` — a complete standalone HTML document (`<!doctype html>…`) with embedded screen+print CSS, in schedule-forward order, branded, no edit chrome.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_report_present.py
from dashboard.biofield_report_present import render_present

REPORT = {
    "test_id": "a2",
    "client": {"name": "Kauilani Perdomo", "email": "k@example.com"},
    "date": "2026-06-24",
    "layers": [
        {"layer": 1, "head": "ET4", "most_affected": "ET4", "remedy": "MSM Lotion",
         "dosage": "1 application", "frequency": "daily", "timing": "morning"},
        {"layer": 2, "head": "ED7 Lungs", "most_affected": "Pancreas Cauda",
         "remedy": "Sulfur Syntropy", "dosage": "10 drops", "frequency": "3x a day", "timing": "before food"},
    ],
    "schedule": {
        "slots": ["On rising", "Breakfast", "Bedtime"],
        "entries": [
            {"name": "Sulfur Syntropy", "dosage": "10 drops", "frequency": "3x a day",
             "timing": "before food", "slots": ["On rising"], "food": "before food", "as_directed": False},
            {"name": "MSM Lotion", "dosage": "1 application", "frequency": "daily",
             "timing": "", "slots": ["Breakfast"], "food": "", "as_directed": False},
            {"name": "Reverse AGE", "dosage": "1 cap", "frequency": "as needed",
             "timing": "", "slots": [], "food": "", "as_directed": True},
        ],
    },
}

def test_full_document_with_branding_and_order():
    html = render_present(REPORT, narrative="You are healing beautifully.")
    assert html.lstrip().lower().startswith("<!doctype html")
    # branding verbatim
    assert "Accelerated Self Healing™" in html
    assert "In wellness, Dr. Glen &amp; Rae · illtowell.com" in html
    # schedule-forward: Remedy Schedule heading appears before Narrative, before Causal Chain
    i_sched = html.index("Remedy Schedule")
    i_narr = html.index("Narrative")
    i_chain = html.index("Causal Chain")
    assert i_sched < i_narr < i_chain
    # client + date in masthead
    assert "Kauilani Perdomo" in html and "2026-06-24" in html

def test_no_edit_chrome():
    html = render_present(REPORT, narrative="x")
    for forbidden in ("<button", "<textarea", "<input", "<nav", "onclick"):
        assert forbidden not in html.lower()

def test_schedule_slots_and_as_directed():
    html = render_present(REPORT, narrative="x")
    assert "On rising" in html and "Sulfur Syntropy" in html and "before food" in html
    assert "As directed" in html and "Reverse AGE" in html

def test_chain_table_and_escaping():
    rep = {**REPORT, "client": {"name": "A & B <x>", "email": ""}}
    html = render_present(rep, narrative="")
    assert "A &amp; B &lt;x&gt;" in html           # escaped
    assert "Sulfur Syntropy" in html and "ED7 Lungs" in html

def test_empty_narrative_section_omitted_or_placeholder():
    html = render_present(REPORT, narrative="")
    # narrative heading still present (section exists), no crash
    assert "Narrative" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_biofield_report_present.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.biofield_report_present'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/biofield_report_present.py
"""Clean, client-facing presentation of a Biofield report — no editing chrome.
Shared renderer for the print/PDF (and later portal) skins. Pure function:
takes the report dict + narrative text, returns a complete print-styled HTML doc.
Section order is schedule-forward (the printed sheet ships with the bottles)."""

WORDMARK = "Accelerated Self Healing™"
FOOTER = "In wellness, Dr. Glen & Rae · illtowell.com"


def _e(s):
    return (str("" if s is None else s)
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


_CSS = """
  :root { --ink:#1a1a1a; --muted:#6b6b6b; --gold:#9a7a1f; --hair:#e6e1d5; }
  * { box-sizing: border-box; }
  body { font: 14px/1.5 Georgia, 'Times New Roman', serif; color: var(--ink);
         max-width: 760px; margin: 0 auto; padding: 28px; }
  .masthead { text-align: center; border-bottom: 2px solid var(--gold); padding-bottom: 12px; margin-bottom: 18px; }
  .masthead .wordmark { font-weight: 700; letter-spacing: .04em; font-size: 20px; }
  .masthead .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
  h2 { color: var(--gold); font-size: 16px; border-bottom: 1px solid var(--hair); padding-bottom: 3px; margin: 22px 0 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 4px 7px; border-bottom: 1px solid var(--hair); vertical-align: top; }
  th { background: #faf7ef; }
  .slot { white-space: nowrap; font-weight: 600; }
  .food { color: var(--muted); }
  .narrative p { margin: 0 0 9px; }
  .footer { margin-top: 26px; border-top: 1px solid var(--hair); padding-top: 8px;
            text-align: center; color: var(--muted); font-size: 12px; }
  @media print {
    body { padding: 0; max-width: none; }
    h2 { page-break-after: avoid; }
    tr, .narrative p { page-break-inside: avoid; }
  }
"""


def _masthead(report):
    c = report.get("client") or {}
    sub = " · ".join(x for x in (_e(c.get("name")), _e(report.get("date"))) if x)
    return (f'<div class="masthead"><div class="wordmark">{_e(WORDMARK)}</div>'
            f'<div class="sub">Biofield Analysis · {sub}</div></div>')


def _schedule(report):
    sched = report.get("schedule") or {}
    entries = sched.get("entries") or []
    placed = [e for e in entries if not e.get("as_directed")]
    rows = ""
    for slot in sched.get("slots") or []:
        here = [e for e in placed if slot in (e.get("slots") or [])]
        if not here:
            continue
        cells = "; ".join(
            f"{_e(e.get('name'))} <span class=food>({_e(e.get('dosage'))}"
            + (f", {_e(e.get('food'))}" if e.get("food") else "") + ")</span>"
            for e in here)
        rows += f"<tr><td class=slot>{_e(slot)}</td><td>{cells}</td></tr>"
    asdir = [e for e in entries if e.get("as_directed")]
    if asdir:
        cells = "; ".join(
            f"{_e(e.get('name'))} <span class=food>({_e(e.get('timing') or 'as directed')})</span>"
            for e in asdir)
        rows += f"<tr><td class=slot>As directed</td><td>{cells}</td></tr>"
    return ("<h2>Remedy Schedule</h2>"
            "<table><tr><th>When</th><th>Take</th></tr>" + rows + "</table>")


def _narrative(narrative):
    paras = "".join(f"<p>{_e(p.strip())}</p>" for p in (narrative or "").split("\n") if p.strip())
    return f'<h2>Narrative</h2><div class="narrative">{paras}</div>'


def _chain(report):
    rows = ""
    for l in report.get("layers") or []:
        ln = l.get("layer")
        rows += ("<tr>"
                 f"<td>{_e(ln) if ln is not None else '·'}</td>"
                 f"<td>{_e(l.get('head'))}</td>"
                 f"<td>{_e(l.get('most_affected'))}</td>"
                 f"<td>{_e(l.get('remedy'))}</td>"
                 f"<td>{_e(l.get('dosage'))}</td>"
                 f"<td>{_e(l.get('frequency'))}</td>"
                 f"<td>{_e(l.get('timing'))}</td>"
                 "</tr>")
    return ("<h2>Causal Chain</h2>"
            "<table><tr><th>Layer</th><th>Head</th><th>Most Affected</th><th>Remedy</th>"
            "<th>Dosage</th><th>Frequency</th><th>Timing</th></tr>" + rows + "</table>")


def render_present(report, narrative=""):
    body = (_masthead(report) + _schedule(report) + _narrative(narrative)
            + _chain(report) + f'<div class="footer">{_e(FOOTER)}</div>')
    return (f"<!doctype html><html><head><meta charset=utf-8>"
            f"<title>Biofield Analysis</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_biofield_report_present.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_report_present.py tests/test_biofield_report_present.py
git commit -m "biofield: clean schedule-forward presentation renderer (print skin)"
```

---

### Task 2: PDF generation via headless Chromium

**Files:**
- Create: `dashboard/biofield_report_pdf.py`
- Test: `tests/test_biofield_report_pdf.py`

**Interfaces:**
- Consumes: an HTML string (from `render_present`).
- Produces: `report_pdf_bytes(html: str) -> bytes` (Letter, backgrounds on) and `save_report_pdf(html: str, out_path: str) -> str` (writes the file, returns the path).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_report_pdf.py
import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="playwright not installed")

from dashboard.biofield_report_pdf import report_pdf_bytes, save_report_pdf

HTML = "<!doctype html><html><body><h1>Hello PDF</h1></body></html>"

def test_returns_real_pdf_bytes():
    data = report_pdf_bytes(HTML)
    assert isinstance(data, (bytes, bytearray))
    assert data[:5] == b"%PDF-"          # real PDF magic
    assert len(data) > 800

def test_save_writes_file(tmp_path):
    out = str(tmp_path / "r.pdf")
    p = save_report_pdf(HTML, out)
    assert p == out
    with open(out, "rb") as f:
        assert f.read(5) == b"%PDF-"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_biofield_report_pdf.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.biofield_report_pdf'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/biofield_report_pdf.py
"""Render a presentation HTML doc to a real PDF via headless Chromium (Playwright,
already installed in the local app's python3). Used for the printable/shippable
Biofield report. Local only; no network."""


def report_pdf_bytes(html: str) -> bytes:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            return page.pdf(
                format="Letter",
                print_background=True,
                margin={"top": "0.6in", "bottom": "0.6in", "left": "0.6in", "right": "0.6in"},
            )
        finally:
            browser.close()


def save_report_pdf(html: str, out_path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(report_pdf_bytes(html))
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_biofield_report_pdf.py -q`
Expected: PASS (2 tests). If Chromium isn't downloaded, run once: `python3 -m playwright install chromium`, then re-run.

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_report_pdf.py tests/test_biofield_report_pdf.py
git commit -m "biofield: HTML->PDF via headless Chromium (print artifact)"
```

---

### Task 3: Local-app routes — clean report view + downloadable/saved PDF

**Files:**
- Modify: `biofield_local_app.py` (add two routes near the existing `@app.route("/test/<test_id>")`, ~line 185; reuse `_report_for`, `get_narrative`)
- Test: `tests/test_biofield_report_routes.py`

**Interfaces:**
- Consumes: `render_present` (Task 1), `report_pdf_bytes`/`save_report_pdf` (Task 2), existing `_report_for(cx, test_id)` and `get_narrative(cx, test_id)`.
- Produces: `GET /test/<id>/report` → clean HTML (text/html); `GET /test/<id>/report.pdf` → `application/pdf` download, also saved to `~/biofield-reports/report_<id>_<date>.pdf`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_report_routes.py
import importlib, sqlite3, pytest
from datetime import datetime, timezone

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOFIELD_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setenv("BIOFIELD_REPORTS_DIR", str(tmp_path / "reports"))
    import biofield_local_app as bla
    importlib.reload(bla)
    app = bla.create_app(db_path=str(tmp_path / "chat_log.db"))
    # seed an authored test with one chain row + narrative
    from dashboard.biofield_authoring import create_test, add_chain_row
    from dashboard.biofield_narrative import save_narrative
    with sqlite3.connect(tmp_path / "chat_log.db") as cx:
        tid = create_test(cx, "Kauilani", "k@x.com", "2026-06-24")
        add_chain_row(cx, tid, 1, "ET4", "ET4", "MSM Lotion", "1 app", "daily", "am")
        save_narrative(cx, tid, "You are healing.")
    return app.test_client(), tid

def test_report_view_is_clean_html(client):
    c, tid = client
    r = c.get(f"/test/{tid}/report")
    assert r.status_code == 200 and r.mimetype == "text/html"
    body = r.get_data(as_text=True)
    assert "Accelerated Self Healing™" in body and "MSM Lotion" in body
    assert "<button" not in body.lower()

def test_report_pdf_downloads_and_saves(client, tmp_path):
    pytest.importorskip("playwright")
    c, tid = client
    r = c.get(f"/test/{tid}/report.pdf")
    assert r.status_code == 200 and r.mimetype == "application/pdf"
    assert r.get_data()[:5] == b"%PDF-"
    saved = list((tmp_path / "reports").glob("report_*.pdf"))
    assert saved, "PDF should also be saved locally for printing/shipping"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_biofield_report_routes.py -q`
Expected: FAIL — 404 on `/test/<id>/report` (route not defined)

- [ ] **Step 3: Write minimal implementation**

Add near the existing `report` route in `biofield_local_app.py` (after the `@app.route("/test/<test_id>")` block, ~line 193). Add imports at the top with the other `dashboard.biofield_*` imports:

```python
from dashboard.biofield_report_present import render_present
from dashboard.biofield_report_pdf import report_pdf_bytes, save_report_pdf
```

Routes (inside `create_app`, alongside the other `@app.route`s):

```python
    @app.route("/test/<test_id>/report")
    def report_present(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        return Response(render_present(rep, narrative), mimetype="text/html")

    @app.route("/test/<test_id>/report.pdf")
    def report_present_pdf(test_id):
        import os
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        html = render_present(rep, narrative)
        reports_dir = os.environ.get("BIOFIELD_REPORTS_DIR",
                                     os.path.join(os.path.expanduser("~"), "biofield-reports"))
        date = (rep.get("date") or "").replace("/", "-") or "undated"
        out = os.path.join(reports_dir, f"report_{test_id}_{date}.pdf")
        try:
            save_report_pdf(html, out)          # keep a local copy to print/ship
            data = open(out, "rb").read()
        except Exception as e:
            return Response(f"PDF generation failed: {e}", status=500)
        return Response(data, mimetype="application/pdf", headers={
            "Content-Disposition": f'inline; filename="biofield-{test_id}.pdf"'})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_biofield_report_routes.py -q`
Expected: PASS (PDF test skips if playwright/chromium unavailable)

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py tests/test_biofield_report_routes.py
git commit -m "biofield: /test/<id>/report (clean view) + /report.pdf (saved + download)"
```

---

### Task 4: "Print / PDF" link on the existing report page

**Files:**
- Modify: `dashboard/biofield_report_html.py` (the existing authoring/report screen — add a link to the clean view + PDF near the top header, ~after the `<h1>` block, ~line 112)
- Test: `tests/test_biofield_report_html_printlink.py`

**Interfaces:**
- Consumes: `report["test_id"]` (already in the render context).
- Produces: visible links to `/test/<id>/report` and `/test/<id>/report.pdf` on the working screen.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_report_html_printlink.py
from dashboard.biofield_report_html import render_report_html

REPORT = {"test_id": "a2", "client": {"name": "K", "email": ""}, "date": "2026-06-24",
          "layers": [], "schedule": {"slots": [], "entries": []}}

def test_has_print_and_pdf_links():
    html = render_report_html(REPORT, "", "", "")
    assert "/test/a2/report" in html        # clean view
    assert "/test/a2/report.pdf" in html    # printable PDF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_biofield_report_html_printlink.py -q`
Expected: FAIL — links not present

- [ ] **Step 3: Write minimal implementation**

In `dashboard/biofield_report_html.py`, find the header block that emits `f"<h1>{name}</h1>"` (~line 112) and append a links line right after the `<p class=sub>…</p>` that follows it:

```python
    tid_link = _e(report.get("test_id") or "")
    head += (f'<p class=sub><a href="/test/{tid_link}/report" target="_blank">Open clean report</a>'
             f' &nbsp;·&nbsp; <a href="/test/{tid_link}/report.pdf" target="_blank">Download printable PDF</a></p>')
```

(Insert where `head` is being assembled, after the existing email/date `<p class=sub>` line and before `chain`/`schedule` are concatenated.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_biofield_report_html_printlink.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_report_html.py tests/test_biofield_report_html_printlink.py
git commit -m "biofield: link to clean report + printable PDF from the working screen"
```

---

## Phase 1 acceptance

- `GET /test/<id>/report` shows the 3 sections clean, schedule-forward, branded, no edit chrome.
- `GET /test/<id>/report.pdf` returns a real PDF AND saves a copy under `~/biofield-reports/` to print + ship.
- The working screen links to both.
- Full local suite green: `python3 -m pytest tests/test_biofield_report_present.py tests/test_biofield_report_pdf.py tests/test_biofield_report_routes.py tests/test_biofield_report_html_printlink.py -q`

## Out of scope (later phases, per the spec)

Phase 2 publish→portal (magic-link, collapse + light/dark memory, email notification); Phase 3 audio; Phase 4 chat Q&A; Phase 5 on-request server-side video. This plan changes nothing on the server and no DB schema.
