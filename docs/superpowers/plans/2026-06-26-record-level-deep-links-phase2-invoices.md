# Record-Level Deep-Links — Phase 2 (QBO Invoice/AR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make QBO accounts-receivable mentions in the money-cash briefing link to their exact row on `/console/money#receivables` (scrolled-to + highlighted).

**Architecture:** Reuses the Phase-1 citation-token registry verbatim. Add QBO AR to the money snapshot; mint `type:"invoice"` linkables (`/console/money?invoice=<id>#receivables`); the existing `resolveRefLinks`/`recNavigate`/sidecar path carries them with NO dashboard change. Only new UI work is a `?invoice=` row-highlight on the AR board.

**Tech Stack:** Python 3 (Flask app, plain-function modules), pytest, vanilla JS in `static/console-money.html`.

**Spec:** `docs/superpowers/specs/2026-06-26-record-level-deep-links-phase2-invoices.md`
**Builds on (already merged):** Phase 1 — `dashboard/briefing_links.py` (`person_url`, `_is_person_email`, `build_linkables`), `intelligence.py` links sidecar, `dashboard.html` `resolveRefLinks`+`recNavigate`.

## Global Constraints

- **Reuse the mechanism — no dashboard render changes.** Do NOT touch `static/dashboard.html`, `resolveRefLinks`, or `recNavigate`. Invoice links are just registry entries resolving to a URL.
- **No console key in stored files or the resolved href.** `invoice_url` returns a bare `/console/money?invoice=<id>#receivables`; the key is appended at click time by the existing `recNavigate`.
- **Graceful degradation.** `money.qbo_ar` may be a list (success) OR a `{"_error": …}` dict (off-prod / QBO token failure). No invoice linkables minted on `_error`; no crash. Unknown refs already unwrap client-side (Phase 1).
- **Real-shape data.** QBO AR rows match `finance.aging()` output: `{id, doc, customer, email, total, balance, due_date, days_overdue}`. `open_invoices()` is **no-arg** and returns a **list** of these rows.
- **No double-counting in the prompt.** QBO AR (`money.qbo_ar`) is the accounts-receivable/overdue/collections source (linkable); Practice Better (`money.practice_better`) is separate clinical-billing activity (collected/outstanding, person-linked). Do not report the same dollars twice.
- **Ref numbering:** people are minted first (preserving Phase-1 `r1..` for person-only snapshots), then invoices; dedup by url; shared counter.

---

### Task 1: `briefing_links.py` — invoice linkables

**Files:**
- Modify: `dashboard/briefing_links.py` (add `invoice_url`, `_iter_invoice_records`; refactor `build_linkables` to a shared mint + an invoice pass)
- Test: `tests/test_briefing_links.py` (append)

**Interfaces:**
- Consumes: existing `person_url`, `_is_person_email`, `_iter_person_records`.
- Produces:
  - `invoice_url(qbo_id) -> str` → `"/console/money?invoice=<urlencoded-id>#receivables"`.
  - `build_linkables(snapshot)` now also stamps `rec["ref"]` on `money.qbo_ar` rows and adds `{type:"invoice", display, url}` registry entries (people minted first, then invoices; dedup by url).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_briefing_links.py
def test_invoice_url_encodes_id():
    assert bl.invoice_url("123") == "/console/money?invoice=123#receivables"


def test_build_linkables_mints_invoice_links_from_qbo_ar():
    snap = {"money": {"qbo_ar": [
        {"id": "501", "doc": "1024", "customer": "Acme Co",
         "email": "ar@acme.com", "balance": 5000.0, "days_overdue": 32},
        {"id": "502", "doc": "1025", "customer": "", "balance": 90.0,
         "days_overdue": 3},
    ]}}
    reg = bl.build_linkables(snap)
    rows = snap["money"]["qbo_ar"]
    assert reg[rows[0]["ref"]] == {"type": "invoice", "display": "Acme Co",
                                   "url": "/console/money?invoice=501#receivables"}
    # no customer -> display falls back to "Invoice <doc>"
    assert reg[rows[1]["ref"]]["display"] == "Invoice 1025"
    assert reg[rows[1]["ref"]]["url"] == "/console/money?invoice=502#receivables"
    assert all(v["type"] == "invoice" for v in reg.values())


def test_build_linkables_qbo_ar_error_block_is_safe():
    snap = {"money": {"qbo_ar": {"_error": "qbo_ar: HTTPError"}}}
    assert bl.build_linkables(snap) == {}  # _error is a dict, not a list -> skipped


def test_build_linkables_mixed_person_and_invoice_share_counter():
    snap = {
        "inbox": {"oldest": [{"from": "Real Client <client@example.com>", "age_days": 4}]},
        "money": {"qbo_ar": [{"id": "77", "doc": "9", "customer": "Beta LLC",
                              "balance": 200.0, "days_overdue": 10}]},
    }
    reg = bl.build_linkables(snap)
    # person minted first (r1), invoice second (r2)
    assert snap["inbox"]["oldest"][0]["ref"] == "r1"
    assert snap["money"]["qbo_ar"][0]["ref"] == "r2"
    assert reg["r1"]["type"] == "person"
    assert reg["r2"]["type"] == "invoice"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_briefing_links.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.briefing_links' has no attribute 'invoice_url'`.

- [ ] **Step 3: Add `invoice_url` + `_iter_invoice_records`**

In `dashboard/briefing_links.py`, add `invoice_url` right after `person_url`:

```python
def invoice_url(qbo_id):
    """Canonical console destination for a QBO accounts-receivable invoice: the
    receivables board, deep-linked to highlight one row. No console key (appended
    client-side at click time)."""
    return "/console/money?invoice=" + quote(str(qbo_id or ""), safe="") + "#receivables"
```

And add an invoice iterator next to `_iter_person_records`:

```python
def _iter_invoice_records(snapshot):
    """Yield (record_dict, display, qbo_id) for each QBO accounts-receivable
    invoice in the snapshot. `money.qbo_ar` is a list of finance.aging() rows on
    success, or a {"_error": ...} dict on failure (skipped)."""
    ar = (snapshot.get("money") or {}).get("qbo_ar")
    rows = ar if isinstance(ar, list) else None
    for rec in (rows or []):
        if isinstance(rec, dict) and rec.get("id"):
            display = rec.get("customer") or ("Invoice " + str(rec.get("doc") or rec.get("id")))
            yield rec, display, rec.get("id")
```

- [ ] **Step 4: Refactor `build_linkables` to add the invoice pass**

Replace the existing `build_linkables` body with a shared mint + two passes:

```python
def build_linkables(snapshot):
    """Stamp `ref` onto each linkable record and return the registry
    {ref: {type, display, url}}. People (inbox senders + PB invoice clients) are
    minted first, then QBO accounts-receivable invoices. Dedup by url. Mutates
    `snapshot`."""
    registry = {}
    url_to_ref = {}
    state = {"n": 0}

    def mint(rec, kind, display, url):
        ref = url_to_ref.get(url)
        if ref is None:
            state["n"] += 1
            ref = "r%d" % state["n"]
            url_to_ref[url] = ref
            registry[ref] = {"type": kind, "display": display, "url": url}
        rec["ref"] = ref

    for rec, display, email in _iter_person_records(snapshot):
        email = (email or "").strip().lower()
        if not _is_person_email(email):
            continue
        mint(rec, "person", (display or email).strip() or email, person_url(email))

    for rec, display, qid in _iter_invoice_records(snapshot):
        qid = str(qid or "").strip()
        if not qid:
            continue
        mint(rec, "invoice", display or ("Invoice " + qid), invoice_url(qid))

    return registry
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_briefing_links.py -q`
Expected: PASS (all Phase-1 tests + the 4 new ones). Phase-1 person-only snapshots still mint `r1..` (invoice pass adds nothing).

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add dashboard/briefing_links.py tests/test_briefing_links.py
git commit -m "feat: invoice (QBO AR) linkables in briefing_links registry"
```

---

### Task 2: Add QBO AR to the snapshot + reword the money-cash prompt

**Files:**
- Modify: `dashboard/briefing_runner.py` (import `finance`; add `qbo_ar` to the money block in `gather_snapshot`; reword `SLUG_PROMPTS["money-cash"]`)
- Test: `tests/test_briefing_runner_links.py` (append source-assert tests — `gather_snapshot` hits QBO/network so it is not run offline)

**Interfaces:**
- Consumes: `dashboard.finance.open_invoices` (no-arg, returns a list of AR rows); `briefing_links.build_linkables` (Task 1, stamps refs on `money.qbo_ar`).
- Produces: `gather_snapshot()["money"]["qbo_ar"]` present; the money-cash prompt instructs QBO-AR-as-receivables.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_briefing_runner_links.py
def test_money_snapshot_includes_qbo_ar():
    src = (_repo() / "dashboard" / "briefing_runner.py").read_text()
    assert "import finance as _finance" in src
    assert '"qbo_ar"' in src
    assert "_finance.open_invoices" in src


def test_money_prompt_uses_qbo_ar_for_receivables():
    from dashboard import briefing_runner as br
    p = br.SLUG_PROMPTS["money-cash"]
    assert "qbo_ar" in p                      # AR comes from the QBO block
    assert "practice_better" in p             # PB still named, as separate activity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_runner_links.py -q`
Expected: FAIL — `qbo_ar`/`_finance` not present yet.
(`PINECONE_API_KEY=dummy` because importing `briefing_runner` constructs `Pinecone()` at import.)

- [ ] **Step 3a: Import finance + add `qbo_ar` to the snapshot**

In `dashboard/briefing_runner.py`, add to the imports block (after `from . import money as _money`):

```python
from . import finance as _finance
```

In `gather_snapshot()`, the money block currently ends:

```python
            "practice_better": _safe(lambda: _money.pb_data(days=30), label="pb_data"),
            "authorize_net":   _safe(lambda: _money.an_data(days=30), label="an_data"),
        },
```

Change it to add `qbo_ar`:

```python
            "practice_better": _safe(lambda: _money.pb_data(days=30), label="pb_data"),
            "authorize_net":   _safe(lambda: _money.an_data(days=30), label="an_data"),
            "qbo_ar":          _safe(_finance.open_invoices, label="qbo_ar"),
        },
```

- [ ] **Step 3b: Reword the money-cash prompt**

In `dashboard/briefing_runner.py`, `SLUG_PROMPTS["money-cash"]` currently reads:

```python
    "money-cash": (
        "You write the Finance card. The first line must be exactly '# Finance' "
        "and nothing else. "
        "Cover ONLY money, from the snapshot's `money` block: the combined cash "
        "position (bank balances + Wise together), money in today and over 7 "
        "days, accounts receivable / overdue, and any Practice Better or "
        "Authorize.net processor issues. Name the single revenue constraint "
        "(the Schwerpunkt) limiting cash this week. Do NOT cover pipeline, "
        "leads, or system health; other cards own those. State each figure "
        "once. Then one short '## Insight' line on runway / trend. Voice: "
        "calm, precise, money-first."
    ),
```

Replace it with:

```python
    "money-cash": (
        "You write the Finance card. The first line must be exactly '# Finance' "
        "and nothing else. "
        "Cover ONLY money, from the snapshot's `money` block: the combined cash "
        "position (bank balances + Wise together) and money in today and over 7 "
        "days. ACCOUNTS RECEIVABLE / OVERDUE comes from `money.qbo_ar` "
        "(QuickBooks open invoices; each row has customer, balance, days_overdue, "
        "doc); name the most overdue by customer and amount, oldest first. "
        "`money.practice_better` is SEPARATE clinical-billing activity "
        "(collected / outstanding) and `money.authorize_net` is processor "
        "settlement; mention processor issues if any, but do NOT report Practice "
        "Better or Authorize.net figures as accounts receivable and do NOT repeat "
        "the same dollars twice. Name the single revenue constraint (the "
        "Schwerpunkt) limiting cash this week. Do NOT cover pipeline, leads, or "
        "system health; other cards own those. State each figure once. Then one "
        "short '## Insight' line on runway / trend. Voice: calm, precise, "
        "money-first."
    ),
```

(The Phase-1 RECORD LINKS instruction in `_build_user_prompt` already makes the LLM link any record carrying a `ref`, so the QBO AR rows — now stamped by Task 1 — get linked once the card discusses them. No prompt change needed for linking itself.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_runner_links.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add dashboard/briefing_runner.py tests/test_briefing_runner_links.py
git commit -m "feat: add QBO AR to money snapshot; money-cash prompt reports AR from QBO"
```

---

### Task 3: `console-money.html` — receivables row highlight (`?invoice=`)

**Files:**
- Modify: `static/console-money.html` (`data-inv` on AR rows; `?invoice=` scroll+flash after receivables render; flash CSS)
- Test: `tests/test_console_money_invoice_highlight.py` (source-assert; the live DOM behavior is render-verified in Task 4)

**Interfaces:**
- Consumes: arrival URL `/console/money?invoice=<id>#receivables` produced by `invoice_url` + `recNavigate`.
- Produces: the AR row with matching `data-inv` scrolls into view and flashes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_money_invoice_highlight.py
from pathlib import Path


def _html():
    return (Path(__file__).resolve().parent.parent / "static" / "console-money.html").read_text()


def test_ar_row_has_data_inv():
    assert 'data-inv="' in _html()


def test_receivables_reads_invoice_param_and_flashes():
    html = _html()
    assert "URLSearchParams(location.search).get('invoice')" in html
    assert "scrollIntoView" in html
    assert "inv-flash" in html  # the transient highlight class
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_console_money_invoice_highlight.py -q`
Expected: FAIL — none of those strings present yet.

- [ ] **Step 3a: Add `data-inv` to the AR row**

In `static/console-money.html`, the receivables `rowHtml` (the `MoneyReceivables` module) currently opens the row with:

```javascript
      return '<div class="ar-row">'
```

Change it to include the invoice id (the `iid` var is already computed just above as `var iid = Number(r.id);`):

```javascript
      return '<div class="ar-row" data-inv="'+iid+'">'
```

- [ ] **Step 3b: Add the scroll+flash after rows render**

In `MoneyReceivables.load()`, the rows currently render at:

```javascript
      var rows = res.data||[];
      if (!rows.length){ board.innerHTML=''; emptyEl.style.display='block'; return; }
      board.innerHTML = rows.map(rowHtml).join('');
    }
```

Change the tail to call a highlight helper after rendering, and add the helper inside the `MoneyReceivables` IIFE (e.g. just before `return { load: load, ... };`):

```javascript
      var rows = res.data||[];
      if (!rows.length){ board.innerHTML=''; emptyEl.style.display='block'; return; }
      board.innerHTML = rows.map(rowHtml).join('');
      maybeHighlight();
    }

    function maybeHighlight(){
      var id = new URLSearchParams(location.search).get('invoice');
      if (!id) return;
      var row = document.querySelector('.ar-row[data-inv="' + (window.CSS && CSS.escape ? CSS.escape(id) : id) + '"]');
      if (!row) return;
      row.scrollIntoView({behavior:'smooth', block:'center'});
      row.classList.add('inv-flash');
      setTimeout(function(){ row.classList.remove('inv-flash'); }, 2600);
    }
```

- [ ] **Step 3c: Add the flash CSS**

In the `<style>` block of `static/console-money.html`, add near the `.ar-row` rules:

```css
@keyframes inv-flash {
  0%   { background: rgba(212,175,55,0.45); }
  100% { background: transparent; }
}
.ar-row.inv-flash { animation: inv-flash 2.4s ease-out; border-radius: 8px; }
```

(Gold flash to match the console palette. If a different accent variable is used by surrounding rules, match it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_console_money_invoice_highlight.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e0f30eb0
git add static/console-money.html tests/test_console_money_invoice_highlight.py
git commit -m "feat: highlight the targeted AR row on /console/money?invoice="
```

---

### Task 4: Render-verify + prod go-live gate

Verification only (no code). Confirm the real DOM behavior and the live path, per the render-verify rule.

**Files:** none (capture findings in the PR description).

- [ ] **Step 1: Run the full Phase-2-touched suite**

Run: `cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_links.py tests/test_briefing_runner_links.py tests/test_console_money_invoice_highlight.py -q`
Expected: all PASS.

- [ ] **Step 2: Boot the app locally with a scratch DATA_DIR**

```bash
mkdir -p /tmp/dc2-scratch/intelligence
cat > /tmp/run-dc2.sh <<'SH'
cd /tmp/wt-deploy-chat-e0f30eb0
exec doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc2-scratch PORT=5078 python3 app.py
SH
bash /tmp/run-dc2.sh
```
(Run in the background; `app.py` validates Pinecone at import, so it needs Doppler secrets and a writable `DATA_DIR`.)

- [ ] **Step 3: Render-verify the AR row highlight in a browser**

Using the claude-in-chrome tools: the `/api/finance/ar` endpoint serves live QBO AR. Open `http://localhost:5078/console/money?invoice=<id>&key=<CONSOLE_SECRET>#receivables` where `<id>` is the `id` of a real row returned by `/api/finance/ar` (fetch that first, authed, to pick a real id). Assert:
- the receivables tab is active (panel-receivables visible),
- the `.ar-row[data-inv="<id>"]` exists, is scrolled into view, and received the `inv-flash` class (check shortly after load),
- zero console errors.
If QBO has no open invoices to target, note that and verify instead that `?invoice=` with a non-existent id is a safe no-op (no error, tab still renders).

- [ ] **Step 4: Stop the app + clean up**

```bash
pkill -f "DATA_DIR=/tmp/dc2-scratch" 2>/dev/null; lsof -ti tcp:5078 | xargs kill 2>/dev/null
rm -rf /tmp/dc2-scratch /tmp/run-dc2.sh
```

- [ ] **Step 5: Prod go-live (after merge + deploy)**

After the PR merges and Render deploys, trigger a regen (authed `POST /api/regenerate-briefings`) and confirm the money-cash registry now contains `type:"invoice"` entries with `/console/money?invoice=…#receivables` urls — only when QBO AR has open invoices. Record the result. If the live Haiku under-links AR despite the reworded prompt, that's a prompt tweak, not a code change (unlinked mentions render as plain text).

---

## Notes for the implementer

- **No `dashboard.html` change.** Invoice links resolve through the existing `resolveRefLinks` and navigate through the existing `recNavigate` (which already produces `/console/money?invoice=123&key=…#receivables` from a query+hash href). Do not modify the dashboard renderer.
- **Run pytest from the worktree** (`/tmp/wt-deploy-chat-e0f30eb0`). `briefing_links` and the HTML source-assert tests are pure; the `briefing_runner` import needs `PINECONE_API_KEY=dummy`.
- **`money.qbo_ar` shape:** a list on success, `{"_error": …}` on failure — `_iter_invoice_records` already guards with `isinstance(ar, list)`.
- **No app.py change** — `/api/finance/ar` and the serve route already exist.
