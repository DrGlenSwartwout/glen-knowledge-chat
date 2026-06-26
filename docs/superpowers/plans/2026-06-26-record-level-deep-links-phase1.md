# Record-Level Dashboard Deep-Links — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make person/client mentions in the dashboard's LLM briefings render as links that land on that person's record in the console.

**Architecture:** The snapshot fed to Claude carries a per-record `ref` token; the prompt tells Claude to write those mentions as markdown links (`[Jane](ref:r3)`). The registry (`ref → {type, display, url}`) is persisted beside the briefing markdown, served in the same JSON envelope, and resolved client-side in `mdRender`'s output — unknown refs unwrap to plain text. The console person destination is `/console/crm?email=`, which gains a tiny `?email=` autoload.

**Tech Stack:** Python 3 (Flask app, no framework for the briefing modules — plain functions), pytest (+ `subprocess` to run `node` for the one JS behavioral test), vanilla JS embedded in `static/dashboard.html`.

**Spec:** `docs/superpowers/specs/2026-06-26-record-level-deep-links-design.md`

## Global Constraints

These apply to every task:

- **Graceful degradation, always.** An unknown/missing `ref`, a missing `links` map, a record with no email, or an `_error` block in the snapshot must never break rendering or generation — it falls back to today's plain-text behavior. Never emit a broken or fabricated link.
- **No secret in stored files.** The persisted `url` is a path+query with **no** console key. The key is appended client-side at click time (mirroring the existing `actNavigate`), never written into `{slug}.md`, `{slug}.links.json`, or the DOM `href` attribute.
- **The model never writes URLs.** It only emits opaque `ref:rN` tokens that already appear in the snapshot. Resolution happens registry-side.
- **Real-shape mocks.** Test data must mirror the real structures verified in the spec investigation: inbox senders use the field name `from` (NOT `email`); PB invoice entries use `name` + (new) `email`. A mock that mirrors a wrong assumption passes while prod breaks.
- **Render-verify the frontend (Task 6).** The frontend is confirmed by rendering the actual DOM in a browser with zero console errors — not by confirming the script is served.
- **Person destination:** `/console/crm?email=<urlencoded-email>` (no key). One source of truth: `briefing_links.person_url`.
- **Scope:** people-by-email only. Money receivables/invoices, orders, payments are later phases that reuse this mechanism — do NOT build them here.

---

### Task 1: `dashboard/briefing_links.py` — registry + `person_url` + `build_linkables`

This is the shared foundation: a pure module (no network, no Flask) that turns a snapshot into a registry and stamps `ref`s onto records.

**Files:**
- Create: `dashboard/briefing_links.py`
- Test: `tests/test_briefing_links.py`

**Interfaces:**
- Produces:
  - `person_url(email: str) -> str` — returns `"/console/crm?email=" + urlencode(email)`.
  - `build_linkables(snapshot: dict) -> dict` — mutates `snapshot` in place, stamping `rec["ref"] = "rN"` onto each person record that has an email; returns the registry `{ "rN": {"type": "person", "display": str, "url": str}, ... }`. Deduplicates by `url` (same person → same ref). Records without a valid email get no ref. `_error` blocks are skipped safely.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_briefing_links.py
from dashboard import briefing_links as bl


def test_person_url_encodes_email():
    assert bl.person_url("jane@x.com") == "/console/crm?email=jane%40x.com"


def test_build_linkables_stamps_inbox_senders():
    snap = {"inbox": {"oldest": [
        {"subject": "Re: order", "from": "jane@x.com", "age_days": 5},
        {"subject": "hi", "from": "bob@y.com", "age_days": 2},
    ]}}
    reg = bl.build_linkables(snap)
    refs = [r["ref"] for r in snap["inbox"]["oldest"]]
    assert refs == ["r1", "r2"]
    assert reg["r1"] == {"type": "person", "display": "jane@x.com",
                         "url": "/console/crm?email=jane%40x.com"}


def test_build_linkables_stamps_pb_invoice_clients_with_email():
    snap = {"money": {"practice_better": {"invoices": [
        {"name": "Jane Doe", "email": "jane@x.com", "invoice": "INV-1", "due": 50},
    ]}}}
    reg = bl.build_linkables(snap)
    ref = snap["money"]["practice_better"]["invoices"][0]["ref"]
    assert reg[ref]["display"] == "Jane Doe"
    assert reg[ref]["url"] == "/console/crm?email=jane%40x.com"


def test_build_linkables_dedupes_same_person_across_blocks():
    snap = {
        "inbox": {"oldest": [{"from": "jane@x.com", "age_days": 5}]},
        "money": {"practice_better": {"invoices": [
            {"name": "Jane Doe", "email": "jane@x.com", "due": 50}]}},
    }
    reg = bl.build_linkables(snap)
    assert snap["inbox"]["oldest"][0]["ref"] == "r1"
    assert snap["money"]["practice_better"]["invoices"][0]["ref"] == "r1"
    assert list(reg.keys()) == ["r1"]


def test_build_linkables_skips_records_without_email():
    snap = {"money": {"practice_better": {"invoices": [
        {"name": "No Email Client", "invoice": "INV-9", "due": 99},
    ]}}}
    reg = bl.build_linkables(snap)
    assert "ref" not in snap["money"]["practice_better"]["invoices"][0]
    assert reg == {}


def test_build_linkables_survives_error_blocks():
    snap = {"inbox": {"_error": "inbox: TimeoutError: boom"},
            "money": {"practice_better": {"_error": "pb_data: KeyError"}}}
    assert bl.build_linkables(snap) == {}  # no crash, no refs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_briefing_links.py -v`
(Run from the worktree path you are working in.)
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.briefing_links'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/briefing_links.py
"""Linkable-record registry for the dashboard briefings.

Pure module (no network, no Flask). The briefing runner builds a registry from
the live snapshot, stamps a short `ref` token onto each person record, and the
LLM is told to cite those records by ref. The registry (ref -> url) is persisted
beside the briefing and resolved client-side. No console key is ever stored in
the url; it is appended at click time in the dashboard.
"""

from urllib.parse import quote


def person_url(email):
    """Canonical console destination for a person, keyed by email. The console
    key is intentionally absent; it is appended client-side at click time."""
    return "/console/crm?email=" + quote(email or "", safe="")


def _iter_person_records(snapshot):
    """Yield (record_dict, display, email) for each person-bearing record.
    Phase 1 sources: inbox oldest senders (email in `from`) and Practice Better
    invoice clients (email in `email`). `_error` blocks and missing keys are
    skipped. Add later-phase sources here."""
    inbox = snapshot.get("inbox") or {}
    for rec in (inbox.get("oldest") or []):
        if isinstance(rec, dict):
            yield rec, rec.get("from"), rec.get("from")
    pb = ((snapshot.get("money") or {}).get("practice_better") or {})
    for rec in (pb.get("invoices") or []):
        if isinstance(rec, dict):
            yield rec, rec.get("name"), rec.get("email")


def build_linkables(snapshot):
    """Stamp `ref` onto each person record that has an email and return the
    registry {ref: {type, display, url}}. Dedup by url. Mutates `snapshot`."""
    registry = {}
    url_to_ref = {}
    n = 0
    for rec, display, email in _iter_person_records(snapshot):
        email = (email or "").strip()
        if "@" not in email:
            continue
        url = person_url(email)
        ref = url_to_ref.get(url)
        if ref is None:
            n += 1
            ref = "r%d" % n
            url_to_ref[url] = ref
            registry[ref] = {"type": "person",
                             "display": (display or email).strip() or email,
                             "url": url}
        rec["ref"] = ref
    return registry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_briefing_links.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/briefing_links.py tests/test_briefing_links.py
git commit -m "feat: briefing_links registry + person_url (deep-link foundation)"
```

---

### Task 2: Surface PB email + wire registry into generation + prompt instruction

Makes `regenerate_all` build the registry, persist it, and instructs the LLM to cite records by ref. Also surfaces the PB client email so PB clients become linkable.

**Files:**
- Modify: `dashboard/money.py` (`pb_data`, the `recent.append({...})` block — add `email`)
- Modify: `dashboard/briefing_runner.py` (`_build_user_prompt` prompt text; `regenerate_all` registry build + `write_links` call; new import)
- Test: `tests/test_briefing_runner_links.py`

**Interfaces:**
- Consumes: `briefing_links.build_linkables` (Task 1); `intelligence.write_links` (Task 3 — wired here, lands when Task 3 ships).
- Produces: `_build_user_prompt(snapshot, slug)` output now contains the ref-citation instruction; `regenerate_all` calls `_intel.write_links(slug, registry)` after `write_briefing`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_briefing_runner_links.py
import sys
from pathlib import Path

from dashboard import briefing_runner as br


def _repo():
    return Path(__file__).resolve().parent.parent


def test_prompt_includes_ref_citation_instruction():
    snap = {"inbox": {"oldest": [{"from": "jane@x.com", "age_days": 5}]}}
    from dashboard import briefing_links as bl
    bl.build_linkables(snap)  # stamps ref so the snapshot shows it
    prompt = br._build_user_prompt(snap, "clients-pipeline")
    assert "ref" in prompt
    assert "(ref:" in prompt              # shows the markdown-link form
    assert "never write a real" in prompt.lower()
    assert '"ref": "r1"' in prompt        # the stamped ref is visible to the LLM


def test_regenerate_all_persists_links():
    # source-assert: the runner builds a registry and writes it per slug
    src = (_repo() / "dashboard" / "briefing_runner.py").read_text()
    assert "build_linkables" in src
    assert "write_links" in src


def test_pb_data_surfaces_client_email():
    # source-assert: pb_data's recent entry includes an email field
    src = (_repo() / "dashboard" / "money.py").read_text()
    assert 'client.get("email"' in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_briefing_runner_links.py -v`
Expected: FAIL — `assert "(ref:" in prompt` fails (instruction not yet added); source-assert tests fail.

- [ ] **Step 3a: Add the email field to `pb_data`**

In `dashboard/money.py`, inside `pb_data`, the `recent.append({...})` block currently is:

```python
            recent.append({
                "date": date_str[:10],
                "name": f"{client.get('firstName','')} {client.get('lastName','')}".strip(),
                "amount": total, "paid": paid, "due": due,
                "invoice": inv.get("invoiceNumber", "—"),
            })
```

Change it to add the email line:

```python
            recent.append({
                "date": date_str[:10],
                "name": f"{client.get('firstName','')} {client.get('lastName','')}".strip(),
                "email": client.get("email", ""),
                "amount": total, "paid": paid, "due": due,
                "invoice": inv.get("invoiceNumber", "—"),
            })
```

- [ ] **Step 3b: Add the ref-citation instruction to `_build_user_prompt`**

In `dashboard/briefing_runner.py`, in `_build_user_prompt`, find the line that ends the acronym/retired-pipeline rule:

```python
        f"Only reference pipelines and sources that appear in the snapshot; never "
        f"mention retired ones (e.g. MCTB, Email Paramedic).\n\n"
```

Immediately after it (before the `## Recommended actions` paragraph), insert:

```python
        f"RECORD LINKS: some records in the snapshot include a `ref` field (an "
        f"inbox sender, an invoice client). Whenever you mention such a record by "
        f"name or email in your prose or actions, write that mention as a markdown "
        f"link using its ref as the URL, e.g. `[Jane Doe](ref:r3)` or "
        f"`[jane@example.com](ref:r3)`. Use ONLY a `ref` value that actually appears "
        f"in the snapshot; never invent one and never write a real web address. A "
        f"record with no `ref` is mentioned as plain text.\n\n"
```

- [ ] **Step 3c: Build + persist the registry in `regenerate_all`**

In `dashboard/briefing_runner.py`, add to the imports block (near the other `from . import ...` lines):

```python
from . import briefing_links as _links
```

In `regenerate_all`, the current lines are:

```python
    snapshot = gather_snapshot()
    client = anthropic.Anthropic()
```

Change to:

```python
    snapshot = gather_snapshot()
    registry = _links.build_linkables(snapshot)   # stamps refs onto the snapshot
    client = anthropic.Anthropic()
```

Then in the result loop, the current success branch is:

```python
                markdown = fut.result()
                _intel.write_briefing(slug, markdown)
                _ba.reset_slug(slug)   # fresh briefing => clear handled-action state
                results[slug] = {"ok": True, "bytes": len(markdown)}
```

Change to:

```python
                markdown = fut.result()
                _intel.write_briefing(slug, markdown)
                _intel.write_links(slug, registry)
                _ba.reset_slug(slug)   # fresh briefing => clear handled-action state
                results[slug] = {"ok": True, "bytes": len(markdown)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_briefing_runner_links.py -v`
Expected: PASS (3 passed). Note: `_build_user_prompt` is pure (no network), so the prompt test runs offline.

- [ ] **Step 5: Commit**

```bash
git add dashboard/money.py dashboard/briefing_runner.py tests/test_briefing_runner_links.py
git commit -m "feat: build+persist link registry in briefing runner; cite-by-ref prompt; PB client email"
```

---

### Task 3: `intelligence.py` — persist + serve the links sidecar

Stores the registry as `{slug}.links.json` and includes it in `read_briefing`, so the existing `/api/intelligence/<slug>` route serves it with no route change.

**Files:**
- Modify: `dashboard/intelligence.py` (add `_links_path`, `write_links`, `read_links`; include `links` in `read_briefing`)
- Test: `tests/test_intelligence_links.py`

**Interfaces:**
- Produces:
  - `write_links(slug: str, links: dict) -> dict` — writes `{slug}.links.json`.
  - `read_links(slug: str) -> dict` — returns the registry, `{}` if absent or unreadable.
  - `read_briefing(slug)` return dict now includes `"links": <registry>` when the briefing exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intelligence_links.py
import json
from pathlib import Path

from dashboard import intelligence as intel


def _use_tmp(monkeypatch, tmp_path):
    d = tmp_path / "intelligence"
    d.mkdir()
    monkeypatch.setattr(intel, "DATA_DIR", d)
    return d


def test_write_then_read_links_roundtrips(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    reg = {"r1": {"type": "person", "display": "Jane",
                  "url": "/console/crm?email=jane%40x.com"}}
    intel.write_links("money-cash", reg)
    assert intel.read_links("money-cash") == reg


def test_read_links_missing_returns_empty(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    assert intel.read_links("money-cash") == {}


def test_read_briefing_includes_links(monkeypatch, tmp_path):
    d = _use_tmp(monkeypatch, tmp_path)
    (d / "money-cash.md").write_text("# Finance\n\n[Jane](ref:r1)")
    intel.write_links("money-cash", {"r1": {"type": "person", "display": "Jane",
                                            "url": "/console/crm?email=jane%40x.com"}})
    data = intel.read_briefing("money-cash")
    assert data["links"]["r1"]["url"] == "/console/crm?email=jane%40x.com"


def test_read_briefing_empty_has_no_links(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    data = intel.read_briefing("money-cash")
    assert data["empty"] is True
    assert "links" not in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_intelligence_links.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.intelligence' has no attribute 'write_links'`

- [ ] **Step 3: Write minimal implementation**

In `dashboard/intelligence.py`, add after `_slug_path`:

```python
def _links_path(slug):
    if slug not in VALID_SLUGS:
        raise ValueError(f"Unknown slug: {slug}. Valid: {sorted(VALID_SLUGS)}")
    return DATA_DIR / f"{slug}.links.json"


def write_links(slug, links):
    """Persist the link registry (ref -> {type, display, url}) for a briefing."""
    p = _links_path(slug)
    p.write_text(json.dumps(links or {}))
    return {"slug": slug, "links": len(links or {})}


def read_links(slug):
    """Return the link registry for a briefing, or {} if absent/unreadable."""
    p = _links_path(slug)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
```

Then in `read_briefing`, the current non-empty return is:

```python
    return {
        "slug": slug,
        "empty": False,
        "markdown": p.read_text(),
        "generated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "bytes": stat.st_size,
    }
```

Change to add the `links` key:

```python
    return {
        "slug": slug,
        "empty": False,
        "markdown": p.read_text(),
        "links": read_links(slug),
        "generated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "bytes": stat.st_size,
    }
```

(`json` is already imported at the top of `intelligence.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_intelligence_links.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/intelligence.py tests/test_intelligence_links.py
git commit -m "feat: persist + serve briefing link registry sidecar"
```

---

### Task 4: `dashboard.html` — resolve ref links + key-appending navigation

Resolves `ref:rN` anchors (produced by `mdRender`) against the served `links` map, adds the `.rec-link` class, unwraps unknown refs, and navigates with the console key appended at click time.

**Files:**
- Modify: `static/dashboard.html` (add `resolveRefLinks` + `recNavigate`; wire `R.briefing`; optional `.rec-link` style)
- Test: `tests/test_record_link_resolver_js.py` (extracts `resolveRefLinks` and runs it under `node`)

**Interfaces:**
- Consumes: `mdRender(src)` output (existing, line ~917 — already renders `[text](url)` to `<a href="url" target="_blank" rel="noopener">text</a>`); `d.links` from the briefing envelope (Task 3); the existing `consoleKey` global.
- Produces: `resolveRefLinks(html, links)`, `recNavigate(el, e)`. `R.briefing` now calls `resolveRefLinks(mdRender(...), d.links)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_record_link_resolver_js.py
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def _repo():
    return Path(__file__).resolve().parent.parent


def _extract_resolver():
    html = (_repo() / "static" / "dashboard.html").read_text()
    m = re.search(
        r"/\* === record-link resolver \(test-extracted\) === \*/(.*?)"
        r"/\* === end record-link resolver === \*/",
        html, re.S)
    assert m, "resolver marker block not found in dashboard.html"
    return m.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_resolver_behaviour():
    fn = _extract_resolver()
    script = fn + textwrap.dedent("""
      const links = {r1: {type:"person", display:"Jane",
                          url:"/console/crm?email=jane%40x.com"}};
      function assert(c, m){ if(!c){ console.error("FAIL: "+m); process.exit(1); } }

      // known ref -> anchor with rec-link class + url, no target=_blank
      let out = resolveRefLinks(
        '<a href="ref:r1" target="_blank" rel="noopener">Jane</a>', links);
      assert(out.indexOf('class="rec-link"') >= 0, "rec-link class");
      assert(out.indexOf('href="/console/crm?email=jane%40x.com"') >= 0, "resolved url");
      assert(out.indexOf("Jane") >= 0, "link text kept");
      assert(out.indexOf("target=") < 0, "no target=_blank on record link");

      // unknown ref -> unwrapped to plain text
      out = resolveRefLinks(
        '<a href="ref:r9" target="_blank" rel="noopener">Bob</a>', links);
      assert(out === "Bob", "unknown ref unwrapped, got: " + out);

      // real (non-ref) link untouched
      const ext = '<a href="https://x.com" target="_blank" rel="noopener">x</a>';
      assert(resolveRefLinks(ext, links) === ext, "external link untouched");

      // missing links map -> unwrap, no crash
      assert(resolveRefLinks('<a href="ref:r1">Jane</a>') === "Jane", "no map unwraps");
      console.log("OK");
    """)
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_briefing_renderer_wires_resolver():
    html = (_repo() / "static" / "dashboard.html").read_text()
    assert "resolveRefLinks(mdRender(" in html
    assert "function recNavigate(" in html
    assert "encodeURIComponent(consoleKey)" in html  # key appended at click
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_record_link_resolver_js.py -v`
Expected: FAIL — `assert m, "resolver marker block not found"` (markers/functions not added yet).

- [ ] **Step 3a: Add the resolver + navigation functions**

In `static/dashboard.html`, immediately before the `R.briefing = (d) => {` line (~966), insert:

```javascript
/* === record-link resolver (test-extracted) === */
function resolveRefLinks(html, links){
  links = links || {};
  return String(html).replace(
    /<a href="ref:([^"]+)"[^>]*>([\s\S]*?)<\/a>/g,
    function(_m, ref, text){
      var rec = links[ref];
      if(!rec || !rec.url) return text;            // unknown/missing ref -> plain text
      var safeUrl = String(rec.url).replace(/&/g, "&amp;").replace(/"/g, "&quot;");
      return '<a href="' + safeUrl + '" class="rec-link" '
           + 'onclick="recNavigate(this,event)">' + text + '</a>';
    });
}
/* === end record-link resolver === */

// Record links navigate to the console with the console key appended at click
// time (mirrors actNavigate), so the secret never lives in the stored markdown.
function recNavigate(el, e){
  if(e) e.preventDefault();
  var href = el.getAttribute("href") || "/console";
  var hash = "";
  var hi = href.indexOf("#");
  if(hi >= 0){ hash = href.slice(hi); href = href.slice(0, hi); }
  var sep = href.indexOf("?") >= 0 ? "&" : "?";
  location.assign(href + sep + "key=" + encodeURIComponent(consoleKey) + hash);
}
```

- [ ] **Step 3b: Wire `R.briefing` to resolve**

The current `R.briefing` is:

```javascript
R.briefing = (d) => {
  if(d.empty) return `<div class="empty">${escapeHtml(d.message||"awaiting first run")}</div>`;
  return `<div class="intel-md">${mdRender(d.markdown||"")}</div>`;
};
```

Change the return to wrap with the resolver:

```javascript
R.briefing = (d) => {
  if(d.empty) return `<div class="empty">${escapeHtml(d.message||"awaiting first run")}</div>`;
  return `<div class="intel-md">${resolveRefLinks(mdRender(d.markdown||""), d.links||{})}</div>`;
};
```

- [ ] **Step 3c: Add a small style for record links**

Find the `.intel-md` style rules in the `<style>` block (search `.intel-md`). Add one rule near them:

```css
.intel-md a.rec-link{ color:var(--accent,#7aa2f7); text-decoration:underline; text-underline-offset:2px; cursor:pointer; }
```

(If no `--accent` var exists, use the same color the card links already use — match the surrounding `.intel-md a` rule if present.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_record_link_resolver_js.py -v`
Expected: PASS (2 passed; the node test runs the real extracted function).

- [ ] **Step 5: Commit**

```bash
git add static/dashboard.html tests/test_record_link_resolver_js.py
git commit -m "feat: resolve briefing record-link refs + key-appending recNavigate"
```

---

### Task 5: `console-crm.html` — `?email=` autoload

Makes the CRM destination pre-fill the contact email from the URL, so a record link lands ready to act.

**Files:**
- Modify: `static/console-crm.html`
- Test: `tests/test_crm_email_autoload.py`

**Interfaces:**
- Consumes: existing `#email` input (line ~60) and the existing `?key=` auto-unlock IIFE (line ~85).
- Produces: on page load, `?email=` pre-fills `#email`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crm_email_autoload.py
from pathlib import Path


def test_crm_reads_email_param():
    html = (Path(__file__).resolve().parent.parent / "static" / "console-crm.html").read_text()
    assert "URLSearchParams(location.search).get('email')" in html
    assert "getElementById('email')" in html  # input pre-filled from the param
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_crm_email_autoload.py -v`
Expected: FAIL — the `?email=` param read is not present yet.

- [ ] **Step 3: Add the autoload**

In `static/console-crm.html`, the file ends its script with:

```javascript
  if (key()) { document.getElementById('gate').style.display='none'; loadQueue(); }
```

Immediately after that line, add:

```javascript
  // Deep-link: ?email= pre-fills the contact so a briefing record-link lands ready.
  (function(){
    var em = new URLSearchParams(location.search).get('email');
    if(em){ var f = document.getElementById('email'); if(f) f.value = em; }
  })();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_crm_email_autoload.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add static/console-crm.html tests/test_crm_email_autoload.py
git commit -m "feat: console-crm ?email= autoload for record deep-links"
```

---

### Task 6: Render-verify the full path in a real browser (go-live gate)

The mandated render-verify: confirm the rendered DOM, not that scripts are served. This is a manual/agent verification, not a pytest. Do it after Tasks 1-5 are merged-to-branch and the suite is green.

**Files:** none (verification only). Capture findings in the PR description.

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest tests/test_briefing_links.py tests/test_briefing_runner_links.py tests/test_intelligence_links.py tests/test_record_link_resolver_js.py tests/test_crm_email_autoload.py -v`
Expected: all PASS.

- [ ] **Step 2: Boot the app locally with a scratch DATA_DIR**

Per the deploy-chat local-test pattern, the app validates secrets at import, so run under Doppler with a writable scratch `DATA_DIR`. Write this to a script (terminal-paste-wrap safe) and run it:

```bash
mkdir -p /tmp/dc-scratch/intelligence
cat > /tmp/run-dc.sh <<'SH'
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/dc-scratch python3 app.py
SH
bash /tmp/run-dc.sh
```

- [ ] **Step 3: Seed a briefing + links sidecar with a known ref and an unknown ref**

```bash
cat > /tmp/dc-scratch/intelligence/money-cash.md <<'MD'
# Finance
*Friday, June 26, 2026*

Oldest client waiting: [Test Client](ref:r1), 30+ days aged.
Also pinging [Ghost Ref](ref:r9) who has no registry entry.

## Recommended actions
[HIGH] Follow up with [Test Client](ref:r1) on the $380 invoice.
MD
cat > /tmp/dc-scratch/intelligence/money-cash.links.json <<'JSON'
{"r1": {"type":"person","display":"Test Client","url":"/console/crm?email=test%40x.com"}}
JSON
```

- [ ] **Step 4: Render the dashboard headless and assert the DOM**

Use the claude-in-chrome browser tools (load the core set via ToolSearch first). Navigate to the dashboard with the console key (the dashboard reads `consoleKey`; supply it the same way the app expects — via the key prompt / localStorage / `?key=` as the page supports). Then assert:

- The Finance card contains an anchor `a.rec-link` whose `href` is `/console/crm?email=test%40x.com` and whose text is `Test Client`.
- The `Ghost Ref` mention rendered as **plain text** (no anchor) — unknown ref unwrapped.
- **Zero console errors** (read_console_messages).
- Clicking the `Test Client` link navigates to `/console/crm?email=test%40x.com&key=...` (key appended). Verify the resulting URL / that `recNavigate` ran without error.

- [ ] **Step 5: Verify the CRM destination autoloads**

Navigate to `/console/crm?email=test%40x.com&key=<console secret>` and assert the `#email` input is pre-filled with `test@x.com` and there are zero console errors.

- [ ] **Step 6: Record the result**

If all assertions pass, the path renders correctly end-to-end. Note the verification (with what was asserted) in the PR description. If anything fails, fix and re-run from Step 1 — do NOT mark the feature done on injection-only evidence.

---

## Notes for the implementer

- **Run pytest from the worktree you are in.** The repo has no `pytest.ini`; pure-module tests (Tasks 1, 3, 4-JS-extract, 5) run with plain `python3 -m pytest`. The prompt test in Task 2 is also pure (no network). No test in Tasks 1-5 hits the network.
- **`node` is required** only for the one JS behavioral test in Task 4 (it is `skipif` when node is absent, but node IS installed in this environment — do not let it skip silently in CI; confirm it ran).
- **Ordering:** Task 2 wires `_intel.write_links`, which Task 3 defines. If you run Task 2's app-level generation before Task 3 lands, `write_links` won't exist — but Task 2's *tests* are source-asserts + a pure prompt test and pass independently. Implement in order; the full generation path is only exercised in Task 6.
- **Do not change `app.py`.** The serve route already returns `read_briefing(slug)` verbatim via `ok(data)`, so the `links` key flows through automatically.
