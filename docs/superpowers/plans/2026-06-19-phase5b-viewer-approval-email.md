# Phase 5b — Viewer Approval Email Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture every known-email viewer of a not-yet-approved product page, and email each one once as Dr. Glen when the page's copy is approved.

**Architecture:** A `sales_page_viewers` store + a `notify_on_approve` sender; capture in `begin_product_page_data` (known-email viewers of a draft page); the email step inside the existing `sales_pages.approve` action. Ships live, no flag.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db`/`LOG_DB`), `dashboard/inbox.send_email`, pytest.

## Global Constraints

- **Recipients = every known-email viewer of the draft**, emailed **once** on approval (re-approval emails nobody new). Anonymous viewers skipped.
- **Capture only while `_ai_state != "approved"`** and only when `get_authenticated_user(request)` yields an email; best-effort (never breaks page-data).
- **Email never fails the approve** — the notify step is wrapped; per-recipient try/except inside `notify_on_approve` (a failed send leaves that viewer un-emailed, others proceed).
- **Idempotent at-most-once:** `record_viewer` is `INSERT OR IGNORE`; `viewers_to_email` returns only `emailed_at`-empty rows; `mark_emailed` stamps on send.
- **Email voice:** subject `f"Your {product_name} page is ready, reviewed by Dr. Glen"`; body opens `Aloha {name},` (or `Aloha,` with no name), one line it's now personally reviewed, the `{base_url}/begin/product/{slug}` link, close `In wellness,\nDr. Glen & Rae`; run through `strip` (no em dashes); `from_name="Dr. Glen Swartwout"`.
- **Ships LIVE, no flag.** Capture rides under `_SALES_AI_COPY_ENABLED` (already live). NO emoji.
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_page_viewers_5b.py -v`
- Tests reload `app` via `importlib`; emails lowercased.

---

### Task 1: `dashboard/sales_page_viewers.py` store + notify

**Files:**
- Create: `dashboard/sales_page_viewers.py`
- Test: `tests/test_sales_page_viewers_5b.py` (create)

**Interfaces:**
- Produces:
  - `init_table(cx)`
  - `record_viewer(cx, slug, email, name="") -> None` (INSERT OR IGNORE; repeat view never resets)
  - `viewers_to_email(cx, slug) -> list[dict]` (`{email, name}` for `emailed_at`-empty rows)
  - `mark_emailed(cx, slug, emails) -> None`
  - `notify_on_approve(cx, slug, product_name, base_url, *, send, strip=lambda s: s) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sales_page_viewers_5b.py`:

```python
import sqlite3
from dashboard import sales_page_viewers as spv


def _cx():
    return sqlite3.connect(":memory:")


def test_record_idempotent_and_to_email():
    cx = _cx()
    spv.record_viewer(cx, "longevity", "A@x.com", "Ann")
    spv.record_viewer(cx, "longevity", "a@x.com", "Ann2")   # same (lowercased) -> ignored
    rows = spv.viewers_to_email(cx, "longevity")
    assert len(rows) == 1 and rows[0]["email"] == "a@x.com" and rows[0]["name"] == "Ann"
    spv.record_viewer(cx, "longevity", "b@x.com")            # no name
    assert {r["email"] for r in spv.viewers_to_email(cx, "longevity")} == {"a@x.com", "b@x.com"}


def test_mark_emailed_excludes():
    cx = _cx()
    spv.record_viewer(cx, "x", "a@x.com", "Ann")
    spv.record_viewer(cx, "x", "b@x.com", "Bob")
    spv.mark_emailed(cx, "x", ["a@x.com"])
    assert [r["email"] for r in spv.viewers_to_email(cx, "x")] == ["b@x.com"]


def test_notify_on_approve_sends_once_and_marks():
    cx = _cx()
    spv.record_viewer(cx, "x", "a@x.com", "Ann")
    spv.record_viewer(cx, "x", "b@x.com", "")
    sent = []
    def fake_send(to, subject, body, from_name=None):
        sent.append({"to": to, "subject": subject, "body": body, "from_name": from_name})
    n = spv.notify_on_approve(cx, "x", "Longevity", "https://illtowell.com",
                              send=fake_send, strip=lambda s: s.replace("—", ","))
    assert n == 2 and len(sent) == 2
    by = {s["to"]: s for s in sent}
    assert by["a@x.com"]["body"].startswith("Aloha Ann,")
    assert by["b@x.com"]["body"].startswith("Aloha,")           # no name
    assert "Your Longevity page is ready, reviewed by Dr. Glen" == by["a@x.com"]["subject"]
    assert "https://illtowell.com/begin/product/x" in by["a@x.com"]["body"]
    assert by["a@x.com"]["from_name"] == "Dr. Glen Swartwout"
    assert "—" not in by["a@x.com"]["body"]                # em dash stripped
    # idempotent: a re-run sends nobody
    assert spv.notify_on_approve(cx, "x", "Longevity", "https://illtowell.com", send=fake_send) == 0
    assert len(sent) == 2


def test_notify_per_recipient_failure_isolated():
    cx = _cx()
    spv.record_viewer(cx, "x", "good@x.com", "G")
    spv.record_viewer(cx, "x", "bad@x.com", "B")
    def flaky_send(to, subject, body, from_name=None):
        if to == "bad@x.com":
            raise RuntimeError("smtp down")
    n = spv.notify_on_approve(cx, "x", "P", "https://b", send=flaky_send)
    assert n == 1                                              # only good@x.com emailed
    assert [r["email"] for r in spv.viewers_to_email(cx, "x")] == ["bad@x.com"]   # bad left for retry
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`ModuleNotFoundError: dashboard.sales_page_viewers`).

- [ ] **Step 3: Implement `dashboard/sales_page_viewers.py`**

```python
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS sales_page_viewers ("
        "product_slug TEXT, email TEXT, name TEXT, first_seen_at TEXT, "
        "emailed_at TEXT DEFAULT '', PRIMARY KEY(product_slug, email))")
    cx.commit()


def record_viewer(cx, slug, email, name=""):
    init_table(cx)
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT OR IGNORE INTO sales_page_viewers (product_slug, email, name, first_seen_at, emailed_at) "
        "VALUES (?,?,?,?,'')", (slug, e, name or "", _now()))
    cx.commit()


def viewers_to_email(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = __import__("sqlite3").Row
    rows = cur.execute(
        "SELECT email, name FROM sales_page_viewers WHERE product_slug=? AND COALESCE(emailed_at,'')='' "
        "ORDER BY first_seen_at", (slug,)).fetchall()
    return [{"email": r["email"], "name": r["name"] or ""} for r in rows]


def mark_emailed(cx, slug, emails):
    init_table(cx)
    now = _now()
    for e in emails:
        cx.execute("UPDATE sales_page_viewers SET emailed_at=? WHERE product_slug=? AND email=?",
                   (now, slug, _norm(e)))
    cx.commit()


def notify_on_approve(cx, slug, product_name, base_url, *, send, strip=lambda s: s):
    """Email each un-emailed viewer of this slug once as Dr. Glen; mark them; return the count sent."""
    viewers = viewers_to_email(cx, slug)
    subject = f"Your {product_name} page is ready, reviewed by Dr. Glen"
    link = f"{base_url}/begin/product/{slug}"
    emailed = []
    for v in viewers:
        name = (v.get("name") or "").strip()
        greeting = f"Aloha {name}," if name else "Aloha,"
        body = strip(
            f"{greeting}\n\nThe {product_name} page you looked at has now been personally reviewed "
            f"by Dr. Glen and is ready:\n\n{link}\n\nIn wellness,\nDr. Glen & Rae")
        try:
            send(v["email"], subject, body, from_name="Dr. Glen Swartwout")
            emailed.append(v["email"])
        except Exception as e:  # noqa: BLE001 - one bad send must not stop the rest
            print(f"[sales-viewers] send failed for {v['email']}: {e}", flush=True)
    if emailed:
        mark_emailed(cx, slug, emailed)
    return len(emailed)
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_page_viewers.py tests/test_sales_page_viewers_5b.py
git commit -m "feat(5b): sales_page_viewers store + notify_on_approve (once-per-viewer, as Dr. Glen)"
```

---

### Task 2: Capture in page-data + email on approve

**Files:**
- Modify: `app.py` (capture in `begin_product_page_data`; add `base_url` to the `sales_pages_actions.configure(...)` call)
- Modify: `dashboard/sales_pages_actions.py` (`_exec_approve` calls `notify_on_approve`)
- Test: `tests/test_sales_page_viewers_5b.py` (append)

**Interfaces:**
- Consumes: Task 1 `sales_page_viewers.{record_viewer, notify_on_approve}`; `dashboard/inbox.send_email`; the Phase-5 `_DEPS` (`get_product`, `strip_dash`) + the new `base_url`; `get_authenticated_user`, `PUBLIC_BASE_URL`.
- Produces: known-email viewers of a draft page recorded; `sales_pages.approve` emails un-emailed viewers once.

- [ ] **Step 1: Write the failing tests**

Append:

```python
import importlib


def _reload_5b_app(monkeypatch, tmp_path, copy="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_COPY", copy)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_page_data_captures_known_email_viewer(monkeypatch, tmp_path):
    appmod = _reload_5b_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    # a draft exists (page not approved) so capture runs
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "draft copy")
    # force a known viewer email
    monkeypatch.setattr(appmod, "get_authenticated_user",
                        lambda req: {"email": "viewer@x.com", "name": "Vi"})
    appmod.app.test_client().get(f"/begin/product-page-data/{slug}")
    from dashboard import sales_page_viewers as spv
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert [r["email"] for r in spv.viewers_to_email(cx, slug)] == ["viewer@x.com"]


def test_page_data_no_capture_when_approved(monkeypatch, tmp_path):
    appmod = _reload_5b_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "draft copy")
        sp.set_state(cx, slug, "approved", by="Glen")
    monkeypatch.setattr(appmod, "get_authenticated_user",
                        lambda req: {"email": "viewer@x.com", "name": "Vi"})
    appmod.app.test_client().get(f"/begin/product-page-data/{slug}")
    from dashboard import sales_page_viewers as spv
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert spv.viewers_to_email(cx, slug) == []


def test_approve_emails_viewers(monkeypatch, tmp_path):
    appmod = _reload_5b_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp, sales_page_viewers as spv
    from dashboard import dispatch as d
    from dashboard.rbac import Actor, OWNER
    sent = []
    from dashboard import inbox as _inbox
    monkeypatch.setattr(_inbox, "send_email",
                        lambda to, subject, body, from_name=None: sent.append(to))
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "draft copy")
        spv.record_viewer(cx, slug, "viewer@x.com", "Vi")
        res = d.dispatch_action(cx, "sales_pages.approve", {"slug": slug},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done"
        assert sent == ["viewer@x.com"]
        assert spv.viewers_to_email(cx, slug) == []            # marked emailed
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (no capture / approve doesn't email).

- [ ] **Step 3: Implement**

In `app.py` `begin_product_page_data`, inside the `with _sq.connect(LOG_DB) as _cx:` block, right after `_ai_state = _pg["state"] if _pg else "none"`, add the capture:

```python
                if _ai_state != "approved":
                    try:
                        from dashboard import sales_page_viewers as _spv
                        _vau = get_authenticated_user(request) or {}
                        _vemail = (_vau.get("email") or "").strip().lower()
                        if _vemail:
                            _spv.record_viewer(_cx, slug, _vemail, _vau.get("name", ""))
                    except Exception as _ve:
                        print(f"[sales-viewers] capture skipped: {_ve}", flush=True)
```

In `app.py`, add `base_url=PUBLIC_BASE_URL` to the existing `sales_pages_actions.configure(...)` call (search `_spa.configure(`):

```python
_spa.configure(client=_cl, get_product=_get_product,
               product_card=_product_card, strip_dash=_strip_dash, base_url=PUBLIC_BASE_URL)
```

In `dashboard/sales_pages_actions.py`, extend `_exec_approve`:

```python
def _exec_approve(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _sp.set_state(ctx["cx"], slug, "approved", by=_actor_name(ctx.get("actor")))
    try:
        from dashboard import sales_page_viewers as _spv
        from dashboard import inbox as _inbox
        get_product = _DEPS.get("get_product")
        strip = _DEPS.get("strip_dash") or (lambda s: s)
        pname = ((get_product(slug) or {}).get("name", slug) if get_product else slug)
        base = _DEPS.get("base_url", "")
        _spv.notify_on_approve(ctx["cx"], slug, pname, base, send=_inbox.send_email, strip=strip)
    except Exception as e:  # noqa: BLE001 - notifying viewers must never fail the approve
        print(f"[sales-pages] viewer notify skipped: {e}", flush=True)
    return {"slug": slug, "state": "approved"}
```

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the sweep `pytest tests/ -k "sales or viewer" -q`. Expected: all pass, no regressions (Phase-5 sales-pages tests included).

- [ ] **Step 5: Commit**

```bash
git add app.py dashboard/sales_pages_actions.py tests/test_sales_page_viewers_5b.py
git commit -m "feat(5b): capture known-email draft viewers + email them on approve (as Dr. Glen)"
```

---

## Self-Review (plan author)

- **Spec coverage:** store + notify (T1) → spec Store/Email; capture + approve email + configure base_url (T2) → spec Capture/Email-on-approve. No flag (live).
- **Decisions honored:** every known-email draft viewer, once (T1 idempotent + T2 capture-when-not-approved); Aloha-no-name greeting (T1); no em dash via strip (T1); notify never fails approve (T2 wrapped) + per-recipient isolation (T1); INSERT OR IGNORE + emailed_at at-most-once (T1); live/no-flag.
- **Type consistency:** `record_viewer(cx, slug, email, name="")`, `viewers_to_email -> [{email,name}]`, `mark_emailed(cx, slug, emails)`, `notify_on_approve(cx, slug, product_name, base_url, *, send, strip=)`, `_DEPS["base_url"]` — used identically across tasks.
- **Confirm-then-use flagged in-task:** the exact `_spa.configure(...)` call in app.py (T2); the capture insertion point after `_ai_state` inside the `with _cx` block (T2).
