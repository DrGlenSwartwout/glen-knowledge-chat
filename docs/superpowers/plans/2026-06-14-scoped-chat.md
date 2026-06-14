# Practitioner-Scoped Support Chat — Implementation Plan (Plan 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A self-contained product-selection chat on the practitioner portal pages. It recommends **only the practitioner's Functional Formulations**, returns add-to-cart suggestions (never RM-direct links or RM retail pricing — self-contained by construction), is a collapsible widget at the bottom, is **available by default on the practitioner pages**, and is a **practitioner toggle (default OFF) on the client page** (consent-gated for the patient).

**Architecture:** A new `dashboard/practitioner_chat.py` — `scoped_reply(message, history, catalog)` makes ONE LLM call grounded ONLY in the supplied catalog and returns `{reply, suggested_slugs}`. Two thin endpoints (`/api/practitioner/chat` authed; `/api/client/<code>/chat` consent-gated, scoped to that practitioner's FF). A collapsible widget added to the three pages, gated by the `chat_enabled` setting on the client page. NO reuse of the existing concierge synthesis (which surfaces truly.vip/RM links) — this is fresh + scoped.

**Tech Stack:** Python 3.11, Flask, Anthropic (`_cl`, Haiku), pytest.

**Spec/decision:** Glen-confirmed 2026-06-14 — practitioner-scoped, **self-contained, no RM links**. Toggle on the client page (Plan 4 settings `branding`/a new `chat_enabled`), default available on practitioner pages, collapsible bottom widget.

**Reuse:** `_cl.messages.create` (Haiku — see app.py existing calls), `practitioner_settings.get_settings` (chat_enabled), `dropship_checkout.practitioner_price_for` (price suggested items), `_get_product` (FF catalog), `_practitioner_session_pid`, `_pp.practitioner_id_by_dispensary_code`, the patient consent gate (`is_member`), the page cart JS.

---

### Task 1: `scoped_reply` — catalog-grounded, no external links

**Files:** Create `dashboard/practitioner_chat.py`; Test `tests/test_practitioner_chat.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_practitioner_chat.py
from dashboard import practitioner_chat as pc

CATALOG = [
    {"slug": "brain-boost", "name": "Brain Boost", "description": "nootropic support"},
    {"slug": "bone-builder", "name": "Bone Builder", "description": "bone density"},
]

def test_scoped_reply_returns_reply_and_validated_slugs(monkeypatch):
    # stub the LLM to return a reply + a suggested slug that IS in the catalog, plus a
    # hallucinated slug that is NOT — only the valid one survives.
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {
        "reply": "For focus, Brain Boost is a good fit.",
        "suggested_slugs": ["brain-boost", "not-a-real-slug"]})
    out = pc.scoped_reply("something for focus", [], CATALOG)
    assert "Brain Boost" in out["reply"]
    assert out["suggested_slugs"] == ["brain-boost"]      # hallucination dropped

def test_scoped_reply_strips_external_links(monkeypatch):
    # even if the model leaks a URL, the reply is scrubbed of links/store mentions.
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {
        "reply": "Try it here: https://remedymatch.com/x or truly.vip/y", "suggested_slugs": []})
    out = pc.scoped_reply("hi", [], CATALOG)
    assert "http" not in out["reply"] and "truly.vip" not in out["reply"] and "remedymatch" not in out["reply"].lower()

def test_empty_catalog_safe(monkeypatch):
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {"reply": "ok", "suggested_slugs": []})
    out = pc.scoped_reply("hi", [], [])
    assert out["suggested_slugs"] == []
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**

```python
# dashboard/practitioner_chat.py
"""Self-contained product-selection chat scoped to a single practitioner's catalog.
Recommends ONLY from the supplied catalog; never emits external/RM links or prices."""
import json, re

_SYSTEM = (
    "You are a product-selection assistant for a natural-health practitioner's own "
    "dispensary. You may ONLY discuss and recommend products from the CATALOG provided "
    "below. Never mention, link to, or price any other store, website, or 'online' option. "
    "Never invent products not in the catalog. Keep replies short and supportive. "
    "Return JSON: {\"reply\": str, \"suggested_slugs\": [slugs from the catalog]}.\n\nCATALOG:\n"
)
_URL_RE = re.compile(r"https?://\S+", re.I)
_BANNED = re.compile(r"\b(truly\.vip|truly\.so|remedymatch|illtowell)\S*", re.I)


def _llm_json(system, messages):
    """One Haiku call returning parsed JSON. Monkeypatched in tests."""
    import app as _app
    r = _app._cl.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=400,
        system=system, messages=messages)
    txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {"reply": txt.strip(), "suggested_slugs": []}


def _scrub(text):
    text = _URL_RE.sub("", text or "")
    text = _BANNED.sub("our selection", text)
    return text.strip()


def scoped_reply(message, history, catalog):
    """Return {reply, suggested_slugs} grounded only in `catalog`
    ([{slug,name,description}]). Suggested slugs are validated against the catalog;
    the reply is scrubbed of any external links/store mentions."""
    valid = {c["slug"] for c in (catalog or [])}
    cat_txt = "\n".join(f"- {c['slug']}: {c.get('name','')} — {c.get('description','')}"
                        for c in (catalog or []))
    msgs = list(history or []) + [{"role": "user", "content": str(message or "")}]
    try:
        out = _llm_json(_SYSTEM + cat_txt, msgs)
    except Exception:
        return {"reply": "Sorry, I had trouble — please pick from the list.", "suggested_slugs": []}
    slugs = [s for s in (out.get("suggested_slugs") or []) if s in valid]
    return {"reply": _scrub(out.get("reply", "")), "suggested_slugs": slugs}
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(scoped-chat): scoped_reply (catalog-grounded, no external links)`

---

### Task 2: chat endpoints

**Files:** Modify `app.py`; Test `tests/test_scoped_chat_routes.py`

- [ ] **Step 1: Failing test** — `POST /api/practitioner/chat` (authed → 401 without session) returns `{reply, suggestions:[{slug,name,price_cents}]}` (prices via `practitioner_price_for`). `POST /api/client/<code>/chat` resolves the practitioner by code (404 unknown), **consent-gated** (403 `need_optin` if not a member), scoped to that practitioner's FF, suggestions priced at the practitioner's price. Stub `practitioner_chat.scoped_reply`.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — both routes build the FF catalog (`_get_product` over FF, exclude Pure Powders/info_only: `[{slug,name,description}]`), call `practitioner_chat.scoped_reply(message, history, catalog)`, then map `suggested_slugs` → `[{slug, name, price_cents: practitioner_price_for(pid, slug)}]`. `/api/practitioner/chat` authed via `_practitioner_session_pid`; `/api/client/<code>/chat` resolves `pid` by code + applies the patient consent gate (like the client checkout). Import `practitioner_chat` locally.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(scoped-chat): /api/practitioner/chat + /api/client/<code>/chat`

---

### Task 3: widget + client-page toggle

**Files:** Modify `static/practitioner-client.html`, `static/practitioner-dropship.html`, the wholesale page, `static/practitioner-settings.html`; `dashboard/practitioner_settings.py` (chat_enabled default)

- [ ] **Step 1:** A small reusable **collapsible chat widget** at the **bottom** of each page (a launcher button that opens/closes a panel): input box → POST the page's chat endpoint → render the reply + suggestion chips, each an **"Add to cart"** that adds the slug to that page's cart (reuse the page's cart JS). No external links rendered.
  - **Practitioner pages (drop-ship, wholesale):** widget available by default.
  - **Client page:** show the widget only when the practitioner's `chat_enabled` setting is true (surface `chat_enabled` in `branding`/settings, default **false**). Add a **chat on/off toggle** to `practitioner-settings.html` (saved via the settings API).
- [ ] **Step 2:** Add `chat_enabled` (bool, default false) to the settings branding/pricing shape (a `chat` section or a branding key); the client `catalog` endpoint exposes it so the page knows whether to show the widget. Verify settings + chat route tests stay green.
- [ ] **Step 3:** Commit — `feat(scoped-chat): collapsible widget + client-page toggle`

---

### Task 4: suite + doc

- [ ] **Step 1:** Run scoped-chat + settings + client + dropship tests — green.
- [ ] **Step 2:** Create `docs/scoped-chat.md`: self-contained, catalog-grounded, no RM links (scrubbed + slug-validated); endpoints; widget (collapsible, bottom); client-page toggle (default off) + consent gate; practitioner pages default-on.
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec/decision coverage:** practitioner-scoped (catalog-only), self-contained, **no RM links** (scrub + slug-validation, fresh endpoint not the concierge) — Task 1,2; toggle on client (default off) + default-on practitioner pages + collapsible bottom widget — Task 3; consent on the client page — Task 2.
- **Deferred:** richer chat (memory across sessions, voice); using the full concierge knowledge base (intentionally NOT — scoped to the catalog to guarantee no RM leakage).
- **Risk:** the recommendation could leak an RM reference — mitigated two ways (the model only sees the practitioner's catalog; the reply is regex-scrubbed of URLs + RM-brand mentions; suggested slugs are validated against the catalog). No money path; no live-checkout impact.
- **Type consistency:** `scoped_reply(message, history, catalog) -> {reply, suggested_slugs}`, `/api/practitioner/chat`, `/api/client/<code>/chat`, `chat_enabled` setting.

## Done
Completes the practitioner drop-ship + white-label portal + scoped chat (Plans 1–5).
