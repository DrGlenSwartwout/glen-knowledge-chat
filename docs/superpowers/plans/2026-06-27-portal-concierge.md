# Portal Concierge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.
> **Spec:** `docs/superpowers/specs/2026-06-27-portal-concierge-design.md`.
> **Base:** branch off current `main` (`portal-concierge`). Independent of the chat-UX PRs (#366/#368).

**Goal:** An ongoing health concierge chat inside the client portal — token-authed, grounded in the client's biofield findings + order history — reusing the existing concierge engine; suggestions link to the existing store/checkout.

**Architecture:** New `POST /api/portal/<token>/chat` SSE endpoint authed by the portal token (no TOS gate). It reuses the concierge engine (RAG + Haiku stream + the `_CONCIERGE_EXTRACT_SYSTEM` post-stream suggestion call + `_resolve_complement`), but builds its system prompt from the client's portal data via a new pure `dashboard/portal_concierge.py`. A new "Ask Dr. Glen" section in `static/client-portal.html` streams it.

**Tech Stack:** Flask, raw sqlite3, Anthropic Haiku 4.5 (SSE), vanilla JS in `client-portal.html`. No new dependencies.

## Global Constraints

- No new dependencies. Reuse the existing concierge engine pieces — do NOT fork a new chat stack.
- **No TOS gate** on the portal chat: possessing a portal token IS the post-TOS authorization. (Do NOT call `is_member` here, unlike `/begin/concierge/chat`.)
- FAIL OPEN: a chat / RAG / suggestion-extraction error must never break the portal page or other portal APIs; the answer still streams if extraction fails.
- The `suggestion` is extracted by a SEPARATE post-stream `messages.create` (NOT a trailing directive) — the stream loop accumulates `full` with no early return, so no `stream_visible`/drain is needed here.
- Buy path = the resolved suggestion's `url` (store/product page) for a *suggested* complement; the portal's existing bundle `/api/portal/<token>/checkout` button is unchanged.
- LLM-output tests assert pass-RATES over multiple samples, never single-shot.
- Straight ASCII quotes only in JS/Python (smart quotes as delimiters have broken this project's chat script twice).
- Reuse existing helpers (exact anchors below); don't reimplement.

## Existing anchors to reuse (verified)

- `_CONCIERGE_SYSTEM` `app.py:6641`; `_CONCIERGE_EXTRACT_SYSTEM` `app.py:6661`; concierge route `app.py:6669-6745`.
- Suggestion extraction (post-stream): `app.py:6724-6741` → `_resolve_complement(name)` `app.py:6617-6638` → dict `{name,title,url,price,slug,in_catalog}`.
- RAG: `_match_query_namespaces(embed(...))` + `build_context(matches)` (`app.py:6693-6694`).
- `_portal_record_for(cx, token)` `app.py:11263-11281` → `{email, name, content}`; `content` has `layers`, `findings`, `reorder_items`.
- Orders: `dashboard/orders.py` `list_orders_by_email(cx, email)` → each order `{items:[{name,qty}], status, created_at, ...}`.
- Portal token in page JS: `client-portal.html:155-161` (`location.pathname` last segment → `token`).
- Chat section placement: after biofield (`client-portal.html:427`), before reorder (`:429`), `.card`+`<h2>` pattern. NO existing SSE reader in the file (net-new).

---

### Task 1: `dashboard/portal_concierge.py` — context + prompt (pure)

**Files:** Create `dashboard/portal_concierge.py`; Test `tests/test_portal_concierge.py`.

**Interfaces — Produces:**
- `build_context(content: dict, orders: list) -> dict` — pulls grounding facts from the portal `content` (biofield `layers`/`findings`) and `orders` (owned remedy names), returns `{"findings": [...], "layers": [...], "owned": [names], "has_data": bool}`.
- `system_prompt(ctx: dict) -> str` — the ongoing-concierge instruction (widened from `_CONCIERGE_SYSTEM`), embedding the grounding facts as plain text.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portal_concierge.py
from dashboard.portal_concierge import build_context, system_prompt

SAMPLE_CONTENT = {
    "layers": [{"n": 3, "title": "Liver terrain", "meaning": "detox load", "remedy": "Terrain Restore"}],
    "findings": [{"code": "EI8", "name": "stress", "rank": 1}],
    "reorder_items": [{"slug": "terrain-restore", "qty": 1}],
}
SAMPLE_ORDERS = [{"items": [{"name": "Neuro-Magnesium", "qty": 1}], "status": "shipped"}]

def test_build_context_collects_grounding_facts():
    ctx = build_context(SAMPLE_CONTENT, SAMPLE_ORDERS)
    assert ctx["has_data"] is True
    assert any("stress" in f.get("name", "") for f in ctx["findings"])
    assert any("Terrain Restore" in (l.get("remedy") or "") for l in ctx["layers"])
    assert "Neuro-Magnesium" in ctx["owned"]

def test_build_context_no_data_degrades():
    ctx = build_context({}, [])
    assert ctx["has_data"] is False
    assert ctx["owned"] == [] and ctx["findings"] == [] and ctx["layers"] == []

def test_system_prompt_embeds_facts_and_is_ongoing():
    p = system_prompt(build_context(SAMPLE_CONTENT, SAMPLE_ORDERS))
    assert "Terrain Restore" in p and "Neuro-Magnesium" in p
    assert "post-purchase" not in p.lower()      # widened away from the one-purchase framing
    assert "ongoing" in p.lower() or "your scan" in p.lower()

def test_system_prompt_no_data_still_valid():
    p = system_prompt(build_context({}, []))
    assert isinstance(p, str) and len(p) > 100   # generic-but-valid
```

- [ ] **Step 2: Run → FAIL** — `python3 -m pytest tests/test_portal_concierge.py -q`
- [ ] **Step 3: Implement `dashboard/portal_concierge.py`** (pure; no Flask/network):

```python
"""Portal concierge context + prompt assembly. Pure / Flask-free / unit-testable.
Grounds the ongoing-concierge chat in the client's biofield findings + owned remedies."""

def build_context(content, orders):
    content = content or {}
    layers = [l for l in (content.get("layers") or []) if isinstance(l, dict)]
    findings = [f for f in (content.get("findings") or []) if isinstance(f, dict)]
    owned = []
    for o in (orders or []):
        for it in (o.get("items") or []):
            nm = (it.get("name") or "").strip()
            if nm and nm not in owned:
                owned.append(nm)
    return {"layers": layers, "findings": findings, "owned": owned,
            "has_data": bool(layers or findings or owned)}

_BASE = (
    "You are Dr. Glen Swartwout's warm, ongoing health concierge (naturopathic physician, "
    "Hilo Hawai'i) inside this client's private portal. They are a known client; help them "
    "with their scan findings, their remedies and protocol (what to take when), reorders, and "
    "well-matched complements. Calm, consultative, never pushy: they are served and in control.\n"
    "- Ground every answer in THEIR data below; reference their actual findings/remedies by name.\n"
    "- Ask ONE gentle question at a time when you need more. Functional Formulations first.\n"
    "- When it fits, suggest ONE complementary remedy at a time with a short plain reason.\n"
    "- Keep replies short and warm. Do not invent prices or URLs. No em dashes, no ALL CAPS, "
    "never prefix anything with 'Hook:'. Sign off as Dr. Glen only when concluding."
)

def system_prompt(ctx):
    ctx = ctx or {}
    parts = [_BASE, "\n\nTHIS CLIENT'S DATA:"]
    if ctx.get("owned"):
        parts.append("Remedies they already own: " + ", ".join(ctx["owned"]) + ".")
    fnd = [f.get("name") or f.get("code") for f in (ctx.get("findings") or []) if (f.get("name") or f.get("code"))]
    if fnd:
        parts.append("Scan findings: " + ", ".join(str(x) for x in fnd) + ".")
    for l in (ctx.get("layers") or []):
        seg = f"Layer {l.get('n','?')}: {l.get('title','')}".strip()
        if l.get("meaning"): seg += f" ({l['meaning']})"
        if l.get("remedy"): seg += f" - remedy: {l['remedy']}"
        parts.append(seg)
    if not ctx.get("has_data"):
        parts.append("(No scan or order data on file yet - answer generally and invite them to share.)")
    return "\n".join(parts)
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(portal-concierge): pure context + prompt assembly`

---

### Task 2: `app.py` — `POST /api/portal/<token>/chat` route

**Files:** Modify `app.py` (new route near the other `/api/portal/<token>/*` routes, ~`app.py:11352`). Test `tests/test_portal_concierge_route.py`.

**Interfaces — Consumes:** `portal_concierge.build_context`/`system_prompt` (Task 1); existing `_portal_record_for`, `list_orders_by_email`, `_match_query_namespaces`/`embed`/`build_context`, `_CONCIERGE_EXTRACT_SYSTEM`, `_resolve_complement`, `_strip_dash`, `sse`. **Produces:** SSE `{"token"}` / `{"suggestion"}` / `{"done"}` (NO `{"gate"}`).

- [ ] **Step 1: Add the route** (mirror the concierge route's structure; token auth, no gate):

```python
@app.route("/api/portal/<token>/chat", methods=["POST", "OPTIONS"])
def api_portal_chat(token):
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    history = data.get("history") or []
    with sqlite3.connect(LOG_DB) as cx:
        from dashboard import client_portal as _cp
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
    if not portal:
        return jsonify({"error": "not found"}), 404
    email = (portal.get("email") or "").strip().lower()
    content = portal.get("content") or {}
    try:
        with sqlite3.connect(LOG_DB) as ocx:
            ocx.row_factory = sqlite3.Row
            from dashboard import orders as _o
            client_orders = _o.list_orders_by_email(ocx, email)
    except Exception:
        client_orders = []
    from dashboard import portal_concierge as _pcz
    ctx = _pcz.build_context(content, client_orders)
    _sys = _pcz.system_prompt(ctx)
    # RAG (best-effort, fail-open)
    context_str = ""
    try:
        matches = _match_query_namespaces(embed(query))
        context_str, _ = build_context(matches) if matches else ("", [])
    except Exception as e:
        print(f"[portal-concierge] retrieval: {e}", flush=True)
    messages = []
    for m in history[-8:]:
        r = "user" if m.get("role") == "user" else "assistant"
        messages.append({"role": r, "content": (m.get("content") or "")[:2000]})
    user_block = (f"CONTEXT:\n{context_str}\n\n" if context_str else "") + query
    messages.append({"role": "user", "content": user_block})

    def generate():
        full = []
        try:
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=700,
                                     system=_sys, messages=messages) as stream:
                for tok in stream.text_stream:
                    tok = _strip_dash(tok); full.append(tok); yield sse({"token": tok})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"}); return
        answer = "".join(full)
        try:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-2:]) + f"\nassistant: {answer}"
            mx = _cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=120,
                                     system=_CONCIERGE_EXTRACT_SYSTEM,
                                     messages=[{"role": "user", "content": convo[:3500]}])
            txt = mx.content[0].text.strip()
            if txt.startswith("```"):
                txt = txt.split("```", 2)[1]
                if txt.startswith("json\n"): txt = txt[5:]
            obj = json.loads(txt)
            if obj.get("suggest") and obj.get("name"):
                c = _resolve_complement(obj["name"])
                if c and (c["in_catalog"] or c["url"]):
                    yield sse({"suggestion": c})
        except Exception as e:
            print(f"[portal-concierge] extract: {e!r}", flush=True)
        yield sse({"done": True})

    return Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

- [ ] **Step 2: Test** `tests/test_portal_concierge_route.py` (reload-app convention; under Doppler). Seed a portal: `init_client_portal_table` + `client_portal.upsert_portal(cx, email="t@x", name="T", content={...layers/findings...})` to mint a token; POST `/api/portal/<token>/chat` `{"query":"what should I take for stress?"}`; assert the SSE body has `"token"` events and a `"done"` event and NO `"gate"`. Invalid token → 404. (Streams real Haiku under Doppler; keep the query short.)

Run: `S=/tmp/pcz; mkdir -p $S; doppler run -p remedy-match -c prd -- env DATA_DIR=$S python3 -m pytest tests/test_portal_concierge_route.py -q -p no:cacheprovider` → PASS.

- [ ] **Step 3: Commit** — `feat(portal-concierge): token-authed /api/portal/<token>/chat (no TOS gate)`

---

### Task 3: Grounding eval (pass-rate, Doppler)

**Files:** Create `tests/test_portal_concierge_eval.py`.

- [ ] **Step 1: Write the eval** — skip if no `ANTHROPIC_API_KEY`. Build a context with a distinctive seeded finding + owned remedy via `portal_concierge.build_context`/`system_prompt`, then sample Haiku N=5 times on a question that should pull from it (e.g. "what's going on with my stress, and what do I already have for it?"). Assert as RATES:
  - In ≥3/5 samples the answer references the seeded **owned remedy name** or the seeded **finding** (grounding works).
  - In ≥4/5 samples there are no em dashes and no "Hook:" (style rules hold).
  Print per-sample snippets. This catches a prompt that ignores the grounding. (Pass-rate, not single-shot — per `feedback_mock_masked_green_tests`.)

Run under Doppler 2-3 times to confirm STABLE; tune thresholds/wording if flaky.

- [ ] **Step 2: Commit** — `test(portal-concierge): pass-rate grounding eval`

---

### Task 4: `client-portal.html` — "Ask Dr. Glen" chat section (+ render-verify)

**Files:** Modify `static/client-portal.html` (new section after biofield ~`:427`, before reorder ~`:429`; new SSE reader JS). Controller render-verify.

- [ ] **Step 1: Add the section + streaming chat JS.** A `.card` titled "Ask Dr. Glen" with a messages container, a text input + send button. On send: POST `/api/portal/${encodeURIComponent(token)}/chat` (`token` already derived at `:155-161`), read the SSE stream (net-new reader — there is none in this file), append assistant tokens to a bubble, and on a `suggestion` event render a card: product `name` + (if `price`) price + a button to `suggestion.url` (`<a target="_blank" rel="noopener">`) — falls back to no card if `url` is empty. Match the page's `.card`/theme styles. Straight ASCII quotes only; guard everything; a chat error shows an error bubble and never breaks the rest of the portal.
- [ ] **Step 2: RENDER-VERIFY (controller, headless, gevent server)** per `feedback_render_verify_not_just_inject` + `feedback_streaming_directive_endtoend_test` (boot gunicorn with `--worker-class gevent` or several workers — a single sync worker yields `chrome-error://` on the browser's concurrent sub-resource loads): load a seeded `/portal/<token>`, send a question, assert assistant tokens stream into a bubble, assert a `suggestion` event renders a card with a working buy link (stub `fetch` to feed a scripted SSE incl. a suggestion), assert an error degrades gracefully, and **zero console errors**. Do NOT rely on injection-only checks.
- [ ] **Step 3: Commit** — `feat(portal): Ask Dr. Glen concierge chat section`

---

## Verification (end-to-end)

- **Unit (plain):** `tests/test_portal_concierge.py` — context + prompt assembly incl. the no-data case.
- **Under Doppler:** the route test (streams, no gate, 404 on bad token) and the **stable pass-rate** grounding eval.
- **Render-verify (headless, gevent):** the portal chat section streams, renders a suggestion→buy card, degrades on error, zero console errors.
- **Manual smoke:** open a real `/portal/<token>`, ask about your scan/remedies — the answer references your actual findings/owned remedies; a relevant complement renders a buy card; the existing bundle reorder button still works.
- **Pre-PR hygiene:** `git diff --name-only origin/main..HEAD | grep -i superpowers` and `git rm --cached` any leaked `.superpowers/sdd` scratch (`feedback_sdd_scratch_git_leak`).

## Out of scope

Funnel concierge (`/begin/concierge/chat`, `concierge.html`) unchanged. No portal cart/add-to-invoice (suggested complements link to the store `url`; the bundle reorder keeps its `/checkout` button). Sub-project A (universal portal + every-buyer-TOS) is separate.
