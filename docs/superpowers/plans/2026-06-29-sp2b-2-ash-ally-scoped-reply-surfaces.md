# SP2b-2 — ASH ally on scoped_reply client surfaces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the ASH ally memory layer (shipped in SP2b-1) to the two non-streaming `scoped_reply` client chat surfaces — the dispensary client widget and the invoice pay-link chat.

**Architecture:** Add a defaulted `overlay=""` param to `dashboard/practitioner_chat.scoped_reply` and inject it into the model system prompt. Compute the overlay in `app.py` at each of the 2 client call sites (where `email` + `ash_ally` are already available — avoids a `dashboard → app → dashboard` import cycle) and fire `ash_ally.record_turn` on a background daemon thread after the reply. The practitioner caller passes no overlay (default), so it is unchanged. Same `ASH_ALLY_ENABLED` flag, dark by default.

**Tech Stack:** Python 3, the existing `dashboard/ash_ally.py` + `dashboard/ash_map.py`, Anthropic Haiku (via `scoped_reply` → `_llm_json`). Test: plain pytest on `practitioner_chat` (lazy app import → no secrets/network needed).

## Global Constraints

- Modify: `dashboard/practitioner_chat.py`, `app.py`. New test: `tests/test_practitioner_chat_overlay.py`.
- `scoped_reply` signature becomes `scoped_reply(message, history, catalog, overlay="")`; system prompt becomes `_SYSTEM + (overlay + "\n\n" if overlay else "") + cat_txt`. Backward-compatible: a 3-arg call is byte-identical to today.
- Do NOT import `ash_map`/`ash_ally` into `dashboard/practitioner_chat.py` (circular-import risk). The overlay is computed in `app.py` and passed in.
- `from dashboard import ash_ally` is already imported in `app.py` (SP2b-1) — do not re-add.
- Subject email = the resolved `email` (speaker == subject on both surfaces). Dispensary: `email` already gated. Invoice: load it via `_invoice_order_for_token(token).get("email")`.
- Record dispatch: `threading.Thread(target=ash_ally.record_turn, args=(LOG_DB, _db_lock, email, <message>, result.get("reply","")), daemon=True).start()`, wrapped in try/except so a spawn failure can't break the response.
- The practitioner surface `/api/practitioner/chat` (~11556) MUST remain a 3-arg `scoped_reply(message, history, catalog)` call — untouched.
- Tests run with plain pytest, no doppler/network: `python3 -m pytest tests/test_practitioner_chat_overlay.py -v`.

---

### Task 1: `scoped_reply` gains the `overlay` param

**Files:**
- Modify: `dashboard/practitioner_chat.py` (`scoped_reply`, ~line 33-42)
- Test: `tests/test_practitioner_chat_overlay.py` (new)

**Interfaces:**
- Produces: `scoped_reply(message, history, catalog, overlay="") -> dict` — same `{"reply", "suggested_slugs"}` return; system prompt is now `_SYSTEM + (overlay + "\n\n" if overlay else "") + cat_txt`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_practitioner_chat_overlay.py
import dashboard.practitioner_chat as pc


def _capture(monkeypatch):
    """Monkeypatch _llm_json to capture the system prompt and return a fixed dict."""
    seen = {}
    def fake(system, messages):
        seen["system"] = system
        return {"reply": "ok", "suggested_slugs": []}
    monkeypatch.setattr(pc, "_llm_json", fake)
    return seen


def test_overlay_injected_after_system_before_catalog(monkeypatch):
    seen = _capture(monkeypatch)
    cat = [{"slug": "neuro-magnesium", "name": "Neuro Magnesium", "description": "x"}]
    pc.scoped_reply("hi", [], cat, overlay="OVERLAY-TEXT")
    sys = seen["system"]
    assert "OVERLAY-TEXT" in sys
    # positioned after _SYSTEM and before the catalog line
    assert sys.index(pc._SYSTEM) < sys.index("OVERLAY-TEXT") < sys.index("neuro-magnesium")


def test_no_overlay_is_backward_compatible(monkeypatch):
    seen = _capture(monkeypatch)
    cat = [{"slug": "neuro-magnesium", "name": "Neuro Magnesium", "description": "x"}]
    pc.scoped_reply("hi", [], cat)  # 3-arg legacy call
    sys = seen["system"]
    assert "OVERLAY" not in sys
    # byte-identical to the legacy _SYSTEM + cat_txt assembly
    cat_txt = "\n".join(f"- {c['slug']}: {c.get('name','')} — {c.get('description','')}" for c in cat)
    assert sys == pc._SYSTEM + cat_txt


def test_return_shape_unchanged(monkeypatch):
    _capture(monkeypatch)
    out = pc.scoped_reply("hi", [], [], overlay="X")
    assert set(out.keys()) == {"reply", "suggested_slugs"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_practitioner_chat_overlay.py -v`
Expected: FAIL — `scoped_reply()` got an unexpected keyword argument `overlay` (and/or the injection assertions fail).

- [ ] **Step 3: Write minimal implementation**

In `dashboard/practitioner_chat.py`, change the `scoped_reply` signature and the system-prompt line. Current (around line 33-42):

```python
def scoped_reply(message, history, catalog):
    """..."""
    cat_txt = "\n".join(f"- {c['slug']}: {c.get('name','')} — {c.get('description','')}"
                        for c in (catalog or []))
    ...
    out = _llm_json(_SYSTEM + cat_txt, msgs)
```

Change to:

```python
def scoped_reply(message, history, catalog, overlay=""):
    """..."""
    cat_txt = "\n".join(f"- {c['slug']}: {c.get('name','')} — {c.get('description','')}"
                        for c in (catalog or []))
    ...
    out = _llm_json(_SYSTEM + (overlay + "\n\n" if overlay else "") + cat_txt, msgs)
```

(Keep the existing docstring and all other lines unchanged; only the signature and the `_llm_json(...)` system argument change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_practitioner_chat_overlay.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_chat.py tests/test_practitioner_chat_overlay.py
git commit -m "feat(practitioner_chat): scoped_reply accepts an optional system-prompt overlay"
```

---

### Task 2: Wire `/api/client/<code>/chat` (dispensary client widget)

Thin additive edit in `app.py`. Behavioral proof at go-live render-verify.

**Files:**
- Modify: `app.py` (`api_client_chat`, ~11572-11607)

**Interfaces:**
- Consumes: `ash_ally.ally_overlay(LOG_DB, email)`, `ash_ally.record_turn(LOG_DB, _db_lock, email, message, result.get("reply",""))`, and the `overlay=` param from Task 1.

- [ ] **Step 1: Add the overlay + pass it into scoped_reply**

The call site is `result = _chat.scoped_reply(message, history, catalog)` (~line 11594), with `email` already resolved + gated above it. Insert the overlay computation immediately before the call and add the `overlay=` kwarg:

```python
        _ally_ov = ash_ally.ally_overlay(LOG_DB, email)
        result = _chat.scoped_reply(message, history, catalog, overlay=_ally_ov)
```

(Match the existing indentation of the `result = _chat.scoped_reply(...)` line.)

- [ ] **Step 2: Add the record touch**

Immediately after that `result = _chat.scoped_reply(...)` line, add the background record dispatch:

```python
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock, email, message, result.get("reply", "")),
                      daemon=True).start()
        except Exception:
            pass
```

- [ ] **Step 3: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "scoped_reply(message, history, catalog, overlay=_ally_ov)" app.py`
Expected: present (the dispensary call site).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(dispensary): wire ASH ally into /api/client/<code>/chat"
```

---

### Task 3: Wire `/api/invoice/<token>/chat` (load email + overlay + record)

**Files:**
- Modify: `app.py` (`api_invoice_chat`, ~26421-26435)

**Interfaces:**
- Consumes: `_invoice_order_for_token(token)` (~26145, returns the order dict with an `"email"` key), `ash_ally.ally_overlay`, `ash_ally.record_turn`, the `overlay=` param.

- [ ] **Step 1: Load the order + email (replacing the bare token guard)**

Replace the current guard:

```python
    if not _pp.order_id_from_invoice_token(token):
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
```

with an order lookup that also yields the email (`_invoice_order_for_token` calls
`order_id_from_invoice_token` internally, so this removes the double call):

```python
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    email = (order.get("email") or "").strip().lower()
```

- [ ] **Step 2: Add overlay + pass into scoped_reply + record**

The call site is `result = _chat.scoped_reply(body.get("message") or "", body.get("history") or [], catalog)` (~line 26427). Change it to compute + pass the overlay and fire the record:

```python
    _ally_ov = ash_ally.ally_overlay(LOG_DB, email)
    _msg = body.get("message") or ""
    result = _chat.scoped_reply(_msg, body.get("history") or [], catalog, overlay=_ally_ov)
    try:
        import threading as _t
        _t.Thread(target=ash_ally.record_turn,
                  args=(LOG_DB, _db_lock, email, _msg, result.get("reply", "")),
                  daemon=True).start()
    except Exception:
        pass
```

(This replaces the single original `result = _chat.scoped_reply(body.get("message") or "", body.get("history") or [], catalog)` line; the `catalog = _build_ff_catalog()` line above it is unchanged.)

- [ ] **Step 3: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "_invoice_order_for_token(token)" app.py`
Expected: now appears in the `api_invoice_chat` handler (in addition to the helper definition).

Run: `grep -c "ash_ally.ally_overlay" app.py`
Expected: 6 (4 from SP2b-1 + 2 new client surfaces).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(invoice): load order email + wire ASH ally into /api/invoice/<token>/chat"
```

---

### Task 4: Full-suite green + verification

**Files:**
- Test: run-only.

- [ ] **Step 1: Run the relevant suites**

Run: `python3 -m pytest tests/test_practitioner_chat_overlay.py tests/test_ash_ally.py tests/test_ash_map.py -v`
Expected: ALL passing (Task 1's 3 + SP2b-1's helper/seam tests). Report the exact count.

- [ ] **Step 2: Confirm app parses + wiring counts**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('app ok')"`
Expected: `app ok`.

Run: `grep -c "ash_ally.ally_overlay" app.py` and `grep -c "ash_ally.record_turn" app.py`
Expected: 6 and 6 (4 SSE from SP2b-1 + 2 scoped_reply client surfaces).

- [ ] **Step 3: Confirm the practitioner caller is untouched**

Run: `grep -n "scoped_reply(message, history, catalog)" app.py`
Expected: still present for `/api/practitioner/chat` (~11556) — a 3-arg call with NO overlay (proves SP2b-3's surface was left alone).

- [ ] **Step 4: Commit (if any verification doc changes; otherwise skip)**

No code changes in this task. If nothing to commit, this task is complete at Step 3.

---

## Self-Review

**Spec coverage:**
- `scoped_reply` `overlay` param + injection → Task 1 ✓
- Backward-compat for the practitioner (3-arg) caller → Task 1 test + Task 4 Step 3 ✓
- Dispensary wiring (overlay + record) → Task 2 ✓
- Invoice wiring (email lookup + overlay + record) → Task 3 ✓
- Compute-overlay-in-app.py (no ash_map import in practitioner_chat) → Global Constraints + Tasks 2-3 ✓
- Background daemon-thread record, try/except-wrapped → Tasks 2-3 ✓
- Same flag, dark, go-live render-verify → spec Verification; suite/parse → Task 4 ✓
- Out of scope (practitioner client-search, scan-analysis, Glendalf) → not in any task ✓

**Placeholder scan:** none — every code/test step carries full content. Task 4 is run-only by design.

**Type consistency:** `scoped_reply(message, history, catalog, overlay="")`, `ash_ally.ally_overlay(LOG_DB, email)`, `ash_ally.record_turn(LOG_DB, _db_lock, email, <message>, result.get("reply",""))` — consistent across tasks. The 2 wirings pass `overlay=_ally_ov`; the practitioner call stays 3-arg.

**Wiring-task testing note:** Tasks 2-3 edit app.py and are gated by `ast.parse` + grep (no route harness; same boundary as SP2b-1). The injection logic itself is unit-tested in Task 1 via the `_llm_json` monkeypatch; behavioral proof is the go-live render-verify.
