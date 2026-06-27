# Brief-as-Lead + Readiness-Triaged CTA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.
> **Spec:** `docs/superpowers/specs/2026-06-27-context-aware-chat-ux-design.md` (+ brief-instruction draft alongside it).
> **Base:** branch off current `main` (includes PR #360 — `_resolve_chat_tier`, `_velocity_guard`, the anonymous-full depth gate, `handleGatedFullReport`/`handleVerifyFullReport` in `embed.html` all already exist).

**Goal:** Make the brief answer a deliberate Benson *lead* (5-beat include-and-transcend) whose beat-5 CTA is matched to reader readiness (page-link / email-report / direct-action / inline), and instrument which CTA was shown vs clicked.

**Architecture:** A prompt rewrite produces the 5-beat brief and ends it with a machine-readable CTA directive (`⟦CTA⟧ type | target | label`). A new pure module parses + strips that directive; `chat()` filters it out of the streamed text, emits a structured `cta` SSE event, and logs the rung/type. The frontend renders the single matched CTA control (reusing #360's email-capture for the email rung) and reports clicks to a new logging endpoint.

**Tech Stack:** Flask, raw sqlite3, Anthropic Haiku 4.5 (SSE streaming), vanilla JS in `static/embed.html`. No new dependencies.

## Global Constraints

- No new dependencies. Stdlib + existing helpers only.
- FAIL OPEN: if CTA parsing/classification fails, the answer still renders; default to NO special CTA (inline) rather than erroring. Never 500 the chat over CTA logic.
- The brief stays ~200 words; beats 1-3 must be a genuine standalone quick win.
- SAFETY OVERRIDE: never withhold safety-critical/time-sensitive info behind the loop; tease optimization/depth, never safety.
- Keep ALL existing answer rules: no printed beat labels, no "Hook" label, formulation-first ordering, product links only from the injection table, Speckhart boundary, active discount-code rule, Sources line.
- The brief NAMES an assumption; the full report BREAKS it (beat-4 ↔ full coupling). Never resolve the assumption in the brief.
- CTA directive grammar (verbatim): a final line `⟦CTA⟧ <type> | <target> | <label>` where `<type>` ∈ `page | email | action | inline`. `<target>` is a URL (page/action) or empty (email/inline). `<label>` is button text (may be empty).
- Readiness rung names (verbatim): `curious | engaged | ready | committed`. Mapping to CTA type: curious→page, engaged→email, ready→action, committed→inline.
- DB: additive `ALTER TABLE` wrapped in try/except (idiom in `_init_log_db`).

---

### Task 1: `dashboard/chat_cta.py` — parse + strip the CTA directive (pure)

**Files:**
- Create: `dashboard/chat_cta.py`
- Test: `tests/test_chat_cta.py`

**Interfaces — Produces:**
- `SENTINEL = "⟦CTA⟧"`
- `VALID_TYPES = ("page", "email", "action", "inline")`
- `parse_cta(answer: str) -> tuple[str, dict | None]` — returns `(clean_text, cta)` where `clean_text` is the answer with the directive line removed and trailing whitespace stripped, and `cta` is `{"type","target","label"}` or `None` if absent/malformed. Unknown type → treated as absent (returns `None`, but still strips the sentinel line so it never shows).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chat_cta.py
from dashboard.chat_cta import parse_cta, SENTINEL

def test_parse_email_directive():
    ans = "Here is your brief answer.\n\n⟦CTA⟧ email |  | Send my full report"
    text, cta = parse_cta(ans)
    assert text == "Here is your brief answer."
    assert cta == {"type": "email", "target": "", "label": "Send my full report"}

def test_parse_page_directive_with_url():
    ans = "Body.\n⟦CTA⟧ page | https://x.com/p | Read the full breakdown"
    text, cta = parse_cta(ans)
    assert cta["type"] == "page" and cta["target"] == "https://x.com/p"
    assert SENTINEL not in text

def test_no_directive_returns_none():
    text, cta = parse_cta("Just an answer, no directive.")
    assert cta is None and text == "Just an answer, no directive."

def test_unknown_type_stripped_and_none():
    text, cta = parse_cta("Body.\n⟦CTA⟧ bogus | | x")
    assert cta is None and SENTINEL not in text  # never leak the sentinel

def test_inline_type():
    _, cta = parse_cta("Body.\n⟦CTA⟧ inline | | ")
    assert cta == {"type": "inline", "target": "", "label": ""}
```

- [ ] **Step 2: Run → FAIL** — `python3 -m pytest tests/test_chat_cta.py -q`
- [ ] **Step 3: Implement `dashboard/chat_cta.py`**

```python
"""Pure parser for the trailing CTA directive a brief answer emits.
Grammar: a final line  ⟦CTA⟧ <type> | <target> | <label>
type ∈ page|email|action|inline. Flask-free; unit-testable in isolation."""
SENTINEL = "⟦CTA⟧"
VALID_TYPES = ("page", "email", "action", "inline")

def parse_cta(answer: str):
    text = answer or ""
    idx = text.rfind(SENTINEL)
    if idx == -1:
        return (text.strip(), None)
    directive = text[idx + len(SENTINEL):].strip()
    clean = text[:idx].rstrip()
    parts = [p.strip() for p in directive.split("|")]
    ctype = parts[0].lower() if parts else ""
    if ctype not in VALID_TYPES:
        return (clean, None)          # strip the sentinel, no cta
    target = parts[1] if len(parts) > 1 else ""
    label = parts[2] if len(parts) > 2 else ""
    return (clean, {"type": ctype, "target": target, "label": label})
```

- [ ] **Step 4: Run → PASS**
- [ ] **Step 5: Commit** — `feat(chat-cta): pure parser for the brief CTA directive`

---

### Task 2: Brief synth instruction — the 5-beat lead + readiness rubric + CTA directive

**Files:**
- Modify: `app.py` — the brief branch of `synth_instr` inside `chat()`'s `generate()` (the `else` string beginning `"Produce the DEFAULT EXECUTIVE SUMMARY response …"`), and the system DEFAULT FORMAT block in `_SYSTEM_BASE`.
- Reference (do not retype): `docs/superpowers/specs/2026-06-27-context-aware-chat-ux-brief-instruction-draft.txt` — the validated 5-beat + readiness-rubric text.
- Test: `tests/test_brief_lead_eval.py` (structural eval, runs under Doppler).

**Interfaces — Consumes:** nothing. **Produces:** the brief generator now emits 5-beat prose ending with a `⟦CTA⟧ …` directive line.

- [ ] **Step 1: Replace the brief synth instruction.** Set the brief branch string to the contents of the brief-instruction draft file, with these additions appended to it:
  - "Do NOT print the beat labels — weave the 5 beats into natural prose."
  - "END your entire response with the Sources line, then on a final separate line emit a machine directive (the user will NOT see it rendered as text): `⟦CTA⟧ <type> | <target> | <label>` — `<type>` is the readiness rung's CTA: curious→`page`, engaged→`email`, ready→`action`, committed→`inline`. For `page`/`action`, `<target>` is the exact URL from the retrieved sources or product injection table (never invent one; if you have no real URL, use `email` instead). For `email`, leave target empty and set label to a short report offer. Example: `⟦CTA⟧ email |  | Send my full report`."
  Keep the existing hard rules already in `_SYSTEM_BASE` intact.

- [ ] **Step 2: Nudge the full instruction for beat-4 coupling.** In `_long_form_synth_instr` (both branches) add one sentence: "If the question implies a single hidden assumption a brief answer would credit-then-question, make breaking THAT assumption the central belief you break and rebuild."

- [ ] **Step 3: Write the structural eval test** (not exact-match — invariants only):

```python
# tests/test_brief_lead_eval.py  — runs only with a real API key (Doppler)
import os, re, importlib, sys, pytest
from pathlib import Path

CASES = [
  "What foods are best for macular degeneration?",                    # expect page
  "I have wet AMD; changed my diet since Nov but bloodwork is the same. Try Gundry's lectin protocol?",  # expect email
]

def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as a; importlib.reload(a); return a
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs API key")
def test_brief_emits_valid_cta_and_is_bounded(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    from dashboard.chat_cta import parse_cta, VALID_TYPES
    import anthropic
    cl = anthropic.Anthropic()
    # build the brief instruction exactly as chat() does for an anonymous brief turn
    instr = a._brief_synth_instruction()    # see Step 4
    for q in CASES:
        r = cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=1024,
            system=a.SYSTEM_PROMPT + "\n" + instr,
            messages=[{"role": "user", "content": q}])
        raw = r.content[0].text
        clean, cta = parse_cta(raw)
        assert cta is not None, f"no CTA directive emitted for: {q}"
        assert cta["type"] in VALID_TYPES
        assert "⟦CTA⟧" not in clean                       # stripped from visible
        assert len(clean.split()) <= 240                  # bounded (~200 target)
        assert "Hook" not in clean                        # no label leak
```

- [ ] **Step 4: Extract the brief instruction into a helper** `_brief_synth_instruction() -> str` in `app.py` returning the brief instruction string, and call it from both `chat()` and the eval test, so the test exercises the real prompt (DRY — one source of truth).

- [ ] **Step 5: Run the eval under Doppler**

Run: `SCRATCH=/tmp/ev && mkdir -p $SCRATCH && doppler run -p remedy-match -c prd -- env DATA_DIR=$SCRATCH python3 -m pytest tests/test_brief_lead_eval.py -q -p no:cacheprovider`
Expected: PASS (both cases emit a valid, bounded, label-free brief with a CTA directive). If a case's rung differs from the comment, that's acceptable (rung choice is model judgment) — the asserted invariants are structure, not the specific rung.

- [ ] **Step 6: Commit** — `feat(chat): brief is a 5-beat lead emitting a readiness CTA directive`

---

### Task 3: Wire CTA parsing into `chat()` — strip directive, emit `cta` SSE, log rung/type

**Files:**
- Modify: `app.py` — `chat()`'s `generate()` streaming loop + the `done` SSE event + `log_query` call; `_init_log_db` (migration); `log_query` signature.
- Test: `tests/test_chat_cta_wiring.py` (app test client, under Doppler).

**Interfaces — Consumes:** `parse_cta` (Task 1). **Produces:** the `/chat` SSE `done` event now carries `cta` (or null); `query_log` has `cta_type`, `cta_rung` columns.

- [ ] **Step 1: Migration.** Add to the `_init_log_db` col list: `"cta_type   TEXT"` and `"cta_rung   TEXT"` (try/except ALTER idiom).
- [ ] **Step 2: Stream-safe directive hiding.** In `generate()`, while iterating `stream.text_stream`, do NOT forward any text from the `⟦CTA⟧` sentinel onward to the client. Buffer the full answer as today (`full_answer.append(token)`), but gate visible `yield sse({"token": ...})` so that once the sentinel has appeared in the accumulated text, subsequent tokens are withheld. Minimal approach:

```python
        _emitted = 0   # chars already streamed as visible tokens
        for token in stream.text_stream:
            full_answer.append(token)
            acc = "".join(full_answer)
            cut = acc.find("⟦CTA⟧")
            visible_target = acc if cut == -1 else acc[:cut]
            if len(visible_target) > _emitted:
                yield sse({"token": visible_target[_emitted:]})
                _emitted = len(visible_target)
```

- [ ] **Step 3: After the stream loop**, parse and attach:

```python
        from dashboard.chat_cta import parse_cta
        _clean, _cta = parse_cta("".join(full_answer))
        _rung = {"page":"curious","email":"engaged","action":"ready","inline":"committed"}.get(
                    (_cta or {}).get("type"), None)
        # ... existing log_query call, now passing cta_type/cta_rung and the CLEAN answer:
        log_query(query, level, _clean, session_id=session_id, email=email, name=name,
                  mode=mode, ..., cta_type=(_cta or {}).get("type"), cta_rung=_rung)
```
  Add `cta_type: str = None, cta_rung: str = None` params (LAST) to `log_query` and include them in the INSERT (additive, all existing callers unaffected). Store `_clean` (directive stripped), not the raw answer.

- [ ] **Step 4: Emit the cta on done.** In the final `done` SSE payload add `"cta": _cta` (or `None`). Wrap the parse in try/except → on error, `_cta=None`, `_clean="".join(full_answer)` (fail open).

- [ ] **Step 5: Test** (under Doppler — needs app import). Drive `/chat` with a brief request and assert the `done` event JSON contains a `cta` key (value may be null if the model emitted none); seed-free structural check. Also unit-assert that when `log_query` is called with `cta_type="email"`, the row stores `email`/`engaged`. Keep assertions structural (don't assert the model's rung).

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$SCRATCH python3 -m pytest tests/test_chat_cta_wiring.py -q -p no:cacheprovider` → PASS.

- [ ] **Step 6: Commit** — `feat(chat): strip CTA directive, emit cta SSE event, log rung/type`

---

### Task 4: Frontend — render the matched CTA + report clicks

**Files:**
- Modify: `static/embed.html` — the `/chat` SSE reader(s) that handle the `done` event; a new `renderCta(cta, logId)` helper; a `/api/cta-click` POST on click.
- Modify: `app.py` — new `/api/cta-click` route (logging only).
- Test: `tests/test_cta_click.py` (endpoint) + controller render-verify.

**Interfaces — Consumes:** the `done` event's `cta` field (Task 3). **Produces:** a rendered CTA control per type; click logging.

- [ ] **Step 1: `/api/cta-click` endpoint** (POST JSON `{log_id, cta_type}`) — append to a new `cta_clicks(ts, log_id, cta_type)` table (or set `query_log.cta_clicked_at` by log_id). Console-key NOT required (public, but rate-limit-exempt, minimal data). Fail open. Unit-test it inserts a row.
- [ ] **Step 2: `renderCta(cta, logId)`** in `embed.html`, called when the `done` event has a non-null `cta`:
  - `email` → reuse the #360 email-capture path (`handleGatedFullReport(logId)` style) under the directive's label.
  - `page` / `action` → render a button/link to `cta.target` with `cta.label`; on click POST `/api/cta-click` then open the target.
  - `inline` → render nothing extra.
  Define `renderCta` before the reader references it. Guard against missing fields; never throw.
- [ ] **Step 3: Wire** the relevant SSE reader(s)' `done` handling to call `renderCta(evt.cta, evt.log_id)` when `evt.cta`.
- [ ] **Step 4: Endpoint test** — `tests/test_cta_click.py` posts to `/api/cta-click` and asserts a row is recorded (reload-app convention; under Doppler).
- [ ] **Step 5: RENDER-VERIFY (controller, headless)** per `feedback_render_verify_not_just_inject`: boot the app under Doppler, load `/embed`, simulate a `done` event with each `cta.type`, assert the right control renders and ZERO console errors. Do NOT rely on injection-only checks. (This is exactly the discipline that caught the smart-quote SyntaxError in #360.)
- [ ] **Step 6: Commit** — `feat(embed): render readiness-matched CTA + log clicks`

---

## Verification (end-to-end)

- **Unit (plain pytest):** `tests/test_chat_cta.py` — the pure parser.
- **Under Doppler:** the brief eval (`test_brief_lead_eval.py`), the chat wiring (`test_chat_cta_wiring.py`), the click endpoint (`test_cta_click.py`).
- **Render-verify (headless):** each `cta.type` renders the correct control on `/embed` with zero console errors.
- **Manual smoke:** a cold "what is X" question → page-link CTA, no email ask; a personal/plateau question → email-capture CTA; a member → inline (no gate). Instrumentation rows appear in `query_log.cta_type`/`cta_rung` and `cta_clicks`.
- **Pre-PR hygiene:** `git diff --name-only origin/main..HEAD | grep -i superpowers` and `git rm --cached` any leaked SDD scratch (per `feedback_sdd_scratch_git_leak`).

## Out of scope (follow-on plan — Component 3)

Context-aware interface: the UI profile (surface × identity), per-surface control simplification, and funnel quick-reply chips. Tracked in the spec; separate plan after this lands.

## Notes on testing prompt work

The brief is LLM output — we assert *structural invariants* (valid CTA directive, bounded length, no label leak, sentinel stripped), never exact text. The model's choice of rung is judgment, validated by eval + manual review, not unit asserts.
