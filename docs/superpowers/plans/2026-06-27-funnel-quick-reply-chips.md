# Funnel Quick-Reply Chips (Remedy-Match Socratic flow) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.
> **Spec:** Component 3 of `docs/superpowers/specs/2026-06-27-context-aware-chat-ux-design.md` (narrowed during planning: chips-only, on the remedy-match Socratic flow; per-surface UI-profile consolidation dropped as low-reward since surface gating already works ad-hoc).
> **Depends on:** PR #366 (the `dashboard/chat_cta.py` directive infra — `parse_cta`, `stream_visible`, `SENTINEL`). **Sequence AFTER #366 merges; branch off the resulting main.** This plan generalizes that module.
> **Surface model:** chips live in the **funnel's remedy-match Socratic flow** (`/begin/match`), where users tap out one-word replies (Yes/None/Agree). Not the open `/chat` Q&A. See `project_chat_surface_model`.

**Goal:** When the remedy-match bot asks a discrete-answer clarifying question (yes/no, confirm, or 2-4 choices), render tap-able quick-reply chips under it; tapping submits that answer. Always keep the free-text input.

**Architecture:** The answer-generating model emits a hidden trailing directive `⟦CHIPS⟧ opt1 | opt2 | ...` (mirroring the CTA directive). The backend hides it from the streamed text (generalized `stream_visible`), parses it (`parse_chips`), and returns `chips` in the `/begin/match/chat` done event. `begin-match.html` renders the chips and submits the tapped text via its existing `fillQuery`+`sendQuery`.

**Tech Stack:** Flask, raw sqlite3, Anthropic Haiku 4.5 (SSE), vanilla JS in `static/begin-match.html`. No new dependencies.

## Global Constraints

- No new dependencies. Stdlib + existing helpers only.
- FAIL OPEN: a chip parse/classify error must never break the answer or the chat; default to NO chips.
- The streamed visible text must NEVER contain the `⟦CHIPS⟧` sentinel or anything after it (it is hidden, like `⟦CTA⟧`).
- Chips: 0-4 per question; emit ONLY for discrete-answer questions (yes/no, confirm, 2-4 choices); OMIT for open-ended questions. Each chip ≤ ~4 words, phrased as the person's reply.
- Free-text input always remains available; chips are an accelerator, not a gate.
- Sentinel is literal `⟦CHIPS⟧` (U+27E6 ⟦ ... U+27E7 ⟧, same brackets as CTA). Straight ASCII quotes only — NO smart quotes (curly quotes as JS string delimiters are a real bug hit twice on this project).
- Only `/begin/match/chat` + `begin-match.html` change. `/chat` and `embed.html` (the CTA path) are untouched except the shared `stream_visible` signature staying backward-compatible.

## File Structure

- **`dashboard/chat_cta.py`** — generalize `stream_visible(tokens, sentinel=SENTINEL)`; add `CHIPS_SENTINEL` + `parse_chips`. (Same module; it's the directive toolkit now.)
- **`app.py`** — `_REMEDY_MATCH_SYSTEM` gains the chips rule; `/begin/match/chat` streaming uses `stream_visible(..., sentinel=CHIPS_SENTINEL)`, parses chips, sets `answer=clean`, adds `chips` to the done event; optional `/api/chip-tap` + `chip_taps` table.
- **`static/begin-match.html`** — `renderChips(chips)` + wire into the done handler; tap → `fillQuery`+`sendQuery`; optional tap logging.

---

### Task 1: Generalize `chat_cta.py` — `stream_visible(sentinel=...)` + `parse_chips`

**Files:**
- Modify: `dashboard/chat_cta.py`
- Test: `tests/test_chat_cta.py` (extend)

**Interfaces — Produces:**
- `CHIPS_SENTINEL = "⟦CHIPS⟧"`
- `stream_visible(tokens, sentinel=SENTINEL)` — now parameterized; hides from `sentinel` onward, holding back `len(sentinel)-1` trailing chars to avoid flashing a partial. Default `SENTINEL` keeps `/chat`'s existing call working unchanged.
- `parse_chips(answer) -> tuple[str, list[str]]` — returns `(clean_text, chips)` where `chips` is up to 4 trimmed non-empty options from a `⟦CHIPS⟧ a | b | c` directive (empty list if absent/malformed), and `clean_text` has the directive line stripped.

- [ ] **Step 1: Add failing tests** (extend `tests/test_chat_cta.py`):

```python
from dashboard.chat_cta import (parse_chips, stream_visible, CHIPS_SENTINEL, SENTINEL)

def test_parse_chips_basic():
    text, chips = parse_chips("Are you hyper or hypo?\n⟦CHIPS⟧ Overactive | Underactive | Not sure")
    assert text == "Are you hyper or hypo?"
    assert chips == ["Overactive", "Underactive", "Not sure"]

def test_parse_chips_absent():
    text, chips = parse_chips("What is your main concern?")
    assert chips == [] and text == "What is your main concern?"

def test_parse_chips_caps_at_4_and_trims_empties():
    _, chips = parse_chips("Q\n⟦CHIPS⟧ a | b | c | d | e |  ")
    assert chips == ["a", "b", "c", "d"]      # cap 4, drop empties

def test_stream_visible_param_sentinel_hides_chips():
    out = "".join(stream_visible(["Pick one.", "⟦", "CHIPS", "⟧", " a | b"], sentinel=CHIPS_SENTINEL))
    assert out == "Pick one." and "⟦" not in out      # split-token safe

def test_stream_visible_default_still_hides_cta():
    out = "".join(stream_visible(["Body.", "⟦CTA⟧ email |  | x"]))
    assert out == "Body." and SENTINEL not in out
```

- [ ] **Step 2: Run → FAIL** — `python3 -m pytest tests/test_chat_cta.py -q`
- [ ] **Step 3: Implement** in `dashboard/chat_cta.py`:
  - Add `CHIPS_SENTINEL = "⟦CHIPS⟧"`.
  - Change `def stream_visible(tokens):` to `def stream_visible(tokens, sentinel=SENTINEL):` and replace internal uses of the hard-coded `SENTINEL`/its length with the `sentinel` parameter (the find target and the `hold = len(sentinel) - 1`).
  - Add:

```python
def parse_chips(answer):
    """Extract a trailing ⟦CHIPS⟧ a | b | c directive. Returns (clean_text, chips[<=4])."""
    text = answer or ""
    idx = text.rfind(CHIPS_SENTINEL)
    if idx == -1:
        return (text.strip(), [])
    directive = text[idx + len(CHIPS_SENTINEL):]
    clean = text[:idx].rstrip()
    chips = [c.strip() for c in directive.split("|")]
    chips = [c for c in chips if c][:4]
    return (clean, chips)
```

- [ ] **Step 4: Run → PASS** (new tests + all prior `chat_cta` tests still green).
- [ ] **Step 5: Commit** — `feat(chat-cta): parameterize stream_visible sentinel; add parse_chips`

---

### Task 2: `_REMEDY_MATCH_SYSTEM` chips rule + `/begin/match/chat` wiring + eval

**Files:**
- Modify: `app.py` — `_REMEDY_MATCH_SYSTEM` (~line 3500-3519) and the `/begin/match/chat` streaming block (~line 3635-3641; `answer = "".join(full)` follows).
- Test: `tests/test_match_chips_eval.py` (pass-rate eval, under Doppler).

**Interfaces — Consumes:** `stream_visible`, `parse_chips`, `CHIPS_SENTINEL` (Task 1). **Produces:** `/begin/match/chat` done event carries `chips: [...]`.

- [ ] **Step 1: Add the chips rule to `_REMEDY_MATCH_SYSTEM`.** Append to that system string (straight ASCII quotes, literal ⟦CHIPS⟧):

```
" \nQUICK-REPLY CHIPS: When the question you ask has a few discrete answers"
" (yes/no, a confirm, or 2-4 distinct choices), end your message with a hidden"
" directive on its OWN FINAL LINE: ⟦CHIPS⟧ opt1 | opt2 | ... — 2 to 4 SHORT"
" tap-able answers (each <= 4 words), phrased exactly as the PERSON would reply"
" (e.g. 'Yes' | 'No' | 'Not sure', or 'Overactive' | 'Underactive'). OMIT the"
" directive entirely when your question is open-ended (e.g. 'What is your main"
" concern?'). The person can always type instead — chips are just a shortcut."
```

- [ ] **Step 2: Wire the streaming block.** The current loop is:
```python
        with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=900,
                                 system=_match_system, messages=messages) as stream:
            for tok in stream.text_stream:
                tok = _strip_dash(tok); full.append(tok); yield sse({"token": tok})
        answer = "".join(full)
```
  Replace with (apply `_strip_dash` per token, accumulate into `full`, stream only the visible portion, hiding the CHIPS directive):
```python
        from dashboard.chat_cta import stream_visible, parse_chips, CHIPS_SENTINEL
        with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=900,
                                 system=_match_system, messages=messages) as stream:
            def _toks():
                for tok in stream.text_stream:
                    t = _strip_dash(tok); full.append(t); yield t
            for delta in stream_visible(_toks(), sentinel=CHIPS_SENTINEL):
                yield sse({"token": delta})
        try:
            _clean, _chips = parse_chips("".join(full))
        except Exception:
            _clean, _chips = "".join(full), []
        answer = _clean
```
  `answer` is now the directive-stripped text — so the existing downstream match-detection / history use of `answer` never sees the sentinel (mirrors the `answer = _clean` fix in `/chat`).

- [ ] **Step 3: Add `chips` to the done event.** Find the `/begin/match/chat` done payload (`{"done": True, "session_id": ..., "sources": ..., "chunks_retrieved": ...}`) and add `"chips": _chips`.

- [ ] **Step 4: Eval** (LLM output → assert RATES, not single-shot — per the brief-eval lesson). `tests/test_match_chips_eval.py`, skip if no `ANTHROPIC_API_KEY`, reload-app convention. Build `_match_system` the way the route does (or call a small helper), then for each scenario sample 3x via Haiku with `_REMEDY_MATCH_SYSTEM` and assert with `parse_chips`:
  - Scenario A — a yes/no-ish prompt (history that elicits a confirm question): in >= 2/3 samples, `parse_chips` yields 2-4 chips, each `<= 4 words`, and the sentinel is absent from the cleaned text.
  - Scenario B — an obviously open-ended first turn (e.g. user just said "hi"): in >= 2/3 samples, chips == [] (model omits the directive for open questions).
  Print the per-sample chip lists. Run:
```
S=/tmp/chipseval; mkdir -p "$S"
doppler run -p remedy-match -c prd -- env DATA_DIR="$S" python3 -m pytest tests/test_match_chips_eval.py -q -p no:cacheprovider
```
  Must PASS (not skip). If a scenario is flaky, raise samples to 5 and require >=3, and tune the system-prompt wording until stable; paste observed chip lists in the report.

- [ ] **Step 5: Commit** — `feat(begin-match): emit/parse ⟦CHIPS⟧ quick-reply directive`

---

### Task 3: `begin-match.html` — render chips + submit on tap (+ render-verify)

**Files:**
- Modify: `static/begin-match.html` — the `evt.done` handler (~lines 420-428) + a new `renderChips` helper + a `.quick-chip` style.
- Test: controller headless render-verify (no unit test for pure DOM JS).

**Interfaces — Consumes:** the done event's `chips` array (Task 2). **Produces:** tap-chips under the last assistant message that submit on click.

- [ ] **Step 1: Add `renderChips(chips)`** near the other helpers (after `appendMessage`/`fillQuery`, ~line 244). Straight ASCII quotes; guard everything; never throw:
```javascript
  function renderChips(chips) {
    try {
      if (!Array.isArray(chips) || !chips.length) return;
      var msgs = document.getElementById('messages');
      if (!msgs) return;
      var assist = msgs.querySelectorAll('.message.assistant');
      var lastMsg = assist.length ? assist[assist.length - 1] : null;
      if (!lastMsg) return;
      var wrap = document.createElement('div');
      wrap.className = 'chip-row';
      chips.slice(0, 4).forEach(function (label) {
        if (!label) return;
        var b = document.createElement('button');
        b.className = 'quick-chip';
        b.type = 'button';
        b.textContent = label;
        b.addEventListener('click', function () {
          wrap.remove();                 // consume the chips once one is tapped
          fillQuery(label);
          sendQuery();
        });
        wrap.appendChild(b);
      });
      lastMsg.appendChild(wrap);
      msgs.scrollTop = msgs.scrollHeight;
    } catch (_) {}
  }
```
  (Use `.message.assistant` last-index, NOT `:last-of-type` — the latter is the latent-fragility pattern flagged on this project.)

- [ ] **Step 2: Add `.quick-chip` / `.chip-row` CSS** in the page's `<style>` block (mirror the `.example-q` look, lines ~60-63): a horizontal wrap of small rounded clickable pills.
- [ ] **Step 3: Wire the done handler.** In the `if (evt.done) { ... }` block (~lines 420-428), after the history pushes, add:
```javascript
      if (evt.chips && evt.chips.length) renderChips(evt.chips);
```
- [ ] **Step 4: RENDER-VERIFY (controller, headless)** per `feedback_render_verify_not_just_inject`: boot the app under Doppler, load `/begin/match`, then in the console call `renderChips(['Yes','No','Not sure'])` against a seeded assistant message — assert 3 `.quick-chip` buttons render, a tap calls `fillQuery`+`sendQuery` (stub `sendQuery` to capture), and ZERO console errors. Then assert `renderChips([])` and `renderChips(null)` are silent no-ops. (This discipline caught a script-killing SyntaxError in #360 and a partial-sentinel leak in #366 — do NOT skip it.)
- [ ] **Step 5: Commit** — `feat(begin-match): render quick-reply chips, submit on tap`

---

### Task 4 (light, instrumentation): chip-tap logging

**Files:** Modify `app.py` (new `/api/chip-tap` + `chip_taps` table); Modify `static/begin-match.html` (fire-and-forget POST on tap); Test: `tests/test_chip_tap.py`.

- [ ] **Step 1:** `_init_chip_taps()` → `chip_taps(ts TEXT, session_id TEXT, label TEXT)` (init at module load, `_db_lock` idiom). `POST /api/chip-tap` JSON `{session_id, label}` → INSERT; no console key; FAIL OPEN (always `{"ok": true}` 200). Unit test asserts a row is written.
- [ ] **Step 2:** In `renderChips`'s click handler, before `fillQuery`, fire-and-forget `fetch('/api/chip-tap', {method:'POST', credentials:'same-origin', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId, label: label})}).catch(function(){})`. (`sessionId` is in scope in begin-match.html.)
- [ ] **Step 3:** Test under Doppler: `doppler run ... python3 -m pytest tests/test_chip_tap.py -q -p no:cacheprovider` → PASS. Re-render-verify a tap still works with the POST added (0 console errors).
- [ ] **Step 4: Commit** — `feat(begin-match): log chip taps for instrumentation`

> If trimming scope, ship Tasks 1-3 (the feature) and land Task 4 (measurement) as a fast-follow.

---

## Verification (end-to-end)

- **Unit (plain pytest):** `tests/test_chat_cta.py` — `parse_chips` + parameterized `stream_visible` incl. split-token.
- **Under Doppler:** the chips eval (`test_match_chips_eval.py`, pass-rate), the chip-tap endpoint.
- **Render-verify (headless):** `/begin/match` renders chips, tap submits via `fillQuery`+`sendQuery`, empty/null no-op, zero console errors.
- **Manual smoke:** open `/begin/match`, answer a couple turns — a yes/no or "overactive or underactive?" question shows tap-chips; tapping one sends it; an open question ("what's your main concern?") shows none; typing still works.
- **Pre-PR hygiene:** `git diff --name-only origin/main..HEAD | grep -i superpowers` and `git rm --cached` any leaked `.superpowers/sdd` scratch (per `feedback_sdd_scratch_git_leak`).

## Notes

- **Testing LLM output:** the chips eval asserts STRUCTURAL RATES (chip count, length, omitted-when-open), never exact chip text — and uses multi-sample pass-rate, per the lesson that a single-sample LLM eval falsely passed on this project.
- **Out of scope (tracked elsewhere):** the per-surface UI-profile consolidation (low reward; ad-hoc gating already works); migrating `/begin/concierge/chat` into the portal (separate follow-on, see `project_chat_surface_model`); chips on the open `/chat` Q&A (different, lower-friction surface).
