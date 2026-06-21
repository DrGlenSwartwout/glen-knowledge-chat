# Compliance Denylist Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disease-anchor the over-blocking claim verbs in the topic-pages compliance denylist so benign wellness copy ("water treatment", "reverse osmosis", "prevent deficiency") passes while real disease claims ("treats diabetes", "cures cancer") still flag deterministically.

**Architecture:** A surgical change to `_BANNED` in `dashboard/topic_copy.py`: replace the four bare verb patterns (`treat(ment)`, `reverse`, `prevent`, `cure`) with one anchored pattern that requires a disease/condition word within ~30 chars after the verb; drop the bare `treatment` noun; keep `diagnose`/`guarantee` unanchored; `heal` is already anchored and stays. `local_claim_flags`, `compliance_scan`, the model layer, and the approve-gate are untouched.

**Tech Stack:** Python 3 `re`, pytest. No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-21-compliance-denylist-anchor-design.md`. Every task inherits its requirements.
- **Never `import app` in tests** — Pinecone is built at import and fails in the sandbox. Test `dashboard/topic_copy.py` directly. Use `python3`, not `python`.
- **Only `dashboard/topic_copy.py` changes** (the `_BANNED` list + a shared anchor constant). `local_claim_flags`, `compliance_scan` (fail-closed), `COMPLIANCE`, prompts, and the model layer are unchanged.
- The change only **narrows** local false positives. Real disease claims matching an anchored pattern must still flag. Defense-in-depth (anchored local → AI model layer → human approval) is preserved.
- Direction is **verb-then-disease only**; the reverse order ("diabetes can be cured") intentionally falls to the model layer.
- No feature flag (the gate is always on); rollback = `git revert`.
- `local_claim_flags` keeps its signature `(content) -> list[{"phrase","reason"}]` and its iterate-`_BANNED` structure; only the patterns change.

---

### Task 1: Disease-anchor the over-blocking claim verbs

**Files:**
- Modify: `dashboard/topic_copy.py` (the `_BANNED` list near the top, currently 7 patterns)
- Test: `tests/test_topic_copy.py` (append new tests)

**Interfaces:**
- Consumes/Produces: `local_claim_flags(content) -> list[dict]` (signature unchanged); `compliance_scan` unchanged. The change is internal to `_BANNED`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_topic_copy.py`:

```python
# --- disease-anchored denylist (over-blocking fix) ---

def test_benign_treatment_words_pass():
    tc = _mod()
    benign = [
        "Our municipal water treatment removes many additives.",
        "It uses reverse osmosis filtration at home.",
        "Good nutrition helps prevent deficiency over time.",
        "A relaxing spa treatment can feel restorative.",
        "Talk with your provider about a sensible treatment plan.",
    ]
    for text in benign:
        assert tc.local_claim_flags({"overview": text}) == [], text


def test_real_disease_claims_still_flag():
    tc = _mod()
    claims = [
        "this protocol treats diabetes",
        "it cures cancer naturally",
        "this reverses heart disease",
        "proven to prevent Alzheimer's",
        "heals your disease for good",
    ]
    for text in claims:
        assert tc.local_claim_flags({"overview": text}), text


def test_generic_condition_anchor_flags():
    tc = _mod()
    assert tc.local_claim_flags({"overview": "cure any condition fast"})
    assert tc.local_claim_flags({"overview": "treats this disorder"})


def test_diagnose_and_guarantee_still_flag_unanchored():
    tc = _mod()
    assert tc.local_claim_flags({"overview": "we diagnose the root cause"})
    assert tc.local_claim_flags({"overview": "results are guaranteed"})


def test_existing_planted_claim_still_flags():
    # must preserve the pre-existing test_local_claim_flags_catches_disease_claim behavior
    tc = _mod()
    flags = tc.local_claim_flags({"overview": "This protocol cures cancer and treats diabetes."})
    assert flags


def test_existing_clean_copy_still_clean():
    tc = _mod()
    flags = tc.local_claim_flags(
        {"overview": "People exploring low energy often look into sleep and minerals."})
    assert flags == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q -k "benign_treatment or real_disease_claims or generic_condition or diagnose_and_guarantee"`
Expected: `test_benign_treatment_words_pass` FAILS (current bare patterns flag "water treatment", "reverse osmosis", "prevent deficiency"). The claim/diagnose/guarantee tests may already pass under the old patterns — that is fine; the benign test is the one that must go from fail → pass.

- [ ] **Step 3: Replace `_BANNED` with the anchored patterns**

In `dashboard/topic_copy.py`, replace the entire current `_BANNED` block:

```python
# Hard denylist: any of these in draft copy fails the gate locally, no model needed.
_BANNED = [
    (r"\bcure[sd]?\b", "claims to cure"),
    (r"\btreat(s|ed|ing|ment)?\b", "claims to treat"),
    (r"\breverse[sd]?\b", "claims to reverse"),
    (r"\bheal[s]?\s+(your\s+)?(disease|cancer|diabetes)", "claims to heal a disease"),
    (r"\bprevent[s]?\b", "claims to prevent disease"),
    (r"\bdiagnos(e|es|is|ing)\b", "claims to diagnose"),
    (r"\bguarantee[sd]?\b", "outcome guarantee"),
]
```

with this anchored version:

```python
# Disease/condition anchor: category words plus common named conditions. A claim verb only
# trips the local gate when one of these sits just after it (verb-then-disease structure).
_DISEASE = (
    r"(?:disease|illness|condition|disorder|syndrome|infection|ailment|"
    r"cancer|tumou?r|diabetes|arthritis|asthma|eczema|psoriasis|hypertension|"
    r"depression|anxiety|alzheimer'?s?|dementia|parkinson'?s?|autism|adhd|"
    r"ibs|crohn'?s?|colitis|lupus|fibromyalgia|migraine|insomnia|allerg(?:y|ies)|"
    r"influenza|covid|copd|osteoporosis|neuropathy|gout)"
)

# Claim verbs (verb forms only -- NOT the bare noun "treatment", which over-blocked
# "water treatment"). These flag only when a disease/condition word follows within ~30 chars.
_CLAIM_VERBS = r"(?:cure[sd]?|treat(?:s|ed|ing)?|reverse[sd]?|prevent(?:s|ed|ing)?|heal(?:s|ed|ing)?)"

# Hard denylist: any match in draft copy fails the gate locally, no model needed.
_BANNED = [
    (rf"\b{_CLAIM_VERBS}\b[\w\s,'-]{{0,30}}\b{_DISEASE}\b",
     "claims to treat/cure/reverse/prevent a disease"),
    (r"\bdiagnos(e|es|is|ing)\b", "claims to diagnose"),
    (r"\bguarantee[sd]?\b", "outcome guarantee"),
]
```

Leave `local_claim_flags` and everything else in the file unchanged. (`local_claim_flags` already lowercases the text and `re.search`es each pattern, so the case-insensitive matching of "Alzheimer's" etc. works via the existing `.lower()`.)

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q`
Expected: PASS — all tests in the file (the new 6 plus every pre-existing topic_copy test).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_copy.py tests/test_topic_copy.py
git commit -m "fix(compliance): disease-anchor treat/cure/reverse/prevent so benign wellness copy passes"
```

---

### Task 2: Full topic_copy + compliance regression

**Files:** none (verification only)

- [ ] **Step 1: Run the whole topic_copy suite**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q`
Expected: all green (existing + new).

- [ ] **Step 2: Confirm the compliance_scan path still gates a planted claim**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q -k "compliance_scan"`
Expected: the existing `compliance_scan` tests pass — a planted disease claim still blocks without consulting the model, clean copy passes, and an errored client fails closed. (No code changed here; this confirms the `_BANNED` edit did not regress the scan.)

- [ ] **Step 3: Broader sanity (ignore Pinecone app-import errors)**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/ -q -k "topic_copy or topic_pages or topic_page_actions" --continue-on-collection-errors`
Expected: the topic dashboard tests pass; any `PineconeConfigurationError` collection errors are environmental (tests that `import app`), not regressions.

---

## Rollout (post-merge, not a code task)

- Open a PR `compliance-denylist-anchor` → `main`; merge (direct-push guarded). Render deploys; the gate behavior updates on deploy.
- Post-deploy: regenerate `fluoride-exposure` (now titled "Fluoride-Free Living") via `topic_page.regenerate`; it should now pass the gate ("water treatment" no longer trips) → approve it. Spot-check one already-approved page is unaffected.
- Rollback if needed: `git revert` the commit (no flag to flip).

## Self-Review notes (author)

- **Spec coverage:** §4 anchored `_BANNED` + `_DISEASE` + `_CLAIM_VERBS` → Task 1 Step 3; drop bare `treatment` noun → Task 1 (verb group omits `ment`); keep `diagnose`/`guarantee` unanchored → Task 1; §6 testing (benign-pass, claims-still-flag, generic-anchor, diagnose/guarantee, existing-preserved) → Task 1 Steps 1/4 + Task 2; §7 rollout → rollout section. No gaps.
- **Type consistency:** `local_claim_flags(content) -> list[{"phrase","reason"}]` signature unchanged; only `_BANNED` contents change; `compliance_scan` consumes `local_claim_flags` unchanged.
- **Watch-item:** the anchored pattern's flag `phrase` is now the whole matched span (e.g. "treats diabetes") rather than a single word — tests assert truthiness/emptiness of the flag list, not exact phrase text, so this is fine.
