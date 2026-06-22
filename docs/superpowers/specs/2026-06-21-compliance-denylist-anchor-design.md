# Compliance Denylist — disease-anchor the over-blocking claim verbs

**Date:** 2026-06-21
**Status:** Approved design, ready for implementation plan
**Touches:** `dashboard/topic_copy.py` (the topic-pages compliance gate, live on `/learn`)

---

## 1. Purpose

The topic-pages compliance gate's local regex denylist (`_BANNED` in `dashboard/topic_copy.py`)
flags the bare verbs `treat`/`cure`/`reverse`/`prevent` (and the noun `treatment`) regardless of
context. That over-blocks legitimate wellness copy: "water **treatment**", "**reverse** osmosis",
"**prevent** deficiency", "spa **treatment**" all fail the gate even though none is a disease claim.
A real case: the `fluoride-exposure` topic page could not pass because its copy used "water
treatment" and "reverse".

This refinement **disease-anchors** those verbs — they flag only when a disease/condition word is
nearby — exactly how the existing `heal` pattern already works (`heal ... (disease|cancer|diabetes)`).
Benign verb uses pass the local net; the AI model layer still judges anything subtler.

## 2. Decision (locked during brainstorming)

**Disease-anchor the verbs.** Make `treat/cure/reverse/prevent` flag only with a disease/condition
word nearby; drop the bare `treatment` noun from the denylist; keep `diagnose` and `guarantee`
unanchored (rarely benign in educational health copy); `heal` stays as-is (already anchored).

## 3. Scope

- **One file changes: `dashboard/topic_copy.py`** — only the `_BANNED` list (and a small shared
  anchor constant). `local_claim_flags`, `compliance_scan` (fail-closed), `COMPLIANCE`, the model
  layer, and the `topic_page.approve` gate are **unchanged**.
- No feature flag — the gate is always on, so the change takes effect on deploy. That is the intent.
- Defense-in-depth is preserved and unchanged: anchored local net → AI model judgment
  (`compliance_scan`'s haiku call) → Glen's manual approval. The change only *narrows* the local
  net's false positives; it does not remove the model layer or the human gate.

## 4. Design

### Disease/condition anchor

A shared regex fragment used by the anchored verbs:

```
_DISEASE = (
    r"(disease|illness|condition|disorder|syndrome|infection|ailment|"
    r"cancer|tumou?r|diabetes|arthritis|asthma|eczema|psoriasis|hypertension|"
    r"depression|anxiety|alzheimer'?s?|dementia|parkinson'?s?|autism|adhd|"
    r"ibs|crohn'?s?|colitis|lupus|fibromyalgia|migraine|insomnia|allerg(y|ies)|"
    r"infection|influenza|covid|copd|osteoporosis|neuropathy|gout)"
)
```

- The **category words** (`disease, illness, condition, disorder, syndrome, infection, ailment`)
  catch the generic structure ("treats this condition", "cure any disease").
- The **named conditions** catch the common explicit claims.
- A named condition not in the list (e.g. "treats shingles") passes the *local* net and falls to the
  *model* layer — acceptable: local is the high-confidence deterministic catch, the model is the
  nuance net.

### `_BANNED` after the change

```python
# Anchored claim verbs: flag only when a disease/condition word sits within ~30 chars after the verb.
_CLAIM_VERBS = r"(cure[sd]?|treat(?:s|ed|ing)?|reverse[sd]?|prevent(?:s|ed|ing)?|heal(?:s|ed|ing)?)"

_BANNED = [
    (rf"\b{_CLAIM_VERBS}\b[\w\s,'-]{{0,30}}\b{_DISEASE}\b", "claims to treat/cure/reverse/prevent a disease"),
    (r"\bdiagnos(e|es|is|ing)\b", "claims to diagnose"),
    (r"\bguarantee[sd]?\b", "outcome guarantee"),
]
```

Notes:
- The verb group uses `treat(?:s|ed|ing)?` — it matches `treat/treats/treated/treating` but **not the
  bare noun `treatment`** (no `ment` alternative), so "water treatment" no longer matches. (If the
  verb sits next to a disease word, e.g. "treatment of cancer", the model layer remains the net; the
  local pattern intentionally targets the verb form.)
- The window `[\w\s,'-]{0,30}` allows a short span of words between the verb and the disease word
  ("reverses early-stage heart disease") without spanning unrelated sentences.
- **Direction is verb-then-disease only** (the common claim structure "treats <disease>"). The
  reverse order ("diabetes can be cured") is intentionally NOT matched by the local net and relies
  on the model layer — an accepted gap, since the wellness prompt rarely produces that phrasing and
  the model + human approval remain.
- `local_claim_flags` keeps iterating `_BANNED` exactly as today; only the patterns change. The
  flag `phrase` reported is the matched span (e.g. "treats diabetes").

## 5. Error handling & safety

- No change to fail-closed behavior: `compliance_scan` still returns `passed=False` on any exception,
  and still runs the local denylist first; a local flag still short-circuits before the model call.
- The change cannot make the gate *more* permissive for a real disease claim that matches an anchored
  pattern — those still flag deterministically. Claims that dodge the anchor list still face the model
  layer and human approval.
- Pure functions, no new imports beyond `re` (already imported).

## 6. Testing

Unit tests in `tests/test_topic_copy.py` (no `import app`; test the helper directly):

- **Benign now passes** (`local_claim_flags` returns `[]`): "Our water treatment removes additives.",
  "uses reverse osmosis filtration", "supports the body to help prevent deficiency",
  "a relaxing spa treatment", "a sensible treatment plan".
- **Real claims still flag**: "this protocol treats diabetes", "cures cancer", "reverses heart
  disease", "prevents Alzheimer's", "heals your disease".
- **Generic-anchor flag**: "cure any condition", "treats this disorder".
- **Diagnose / guarantee still flag** (unanchored): "we diagnose the cause", "results are guaranteed".
- **Existing cases preserved**: the existing `test_local_claim_flags_catches_disease_claim`
  ("...cures cancer and treats diabetes...") still flags; `test_local_claim_flags_clean_copy_passes`
  ("people exploring low energy often look into sleep and minerals.") still clean; the
  `compliance_scan` tests still pass (planted claim blocks without the model; clean copy passes;
  fail-closed on client error).

## 7. Rollout

- Merge to `main` (PR; direct-push guarded) → Render deploys → the gate behavior updates immediately.
- Post-deploy: regenerate `fluoride-exposure` (now "Fluoride-Free Living") → it should pass the gate →
  approve it. Spot-check that an existing approved page is unaffected.
- Rollback if needed: `git revert` the commit (no flag to flip; the gate has none).

## 8. Out of scope

- The model-layer prompt and `compliance_scan` flow (unchanged).
- The ingredient-page denylist (separate module `ingredient_copy.py`, different `COMPLIANCE` block;
  not part of this change).
- Expanding the named-condition list beyond a reasonable common set (it can grow later; the model
  layer covers the gaps).
