# Glendalf Reaction Library & Real-Time Director — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the shipped Glendalf fireside a palette of expressive reaction clips, an ambient soundscape, and a client-side director that selects and crossfades reactions in near-real-time as the traveler types.

**Architecture:** Pure-logic ES modules (manifest normalize, typing→reaction heuristics, clip selection, rate/ambience math) unit-tested with Node's built-in test runner; a DOM director that drives a two-stacked-`<video>` crossfade + an independent ambience scheduler, wired into `static/begin-fireside.html`; the reaction/ambience assets are manifest-driven so they swap in with **no app.py change**. Ships entirely behind the existing `FIRESIDE_ENABLED` flag and degrades to today's single-loop behavior when assets are absent.

**Tech Stack:** Vanilla ES modules (browser `<script type="module">` + Node 22 `node --test`), Flask static serving (unchanged), Kling/HeyGen video + ElevenLabs/Glen-recorded audio for the asset track.

**Spec:** `docs/superpowers/specs/2026-06-30-glendalf-reaction-library-and-director-design.md`

## Global Constraints

- **Ships dark.** Everything lives within `FIRESIDE_ENABLED` (already gates `/begin/fireside`). No new routes; no app.py change in Track A.
- **No app.py / Python change in Track A.** New files go under `static/fireside/`; new manifest keys are additive. Flask serves them automatically.
- **Graceful degradation is mandatory.** If `reactions`/`resting_loops`/`ambience` are absent or any asset 404s, the page must behave exactly as it does today (single listening/speaking/pondering loop). The normalizer guarantees this; it must never throw.
- **Production model = flat tagged clips + crossfade.** No runtime layer compositing. Gaze/hand are selection *tags*, not composited layers.
- **Director model = hybrid.** Client heuristics drive all real-time reactions in v1 (the shipped async Haiku analysis runs post-reply and cannot feed the pondering gap in time — server emotion is Phase B).
- **Canonical vocabularies (use these exact strings everywhere):**
  - Families (13): `attending, affirming, empathic_concern, curiosity, surprise, delight, recognition, pondering, reassurance, gentle_gravity, awe, invitation, lightness`
  - Gaze (6): `up_right, lat_right, down_right, up_left, lat_left, down_left` — **listening uses the right triad; pondering uses the left triad.** Render note: Glendalf's right = viewer's left of frame.
  - Forms: `silent, voice, audible_action`
  - Tiers: `backchannel, hero, ponder, gaze, interjection, interruption`
- **Taste governance (carry the shipped fireside constants):** ≤1 backchannel per 5s while typing; interjections require idle ≥3.5s, ≤3 per session, never turn 1; exactly one hero per submitted turn (pondering gap only); audible actions only at high intensity and never over the traveler's active typing.
- **ES module setup:** source modules are `static/fireside/*.js` made ESM by a scoped `static/fireside/package.json` containing `{"type":"module"}`; Node tests are `tests/fireside_js/*.test.mjs` run with `node --test tests/fireside_js/`.

---

## File Structure

**New (Track A — code):**
- `static/fireside/package.json` — `{"type":"module"}` (scopes ESM to this dir for Node; Flask ignores it).
- `static/fireside/manifest-normalize.js` — `normalizeManifest(raw)` → a manifest with every field guaranteed present + safe defaults. Never throws.
- `static/fireside/heuristics.js` — pure typing→reaction classifiers: `detectGaze`, `detectFamily`, `detectIntensity`, `classifyTyping`.
- `static/fireside/reaction-select.js` — `selectClip(...)` + `FAMILY_AFFINITY` (the §6.4 selection algorithm, injectable RNG).
- `static/fireside/governance.js` — pure rate/ambience math: `canFireBackchannel`, `canInterject`, `nextGapMs`, `shouldDuck`.
- `static/fireside/ambience.js` — `Ambience` DOM class (bed loop + scheduled one-shots + ducking + spark hook).
- `static/fireside/spark.js` — `emberBurst(ctx, x, y, rng)` canvas particle accent.
- `static/fireside/director.js` — `Director` DOM class tying the pure modules to the crossfade `<video>` pair + state machine.
- `tests/fireside_js/manifest-normalize.test.mjs`, `heuristics.test.mjs`, `reaction-select.test.mjs`, `governance.test.mjs` — Node unit tests.

**Modified (Track A):**
- `static/fireside/fireside-manifest.json` — add `resting_loops`, `reactions`, `ambience` (initially empty/placeholder so prod stays safe).
- `static/begin-fireside.html` — convert the single `<video>` to a stacked crossfade pair; replace inline raw manifest access with the normalizer; mount `Director` + `Ambience`; add a mute control + spark canvas.

**Track B — asset production (NOT TDD; human-in-loop, driven separately):** clips + audio + ambient SFX per spec §7/§9, dropped into `static/fireside/video/` and `static/fireside/audio/` and referenced from the manifest.

---

## Track A — Code

### Task 1: ESM scaffold + manifest normalizer

**Files:**
- Create: `static/fireside/package.json`
- Create: `static/fireside/manifest-normalize.js`
- Test: `tests/fireside_js/manifest-normalize.test.mjs`

**Interfaces:**
- Produces: `normalizeManifest(raw: object|null) -> NormManifest` where
  `NormManifest = { intro_video: string|null, intro_poster: string|null, speaking_loop: string|null, pondering_loops: string[], resting_loops: string[], fillers: object[], interjections: object[], reactions: Reaction[], ambience: { bed: string|null, bed_volume: number, oneshots: Oneshot[] } }`.
  `Reaction = { id, family, form, gaze: string|null, hand: string|null, intensity, tier, duration_s: number, loopable: boolean, file: string, audio: string|null }`.
  `Oneshot = { id, file, volume: number, spark: boolean, min_gap_s: number, max_gap_s: number }`.
  Never throws; unknown fields are dropped; bad entries are skipped.

- [ ] **Step 1: Write the scoped package.json**

Create `static/fireside/package.json`:
```json
{ "type": "module" }
```

- [ ] **Step 2: Write the failing test**

Create `tests/fireside_js/manifest-normalize.test.mjs`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeManifest } from '../../static/fireside/manifest-normalize.js';

test('null/garbage input yields safe empty manifest, never throws', () => {
  for (const bad of [null, undefined, 42, 'x', []]) {
    const m = normalizeManifest(bad);
    assert.deepEqual(m.pondering_loops, []);
    assert.deepEqual(m.resting_loops, []);
    assert.deepEqual(m.reactions, []);
    assert.equal(m.ambience.bed, null);
    assert.deepEqual(m.ambience.oneshots, []);
    assert.equal(m.ambience.bed_volume, 0.18);
  }
});

test('resting_loops falls back to pondering_loops then speaking_loop', () => {
  const a = normalizeManifest({ pondering_loops: ['/p1.mp4'], speaking_loop: '/s.mp4' });
  assert.deepEqual(a.resting_loops, ['/p1.mp4']);
  const b = normalizeManifest({ speaking_loop: '/s.mp4' });
  assert.deepEqual(b.resting_loops, ['/s.mp4']);
  const c = normalizeManifest({ resting_loops: ['/r.mp4'], pondering_loops: ['/p.mp4'] });
  assert.deepEqual(c.resting_loops, ['/r.mp4']);
});

test('reactions: bad entries skipped, defaults filled', () => {
  const m = normalizeManifest({ reactions: [
    { id: 'ok', family: 'surprise', form: 'silent', tier: 'backchannel', file: '/r.mp4' },
    { id: 'nofile', family: 'surprise' },         // dropped: no file
    { family: 'surprise', file: '/x.mp4' },        // dropped: no id
    'garbage',
  ]});
  assert.equal(m.reactions.length, 1);
  const r = m.reactions[0];
  assert.equal(r.gaze, null);
  assert.equal(r.hand, null);
  assert.equal(r.intensity, 'med');
  assert.equal(r.loopable, false);
  assert.equal(typeof r.duration_s, 'number');
});

test('ambience oneshots: defaults + bad entries skipped', () => {
  const m = normalizeManifest({ ambience: { bed: '/b.mp3', oneshots: [
    { id: 'pop', file: '/p.mp3', spark: true, min_gap_s: 25, max_gap_s: 70 },
    { id: 'nofile' },          // dropped
  ]}});
  assert.equal(m.ambience.bed, '/b.mp3');
  assert.equal(m.ambience.oneshots.length, 1);
  assert.equal(m.ambience.oneshots[0].volume, 0.2);   // default
  assert.equal(m.ambience.oneshots[0].spark, true);
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `node --test tests/fireside_js/manifest-normalize.test.mjs`
Expected: FAIL — `Cannot find module '.../manifest-normalize.js'`.

- [ ] **Step 4: Write the implementation**

Create `static/fireside/manifest-normalize.js`:
```javascript
// Normalize a fireside manifest into a fully-populated, safe shape.
// Never throws. Missing/garbage input degrades to today's single-loop behavior.

const str = (v) => (typeof v === 'string' && v ? v : null);
const arrOfStr = (v) => (Array.isArray(v) ? v.filter((x) => typeof x === 'string' && x) : []);
const num = (v, d) => (typeof v === 'number' && isFinite(v) ? v : d);
const bool = (v) => v === true;

function normReaction(r) {
  if (!r || typeof r !== 'object') return null;
  const id = str(r.id), file = str(r.file);
  if (!id || !file) return null;
  return {
    id, file,
    family: str(r.family) || 'attending',
    form: str(r.form) || 'silent',
    gaze: str(r.gaze),
    hand: str(r.hand),
    intensity: str(r.intensity) || 'med',
    tier: str(r.tier) || 'backchannel',
    duration_s: num(r.duration_s, 2.5),
    loopable: bool(r.loopable),
    audio: str(r.audio),
  };
}

function normOneshot(o) {
  if (!o || typeof o !== 'object') return null;
  const id = str(o.id), file = str(o.file);
  if (!id || !file) return null;
  return {
    id, file,
    volume: num(o.volume, 0.2),
    spark: bool(o.spark),
    min_gap_s: num(o.min_gap_s, 60),
    max_gap_s: num(o.max_gap_s, 180),
  };
}

export function normalizeManifest(raw) {
  const m = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const pondering = arrOfStr(m.pondering_loops);
  const speaking = str(m.speaking_loop);

  let resting = arrOfStr(m.resting_loops);
  if (!resting.length) resting = pondering.length ? pondering.slice() : (speaking ? [speaking] : []);

  const amb = m.ambience && typeof m.ambience === 'object' ? m.ambience : {};
  return {
    intro_video: str(m.intro_video),
    intro_poster: str(m.intro_poster),
    speaking_loop: speaking,
    pondering_loops: pondering,
    resting_loops: resting,
    fillers: Array.isArray(m.fillers) ? m.fillers.filter((x) => x && typeof x === 'object') : [],
    interjections: Array.isArray(m.interjections) ? m.interjections.filter((x) => x && typeof x === 'object') : [],
    reactions: Array.isArray(m.reactions) ? m.reactions.map(normReaction).filter(Boolean) : [],
    ambience: {
      bed: str(amb.bed),
      bed_volume: num(amb.bed_volume, 0.18),
      oneshots: Array.isArray(amb.oneshots) ? amb.oneshots.map(normOneshot).filter(Boolean) : [],
    },
  };
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `node --test tests/fireside_js/manifest-normalize.test.mjs`
Expected: PASS — 4 tests.

- [ ] **Step 6: Commit**

```bash
git add static/fireside/package.json static/fireside/manifest-normalize.js tests/fireside_js/manifest-normalize.test.mjs
git commit -m "feat(fireside): manifest normalizer with safe defaults + ESM scaffold"
```

---

### Task 2: Typing→reaction heuristics

**Files:**
- Create: `static/fireside/heuristics.js`
- Test: `tests/fireside_js/heuristics.test.mjs`

**Interfaces:**
- Produces:
  - `detectGaze(text: string) -> 'up_right'|'lat_right'|'down_right'|null` (listening = right/construct triad; null when no predicate dominates).
  - `detectFamily(text: string) -> Family` (one of the 13; default `'attending'`).
  - `detectIntensity(text: string) -> 'low'|'med'|'high'`.
  - `classifyTyping(text: string) -> { gaze, family, intensity }`.

- [ ] **Step 1: Write the failing test**

Create `tests/fireside_js/heuristics.test.mjs`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectGaze, detectFamily, detectIntensity, classifyTyping } from '../../static/fireside/heuristics.js';

test('detectGaze maps rep-system predicates to the right triad', () => {
  assert.equal(detectGaze('I can see it clearly, a dark picture'), 'up_right');
  assert.equal(detectGaze('it sounds loud, I hear ringing'), 'lat_right');
  assert.equal(detectGaze('I feel a heavy pain, so tense'), 'down_right');
  assert.equal(detectGaze('the cat sat on the mat'), null);
});

test('detectFamily picks dominant emotional family, defaults attending', () => {
  assert.equal(detectFamily('I am in so much pain and feel alone'), 'empathic_concern');
  assert.equal(detectFamily('wow that is amazing, I love it!'), 'delight');
  assert.equal(detectFamily('what if it is something else? I wonder'), 'curiosity');
  assert.equal(detectFamily('whoa, I never expected that!'), 'surprise');
  assert.equal(detectFamily('and then I went to the store'), 'attending');
});

test('detectIntensity reads punctuation/emphasis', () => {
  assert.equal(detectIntensity('help!'), 'high');
  assert.equal(detectIntensity('I can NEVER do this'), 'high');
  assert.equal(detectIntensity('ok'), 'low');
  assert.equal(detectIntensity('I went for a walk this morning and it was fine'), 'med');
});

test('classifyTyping combines the three', () => {
  const c = classifyTyping('I feel such heavy pain!');
  assert.equal(c.gaze, 'down_right');
  assert.equal(c.family, 'empathic_concern');
  assert.equal(c.intensity, 'high');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/fireside_js/heuristics.test.mjs`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `static/fireside/heuristics.js`:
```javascript
// Pure, dependency-free classifiers turning the traveler's (partial) text into
// reaction cues. Listening = he builds a picture of THEIR world => right triad.

const VISUAL = ['see', 'saw', 'look', 'looks', 'picture', 'clear', 'dark', 'bright', 'focus', 'vision', 'watch', 'imagine', 'appears', 'view', 'colour', 'color'];
const AUDITORY = ['hear', 'heard', 'sound', 'sounds', 'told', 'loud', 'quiet', 'said', 'listen', 'ringing', 'noise', 'voice', 'silence'];
const KINESTHETIC = ['feel', 'felt', 'feeling', 'heavy', 'tense', 'pain', 'gut', 'tight', 'warm', 'cold', 'ache', 'pressure', 'numb', 'exhausted', 'tired', 'stress'];

const FAMILY_WORDS = {
  empathic_concern: ['pain', 'hurt', 'sad', 'scared', 'afraid', 'alone', 'lost', 'struggle', 'hard', 'difficult', 'cry', 'grief', 'worried', 'overwhelmed', 'exhausted', 'tired', 'cannot', "can't", 'suffer'],
  delight: ['love', 'happy', 'joy', 'joyful', 'excited', 'wonderful', 'great', 'laugh', 'funny', 'amazing', 'glad', 'delighted'],
  surprise: ['whoa', 'wow', 'suddenly', 'shocked', 'unexpected', 'surprised', 'cannot believe', "can't believe"],
  curiosity: ['wonder', 'curious', 'what if', 'maybe', 'not sure', 'question', 'why'],
  recognition: ['realize', 'realise', 'makes sense', "that's why", 'i see now', 'connect', 'understand now'],
  awe: ['incredible', 'mystery', 'miracle', 'beautiful', 'vast', 'profound', 'sacred'],
  gentle_gravity: ['diagnosis', 'chronic', 'serious', 'years', 'terminal', 'disease', 'condition'],
};

const words = (t) => String(t || '').toLowerCase().split(/[^a-z']+/).filter(Boolean);
const hits = (text, list) => {
  const low = String(text || '').toLowerCase();
  return list.reduce((n, w) => n + (low.includes(w) ? 1 : 0), 0);
};

export function detectGaze(text) {
  const v = hits(text, VISUAL), a = hits(text, AUDITORY), k = hits(text, KINESTHETIC);
  const max = Math.max(v, a, k);
  if (max === 0) return null;
  if (k === max) return 'down_right';   // feeling wins ties (healer leans somatic)
  if (v === max) return 'up_right';
  return 'lat_right';
}

export function detectFamily(text) {
  let best = 'attending', bestN = 0;
  for (const fam of Object.keys(FAMILY_WORDS)) {
    const n = hits(text, FAMILY_WORDS[fam]);
    if (n > bestN) { bestN = n; best = fam; }
  }
  if (bestN === 0 && /[!?]/.test(String(text || ''))) {
    return /\?/.test(text) ? 'curiosity' : 'surprise';
  }
  return best;
}

export function detectIntensity(text) {
  const t = String(text || '');
  const w = words(t);
  if (/!/.test(t) || /\b[A-Z]{3,}\b/.test(t) || hits(t, ['never', 'always', 'desperate', 'cannot', "can't"])) return 'high';
  if (w.length <= 3) return 'low';
  return 'med';
}

export function classifyTyping(text) {
  return { gaze: detectGaze(text), family: detectFamily(text), intensity: detectIntensity(text) };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/fireside_js/heuristics.test.mjs`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add static/fireside/heuristics.js tests/fireside_js/heuristics.test.mjs
git commit -m "feat(fireside): typing->reaction heuristics (gaze/family/intensity)"
```

---

### Task 3: Clip selection algorithm

**Files:**
- Create: `static/fireside/reaction-select.js`
- Test: `tests/fireside_js/reaction-select.test.mjs`

**Interfaces:**
- Consumes: `Reaction[]` from Task 1's normalized manifest; family/gaze vocab from Global Constraints.
- Produces:
  - `FAMILY_AFFINITY: Record<Family, Family[]>` — nearest-family fallback order.
  - `selectClip(reactions, query, lastId=null, rng=Math.random) -> Reaction|null` where
    `query = { tier, family?, gaze?, form?, intensity? }`. Algorithm: filter to `tier`; if `gaze` requested, prefer matching-gaze clips but drop the constraint if none; filter to `family`, else walk `FAMILY_AFFINITY[family]`, else keep the tier pool; if `form` requested, filter to it (caller excludes `audible_action` while typing); prefer `intensity` match; avoid `lastId` when alternatives exist; pick with `rng`. Returns `null` only when the tier pool is empty.

- [ ] **Step 1: Write the failing test**

Create `tests/fireside_js/reaction-select.test.mjs`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectClip, FAMILY_AFFINITY } from '../../static/fireside/reaction-select.js';

const R = (o) => ({ id: o.id, family: o.family, form: o.form || 'silent', gaze: o.gaze || null,
  hand: null, intensity: o.intensity || 'med', tier: o.tier || 'backchannel',
  duration_s: 2, loopable: false, file: '/' + o.id + '.mp4', audio: null });

const pool = [
  R({ id: 'att', family: 'attending', tier: 'backchannel' }),
  R({ id: 'con', family: 'empathic_concern', tier: 'backchannel' }),
  R({ id: 'grav', family: 'gentle_gravity', tier: 'backchannel' }),
  R({ id: 'hero-con', family: 'empathic_concern', tier: 'hero', intensity: 'high' }),
  R({ id: 'gaze-dr', family: 'attending', tier: 'gaze', gaze: 'down_right' }),
];

test('exact family + tier match', () => {
  const c = selectClip(pool, { tier: 'backchannel', family: 'empathic_concern' });
  assert.equal(c.id, 'con');
});

test('affinity fallback when family absent (empathic_concern -> gentle_gravity)', () => {
  const noCon = pool.filter((x) => x.id !== 'con');
  const c = selectClip(noCon, { tier: 'backchannel', family: 'empathic_concern' });
  assert.equal(c.id, 'grav');
  assert.ok(FAMILY_AFFINITY.empathic_concern.includes('gentle_gravity'));
});

test('gaze preference within tier', () => {
  const c = selectClip(pool, { tier: 'gaze', gaze: 'down_right' });
  assert.equal(c.id, 'gaze-dr');
});

test('returns null on empty tier pool', () => {
  assert.equal(selectClip(pool, { tier: 'interjection' }), null);
});

test('avoids repeating lastId when an alternative exists', () => {
  const two = [R({ id: 'a', family: 'attending' }), R({ id: 'b', family: 'attending' })];
  const rng = () => 0; // deterministic: would pick index 0
  const c = selectClip(two, { tier: 'backchannel', family: 'attending' }, 'a', rng);
  assert.equal(c.id, 'b');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/fireside_js/reaction-select.test.mjs`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `static/fireside/reaction-select.js`:
```javascript
// Pick the single best-matching reaction clip for a moment. Pure; RNG injectable.

export const FAMILY_AFFINITY = {
  attending: ['affirming', 'curiosity'],
  affirming: ['attending', 'recognition'],
  empathic_concern: ['gentle_gravity', 'reassurance'],
  curiosity: ['attending', 'invitation'],
  surprise: ['curiosity', 'awe'],
  delight: ['lightness', 'affirming'],
  recognition: ['affirming', 'pondering'],
  pondering: ['attending', 'recognition'],
  reassurance: ['empathic_concern', 'gentle_gravity'],
  gentle_gravity: ['reassurance', 'empathic_concern'],
  awe: ['curiosity', 'reassurance'],
  invitation: ['attending', 'curiosity'],
  lightness: ['delight', 'affirming'],
};

function chooseFamilyPool(pool, family) {
  if (!family) return pool;
  const exact = pool.filter((c) => c.family === family);
  if (exact.length) return exact;
  for (const alt of FAMILY_AFFINITY[family] || []) {
    const hit = pool.filter((c) => c.family === alt);
    if (hit.length) return hit;
  }
  return pool; // last resort: anything in this tier
}

export function selectClip(reactions, query, lastId = null, rng = Math.random) {
  const q = query || {};
  let pool = (reactions || []).filter((c) => c && c.tier === q.tier);
  if (!pool.length) return null;

  if (q.gaze) {
    const g = pool.filter((c) => c.gaze === q.gaze);
    if (g.length) pool = g;
  }
  pool = chooseFamilyPool(pool, q.family);
  if (q.form) {
    const f = pool.filter((c) => c.form === q.form);
    if (f.length) pool = f;
  }
  if (q.intensity) {
    const i = pool.filter((c) => c.intensity === q.intensity);
    if (i.length) pool = i;
  }
  if (lastId && pool.length > 1) {
    const alt = pool.filter((c) => c.id !== lastId);
    if (alt.length) pool = alt;
  }
  return pool[Math.floor(rng() * pool.length)] || null;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/fireside_js/reaction-select.test.mjs`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add static/fireside/reaction-select.js tests/fireside_js/reaction-select.test.mjs
git commit -m "feat(fireside): clip selection algorithm with family affinity fallback"
```

---

### Task 4: Rate-governance + ambience scheduling math

**Files:**
- Create: `static/fireside/governance.js`
- Test: `tests/fireside_js/governance.test.mjs`

**Interfaces:**
- Produces:
  - `canFireBackchannel(nowMs, lastMs, minGapMs=5000) -> boolean`
  - `canInterject({ idleMs, sessionCount, turn }) -> boolean` (idle ≥3500, count <3, turn >1)
  - `canInterrupt({ seen, turn, idleMs }) -> boolean` (once-per-client hobbit call: not yet seen, turn ≥3, idle ≥2500 — a natural pause a few turns in, never turn 1)
  - `nextGapMs(oneshot, rng=Math.random) -> number` (ms in [min_gap_s, max_gap_s])
  - `shouldDuck(voicePlaying) -> boolean`

- [ ] **Step 1: Write the failing test**

Create `tests/fireside_js/governance.test.mjs`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { canFireBackchannel, canInterject, canInterrupt, nextGapMs, shouldDuck } from '../../static/fireside/governance.js';

test('backchannel respects 5s min gap', () => {
  assert.equal(canFireBackchannel(10000, 4000), true);   // 6s gap
  assert.equal(canFireBackchannel(10000, 6000), false);  // 4s gap
  assert.equal(canFireBackchannel(10000, null), true);   // never fired
});

test('interjection gate: idle>=3.5s, <3 this session, never turn 1', () => {
  assert.equal(canInterject({ idleMs: 4000, sessionCount: 0, turn: 3 }), true);
  assert.equal(canInterject({ idleMs: 2000, sessionCount: 0, turn: 3 }), false);
  assert.equal(canInterject({ idleMs: 4000, sessionCount: 3, turn: 3 }), false);
  assert.equal(canInterject({ idleMs: 4000, sessionCount: 0, turn: 1 }), false);
});

test('interruption gate: not seen, turn>=3, idle>=2.5s', () => {
  assert.equal(canInterrupt({ seen: false, turn: 3, idleMs: 3000 }), true);
  assert.equal(canInterrupt({ seen: true, turn: 3, idleMs: 3000 }), false);
  assert.equal(canInterrupt({ seen: false, turn: 1, idleMs: 3000 }), false);
  assert.equal(canInterrupt({ seen: false, turn: 3, idleMs: 1000 }), false);
});

test('nextGapMs lands within [min,max] seconds', () => {
  const o = { min_gap_s: 25, max_gap_s: 70 };
  assert.equal(nextGapMs(o, () => 0), 25000);
  assert.equal(nextGapMs(o, () => 1), 70000);
  const mid = nextGapMs(o, () => 0.5);
  assert.ok(mid >= 25000 && mid <= 70000);
});

test('shouldDuck while voice plays', () => {
  assert.equal(shouldDuck(true), true);
  assert.equal(shouldDuck(false), false);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/fireside_js/governance.test.mjs`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `static/fireside/governance.js`:
```javascript
// Pure taste/rate governance + ambience scheduling math.

export function canFireBackchannel(nowMs, lastMs, minGapMs = 5000) {
  if (lastMs == null) return true;
  return (nowMs - lastMs) >= minGapMs;
}

export function canInterject({ idleMs, sessionCount, turn }) {
  return idleMs >= 3500 && sessionCount < 3 && turn > 1;
}

export function canInterrupt({ seen, turn, idleMs }) {
  return !seen && turn >= 3 && idleMs >= 2500;
}

export function nextGapMs(oneshot, rng = Math.random) {
  const lo = oneshot.min_gap_s, hi = oneshot.max_gap_s;
  return Math.round((lo + rng() * (hi - lo)) * 1000);
}

export function shouldDuck(voicePlaying) {
  return voicePlaying === true;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/fireside_js/governance.test.mjs`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add static/fireside/governance.js tests/fireside_js/governance.test.mjs
git commit -m "feat(fireside): pure rate-governance + ambience scheduling math"
```

---

### Task 5: Spark ember-burst canvas accent

**Files:**
- Create: `static/fireside/spark.js`
- Test: `tests/fireside_js/spark.test.mjs`

**Interfaces:**
- Produces: `emberBurst(ctx, x, y, rng=Math.random, opts={}) -> () => void`. Spawns N short-lived ember particles drawn on a 2D canvas context starting at (x,y); returns a `cancel()` that stops the animation. Uses `requestAnimationFrame` when present, else a no-op single draw (so Node import + a headless call don't throw). Particle *math* is a pure helper `spawnEmbers(x, y, n, rng)` that is unit-tested.

- [ ] **Step 1: Write the failing test**

Create `tests/fireside_js/spark.test.mjs`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnEmbers } from '../../static/fireside/spark.js';

test('spawnEmbers makes N particles rising from the origin', () => {
  const ps = spawnEmbers(100, 200, 8, () => 0.5);
  assert.equal(ps.length, 8);
  for (const p of ps) {
    assert.equal(p.x, 100);
    assert.equal(p.y, 200);
    assert.ok(p.vy < 0);            // embers rise
    assert.ok(p.life > 0);
  }
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test tests/fireside_js/spark.test.mjs`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `static/fireside/spark.js`:
```javascript
// A brief ember-burst near the hearth, paired with a fire-pop ambient one-shot.
// Pure particle spawn is unit-tested; the rAF animation is best-effort/DOM-only.

export function spawnEmbers(x, y, n, rng = Math.random) {
  const ps = [];
  for (let i = 0; i < n; i++) {
    ps.push({
      x, y,
      vx: (rng() - 0.5) * 0.6,
      vy: -(0.4 + rng() * 0.8),     // upward
      life: 1.0,
      decay: 0.02 + rng() * 0.03,
      r: 1 + rng() * 1.5,
    });
  }
  return ps;
}

export function emberBurst(ctx, x, y, rng = Math.random, opts = {}) {
  const n = opts.count || 10;
  let ps = spawnEmbers(x, y, n, rng);
  let raf = null, stopped = false;
  const hasRAF = typeof requestAnimationFrame === 'function';

  function frame() {
    if (stopped || !ctx) return;
    ctx.clearRect(x - 60, y - 120, 120, 140);
    ps = ps.filter((p) => p.life > 0);
    for (const p of ps) {
      p.x += p.vx; p.y += p.vy; p.vy += 0.01; p.life -= p.decay;
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.fillStyle = 'rgba(255,170,60,1)';
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (ps.length && hasRAF) raf = requestAnimationFrame(frame);
  }
  if (hasRAF) raf = requestAnimationFrame(frame); else frame();

  return function cancel() { stopped = true; if (raf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(raf); };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --test tests/fireside_js/spark.test.mjs`
Expected: PASS — 1 test.

- [ ] **Step 5: Commit**

```bash
git add static/fireside/spark.js tests/fireside_js/spark.test.mjs
git commit -m "feat(fireside): ember-burst spark accent (pure spawn + rAF draw)"
```

---

### Task 6: Ambience DOM scheduler

**Files:**
- Create: `static/fireside/ambience.js`
- (Render-verified in Task 9; no Node unit test — this is DOM/audio wiring built on Task 4's tested math.)

**Interfaces:**
- Consumes: `governance.nextGapMs`, `governance.shouldDuck`; `spark.emberBurst`; normalized `manifest.ambience`.
- Produces: `class Ambience` with constructor `(ambience, { isVoicePlaying, sparkCtx, sparkXY, muted })` and methods `start()`, `stop()`, `setMuted(bool)`. `isVoicePlaying()` is a caller-supplied predicate (reads the TTS audio element). On each one-shot tick it consults `shouldDuck(isVoicePlaying())` and skips (reschedules) if ducking; otherwise plays at `volume` and, if `spark`, calls `emberBurst`.

- [ ] **Step 1: Implement the class**

Create `static/fireside/ambience.js`:
```javascript
import { nextGapMs, shouldDuck } from './governance.js';
import { emberBurst } from './spark.js';

export class Ambience {
  constructor(ambience, opts = {}) {
    this.amb = ambience || { bed: null, bed_volume: 0.18, oneshots: [] };
    this.isVoicePlaying = opts.isVoicePlaying || (() => false);
    this.sparkCtx = opts.sparkCtx || null;
    this.sparkXY = opts.sparkXY || [0, 0];
    this.muted = !!opts.muted;
    this.bedEl = null;
    this.timers = [];
    this.cancelSpark = null;
  }

  start() {
    if (this.amb.bed) {
      this.bedEl = new Audio(this.amb.bed);
      this.bedEl.loop = true;
      this.bedEl.volume = this.muted ? 0 : this.amb.bed_volume;
      this.bedEl.play().catch(() => {});
    }
    for (const o of this.amb.oneshots) this._schedule(o);
  }

  _schedule(o) {
    const t = setTimeout(() => {
      if (!this.muted && !shouldDuck(this.isVoicePlaying())) this._play(o);
      this._schedule(o); // always reschedule, ducked or not
    }, nextGapMs(o));
    this.timers.push(t);
  }

  _play(o) {
    const a = new Audio(o.file);
    a.volume = o.volume;
    a.play().catch(() => {});
    if (o.spark && this.sparkCtx) {
      if (this.cancelSpark) this.cancelSpark();
      this.cancelSpark = emberBurst(this.sparkCtx, this.sparkXY[0], this.sparkXY[1]);
    }
  }

  setMuted(m) {
    this.muted = !!m;
    if (this.bedEl) this.bedEl.volume = this.muted ? 0 : this.amb.bed_volume;
  }

  stop() {
    this.timers.forEach(clearTimeout); this.timers = [];
    if (this.bedEl) { this.bedEl.pause(); this.bedEl = null; }
    if (this.cancelSpark) { this.cancelSpark(); this.cancelSpark = null; }
  }
}
```

- [ ] **Step 2: Syntax-check the module under Node**

Run: `node --input-type=module -e "import('./static/fireside/ambience.js').then(()=>console.log('OK'))"`
Expected: prints `OK` (imports resolve; `Audio`/`requestAnimationFrame` are only referenced inside methods, not at import time).

- [ ] **Step 3: Commit**

```bash
git add static/fireside/ambience.js
git commit -m "feat(fireside): ambience DOM scheduler (bed loop + ducked one-shots + spark)"
```

---

### Task 7: Director DOM class (crossfade state machine)

**Files:**
- Create: `static/fireside/director.js`
- (Render-verified in Task 9.)

**Interfaces:**
- Consumes: `normalizeManifest`, `classifyTyping`, `selectClip`, `canFireBackchannel`, `canInterject`.
- Produces: `class Director` with constructor `(normManifest, { videoA, videoB, onInterjectionAudio })` and methods:
  - `toResting()` — crossfade to a random `resting_loops` clip, looping.
  - `onType(text, ctx)` — `ctx = { nowMs, turn }`; rate-gated; classifies text, selects a `backchannel`/`gaze` clip (never `audible_action` while typing unless intensity `high`), crossfades in, then back to resting on end.
  - `maybeInterject(ctx)` — `ctx = { idleMs, sessionCount, turn }`; if `canInterject`, picks an `interjection` clip + fires `onInterjectionAudio(clip)`.
  - `onSubmit(text)` — picks a `hero` clip by `detectFamily(text)`, crossfades in, then to a `ponder` clip until `onReplyReady()`.
  - `onReplyReady()` / `toSpeaking()` — return to the existing speaking loop (Phase-B will make this emotion-matched).
  - Crossfade: two stacked `<video>` elements; the incoming one's `.src` is set, then CSS opacity swaps over ~250ms; the outgoing pauses after the transition. Always returns to resting; never throws on a missing clip (selectClip → null ⇒ stay in current state).

- [ ] **Step 1: Implement the class**

Create `static/fireside/director.js`:
```javascript
import { classifyTyping, detectFamily, detectIntensity } from './heuristics.js';
import { selectClip } from './reaction-select.js';
import { canFireBackchannel, canInterject, canInterrupt } from './governance.js';

const FADE_MS = 250;

export class Director {
  constructor(manifest, opts = {}) {
    this.m = manifest;
    this.a = opts.videoA; this.b = opts.videoB;       // two stacked <video>
    this.onInterjectionAudio = opts.onInterjectionAudio || (() => {});
    this.front = this.a; this.back = this.b;
    this.lastReactionId = null;
    this.lastBackchannelMs = null;
    this.state = 'idle';
  }

  _crossfadeTo(file, { loop = false } = {}) {
    if (!file) return;
    const inc = this.back;
    inc.src = file; inc.loop = loop; inc.currentTime = 0;
    const p = inc.play(); if (p && p.catch) p.catch(() => {});
    inc.style.transition = `opacity ${FADE_MS}ms`;
    inc.style.opacity = '1';
    this.front.style.opacity = '0';
    const out = this.front;
    setTimeout(() => { try { if (!out.loop) out.pause(); } catch (e) {} }, FADE_MS + 30);
    this.front = inc; this.back = out;
  }

  toResting() {
    this.state = 'resting';
    const pool = this.m.resting_loops;
    if (!pool.length) return;
    const file = pool[Math.floor(Math.random() * pool.length)];
    this._crossfadeTo(file, { loop: true });
  }

  _playReactionThenRest(clip) {
    if (!clip) return;
    this.lastReactionId = clip.id;
    this._crossfadeTo(clip.file, { loop: false });
    const back = () => this.toResting();
    this.front.onended = back;
    setTimeout(back, Math.max(1200, (clip.duration_s || 2.5) * 1000 + 200)); // safety
  }

  onType(text, ctx) {
    const now = ctx.nowMs;
    if (!canFireBackchannel(now, this.lastBackchannelMs)) return;
    const c = classifyTyping(text);
    const wantAudible = c.intensity === 'high';
    let clip = c.gaze
      ? selectClip(this.m.reactions, { tier: 'gaze', gaze: c.gaze }, this.lastReactionId)
      : null;
    if (!clip) {
      clip = selectClip(this.m.reactions,
        { tier: 'backchannel', family: c.family, intensity: c.intensity,
          form: wantAudible ? undefined : 'silent' },
        this.lastReactionId);
    }
    if (clip) { this.lastBackchannelMs = now; this._playReactionThenRest(clip); }
  }

  maybeInterject(ctx) {
    if (!canInterject(ctx)) return null;
    const clip = selectClip(this.m.reactions, { tier: 'interjection' }, this.lastReactionId);
    if (clip) { this.lastReactionId = clip.id; this._playReactionThenRest(clip); this.onInterjectionAudio(clip); }
    return clip;
  }

  // Once per client: an off-screen hobbit voice calls "Glendalf"; he waves them off.
  // `ctx = { turn, idleMs }`; the seen-flag is persisted by the caller (localStorage
  // 'fireside_interruption_seen'). Returns the clip played, or null.
  maybeInterruption(ctx) {
    const seen = this._interruptionSeen ?? false;
    if (!canInterrupt({ seen, turn: ctx.turn, idleMs: ctx.idleMs })) return null;
    const clip = selectClip(this.m.reactions, { tier: 'interruption' }, null);
    if (!clip) return null;
    this._interruptionSeen = true;
    this.lastReactionId = clip.id;
    this._playReactionThenRest(clip);     // clip carries its own off-screen + reply audio
    this.onInterjectionAudio(clip);
    return clip;
  }

  onSubmit(text) {
    const fam = detectFamily(text), intensity = detectIntensity(text);
    const hero = selectClip(this.m.reactions, { tier: 'hero', family: fam, intensity }, this.lastReactionId);
    if (hero) { this.lastReactionId = hero.id; this._crossfadeTo(hero.file, { loop: false });
      this.front.onended = () => this._ponder(); }
    else this._ponder();
  }

  _ponder() {
    const clip = selectClip(this.m.reactions, { tier: 'ponder' }, this.lastReactionId);
    if (clip) this._crossfadeTo(clip.file, { loop: true });
    else if (this.m.pondering_loops.length)
      this._crossfadeTo(this.m.pondering_loops[0], { loop: true });
  }
}
```

- [ ] **Step 2: Syntax-check the module under Node**

Run: `node --input-type=module -e "import('./static/fireside/director.js').then(()=>console.log('OK'))"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add static/fireside/director.js
git commit -m "feat(fireside): Director DOM class with two-video crossfade state machine"
```

---

### Task 8: Manifest data + page integration

**Files:**
- Modify: `static/fireside/fireside-manifest.json` (add empty/placeholder `resting_loops`, `reactions`, `ambience`)
- Modify: `static/begin-fireside.html` (stacked videos, normalizer, mount Director + Ambience, mute control, spark canvas)

**Interfaces:**
- Consumes: all Track A modules. Loads them via `<script type="module">`.

- [ ] **Step 1: Extend the real manifest (safe placeholders so prod stays unchanged)**

Add these keys to `static/fireside/fireside-manifest.json` (siblings of the existing keys). Empty `reactions` + null `ambience.bed` means the page behaves exactly as today until real assets land:
```json
  "resting_loops": [],
  "reactions": [],
  "ambience": { "bed": null, "bed_volume": 0.18, "oneshots": [] }
```

- [ ] **Step 2: Convert the single video to a stacked crossfade pair**

In `static/begin-fireside.html`, replace the single `<video id="stage" …>` (line ~51) with two layered videos in a positioned wrapper, and add a hearth spark canvas + a mute toggle. Match existing styling/dimensions:
```html
<div id="stage-wrap" style="position:relative;width:100%;height:100%;">
  <video id="stageA" playsinline muted loop style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:1;"></video>
  <video id="stageB" playsinline muted loop style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;"></video>
  <canvas id="sparks" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"></canvas>
  <button id="muteBtn" aria-label="Toggle ambient sound" style="position:absolute;right:12px;bottom:12px;">🔊</button>
</div>
```
Keep any code that referenced `stage` working by pointing the intro/speaking helpers at `stageA` (the initially-visible element). The intro still plays on `stageA`; the Director takes over the pair after intro ends.

- [ ] **Step 3: Load the modules and mount the director**

Convert the page's inline `<script>` (or add a new `<script type="module">` after it) so it imports and wires the modules. Replace direct `manifest.*` reads with the normalized object. Skeleton to integrate with the existing flow (keep the existing fetch + intro logic; swap raw access for `norm`):
```html
<script type="module">
  import { normalizeManifest } from '/static/fireside/manifest-normalize.js';
  import { Director } from '/static/fireside/director.js';
  import { Ambience } from '/static/fireside/ambience.js';

  const stageA = document.getElementById('stageA');
  const stageB = document.getElementById('stageB');
  const canvas = document.getElementById('sparks');
  const ctx = canvas.getContext('2d');

  let director = null, ambience = null;

  window.__firesideInit = function (rawManifest, ttsAudioEl) {
    const norm = normalizeManifest(rawManifest);
    director = new Director(norm, {
      videoA: stageA, videoB: stageB,
      onInterjectionAudio: (clip) => { /* reuse existing interjection audio playback */ },
    });
    ambience = new Ambience(norm.ambience, {
      isVoicePlaying: () => ttsAudioEl && !ttsAudioEl.paused && !ttsAudioEl.ended,
      sparkCtx: ctx, sparkXY: [canvas.width * 0.22, canvas.height * 0.62],
    });
    document.getElementById('muteBtn').addEventListener('click', () => {
      const on = director._muted = !director._muted;
      ambience.setMuted(on);
      document.getElementById('muteBtn').textContent = on ? '🔇' : '🔊';
    });
    return { director, ambience };
  };
</script>
```
Then in the existing flow: after the manifest fetch resolves and the intro ends, call `window.__firesideInit(manifest, <ttsAudioElement>)`, start `ambience.start()` on the first user gesture (the existing "begin"/submit handler — autoplay-policy safe), call `director.toResting()` when entering the listening state, `director.onType(inputValue, {nowMs: Date.now(), turn})` on the composer's `input` event (debounced ~400ms), `director.maybeInterject({idleMs, sessionCount, turn})` on the idle timer, and `director.onSubmit(text)` when the traveler submits.

**Once-per-client interruption wiring:** seed the director's seen-flag from storage before first use — `director._interruptionSeen = localStorage.getItem('fireside_interruption_seen') === '1';` — and on the idle timer (alongside `maybeInterject`) call `if (director.maybeInterruption({turn, idleMs})) localStorage.setItem('fireside_interruption_seen', '1');`. This persists "shown once" across sessions for that browser.

- [ ] **Step 4: Verify the page still parses and serves**

Run (foreground stub server, then curl):
```bash
cd /tmp/wt-deploy-chat-e5fec1df
FIRESIDE_ENABLED=true DATA_DIR=/tmp/fireside_verify_data python3 -c "import app" && echo "app imports OK"
```
Expected: `app imports OK` (no Python change, just confirms nothing broke). Manual: open the page in Task 9.

- [ ] **Step 5: Commit**

```bash
git add static/fireside/fireside-manifest.json static/begin-fireside.html
git commit -m "feat(fireside): wire reaction director + ambience into the page (stacked crossfade, mute, sparks)"
```

---

### Task 9: Headless render-verify (degradation + dance)

**Files:**
- Create: `tests/fireside_render_verify.md` (a documented manual/headless checklist + the local-serve recipe) — there is no committed browser harness in the repo, so this task is a controller-run verification, recorded as a checklist.

**Interfaces:** Consumes the running app + the integrated page.

> **Note on video in headless Chrome:** automated Chrome cannot reliably *decode* video frames (readyState may stay 0). Render-verify therefore asserts **DOM state + console-clean**, not pixel playback: which `<video>.src` is set, opacity crossfade classes, ambience audio elements created, no uncaught errors. Actual visual quality is reviewed by Glen against the real clips (Track B).

- [ ] **Step 1: Write the local stub server (reuse the fireside pattern)**

Create `/tmp/fireside_rv_server.py` (mirrors the existing `/tmp/fireside_verify_server.py` used for the original fireside): sets `FIRESIDE_ENABLED=true`, overrides `DATA_DIR` to a tmp dir, stubs `_cl.messages.stream` to yield two short deltas, stubs `_el_tts` to return a 0.4s silent mp3, then `app.run(port=5099)`.

- [ ] **Step 2: Verify graceful degradation (empty new fields)**

With the shipped manifest (empty `reactions`/`resting_loops`, null `ambience.bed`): load `http://127.0.0.1:5099/begin/fireside` in headless Chrome. Assert: page loads, intro→listening works exactly as today, **zero console errors**, no `Director`/`Ambience` exceptions (they no-op on empty pools), `stageA` is the visible element.

- [ ] **Step 3: Verify the full dance (test manifest with fixture clips)**

Swap in a test manifest pointing `resting_loops`/`reactions`/`ambience` at the existing placeholder mp4s/mp3s (reuse `pondering-1.mp4` etc. as stand-ins), including one `tier: "interruption"` fixture clip. Drive the composer: type a kinesthetic-emotional string ("I feel such heavy pain"), assert a reaction `<video>.src` changes then returns to a resting src; submit, assert a hero/ponder src during the gap; assert an ambience `Audio` was constructed for the bed. Call `director.maybeInterruption({turn:3, idleMs:3000})` twice — assert it plays a clip the **first** time and returns `null` the **second** (once-per-client), and that `localStorage['fireside_interruption_seen']` is `'1'`. **Zero console errors throughout.** No chrome ribbon (page already excluded from the journey shell).

- [ ] **Step 4: Record results + commit the checklist**

```bash
git add tests/fireside_render_verify.md
git commit -m "test(fireside): headless render-verify checklist (degradation + dance)"
```

---

## Track B — Asset Production (NOT TDD; human-in-loop)

> Driven interactively (Replicate/ElevenLabs generation scripts + Glen's taste review), **not** by subagent-TDD. Each clip drops into `static/fireside/video/` or `/audio/` and is added to the manifest `reactions`/`ambience` arrays (Task 1's schema). No code change. Render in the spec §7 waves so the loop comes alive early. Audio sourcing per spec §9 (Glen records audible actions + non-verbal vocalizations; ElevenLabs-with-approval for borderline-verbal backchannels; ambient SFX from library or Glen).

- [ ] **Wave 1.0 — make him present (~13):** 2 resting loops; 5 silent backchannels (attending nod, curiosity eyebrow, empathic-concern, recognition, surprise); 5 hero reactions (empathic_concern, delight, surprise, recognition, awe); 1 extra pondering loop. All from `n11-N1.jpg` start image. Add to manifest, render-verify the dance with real clips, Glen taste-review as inline GIFs.
- [ ] **Wave 1.1 — voice & hands (~12):** 6 vocalized backchannels (mm-hmm, oh?, ahh, oh!, chuckle, hmm); 3 audible actions (snap, table-tap, clap); 3 pondering beats (reach-for-book, sip, page-turn). Tag `form: voice`/`audible_action`.
- [ ] **Wave 1.2 — eyes (~6):** gaze glances, right triad first (`up_right`, `lat_right`, `down_right`), then left triad. Tag `tier: gaze`.
- [ ] **Wave 1.3 — interjection video beats (~5):** match the shipped interjection audio. Tag `tier: interjection`.
- [ ] **Wave 1.4 — once-per-client interruptions (~4–6):** the off-screen hobbit call (spec §3.4b) — Glendalf glances to his side, waves them off, replies; each clip carries its off-screen-voice line (Glen-recorded hobbit voices, or ElevenLabs villager) + his reply. Tag `tier: interruption`. Director plays at most one per client via the persistent seen-flag.
- [ ] **Ambient SFX:** fire bed (seamless loop), fire pop (+spark), dog, birds, wind, creak → `static/fireside/audio/amb-*.mp3`; fill the manifest `ambience` block (spec §5 values).
- [ ] **Final go-live:** with `FIRESIDE_ENABLED` already on in prod, the real clips are live the moment the manifest references them. Render-verify on the real site; Glen confirms continuity + zero console errors.

---

## Notes / Decisions for the Implementer

- **Spec §6.3 refinement (important):** the shipped `_fireside_coverage_async` Haiku analysis runs *after* the reply in a background thread, so it cannot supply a dominant-emotion label in time for the pondering-gap hero reaction. **v1 drives the hero reaction from the client heuristic `detectFamily(text)`** (Task 7 `onSubmit`). A server-side emotion label feeding an emotion-matched **speaking** loop is **Phase B** (separate spec) — do not add it here.
- **Crossfade choice (resolves spec §13 Q2):** two stacked `<video>` elements with a CSS opacity transition (Task 6/7), chosen over single-element `.src` swap to avoid the black flash on the immersive page.
- **Resting-loop seam (spec §13 Q3):** the ~250ms opacity dissolve hides the loop-point seam, so resting variants do **not** need frame-matched ends; if a visible seam appears in Glen's review, address it in Track B by trimming to matched frames.
- **Debounce typing** at ~400ms before calling `onType` so heuristics run on settled partial text, not every keystroke.

---

## Self-Review

**Spec coverage:** Palette channels 1–4 → manifest schema (Task 1) + heuristics/selector (Tasks 2–3) + Director (Task 7); Channel 5 ambience → governance math (Task 4) + Ambience (Task 6) + spark (Task 5); production model A → flat tagged clips + crossfade (Tasks 1, 7); director model A → Tasks 2–4, 7; graceful degradation → Task 1 + Task 9 Step 2; integration/no-route-change → Task 8; voice-input/Phase-B → explicitly out of scope (Notes); asset waves §7 + audio §9 → Track B. Covered.

**Placeholder scan:** No "TBD/TODO/handle edge cases" — every code step has complete code; the one doc-deliverable (Task 9 checklist) is itself the deliverable. Track B is intentionally a production checklist, flagged as non-TDD.

**Type consistency:** `normalizeManifest` shape (Task 1) is consumed verbatim by Director/Ambience (Tasks 6–8); `selectClip(reactions, query, lastId, rng)` signature identical across Tasks 3 and 7; family/gaze/tier/form strings match the Global Constraints vocab throughout; `nextGapMs`/`shouldDuck`/`canFireBackchannel`/`canInterject` signatures identical across Tasks 4, 6, 7.
