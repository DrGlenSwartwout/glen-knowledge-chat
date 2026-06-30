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
