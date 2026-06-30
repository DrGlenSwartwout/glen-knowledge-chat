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
