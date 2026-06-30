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
