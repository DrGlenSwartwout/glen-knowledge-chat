// Pure selection helper for the portal's per-element Glendalf backdrop.
// Given the member's deficient element (element_state.setting) and the loaded
// elements-manifest, return the backdrop entry to render — or null to show nothing
// (the graceful default: members with no computed element see the plain portal).
// No DOM here so it stays unit-testable under node like the other fireside modules.

export const ELEMENT_KEYS = ['water', 'wood', 'earth', 'metal', 'fire'];

// Resolve which element the portal should show, honoring an explicit member
// preference over the automatically computed deficient element. `pref` is the
// saved/override choice ('fire', 'water', … or falsy/'auto' for Automatic);
// `autoSetting` is element_state.setting. Returns an element key or null.
export function resolveBackdrop(pref, autoSetting) {
  const p = (typeof pref === 'string') ? pref.trim().toLowerCase() : '';
  if (p && p !== 'auto' && ELEMENT_KEYS.includes(p)) return p;
  return autoSetting || null;
}

export function pickElement(setting, manifest) {
  if (!setting || typeof setting !== 'string') return null;
  const key = setting.trim().toLowerCase();
  if (!ELEMENT_KEYS.includes(key)) return null;
  const els = manifest && manifest.elements;
  const entry = els && els[key];
  if (!entry || typeof entry !== 'object') return null;
  if (!entry.video) return null;
  return { key, ...entry };
}
