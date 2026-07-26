/* Shared Chinese organ-clock timing helpers for Body Maps and Journal. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.OrganClock = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const WINDOWS = [
    { code: "GB", meridian: "Gall Bladder", element: "Wood", startHour: 23, endHour: 1, label: "11 PM–1 AM", interpretation: "Sleep, cell repair, cleanse brain." },
    { code: "LR", meridian: "Liver", element: "Wood", startHour: 1, endHour: 3, label: "1–3 AM", interpretation: "Deep sleep, detox, recover, plan." },
    { code: "LU", meridian: "Lung", element: "Metal", startHour: 3, endHour: 5, label: "3–5 AM", interpretation: "Deep sleep and memory." },
    { code: "LI", meridian: "Large Intestine", element: "Metal", startHour: 5, endHour: 7, label: "5–7 AM", interpretation: "Wake up to natural light." },
    { code: "ST", meridian: "Stomach", element: "Earth", startHour: 7, endHour: 9, label: "7–9 AM", interpretation: "Breakfast and concentration." },
    { code: "SP", meridian: "Spleen/Pancreas", element: "Earth", startHour: 9, endHour: 11, label: "9–11 AM", interpretation: "Focus and thinking clearly." },
    { code: "HT", meridian: "Heart", element: "Fire", startHour: 11, endHour: 13, label: "11 AM–1 PM", interpretation: "Relate to meaning, lunch." },
    { code: "SI", meridian: "Small Intestine", element: "Fire", startHour: 13, endHour: 15, label: "1–3 PM", interpretation: "Absorb food, take a nap." },
    { code: "BL", meridian: "Bladder", element: "Water", startHour: 15, endHour: 17, label: "3–5 PM", interpretation: "Restore energy, work and study." },
    { code: "KI", meridian: "Kidney", element: "Water", startHour: 17, endHour: 19, label: "5–7 PM", interpretation: "Store nutrients, relax mind." },
    { code: "PC", meridian: "Circulation", classicalMeridian: "Pericardium", element: "Fire", startHour: 19, endHour: 21, label: "7–9 PM", interpretation: "Turn down blue light, reading." },
    { code: "TE", meridian: "Endocrine", classicalMeridian: "Triple Burner", element: "Fire", startHour: 21, endHour: 23, label: "9–11 PM", interpretation: "Metabolic balancing, enjoy life." },
  ];

  function normalizeHour(hour) {
    return ((Number(hour) % 24) + 24) % 24;
  }

  function windowForHour(hour) {
    const h = normalizeHour(hour);
    return WINDOWS.find(w => w.startHour < w.endHour
      ? h >= w.startHour && h < w.endHour
      : h >= w.startHour || h < w.endHour);
  }

  function angleForHour(hour) {
    return normalizeHour(Number(hour) - 3) * 15;
  }

  function hourFromParts(rawHour, meridiem) {
    let hour = Number(rawHour);
    const period = String(meridiem || "").toLowerCase().replace(/\./g, "");
    if (period === "am") return hour === 12 ? 0 : hour;
    if (period === "pm") return hour === 12 ? 12 : hour + 12;
    return hour;
  }

  function rangeWindowCodes(startHour, endHour) {
    const start = normalizeHour(startHour);
    let span = normalizeHour(endHour - startHour);
    if (span === 0) span = 2;
    return WINDOWS.filter(w => {
      for (let offset = 0; offset < span; offset += 0.25) {
        if (windowForHour(start + offset).code === w.code) return true;
      }
      return false;
    }).map(w => w.code);
  }

  function extractTimeMentions(text) {
    const input = String(text || "");
    const found = [];
    const occupied = [];
    const overlaps = (start, end) => occupied.some(r => start < r.end && end > r.start);
    const add = (match, startHour, endHour) => {
      if (overlaps(match.index, match.index + match[0].length)) return;
      occupied.push({ start: match.index, end: match.index + match[0].length });
      const codes = rangeWindowCodes(startHour, endHour);
      found.push({ text: match[0], startHour: normalizeHour(startHour), endHour: normalizeHour(endHour), codes });
    };

    // Explicit ranges: “3–5 PM”, “from 3:30 to 5 am”.
    const rangeRe = /\b(?:from\s+)?(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(?:-|–|—|to)\s*(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b/gi;
    for (const m of input.matchAll(rangeRe)) {
      const endHour = hourFromParts(m[3], m[5]) + Number(m[4] || 0) / 60;
      let startHour = hourFromParts(m[1], m[5]) + Number(m[2] || 0) / 60;
      if (endHour < startHour && endHour + 12 > startHour) startHour -= 12;
      add(m, startHour, endHour);
    }

    // Explicit clock times with AM/PM.
    const timeRe = /\b(?:at|around|about|by|before|after|since)?\s*(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b/gi;
    for (const m of input.matchAll(timeRe)) {
      const hour = hourFromParts(m[1], m[3]) + Number(m[2] || 0) / 60;
      add(m, hour, hour + 0.01);
    }

    const named = [
      { re: /\bmidnight\b/gi, start: 0, end: 0.01 },
      { re: /\bnoon\b/gi, start: 12, end: 12.01 },
      { re: /\bearly morning\b/gi, start: 5, end: 9 },
      { re: /\bmorning\b/gi, start: 5, end: 12 },
      { re: /\bafternoon\b/gi, start: 12, end: 17 },
      { re: /\bevening\b/gi, start: 17, end: 21 },
      { re: /\b(?:late at night|overnight)\b/gi, start: 23, end: 3 },
      { re: /\b(?:at night|tonight)\b/gi, start: 21, end: 23 },
    ];
    named.forEach(spec => {
      for (const m of input.matchAll(spec.re)) add(m, spec.start, spec.end);
    });

    return found.sort((a, b) => input.indexOf(a.text) - input.indexOf(b.text));
  }

  return { WINDOWS, normalizeHour, windowForHour, angleForHour, rangeWindowCodes, extractTimeMentions };
});
