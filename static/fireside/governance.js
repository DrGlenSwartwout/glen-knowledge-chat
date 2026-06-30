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
