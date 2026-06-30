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
