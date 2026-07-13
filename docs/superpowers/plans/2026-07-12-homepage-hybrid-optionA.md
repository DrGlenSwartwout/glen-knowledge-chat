# Homepage Hybrid (Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a visual re-skin, so most tasks verify by rendering the page in a headless browser, not by unit tests.

**Goal:** Re-skin the illtowell.com homepage hero for clarity and brand, keeping the live chat concierge intact, and add a new always-visible Scan / Find / Heal explainer built from living waveform animations.

**Architecture:** All changes are in the single static file `static/begin.html` (served as-is at `/`, no templating). The page already ships the committed deep-green palette; this is a re-skin plus one new section, not a repaint. The chat concierge is a hard contract that must not break.

**Tech Stack:** Static HTML, one inline `<style>` block (lines 11-632), vanilla JS, Canvas 2D for the waveform motifs. No build step, no framework, no external assets.

## Global Constraints

- **Do not break the chat concierge.** The `heroChat()` IIFE returns early and silently dies if any of these IDs are missing: `#hero-chat`, `#hero-messages`, `#hero-activate`, `#hero-input`, `#hero-send`. Keep all five, with those exact ids, inside `.hero`.
- Keep the personalization spans `[data-name-prefix]` and `[data-name]` in the hero headline, and `[data-fi-sub]` in the fireside invite. JS fills them from `/begin/state`.
- Keep the `data-layer`, `locked`, and `reveal` classes on gated sections untouched. New elements use the existing `.reveal` IntersectionObserver system (class `reveal`, becomes `reveal in`).
- Preserve the `unlock('video')` funnel signal: the current `.video` element's click fires it (app-script). If the video block is replaced, re-wire that click (or consciously drop it and note it).
- Palette tokens already exist in `:root` (line 12): `--bg #0a150d`, `--surface #111f16`, `--border #21472d`, `--cream #fdf4d8`, `--muted #a89870`, `--gold #d4a843`, `--green #3d8a52`. Use them; do not introduce new hex values except where the mockup adds a darker inset (`#0c1e14`).
- Copy rules (Glen's voice): no em dashes, no ALL CAPS, no emojis (SVG/canvas only), lead with validation. Serif display face is a system stack (no webfont): `"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif`.
- Respect `prefers-reduced-motion`: every canvas animation renders a static representative frame when reduced.
- Reference mockup (approved): artifact `optionA-healing-oasis-link`. Match its hero copy, explainer copy, and the three waveform behaviors.

## File Structure

Everything is in `static/begin.html`:
- The `<style>` block (11-632): add the serif face var, the explainer section styles, and the hero-proof restyle.
- The hero markup (713-743): re-skinned head + grid.
- A new `<section class="how">` inserted between `.hero` (ends 743) and `#journey-strip` (748).
- The inline app script (813-1346): add one self-contained waveform IIFE near the end; do not entangle it with `heroChat()`.

---

### Task 1: Serif display face + hero headline/subhead re-skin

**Files:** Modify `static/begin.html` (style block near the `:root`; hero-head at 714-717)

- [ ] **Step 1: Add the serif stack as a token.** In `:root` (after line 13's `--radius`), add:

```css
    --serif: "Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
```

- [ ] **Step 2: Restyle `.hero-head`.** Replace the `.hero-head h1`/`p` rules (577-578) with the mockup's treatment:

```css
  .hero-head { text-align: center; max-width: 760px; margin: 0 auto 22px; }
  .hero-head .eyebrow-k { font-size: 12px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--gold); font-weight: 700; }
  .hero-head h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(30px, 4.2vw, 46px); line-height: 1.06; letter-spacing: -0.01em; margin: 10px 0 8px; text-wrap: balance; }
  .hero-head h1 em { font-style: italic; color: var(--gold); }
  .hero-head p { font-size: 16px; color: var(--muted); line-height: 1.55; margin: 0 auto; max-width: 52ch; }
```

- [ ] **Step 3: Rewrite the hero-head markup** (714-717), keeping the `[data-name]` spans so personalization still works:

```html
    <div class="hero-head">
      <div class="eyebrow-k">Bioenergetic voice analysis</div>
      <h1><span data-name-prefix></span><span data-name></span>Your body is always speaking. <em>Start by telling me what&rsquo;s on your mind.</em></h1>
      <p>A short, private conversation reads your body&rsquo;s signal and points you toward the remedies it is actually asking for. No guesswork, no generic supplements.</p>
    </div>
```

(The `[data-name-prefix]`/`[data-name]` spans stay first and empty; `personalize()` fills them for returning visitors. `[data-name-prefix]` is `display:none` until a name exists, per rule at line 590.)

- [ ] **Step 4: Render-verify.** Serve the app locally and load `/`. Confirm: the headline is serif, reads clearly, the eyebrow shows, and the concierge still greets (chat not broken). No console errors.

- [ ] **Step 5: Commit.**

```bash
git add static/begin.html
git commit -m "feat(home): serif hero headline + clarified copy"
```

---

### Task 2: Re-skin the hero grid (proof/signature left, chat right)

**Files:** Modify `static/begin.html` (hero-grid 718-742; style block)

**Interfaces:** Must preserve `#hero-chat`, `#hero-messages`, `#hero-activate`, `#hero-input`, `#hero-send`.

- [ ] **Step 1: Replace the `.video-wrap` block (719-733)** with a proof card carrying the hero voice-signature canvas plus the USA Today proof line (reuses the existing lower-third copy). Keep an interactive hook so `unlock('video')` still fires:

```html
      <div class="proof reveal">
        <div class="sig" role="button" tabindex="0" aria-label="Your body&rsquo;s voice signature">
          <canvas id="hero-signature" width="480" height="120" aria-hidden="true"></canvas>
        </div>
        <p class="lt">Guided by the lifework of <strong>Dr. Glen Swartwout, O.D., N.D.</strong>, featured in USA&nbsp;Today: <em>&ldquo;Revolutionizing the World of Medicine with Natural Therapies.&rdquo;</em></p>
      </div>
```

Wire the `unlock('video')` call onto `.proof .sig` (find the existing handler that binds `.video` click and repoint it, or add a click/keydown on `#hero-signature`'s parent that calls the same `unlock('video')` path). If the video-unlock is intentionally dropped, say so in the report.

- [ ] **Step 2: Add proof + chat styles** to the style block:

```css
  .hero-grid { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 22px; align-items: stretch; }
  .proof { border: 1px solid var(--border); border-radius: var(--radius); background: linear-gradient(160deg, rgba(212,168,67,0.05), rgba(0,0,0,0.18)); padding: 18px; display: flex; flex-direction: column; gap: 14px; }
  .proof .sig { border: 1px solid var(--border); border-radius: 10px; background: #0c1e14; overflow: hidden; cursor: pointer; }
  .proof .sig canvas { width: 100%; height: 120px; display: block; }
  .proof .lt { font-size: 14px; line-height: 1.5; color: var(--muted); margin: 0; }
  .proof .lt strong { color: var(--cream); font-weight: 600; }
  .proof .lt em { color: var(--gold); font-style: italic; }
  @media (max-width: 720px) { .hero-grid { grid-template-columns: 1fr; } }
```

(Leave the existing `.hero-chat`, `.hero-messages`, `.hero-inputbar` rules 580-588 as-is; they already match the palette.)

- [ ] **Step 3: Render-verify.** Load `/`. Confirm the two-column hero (signature card left, working chat right), the concierge greets and accepts input, and the signature canvas animates. Clicking the signature still fires the journey unfold if you kept the hook. No console errors.

- [ ] **Step 4: Commit.**

```bash
git add static/begin.html
git commit -m "feat(home): re-skin hero grid with voice-signature proof card"
```

---

### Task 3: Add the Scan / Find / Heal explainer section

**Files:** Modify `static/begin.html` (insert after `.hero` closes at 743, before `#journey-strip` at 748; style block)

- [ ] **Step 1: Insert the section markup** right after the hero `</section>`:

```html
  <section class="how reveal" aria-label="How it works">
    <p class="how-cap">How it works</p>
    <h2 class="how-h">Three steps to a protocol built around you</h2>
    <p class="how-lede">Not another shelf of generic supplements. Your body&rsquo;s own signal, matched to what it is asking for, guided all the way to coherence.</p>
    <div class="how-flow">
      <div class="how-cell">
        <canvas class="how-motif" id="m-scan" width="240" height="76" aria-hidden="true"></canvas>
        <div class="how-n">Step 01</div><b>Scan</b>
        <span>A short voice sample captures your body&rsquo;s raw energy signature in seconds.</span>
      </div>
      <div class="how-arw" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 12h14M13 6l6 6-6 6"/></svg></div>
      <div class="how-cell">
        <canvas class="how-motif" id="m-find" width="240" height="76" aria-hidden="true"></canvas>
        <div class="how-n">Step 02</div><b>Find</b>
        <span>Your signal is compared to remedy after remedy until the right ones resonate.</span>
      </div>
      <div class="how-arw" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 12h14M13 6l6 6-6 6"/></svg></div>
      <div class="how-cell">
        <canvas class="how-motif" id="m-heal" width="240" height="76" aria-hidden="true"></canvas>
        <div class="how-n">Step 03</div><b>Heal</b>
        <span>Your field settles into coherence, the body&rsquo;s own return to order and syntropy.</span>
      </div>
    </div>
    <p class="how-foot">A <b>welcome call</b> is offered right after you start. You are never alone in this.</p>
  </section>
```

- [ ] **Step 2: Add the explainer styles** to the style block:

```css
  .how { max-width: 1080px; margin: 34px auto 0; padding: 26px 20px; border-top: 1px solid var(--border); }
  .how-cap { text-align: center; color: var(--muted); font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; margin: 0; }
  .how-h { font-family: var(--serif); text-align: center; font-weight: 600; font-size: clamp(22px, 2.8vw, 32px); margin: 6px 0 4px; text-wrap: balance; }
  .how-lede { text-align: center; color: var(--muted); max-width: 52ch; margin: 0 auto 26px; font-size: 15px; line-height: 1.55; }
  .how-flow { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: start; gap: 6px; }
  .how-cell { text-align: center; padding: 8px 12px; }
  .how-motif { width: 100%; max-width: 220px; height: 74px; margin: 0 auto 12px; display: block; border: 1px solid var(--border); border-radius: 9px; background: #0c1e14; }
  .how-n { font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--gold); font-weight: 700; }
  .how-cell b { display: block; font-size: 18px; margin: 4px 0 6px; }
  .how-cell span { color: var(--muted); font-size: 15px; line-height: 1.5; }
  .how-arw { align-self: center; color: var(--border); }
  .how-arw svg { width: 26px; height: 26px; display: block; }
  .how-foot { text-align: center; margin-top: 24px; color: var(--muted); font-size: 15px; }
  .how-foot b { color: var(--cream); }
  @media (max-width: 680px) { .how-flow { grid-template-columns: 1fr; } .how-arw { display: none; } }
```

- [ ] **Step 3: Render-verify.** Load `/`. The explainer appears below the hero (above the hidden journey strip), three cells with canvas motifs and copy, arrows between them on desktop, stacked on mobile. Reveals animate in on scroll.

- [ ] **Step 4: Commit.**

```bash
git add static/begin.html
git commit -m "feat(home): add Scan/Find/Heal explainer section"
```

---

### Task 4: Waveform canvas animations

**Files:** Modify `static/begin.html` (add one IIFE near the end of the inline app script, before `</script>` at 1346)

**Interfaces:** Consumes canvases `#hero-signature`, `#m-scan`, `#m-find`, `#m-heal` from Tasks 2-3.

- [ ] **Step 1: Add the animation IIFE.** Paste this self-contained block (it never touches `heroChat()` state). Replace the placeholder remedy names in `REMEDIES` with three real Functional Formulation names Glen confirms; keep `your match` as the resonant one.

```html
<script>
(function(){
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function grab(id){ var c=document.getElementById(id); return c ? {x:c.getContext('2d'),W:c.width,H:c.height} : null; }
  function sumComps(comps,u,t,tphase){ var v=0; for(var k=0;k<comps.length;k++){ v+=comps[k].a*Math.sin(comps[k].f*u*Math.PI*2+comps[k].ph+t*tphase); } return v; }
  function line(o,valueFn,stroke,glow,blur){
    var x=o.x; x.beginPath();
    for(var px=0;px<=o.W;px+=2){ var y=valueFn(px/o.W); if(px===0)x.moveTo(px,y); else x.lineTo(px,y); }
    x.lineWidth=2.2; x.strokeStyle=stroke;
    if(glow){ x.shadowColor=glow; x.shadowBlur=blur||6; } else { x.shadowBlur=0; }
    x.stroke(); x.shadowBlur=0;
  }
  function bars(o,n){
    var seeds=[]; for(var i=0;i<n;i++) seeds.push({p:i*0.5,s:0.6+(i%7)*0.12});
    function draw(t){
      var x=o.x,W=o.W,H=o.H,mid=H/2,gap=W/n; x.clearRect(0,0,W,H);
      for(var i=0;i<n;i++){
        var env=Math.sin((i/n)*Math.PI);
        var a=reduce?0.55:(0.5+0.5*Math.sin(t*0.0016*seeds[i].s+seeds[i].p));
        var h=3+env*(H*0.40)*a, cx=gap*i+gap*0.5, w=gap*0.42;
        var g=x.createLinearGradient(0,mid-h,0,mid+h);
        g.addColorStop(0,'rgba(212,168,67,.95)'); g.addColorStop(1,'rgba(61,138,82,.7)');
        x.fillStyle=g; x.beginPath();
        if(x.roundRect){ x.roundRect(cx-w/2,mid-h,w,h*2,w/2); x.fill(); } else x.fillRect(cx-w/2,mid-h,w,h*2);
      }
      if(!reduce) requestAnimationFrame(draw);
    }
    reduce?draw(0):requestAnimationFrame(draw);
  }
  var YOU=[{f:2,a:1,ph:0.4},{f:4,a:.5,ph:1.1}];
  var REMEDIES=[
    {name:'Neuro Magnesium',comps:[{f:2,a:1,ph:1.7},{f:5,a:.4,ph:1.2}],match:false},
    {name:'Lithospermum',comps:[{f:3,a:.85,ph:2.4},{f:7,a:.3,ph:.4}],match:false},
    {name:'Lutein Complex',comps:[{f:1.5,a:1,ph:.2},{f:4,a:.55,ph:2.9}],match:false},
    {name:'your match',comps:YOU,match:true}
  ];
  function find(o){
    var period=1300,i=0,last=0;
    function draw(t){
      var x=o.x,W=o.W,H=o.H; x.clearRect(0,0,W,H);
      if(!last)last=t;
      if(!reduce && t-last>period){ i=(i+1)%REMEDIES.length; last=t; }
      var cand=REMEDIES[reduce?3:i], matched=cand.match;
      line(o,function(u){return H*0.32 - sumComps(YOU,u,t,0.0012)*H*0.16;},'rgba(212,168,67,'+(matched?'1':'.9')+')',matched?'rgba(212,168,67,.8)':0,matched?6:0);
      line(o,function(u){return H*0.58 - sumComps(cand.comps,u,t,0.0012)*H*0.16;},matched?'rgba(95,174,110,1)':'rgba(120,140,128,.75)',matched?'rgba(95,174,110,.9)':0,matched?9:0);
      x.shadowBlur=0; x.font='600 10px system-ui,sans-serif'; x.textAlign='center';
      x.fillStyle=matched?'rgba(95,174,110,1)':'rgba(168,152,112,.85)';
      x.fillText((matched?'✓ ':'')+cand.name, W/2, H-6);
      if(!reduce) requestAnimationFrame(draw);
    }
    reduce?draw(1500):requestAnimationFrame(draw);
  }
  function heal(o){
    function draw(t){
      var x=o.x,W=o.W,H=o.H; x.clearRect(0,0,W,H);
      var breath=reduce?0.85:(0.74+0.14*Math.sin(t*0.0011));
      var g=x.createLinearGradient(0,0,W,0);
      g.addColorStop(0,'rgba(212,168,67,.9)'); g.addColorStop(1,'rgba(95,174,110,.9)');
      line(o,function(u){return H*0.5 - Math.sin(u*Math.PI*2 + (reduce?0:t*0.0016))*H*0.22*breath;},g,'rgba(95,174,110,.5)',9);
      if(!reduce) requestAnimationFrame(draw);
    }
    reduce?draw(0):requestAnimationFrame(draw);
  }
  var hs=grab('hero-signature'); if(hs) bars(hs,42);
  var s=grab('m-scan'); if(s) bars(s,22);
  var f=grab('m-find'); if(f) find(f);
  var h=grab('m-heal'); if(h) heal(h);
})();
</script>
```

- [ ] **Step 2: Render-verify.** Load `/`. All four canvases animate (hero signature, scan bars, find matching with green lock + checkmark on `your match`, heal coherent breathing wave). Toggle OS reduced-motion and reload: canvases show a static frame, no animation. No console errors; the chat still works.

- [ ] **Step 3: Commit.**

```bash
git add static/begin.html
git commit -m "feat(home): living waveform motifs for scan/find/heal"
```

---

### Task 5: Secondary Healing Oasis label + full-page verification

**Files:** Modify `static/begin.html` (the `#oasis-btn`, line 639) and full render pass.

- [ ] **Step 1: Set the Healing Oasis label.** The header/ribbon rebuild is a separate plan; here only align the existing `#oasis-btn` label. Set its text to `Healing Oasis` (from `My Healing Oasis`). Leave its show/hide flag logic (`/api/healing-oasis/status`) and modal wiring unchanged.

- [ ] **Step 2: Full render-verify** (the whole homepage in a headless browser):
  - Hero: serif headline reads clearly, eyebrow, subhead, signature card + working chat.
  - Concierge: greets, accepts a name, streams a reply (confirm `/begin/match/chat` still reached).
  - Explainer: three animated motifs + copy, arrows on desktop, stacked on mobile at 375px width.
  - Journey strip still hidden until unfold; layers/cards still reveal as before.
  - Reduced-motion: static frames.
  - Mobile (375px) and desktop (1280px): no horizontal scroll, hero grid collapses to one column.
  - Console: no errors on load or after sending a chat message.

- [ ] **Step 3: Commit and open a draft PR.**

```bash
git add static/begin.html
git commit -m "feat(home): secondary Healing Oasis label + homepage hybrid verified"
git push -u origin HEAD
gh pr create --draft --title "Homepage hybrid (Option A): re-skinned hero + Scan/Find/Heal explainer" \
  --body "Implements the homepage section of docs/superpowers/specs/2026-07-12-homepage-header-redesign-design.md, Option A. Keeps the chat concierge; adds living-waveform explainer."
```

## Self-Review

- **Spec coverage:** hero re-skin (Tasks 1-2), the new Scan/Find/Heal explainer (Task 3-4), the living waveforms with the scan/match/coherence story (Task 4), stateful-but-secondary Healing Oasis link (Task 5). Header/ribbon rebuild and the adaptive concierge behavior are explicitly separate plans.
- **Guardrail coverage:** chat IDs preserved (Global Constraints + Task 2 interfaces), `[data-name]` spans preserved (Task 1 Step 3), `unlock('video')` re-wired or consciously dropped (Task 2 Step 1), reduced-motion handled (Task 4).
- **Placeholder scan:** the only intentional placeholders are the three remedy names in `REMEDIES`, flagged in Task 4 Step 1 for Glen to confirm with real Functional Formulation names.
- **Verification reality:** deploy-chat has no CI and this is visual; every task ends in a headless-browser render check, and the live app must be run to do them (the sub-agent should launch it locally, not rely on tests).
