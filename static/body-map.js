// Body Map client. Reference-chart rendering + interaction. No image ever leaves the browser.
(function () {
  const svgNS = "http://www.w3.org/2000/svg";
  const VIEW = 600, CX = 300, CY = 300;
  const state = { payload: null, eye: "right", activeLayers: new Set(), transform: null, depth: "",
                  litZones: new Set(), portalToken: null };

  function zoneSide(z) { return z.side || z.eye; }
  function zoneGroup(z) { return z.group || z.germ_layer; }

  // Embryological germ layer of a zone (for the depth-peel). Prefer an explicit
  // germ_layer / embryo tag, else infer from the anatomy name. Order matters:
  // neural/sensory (ectoderm) and viscera (endoderm) are checked before the
  // structural/circulatory catch-all (mesoderm).
  const EMBRYO_KW = [
    ["ectoderm", /brain|cerebr|cerebell|medulla|brainstem|pineal|pituitar|nerv|sciatic|neuro|sensor|subcortex|occiput|\bmind\b|mental|ego|comprehens|speech|\bmotor|skin|epiderm|\beye|vision|\bear\b|hearing|\bnose\b|sinus|\bface|forehead|scalp|tooth|teeth|mammary|breast/i],
    ["endoderm", /liver|lung|bronch|trachea|pharynx|larynx|thyroid|parathyroid|thymus|stomach|duoden|intestin|colon|cecum|append|ileocecal|sigmoid|rectum|pancrea|gallbladder|\bbladder|urethra|prostate|tonsil|esophag|oesophag|digest/i],
    ["mesoderm", /kidney|adrenal|heart|circulat|aorta|vessel|blood|spleen|lymph|node|muscle|bone|skelet|vertebra|spine|cervical|thoracic|lumbar|sacr|coccyx|\brib|sternum|femur|pelvi|\bhip|joint|knee|elbow|wrist|ankle|shoulder|\barm\b|\bleg\b|limb|clavicle|connective|cartilage|dermis|gonad|ovar|test|uter|reproduct|genital|peritoneum|diaphragm|hernia|articular/i],
  ];
  function embryoLayer(z) {
    const g = z.germ_layer;
    if (g === "endoderm" || g === "mesoderm" || g === "ectoderm") return g;
    const e = z.embryo || (z.layers && z.layers.embryological_depth);
    if (e === "endoderm" || e === "mesoderm" || e === "ectoderm") return e;
    const hay = (z.anatomy || "") + " " + (z.id || "");
    for (const [layer, re] of EMBRYO_KW) if (re.test(hay)) return layer;
    return null; // untagged -> visible at every depth
  }
  function groupsOf(p) { return (p && (p.groups && p.groups.length ? p.groups : p.germ_layers)) || []; }
  function isOutlineFrame() { return state.frame && state.frame !== "unit_circle"; }

  // clock degrees (0=12 o'clock, clockwise) -> normalized unit vector, y-down, 12=up=(0,-1)
  function clockToNormalized(deg) {
    const r = deg * Math.PI / 180;
    return { x: Math.sin(r), y: -Math.cos(r) };
  }

  // normalized annular sector -> array of {x,y} normalized points (polyline approximation)
  function arcSectorPoints(radial, sector) {
    const pts = [], step = 2;
    for (let d = sector.start_deg; d <= sector.end_deg; d += step) {
      const u = clockToNormalized(d); pts.push({ x: u.x * radial.r_outer, y: u.y * radial.r_outer });
    }
    for (let d = sector.end_deg; d >= sector.start_deg; d -= step) {
      const u = clockToNormalized(d); pts.push({ x: u.x * radial.r_inner, y: u.y * radial.r_inner });
    }
    return pts;
  }

  // reference-frame normalized point -> reference-chart screen point
  function refToScreen(p) {
    if (isOutlineFrame()) { return { x: p.x * VIEW, y: p.y * VIEW }; }
    const R = state.chartR || 250; return { x: CX + p.x * R, y: CY + p.y * R };
  }

  // Fit the reference-chart radius to the loaded system's data extent (iris r<=1, sclerology r_outer up to ~3).
  function computeChartR(payload) {
    let maxR = 1;
    (payload.germ_layers || []).forEach(g => { if (g.r_outer > maxR) maxR = g.r_outer; });
    (payload.zones || []).forEach(z => { if (z.radial && z.radial.r_outer > maxR) maxR = z.radial.r_outer; });
    return (CX - 20) / maxR;   // fit within the canvas with a 20px margin
  }

  function pointsToPath(pts, mapFn) {
    return pts.map((p, i) => { const s = mapFn(p); return (i ? "L" : "M") + s.x.toFixed(1) + " " + s.y.toFixed(1); }).join(" ") + " Z";
  }

  // Transform every coordinate pair in a normalized SVG path via mapFn, preserving the curve
  // commands (M/L/C/S/Q/T) so the outline draws as smooth curves rather than a jagged polyline.
  function transformPathD(d, mapFn) {
    let out = d.replace(/([MLCSQT])([^A-Za-z]*)/gi, (m, cmd, nums) => {
      const vals = (nums.match(/-?\d*\.?\d+/g) || []).map(Number);
      let s = cmd.toUpperCase();
      for (let i = 0; i + 1 < vals.length; i += 2) {
        const p = mapFn({ x: vals[i], y: vals[i + 1] });
        s += " " + p.x.toFixed(1) + " " + p.y.toFixed(1);
      }
      return s + " ";
    });
    if (/z\s*$/i.test(d)) out += "Z";
    return out;
  }

  // Distinct region colors, assigned by a group's position in the payload's group list.
  const GROUP_PALETTE = ["#2f6f5e", "#c07f2a", "#7a5aa6", "#3f7cae", "#b0503a", "#5f8a3a", "#9a7b39", "#4a8a86",
    "#b5447a", "#2f8f8f", "#8a6d2f", "#6a4ac0", "#c05a2a", "#3a7a5f", "#a03a6a", "#4a6aae"];
  function groupColor(z) {
    const groups = groupsOf(state.payload);
    const idx = groups.findIndex(g => g.id === zoneGroup(z));
    return GROUP_PALETTE[(idx >= 0 ? idx : 0) % GROUP_PALETTE.length];
  }

  function currentZones() {
    if (!state.payload) return [];
    return state.payload.zones.filter(z =>
      (z.bilateral || zoneSide(z) === state.eye) &&
      (state.activeLayers.size === 0 || state.activeLayers.has(zoneGroup(z))) &&
      (!state.depth || embryoLayer(z) === state.depth));
  }

  function renderChart() {
    if (!state.payload) return;
    const svg = document.getElementById("bm-svg");
    svg.innerHTML = "";
    const mapFn = state.transform ? (p) => state.transform(p) : refToScreen;
    if (isOutlineFrame()) {
      // per-view outline (e.g. front/back share a silhouette, side is a profile);
      // falls back to the single `outline` for systems that have just one.
      const outlineD = (state.payload.outlines && state.payload.outlines[state.eye]) || state.payload.outline;
      if (outlineD) {
        const path = document.createElementNS(svgNS, "path");
        const oMir = state.payload.outline_side && state.payload.outline_side !== state.eye;
        const oMapFn = oMir ? (p) => mapFn({ x: 1 - p.x, y: p.y }) : mapFn;
        path.setAttribute("d", transformPathD(outlineD, oMapFn));
        path.setAttribute("fill", "#00000008"); path.setAttribute("stroke", "#b8a678"); path.setAttribute("stroke-width", "1.5");
        svg.appendChild(path);
      }
    } else {
      (state.payload.germ_layers || []).forEach(g => {
        [g.r_inner, g.r_outer].forEach(rr => {
          const c = document.createElementNS(svgNS, "circle");
          const o = mapFn({ x: 0, y: 0 }), edge = mapFn({ x: rr, y: 0 });
          c.setAttribute("cx", o.x); c.setAttribute("cy", o.y);
          c.setAttribute("r", Math.hypot(edge.x - o.x, edge.y - o.y));
          c.setAttribute("fill", "none"); c.setAttribute("stroke", "#d9cfb8"); c.setAttribute("stroke-width", "1");
          svg.appendChild(c);
        });
      });
    }
    const labelSpecs = [];
    currentZones().forEach(z => {
      const geo = z.geometry || {};
      const mir = z.bilateral && zoneSide(z) !== state.eye;         // mirror the contralateral side
      const N = (x, y) => mapFn({ x: mir ? 1 - x : x, y: y });
      const col = groupColor(z);
      // Collect label anchors now; place them in de-collided columns after all
      // zones are drawn (see placeLabels) so labels never overlap each other.
      function addLabel(sx, sy) { labelSpecs.push({ z, sx, sy }); }
      if (geo.type === "ellipse") {
        const c = N(geo.cx, geo.cy);
        const ex = N(geo.cx + geo.rx, geo.cy), ey = N(geo.cx, geo.cy + geo.ry);
        const el = document.createElementNS(svgNS, "ellipse");
        el.setAttribute("cx", c.x.toFixed(1)); el.setAttribute("cy", c.y.toFixed(1));
        el.setAttribute("rx", Math.hypot(ex.x - c.x, ex.y - c.y).toFixed(1));
        el.setAttribute("ry", Math.hypot(ey.x - c.x, ey.y - c.y).toFixed(1));
        el.setAttribute("class", "bm-zone bm-area"); el.dataset.id = z.id;
        el.setAttribute("fill", col); el.setAttribute("fill-opacity", "0.35");
        el.setAttribute("stroke", col); el.setAttribute("stroke-width", "1.2");
        el.addEventListener("click", () => selectZone(z));
        svg.appendChild(el); addLabel(c.x, c.y);
      } else if (geo.type === "polygon") {
        const pts = geo.points || [];
        const d = pts.map((p, i) => { const s = N(p[0], p[1]); return (i ? "L" : "M") + s.x.toFixed(1) + " " + s.y.toFixed(1); }).join(" ") + " Z";
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", d); path.setAttribute("class", "bm-zone bm-area"); path.dataset.id = z.id;
        path.setAttribute("fill", col); path.setAttribute("fill-opacity", "0.35");
        path.setAttribute("stroke", col); path.setAttribute("stroke-width", "1.2");
        path.addEventListener("click", () => selectZone(z));
        svg.appendChild(path);
        const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
        const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
        const c = N(cx, cy); addLabel(c.x, c.y);
      } else if (geo.type === "point") {
        const s = N(geo.x, geo.y);
        const dot = document.createElementNS(svgNS, "circle");
        dot.setAttribute("cx", s.x); dot.setAttribute("cy", s.y); dot.setAttribute("r", 5);
        dot.setAttribute("class", "bm-zone bm-point"); dot.dataset.id = z.id;
        dot.setAttribute("fill", col); dot.setAttribute("stroke", "#fff"); dot.setAttribute("stroke-width", "1");
        dot.addEventListener("click", () => selectZone(z));
        svg.appendChild(dot); addLabel(s.x, s.y);
      } else if (geo.type === "path") {
        // a stroked line (e.g. an acupuncture meridian) rather than a filled area
        const line = document.createElementNS(svgNS, "path");
        line.setAttribute("d", transformPathD(geo.d, (p) => N(p.x, p.y)));
        line.setAttribute("class", "bm-zone bm-line"); line.dataset.id = z.id;
        line.setAttribute("fill", "none"); line.setAttribute("stroke", col);
        line.setAttribute("stroke-width", "2.5"); line.setAttribute("stroke-opacity", "0.85");
        line.setAttribute("stroke-linecap", "round"); line.setAttribute("stroke-linejoin", "round");
        line.addEventListener("click", () => selectZone(z));
        svg.appendChild(line);
        // label at a coordinate pair near the middle of the path
        const nums = (geo.d.match(/-?\d*\.?\d+/g) || []).map(Number);
        const m = Math.max(0, (Math.floor(nums.length / 4) * 2) - ((Math.floor(nums.length / 4) * 2) % 2));
        const lc = N(nums[m] ?? 0.5, nums[m + 1] ?? 0.5);
        addLabel(lc.x, lc.y);
      } else {
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", pointsToPath(arcSectorPoints(z.radial, z.sector), mapFn));
        path.setAttribute("class", "bm-zone"); path.dataset.id = z.id;
        path.addEventListener("click", () => selectZone(z));
        svg.appendChild(path);
        // label at the sector centroid (mid-angle, mid-radius)
        const midA = (z.sector.start_deg + z.sector.end_deg) / 2;
        const midR = (z.radial.r_inner + z.radial.r_outer) / 2;
        const u = clockToNormalized(midA);
        const sc = mapFn({ x: u.x * midR, y: u.y * midR });
        addLabel(sc.x, sc.y);
      }
    });
    placeLabels(labelSpecs, svg);
    // Personalized portal view: highlight the zones tied to the client's findings.
    if (state.litZones.size) {
      svg.querySelectorAll(".bm-zone, .bm-label, .bm-leader").forEach(e =>
        e.classList.toggle("bm-lit", state.litZones.has(e.dataset.id)));
    }
  }

  // Place zone labels in two vertical columns (left/right of the chart centre),
  // greedily de-collided so no two labels overlap, each tied to its zone by a
  // thin leader line. Long anatomy names stay legible via the label halo.
  function placeLabels(specs, svg) {
    if (!specs.length) return;
    const LINE_H = 13, TOP = 14, BOT = VIEW - 14;
    const LX = 150, RX = 450;                 // inner edges of the two columns
    // Balance the two columns: the foot's zones cluster near the centre line,
    // so a fixed x=CX split lands almost everything on one side. Sort by x and
    // give the left half to the left column, right half to the right column.
    const cols = { L: [], R: [] };
    const byX = [...specs].sort((a, b) => a.sx - b.sx);
    const half = Math.ceil(byX.length / 2);
    byX.forEach((s, i) => (i < half ? cols.L : cols.R).push(s));
    for (const side of ["L", "R"]) {
      const list = cols[side].sort((a, b) => a.sy - b.sy);
      // top-down greedy: keep each label at its anchor y unless that collides
      let prevY = -Infinity;
      list.forEach(s => { s.ly = Math.max(s.sy, prevY + LINE_H); prevY = s.ly; });
      // if the column overran the bottom, slide the whole run up, re-clamping
      const overrun = list.length ? list[list.length - 1].ly - BOT : 0;
      if (overrun > 0) {
        const headroom = list.length ? list[0].ly - TOP : 0;
        const shift = Math.min(overrun, Math.max(0, headroom));
        prevY = TOP - LINE_H;
        list.forEach(s => { s.ly = Math.max(s.ly - shift, prevY + LINE_H); prevY = s.ly; });
      }
      const anchorEnd = side === "L";
      const edgeX = anchorEnd ? LX : RX;
      list.forEach(s => {
        const ly = Math.min(BOT, Math.max(TOP, s.ly));
        const leader = document.createElementNS(svgNS, "line");
        leader.setAttribute("x1", edgeX); leader.setAttribute("y1", ly.toFixed(1));
        leader.setAttribute("x2", s.sx.toFixed(1)); leader.setAttribute("y2", s.sy.toFixed(1));
        leader.setAttribute("class", "bm-leader"); leader.dataset.id = s.z.id;
        svg.appendChild(leader);
        const t = document.createElementNS(svgNS, "text");
        t.setAttribute("x", edgeX); t.setAttribute("y", (ly + 3).toFixed(1));
        t.setAttribute("text-anchor", anchorEnd ? "end" : "start");
        t.setAttribute("class", "bm-label"); t.dataset.id = s.z.id;
        t.textContent = s.z.anatomy;
        t.addEventListener("click", () => selectZone(s.z));
        svg.appendChild(t);
      });
    }
  }

  function selectZone(z) {
    document.querySelectorAll(".bm-zone, .bm-label, .bm-leader").forEach(e => e.classList.toggle("bm-sel", e.dataset.id === z.id));
    const panel = document.getElementById("bm-panel");
    panel.replaceChildren();
    const h = document.createElement("h2"); h.textContent = z.anatomy;
    const groupNoun = (state.payload && state.payload.group_noun) ? (" " + state.payload.group_noun + ", ") : (z.germ_layer ? " layer, " : " region, ");
    const sideNoun = (state.payload && state.payload.side_noun) ? (" " + state.payload.side_noun) : (z.eye ? " eye" : " ear");
    const meta = document.createElement("p");
    const strong = document.createElement("strong"); strong.textContent = zoneGroup(z);
    meta.append(strong, document.createTextNode(groupNoun + zoneSide(z) + sideNoun));
    const body = document.createElement("p"); body.textContent = z.meaning_display || z.meaning_standard;
    panel.append(h, meta, body);
  }

  function renderLayerToggles() {
    const box = document.getElementById("bm-layers"); box.innerHTML = "";
    groupsOf(state.payload).forEach(g => {
      const id = "bml-" + g.id;
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.id = id;
      label.append(cb, document.createTextNode(" " + g.label));
      cb.addEventListener("change", () => {
        if (cb.checked) state.activeLayers.add(g.id); else state.activeLayers.delete(g.id);
        renderChart();
      });
      box.appendChild(label);
    });
  }

  async function loadSystem(system) {
    const res = await fetch("/body-map/data?system=" + encodeURIComponent(system));
    state.payload = await res.json();
    state.frame = state.payload.reference_frame || "unit_circle";
    state.chartR = computeChartR(state.payload);
    state.activeLayers.clear();
    // laterality: relabel + repopulate from the sides present, keep current if still valid.
    // Outline systems mirror via outline_side, so both sides are always available
    // even when every zone is authored bilaterally on one canonical side (e.g. the ear).
    const sides = state.payload.outline_side
      ? ["left", "right"]
      : [...new Set((state.payload.zones || []).map(zoneSide))];
    const sel = document.getElementById("bm-eye");
    sel.replaceChildren();
    sides.forEach(s => { const o = document.createElement("option"); o.value = s; o.textContent = s.charAt(0).toUpperCase() + s.slice(1); sel.appendChild(o); });
    if (!sides.includes(state.eye)) state.eye = sides[0] || "right";
    sel.value = state.eye;
    // Label the view control: outline systems = "Side"; unit-circle systems use
    // their own noun ("eye" for iris/sclera, "clock" for the organ clock) else "Eye".
    const _sn = state.payload.side_noun;
    document.getElementById("bm-side-label").textContent =
      isOutlineFrame() ? "Side" : (_sn ? _sn.charAt(0).toUpperCase() + _sn.slice(1) : "Eye");
    renderLayerToggles(); renderChart();
  }

  function wire() {
    document.getElementById("bm-system").addEventListener("change", e => {
      // In the portal, switching systems re-personalizes (keeps the client's
      // findings lit on the new map) instead of loading a blank reference chart.
      if (state.portalToken) bootstrapPortal(e.target.value, e.target.value === "face");
      else loadSystem(e.target.value);
    });
    document.getElementById("bm-eye").addEventListener("change", e => { state.eye = e.target.value; renderChart(); });
    // embryological depth-peel: isolate one germ layer (endoderm/mesoderm/ectoderm)
    document.querySelectorAll("#bm-depth button").forEach(b => b.addEventListener("click", () => {
      state.depth = b.dataset.depth;
      document.querySelectorAll("#bm-depth button").forEach(x => x.classList.toggle("bm-active", x === b));
      renderChart();
    }));
    wireOverlay();
    // Portal-personalized entry: /portal/<token>/bodymap warps the face map onto
    // the client's own photo with their findings lit, instead of the URL-focus path.
    const portalMatch = location.pathname.match(/^\/portal\/([^/]+)\/bodymap\/?$/);
    if (portalMatch) {
      state.portalToken = decodeURIComponent(portalMatch[1]);
      bootstrapPortal().then(__bmSelfCheck);
      return;
    }
    const params = new URLSearchParams(location.search);
    const sys = params.get("system");
    const sel = document.getElementById("bm-system");
    const known = [...sel.options].map(o => o.value);
    const initialSystem = known.includes(sys) ? sys : "iridology";
    sel.value = initialSystem;
    loadSystem(initialSystem).then(function () { applyFocusFromURL(params); __bmSelfCheck(); });
  }

  // Load the client's personalization for `system` (default face), light their
  // finding zones on that map, and — for the face, auto-warp their own photo.
  // Other systems show the reference figure lit (a face selfie can't warp onto a
  // body/foot map); the client can still upload a matching photo to warp it.
  async function bootstrapPortal(system, autoPhoto) {
    if (system === undefined) { system = "face"; autoPhoto = true; }
    const token = state.portalToken;
    let pz = null;
    try {
      const url = "/api/portal/" + encodeURIComponent(token) + "/bodymap"
        + (system ? "?system=" + encodeURIComponent(system) : "");
      pz = await (await fetch(url)).json();
    } catch (e) { pz = null; }
    const loaded = (pz && pz.system) || system || "face";
    document.getElementById("bm-system").value = loaded;
    await loadSystem(loaded);
    if (pz && !pz.error) {
      if (pz.view) { state.eye = pz.view; document.getElementById("bm-eye").value = pz.view; }
      state.litZones = new Set(pz.lit_zones || []);
      renderChart();
      renderPortalPanel(pz);
      if (autoPhoto && pz.has_photo) loadPortalPhoto(token);
      else setMode(false);
    }
  }

  function loadPortalPhoto(token) {
    const img = document.getElementById("bm-photo");
    img.onload = function () {
      setMode(true);
      document.getElementById("bm-autodetect").hidden = !detectorKind();
      beginAnchoring();              // resets anchors; keeps state.litZones
      autoDetect();                  // MediaPipe face landmarks -> warp
    };
    img.onerror = function () { setMode(false); };
    img.src = "/api/portal/" + encodeURIComponent(token) + "/photo?t=" + Date.now();
  }

  function renderPortalPanel(pz) {
    const panel = document.getElementById("bm-panel");
    panel.replaceChildren();
    const h = document.createElement("h2");
    h.textContent = pz.count ? "Your findings on this map" : "Your Body Map";
    panel.appendChild(h);
    if (!pz.count) {
      const p = document.createElement("p"); p.className = "bm-hint";
      p.textContent = pz.has_photo
        ? "None of your current findings map to the face yet. Explore the map, or get a voice scan to personalize it."
        : "Add a photo in your portal and get a voice scan to see your findings mapped onto your own face.";
      panel.appendChild(p);
      return;
    }
    const intro = document.createElement("p"); intro.className = "bm-hint";
    intro.textContent = "The highlighted zones correspond to your top findings. Tap a zone to read what it maps to.";
    panel.appendChild(intro);
    const ul = document.createElement("ul"); ul.className = "bm-findings";
    pz.findings.forEach(f => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button"; btn.className = "bm-finding";
      btn.textContent = f.label + (f.rank != null ? "  ·  priority " + f.rank : "");
      btn.addEventListener("click", () => {
        const zid = (f.zones || [])[0];
        const z = zid && (state.payload.zones || []).find(x => x.id === zid);
        if (z) selectZone(z);
      });
      li.appendChild(btn); ul.appendChild(li);
    });
    panel.appendChild(ul);
  }

  function __bmSelfCheck() {
    const up = clockToNormalized(0), right = clockToNormalized(90);
    const okUp = Math.abs(up.x) < 1e-9 && Math.abs(up.y + 1) < 1e-9;
    const okRight = Math.abs(right.x - 1) < 1e-9 && Math.abs(right.y) < 1e-9;
    const okArc = arcSectorPoints({ r_inner: 0.1, r_outer: 0.2 }, { start_deg: 0, end_deg: 10 }).length > 0;
    console.log("[bodymap] selfcheck " + (okUp && okRight && okArc ? "ok" : "FAIL"));
  }

  const ANCHOR_STEPS = [
    { key: "pupil", hint: "Tap the center of your pupil." },
    { key: "limbus", hint: "Tap the edge of your iris (where color meets white)." },
    { key: "twelve", hint: "Tap the top of your iris edge (12 o'clock)." },
  ];
  const anchors = {};
  let anchorIdx = 0;

  // Build a normalized-frame -> screen transform from the three tapped anchors.
  function computeSimilarity(P, L, Tw) {
    const scale = Math.hypot(L.x - P.x, L.y - P.y);
    const rot = Math.atan2(Tw.y - P.y, Tw.x - P.x) + Math.PI / 2; // normalized 12 o'clock is up
    const cos = Math.cos(rot), sin = Math.sin(rot);
    return (n) => ({
      x: P.x + scale * (n.x * cos - n.y * sin),
      y: P.y + scale * (n.x * sin + n.y * cos),
    });
  }

  function activeAnchorSteps() {
    const a = (state.payload && state.payload.anchors) || [];
    // multi-view systems tag anchors with a `view`; use the ones for the current view
    const forView = a.filter(s => !s.view || s.view === state.eye);
    if (forView.length) return forView;
    return a.length ? a : ANCHOR_STEPS;
  }

  // Fit a similarity (translation + rotation + uniform scale) mapping template coords -> screen,
  // from the first two anchor correspondences. Exact for 2 points; a third is not required.
  function fitSimilarity(steps) {
    const a0 = steps[0].template, a1 = steps[1].template;
    const b0 = anchors[steps[0].key], b1 = anchors[steps[1].key];
    const dax = a1.x - a0.x, day = a1.y - a0.y;
    const dbx = b1.x - b0.x, dby = b1.y - b0.y;
    const denom = dax * dax + day * day || 1e-9;
    const mx = (dbx * dax + dby * day) / denom;
    const my = (dby * dax - dbx * day) / denom;
    const tx = b0.x - (mx * a0.x - my * a0.y);
    const ty = b0.y - (my * a0.x + mx * a0.y);
    return (n) => ({ x: mx * n.x - my * n.y + tx, y: my * n.x + mx * n.y + ty });
  }

  function setMode(photo) {
    document.getElementById("bm-mode-ref").classList.toggle("bm-active", !photo);
    document.getElementById("bm-mode-photo").classList.toggle("bm-active", photo);
    document.getElementById("bm-photo").hidden = !photo || !document.getElementById("bm-photo").src;
    document.getElementById("bm-photo-tools").hidden = !photo;
    document.getElementById("bm-disclaimer").hidden = !photo;
    if (!photo) { state.transform = null; renderChart(); }
  }

  function beginAnchoring() {
    anchorIdx = 0; Object.keys(anchors).forEach(k => delete anchors[k]);
    state.transform = null; renderChart();
    document.getElementById("bm-anchor-hint").textContent = activeAnchorSteps()[0].hint;
  }

  function onCanvasClick(evt) {
    const steps = activeAnchorSteps();
    if (document.getElementById("bm-photo").hidden || anchorIdx >= steps.length) return;
    const svg = document.getElementById("bm-svg");
    const rect = svg.getBoundingClientRect();
    const x = (evt.clientX - rect.left) / rect.width * VIEW;
    const y = (evt.clientY - rect.top) / rect.height * VIEW;
    anchors[steps[anchorIdx].key] = { x, y };
    anchorIdx++;
    if (anchorIdx < steps.length) {
      document.getElementById("bm-anchor-hint").textContent = steps[anchorIdx].hint;
      drawAnchors();
    } else {
      placeOverlay(steps);
      console.log("[bodymap] overlay placed");
    }
  }

  function drawAnchors() {
    const svg = document.getElementById("bm-svg");
    Object.values(anchors).forEach(a => {
      const c = document.createElementNS(svgNS, "circle");
      c.setAttribute("cx", a.x); c.setAttribute("cy", a.y); c.setAttribute("r", 6);
      c.setAttribute("class", "bm-anchor"); svg.appendChild(c);
    });
  }

  // Two tapped/detected anchors with template coords -> similarity; else iris fallback.
  function placeOverlay(steps) {
    document.getElementById("bm-anchor-hint").textContent = "Overlay placed. Re-upload to redo.";
    state.transform = (steps.length >= 2 && steps[0].template && steps[1].template)
      ? fitSimilarity(steps)
      : computeSimilarity(anchors.pupil, anchors.limbus, anchors.twelve);
    renderChart(); drawAnchors();
  }

  // ---- ML auto-anchoring: detect the anchor landmarks on the photo client-side.
  // MediaPipe runs in-browser (WASM), so the photo still never leaves the device.
  const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
  const MP_MODELS = {
    face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  };
  let _mp = null; const _lmk = {};

  function detectorKind() {
    const sys = state.payload && state.payload.system;
    if (sys === "face") return "face";
    if (sys === "hand") return "hand";
    if (sys === "eav") return state.eye === "foot" ? "pose" : "hand";
    if (sys === "foot" || sys === "meridian" || sys === "neurotome" || sys === "lymph") return "pose";
    return null; // iris / sclera / ear -> manual for now
  }

  async function getLandmarker(kind) {
    if (_lmk[kind]) return _lmk[kind];
    if (!_mp) _mp = await import(MP_BASE);
    const { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker } = _mp;
    const fileset = await FilesetResolver.forVisionTasks(MP_BASE + "/wasm");
    const opts = { baseOptions: { modelAssetPath: MP_MODELS[kind] }, runningMode: "IMAGE" };
    _lmk[kind] = await (kind === "face" ? FaceLandmarker : kind === "pose" ? PoseLandmarker : HandLandmarker)
      .createFromOptions(fileset, opts);
    return _lmk[kind];
  }

  function landmarksFromResult(kind, res) {
    return kind === "face" ? (res.faceLandmarks || [])[0] : (res.landmarks || [])[0];
  }

  // resolve an anchor-step key -> normalized {x,y} from the detected landmarks
  function landmarkFor(kind, lms, key) {
    const k = key.toLowerCase();
    if (kind === "face") {
      if (k.includes("hairline") || k.includes("head")) return lms[10];   // forehead top
      if (k.includes("chin")) return lms[152];
    } else if (kind === "pose") {
      if (k.includes("head") || k.includes("hairline")) {
        const nose = lms[0]; return { x: nose.x, y: Math.max(0, nose.y - 0.06) };
      }
      if (k.includes("feet") || k.includes("foot") || k.includes("ankle")) {
        const a = lms[27], b = lms[28]; return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      }
    } else if (kind === "hand") {
      if (k.includes("wrist")) return lms[0];
      if (k.includes("mid") || k.includes("middle")) return lms[12];
      if (k.includes("toe") || k.includes("tip")) return lms[12];
      if (k.includes("thumb")) return lms[4];
    }
    return null;
  }

  async function autoDetect() {
    const hint = document.getElementById("bm-anchor-hint");
    const kind = detectorKind();
    const steps = activeAnchorSteps();
    if (!kind || !(steps.length >= 2 && steps[0].template)) {
      hint.textContent = "Auto-detect isn't available here — tap the points manually."; return;
    }
    hint.textContent = "Detecting landmarks…";
    try {
      const L = await getLandmarker(kind);
      const lms = landmarksFromResult(kind, L.detect(document.getElementById("bm-photo")));
      if (!lms) throw new Error("no landmarks found");
      Object.keys(anchors).forEach(kk => delete anchors[kk]);
      for (const s of steps) {
        const n = landmarkFor(kind, lms, s.key);
        if (!n) throw new Error("unmapped anchor " + s.key);
        anchors[s.key] = { x: n.x * VIEW, y: n.y * VIEW };
      }
      anchorIdx = steps.length;
      placeOverlay(steps);
    } catch (e) {
      console.warn("[bodymap] auto-detect failed", e);
      anchorIdx = 0; Object.keys(anchors).forEach(kk => delete anchors[kk]);
      state.transform = null; renderChart();
      hint.textContent = "Couldn't auto-detect — tap manually, starting with: " + steps[0].hint;
    }
  }

  function onUpload(evt) {
    const file = evt.target.files && evt.target.files[0];
    if (!file) return;
    const img = document.getElementById("bm-photo");
    img.src = URL.createObjectURL(file); // stays in-browser; never uploaded
    img.hidden = false; setMode(true); beginAnchoring();
    document.getElementById("bm-autodetect").hidden = !detectorKind();
  }

  function wireOverlay() {
    document.getElementById("bm-mode-ref").addEventListener("click", () => setMode(false));
    document.getElementById("bm-mode-photo").addEventListener("click", () => setMode(true));
    document.getElementById("bm-upload").addEventListener("change", onUpload);
    document.getElementById("bm-autodetect").addEventListener("click", autoDetect);
    document.getElementById("bm-svg").addEventListener("click", onCanvasClick);
  }

  function applyFocusFromURL(params) {
    if (!state.payload) return;
    const side = params.get("side") || params.get("eye");
    const sides = state.payload.outline_side
      ? new Set(["left", "right"])
      : new Set((state.payload.zones || []).map(zoneSide));
    if (side && sides.has(side)) {
      state.eye = side; document.getElementById("bm-eye").value = side; renderChart();
    }
    const zoneId = params.get("zone");
    if (zoneId) {
      const z = (state.payload.zones || []).find(x => x.id === zoneId);
      if (z) {
        if (zoneSide(z) !== state.eye) {
          state.eye = zoneSide(z); document.getElementById("bm-eye").value = state.eye; renderChart();
        }
        selectZone(z);
        return;
      }
    }
    const groupId = params.get("group") || params.get("layer");
    if (groupId) {
      const grp = groupsOf(state.payload).find(g => g.id === groupId);
      if (grp) {
        state.activeLayers.clear(); state.activeLayers.add(groupId);
        const cb = document.getElementById("bml-" + groupId);
        if (cb) cb.checked = true;
        renderChart();
        const first = currentZones()[0];
        if (first) selectZone(first);
      }
    }
  }

  // expose for tasks/tests
  window.__bm = { clockToNormalized, arcSectorPoints, computeSimilarity, state };
  document.addEventListener("DOMContentLoaded", wire);
})();
