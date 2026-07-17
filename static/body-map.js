// Body Map client. Reference-chart rendering + interaction. No image ever leaves the browser.
(function () {
  const svgNS = "http://www.w3.org/2000/svg";
  const VIEW = 600, CX = 300, CY = 300;
  const state = { payload: null, eye: "right", activeLayers: new Set(), transform: null };

  function zoneSide(z) { return z.side || z.eye; }
  function zoneGroup(z) { return z.group || z.germ_layer; }
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
  const GROUP_PALETTE = ["#2f6f5e", "#c07f2a", "#7a5aa6", "#3f7cae", "#b0503a", "#5f8a3a", "#9a7b39", "#4a8a86"];
  function groupColor(z) {
    const groups = groupsOf(state.payload);
    const idx = groups.findIndex(g => g.id === zoneGroup(z));
    return GROUP_PALETTE[(idx >= 0 ? idx : 0) % GROUP_PALETTE.length];
  }

  function currentZones() {
    if (!state.payload) return [];
    return state.payload.zones.filter(z =>
      (z.bilateral || zoneSide(z) === state.eye) &&
      (state.activeLayers.size === 0 || state.activeLayers.has(zoneGroup(z))));
  }

  function renderChart() {
    if (!state.payload) return;
    const svg = document.getElementById("bm-svg");
    svg.innerHTML = "";
    const mapFn = state.transform ? (p) => state.transform(p) : refToScreen;
    if (isOutlineFrame()) {
      if (state.payload.outline) {
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", transformPathD(state.payload.outline, mapFn));
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
    currentZones().forEach(z => {
      const geo = z.geometry || {};
      const mir = z.bilateral && zoneSide(z) !== state.eye;         // mirror the contralateral side
      const N = (x, y) => mapFn({ x: mir ? 1 - x : x, y: y });
      const col = groupColor(z);
      function addLabel(sx, sy) {
        const onRight = sx > 300;
        const t = document.createElementNS(svgNS, "text");
        t.setAttribute("x", (sx + (onRight ? -8 : 8)).toFixed(1));
        t.setAttribute("y", (sy + 3).toFixed(1));
        t.setAttribute("text-anchor", onRight ? "end" : "start");
        t.setAttribute("class", "bm-label"); t.dataset.id = z.id;
        t.textContent = z.anatomy; t.addEventListener("click", () => selectZone(z));
        svg.appendChild(t);
      }
      if (geo.type === "polygon") {
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
      } else {
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", pointsToPath(arcSectorPoints(z.radial, z.sector), mapFn));
        path.setAttribute("class", "bm-zone"); path.dataset.id = z.id;
        path.addEventListener("click", () => selectZone(z));
        svg.appendChild(path);
      }
    });
  }

  function selectZone(z) {
    document.querySelectorAll(".bm-zone").forEach(e => e.classList.toggle("bm-sel", e.dataset.id === z.id));
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
    // laterality: relabel + repopulate from the sides present, keep current if still valid
    const sides = [...new Set((state.payload.zones || []).map(zoneSide))];
    const sel = document.getElementById("bm-eye");
    sel.replaceChildren();
    sides.forEach(s => { const o = document.createElement("option"); o.value = s; o.textContent = s.charAt(0).toUpperCase() + s.slice(1); sel.appendChild(o); });
    if (!sides.includes(state.eye)) state.eye = sides[0] || "right";
    sel.value = state.eye;
    document.getElementById("bm-side-label").textContent = isOutlineFrame() ? "Side" : "Eye";
    renderLayerToggles(); renderChart();
  }

  function wire() {
    document.getElementById("bm-system").addEventListener("change", e => loadSystem(e.target.value));
    document.getElementById("bm-eye").addEventListener("change", e => { state.eye = e.target.value; renderChart(); });
    wireOverlay();
    const params = new URLSearchParams(location.search);
    const sys = params.get("system");
    const initialSystem = (sys === "iridology" || sys === "sclerology" || sys === "ear" || sys === "foot") ? sys : "iridology";
    document.getElementById("bm-system").value = initialSystem;
    loadSystem(initialSystem).then(function () { applyFocusFromURL(params); __bmSelfCheck(); });
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
    const a = state.payload && state.payload.anchors;
    return (a && a.length) ? a : ANCHOR_STEPS;
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
      document.getElementById("bm-anchor-hint").textContent = "Overlay placed. Re-upload to redo.";
      state.transform = (state.payload && state.payload.anchors && state.payload.anchors.length)
        ? fitSimilarity(steps)
        : computeSimilarity(anchors.pupil, anchors.limbus, anchors.twelve);
      renderChart(); drawAnchors();
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

  function onUpload(evt) {
    const file = evt.target.files && evt.target.files[0];
    if (!file) return;
    const img = document.getElementById("bm-photo");
    img.src = URL.createObjectURL(file); // stays in-browser; never uploaded
    img.hidden = false; setMode(true); beginAnchoring();
  }

  function wireOverlay() {
    document.getElementById("bm-mode-ref").addEventListener("click", () => setMode(false));
    document.getElementById("bm-mode-photo").addEventListener("click", () => setMode(true));
    document.getElementById("bm-upload").addEventListener("change", onUpload);
    document.getElementById("bm-svg").addEventListener("click", onCanvasClick);
  }

  function applyFocusFromURL(params) {
    if (!state.payload) return;
    const side = params.get("side") || params.get("eye");
    const sides = new Set((state.payload.zones || []).map(zoneSide));
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
