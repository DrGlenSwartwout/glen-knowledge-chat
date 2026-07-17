// Body Map client. Reference-chart rendering + interaction. No image ever leaves the browser.
(function () {
  const svgNS = "http://www.w3.org/2000/svg";
  const VIEW = 600, CX = 300, CY = 300, R = 250; // reference-chart layout: unit circle -> 250px
  const state = { payload: null, eye: "right", activeLayers: new Set(), transform: null };

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
  function refToScreen(p) { return { x: CX + p.x * R, y: CY + p.y * R }; }

  function pointsToPath(pts, mapFn) {
    return pts.map((p, i) => { const s = mapFn(p); return (i ? "L" : "M") + s.x.toFixed(1) + " " + s.y.toFixed(1); }).join(" ") + " Z";
  }

  function currentZones() {
    if (!state.payload) return [];
    return state.payload.zones.filter(z => z.eye === state.eye &&
      (state.activeLayers.size === 0 || state.activeLayers.has(z.germ_layer)));
  }

  function renderChart() {
    if (!state.payload) return;
    const svg = document.getElementById("bm-svg");
    svg.innerHTML = "";
    const mapFn = state.transform ? (p) => state.transform(p) : refToScreen;
    // germ-layer rings (context)
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
    currentZones().forEach(z => {
      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", pointsToPath(arcSectorPoints(z.radial, z.sector), mapFn));
      path.setAttribute("class", "bm-zone");
      path.dataset.id = z.id;
      path.addEventListener("click", () => selectZone(z));
      svg.appendChild(path);
    });
  }

  function selectZone(z) {
    document.querySelectorAll(".bm-zone").forEach(e => e.classList.toggle("bm-sel", e.dataset.id === z.id));
    const panel = document.getElementById("bm-panel");
    panel.replaceChildren();
    const h = document.createElement("h2"); h.textContent = z.anatomy;
    const meta = document.createElement("p");
    const strong = document.createElement("strong"); strong.textContent = z.germ_layer;
    meta.append(strong, document.createTextNode(" layer, " + z.eye + " eye"));
    const body = document.createElement("p"); body.textContent = z.meaning_display || z.meaning_standard;
    panel.append(h, meta, body);
  }

  function renderLayerToggles() {
    const box = document.getElementById("bm-layers"); box.innerHTML = "";
    (state.payload.germ_layers || []).forEach(g => {
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
    state.activeLayers.clear();
    renderLayerToggles(); renderChart();
  }

  function wire() {
    document.getElementById("bm-system").addEventListener("change", e => loadSystem(e.target.value));
    document.getElementById("bm-eye").addEventListener("change", e => { state.eye = e.target.value; renderChart(); });
    loadSystem("iridology").then(__bmSelfCheck);
  }

  function __bmSelfCheck() {
    const up = clockToNormalized(0), right = clockToNormalized(90);
    const okUp = Math.abs(up.x) < 1e-9 && Math.abs(up.y + 1) < 1e-9;
    const okRight = Math.abs(right.x - 1) < 1e-9 && Math.abs(right.y) < 1e-9;
    const okArc = arcSectorPoints({ r_inner: 0.1, r_outer: 0.2 }, { start_deg: 0, end_deg: 10 }).length > 0;
    console.log("[bodymap] selfcheck " + (okUp && okRight && okArc ? "ok" : "FAIL"));
  }

  // expose for tasks/tests
  window.__bm = { clockToNormalized, arcSectorPoints, state };
  document.addEventListener("DOMContentLoaded", wire);
})();
