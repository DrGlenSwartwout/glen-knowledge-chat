// Course Body-Map concern-marking widget (ash-certification, 02-body module homework).
//
// Self-contained, dependency-free client widget. Fetches the SAME public data
// endpoint the clinical portal Body Map uses (GET /body-map/data?system=<s>)
// and lets a learner multi-select "areas of concern" with an optional note
// per area, plus one overall reflection note, then POSTs a small JSON
// payload (system + zone ids + notes only -- never a token, cookie, or
// photo) to a homework submit endpoint.
//
// Rendering contract: a single <svg viewBox="0 0 600 600"> containing one
// <g transform="scale(600)">. The outline path and every zone shape are
// drawn inside that group in their native [0,1] normalized coordinates, so
// the group's scale(600) does all the math -- no path-coordinate parsing
// needed here. Every stroked shape uses vector-effect="non-scaling-stroke"
// with a small stroke width so strokes read the same regardless of the
// 600x scale.
//
// This file is entirely separate from static/body-map.js (the clinical
// portal's Body Map) and must never be confused with it or wired into it.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  // The ten whole-body systems this homework tool covers, in menu order.
  var SYSTEMS = [
    ["organs", "Organs"],
    ["skeleton", "Skeleton"],
    ["muscle", "Muscle"],
    ["nervous", "Nervous"],
    ["endocrine", "Endocrine"],
    ["respiratory", "Respiratory"],
    ["digestive", "Digestive"],
    ["cardiovascular", "Cardiovascular"],
    ["urogenital", "Urogenital"],
    ["lymph", "Lymph"]
  ];

  var STYLE_ID = "cbm-style-tag";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent =
      ".cbm-wrap{font-family:inherit;max-width:920px;}" +
      ".cbm-topbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px;}" +
      ".cbm-system-select{margin-left:6px;padding:4px 8px;}" +
      ".cbm-view-toggle{display:flex;gap:6px;}" +
      ".cbm-view-btn{padding:4px 12px;border:1px solid #999;background:#fff;border-radius:4px;cursor:pointer;}" +
      ".cbm-view-btn.cbm-active{background:#2f6f5e;color:#fff;border-color:#2f6f5e;}" +
      ".cbm-body{display:flex;gap:24px;flex-wrap:wrap;}" +
      ".cbm-map-col{flex:0 0 auto;}" +
      ".cbm-svg{width:320px;height:320px;background:#fafafa;border:1px solid #ddd;border-radius:6px;}" +
      ".cbm-hint{font-size:0.85em;color:#666;max-width:320px;}" +
      ".cbm-list-col{flex:1 1 260px;min-width:240px;}" +
      ".cbm-marks-list{display:flex;flex-direction:column;gap:8px;}" +
      ".cbm-mark-row{display:flex;flex-direction:column;gap:4px;padding:8px;border:1px solid #e0e0e0;border-radius:6px;background:#fff;}" +
      ".cbm-mark-label{font-weight:600;}" +
      ".cbm-mark-note{padding:4px 6px;}" +
      ".cbm-mark-clear{align-self:flex-start;background:none;border:none;color:#b03a2e;cursor:pointer;padding:0;font-size:0.85em;}" +
      ".cbm-empty-msg{color:#777;font-size:0.9em;}" +
      ".cbm-overall-label{display:block;margin-top:18px;font-weight:600;}" +
      ".cbm-overall-note{display:block;margin-top:6px;width:100%;max-width:600px;min-height:70px;padding:8px;box-sizing:border-box;}" +
      ".cbm-msg{color:#a83232;min-height:1.2em;}" +
      ".cbm-save-btn{padding:8px 18px;border-radius:6px;border:none;background:#2f6f5e;color:#fff;cursor:pointer;font-size:1em;}" +
      ".cbm-save-btn:disabled{opacity:0.6;cursor:default;}" +
      ".cbm-result{min-height:1.2em;color:#2f6f5e;}" +
      ".cbm-zone{fill:rgba(80,120,190,0.18);stroke:#4a6fa5;stroke-width:1;cursor:pointer;}" +
      ".cbm-zone:hover{fill:rgba(80,120,190,0.32);}" +
      ".cbm-zone-marked{fill:rgba(214,88,61,0.55);stroke:#a8321f;stroke-width:2.5;}" +
      ".cbm-outline{pointer-events:none;}";
    document.head.appendChild(style);
  }

  // Pure by design: takes a plain state object ({system, marks, note}) and
  // returns { payload: "<json string>" }. marks may be a Map (zone id ->
  // {anatomy, note}) or a plain object keyed the same way. Never reads or
  // writes a token, cookie, or photo -- only system + zone ids + notes.
  function _serialize(state) {
    state = state || {};
    var entries;
    if (state.marks instanceof Map) {
      entries = Array.from(state.marks.entries());
    } else {
      var src = state.marks || {};
      entries = Object.keys(src).map(function (k) { return [k, src[k]]; });
    }
    var marks = entries.map(function (pair) {
      var zone = pair[0], m = pair[1] || {};
      return { zone: zone, anatomy: m.anatomy || "", note: m.note || "" };
    });
    var out = {
      system: state.system || "organs",
      marks: marks,
      note: state.note || ""
    };
    return { payload: JSON.stringify(out) };
  }

  function zoneVisible(z, view) {
    if (z.bilateral) return true;
    if (!z.side) return true;
    return z.side === view;
  }

  function drawShape(z) {
    var geo = z.geometry || {};
    var el;
    if (geo.type === "ellipse") {
      el = document.createElementNS(SVG_NS, "ellipse");
      el.setAttribute("cx", geo.cx);
      el.setAttribute("cy", geo.cy);
      el.setAttribute("rx", geo.rx);
      el.setAttribute("ry", geo.ry);
    } else if (geo.type === "point") {
      el = document.createElementNS(SVG_NS, "circle");
      el.setAttribute("cx", geo.x);
      el.setAttribute("cy", geo.y);
      el.setAttribute("r", "0.008");
    } else if (geo.type === "path") {
      el = document.createElementNS(SVG_NS, "path");
      el.setAttribute("d", geo.d);
    } else {
      return null;
    }
    el.setAttribute("vector-effect", "non-scaling-stroke");
    el.setAttribute("class", "cbm-zone");
    el.setAttribute("data-zone", z.id);
    return el;
  }

  function mount(root, opts) {
    ensureStyle();
    opts = opts || {};
    var submitUrl = opts.submitUrl || "";
    var initialSystem = opts.system || "organs";

    var state = {
      system: initialSystem,
      marks: new Map(),   // zone id -> { anatomy, note }
      note: "",
      view: "front",
      payload: null        // last /body-map/data response
    };

    // Parse priorPayload once up front; applied to state after the matching
    // system's zones have loaded (so anatomy/ids can be cross-checked).
    var priorParsed = null;
    if (opts.priorPayload) {
      try {
        var parsedPrior = JSON.parse(opts.priorPayload);
        if (parsedPrior && typeof parsedPrior === "object" &&
            typeof parsedPrior.system === "string" &&
            Array.isArray(parsedPrior.marks)) {
          priorParsed = parsedPrior;
          initialSystem = parsedPrior.system;
          state.system = initialSystem;
        }
      } catch (e) {
        priorParsed = null;
      }
    }

    root.textContent = "";

    var wrap = document.createElement("div");
    wrap.className = "cbm-wrap";

    // ---- top bar: system select + front/back toggle ----
    var topBar = document.createElement("div");
    topBar.className = "cbm-topbar";

    var selectLabel = document.createElement("label");
    selectLabel.textContent = "Body system:";
    var select = document.createElement("select");
    select.className = "cbm-system-select";
    SYSTEMS.forEach(function (pair) {
      var opt = document.createElement("option");
      opt.value = pair[0];
      opt.textContent = pair[1];
      select.appendChild(opt);
    });
    select.value = initialSystem;
    selectLabel.appendChild(select);
    topBar.appendChild(selectLabel);

    var viewToggle = document.createElement("div");
    viewToggle.className = "cbm-view-toggle";
    viewToggle.style.display = "none";
    var frontBtn = document.createElement("button");
    frontBtn.type = "button";
    frontBtn.textContent = "Front";
    frontBtn.className = "cbm-view-btn cbm-active";
    var backBtn = document.createElement("button");
    backBtn.type = "button";
    backBtn.textContent = "Back";
    backBtn.className = "cbm-view-btn";
    viewToggle.appendChild(frontBtn);
    viewToggle.appendChild(backBtn);
    topBar.appendChild(viewToggle);

    wrap.appendChild(topBar);

    // ---- body: map + marked-areas list ----
    var body = document.createElement("div");
    body.className = "cbm-body";

    var mapCol = document.createElement("div");
    mapCol.className = "cbm-map-col";

    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 600 600");
    svg.setAttribute("class", "cbm-svg");
    var g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("transform", "scale(600)");
    svg.appendChild(g);
    mapCol.appendChild(svg);

    var hint = document.createElement("p");
    hint.className = "cbm-hint";
    hint.textContent = "Tap an area to mark it as a concern. Tap it again to clear it.";
    mapCol.appendChild(hint);

    body.appendChild(mapCol);

    var listCol = document.createElement("div");
    listCol.className = "cbm-list-col";
    var listHeading = document.createElement("h4");
    listHeading.textContent = "Marked areas";
    listCol.appendChild(listHeading);
    var list = document.createElement("div");
    list.className = "cbm-marks-list";
    listCol.appendChild(list);
    var emptyMsg = document.createElement("p");
    emptyMsg.className = "cbm-empty-msg";
    emptyMsg.textContent = "Nothing marked yet.";
    listCol.appendChild(emptyMsg);

    body.appendChild(listCol);
    wrap.appendChild(body);

    // ---- overall reflection ----
    var overallId = "cbm-overall-" + Math.random().toString(36).slice(2);
    var overallLabel = document.createElement("label");
    overallLabel.setAttribute("for", overallId);
    overallLabel.className = "cbm-overall-label";
    overallLabel.textContent = "Anything else you want to note about your body right now?";
    wrap.appendChild(overallLabel);

    var overallNote = document.createElement("textarea");
    overallNote.id = overallId;
    overallNote.className = "cbm-overall-note";
    wrap.appendChild(overallNote);

    // ---- messages + save ----
    var msgLine = document.createElement("p");
    msgLine.className = "cbm-msg";
    wrap.appendChild(msgLine);

    var saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "cbm-save-btn";
    saveBtn.textContent = "Save my body map";
    wrap.appendChild(saveBtn);

    var resultLine = document.createElement("p");
    resultLine.className = "cbm-result";
    wrap.appendChild(resultLine);

    root.appendChild(wrap);

    // ---------------- behavior ----------------

    function anatomyFor(zoneId) {
      var zones = (state.payload && state.payload.zones) || [];
      for (var i = 0; i < zones.length; i++) {
        if (zones[i].id === zoneId) return zones[i].anatomy || zoneId;
      }
      return zoneId;
    }

    function clearMessages() {
      msgLine.textContent = "";
    }

    function renderMap() {
      while (g.firstChild) g.removeChild(g.firstChild);
      var payload = state.payload;
      if (!payload) return;

      var outlineD = "";
      if (payload.outlines && payload.outlines[state.view]) {
        outlineD = payload.outlines[state.view];
      } else if (payload.outline) {
        outlineD = payload.outline;
      }
      if (outlineD) {
        var outlineEl = document.createElementNS(SVG_NS, "path");
        outlineEl.setAttribute("d", outlineD);
        outlineEl.setAttribute("fill", "none");
        outlineEl.setAttribute("stroke", "#333");
        outlineEl.setAttribute("stroke-width", "1.5");
        outlineEl.setAttribute("vector-effect", "non-scaling-stroke");
        outlineEl.setAttribute("class", "cbm-outline");
        g.appendChild(outlineEl);
      }

      (payload.zones || []).forEach(function (z) {
        if (!zoneVisible(z, state.view)) return;
        var el = drawShape(z);
        if (!el) return;
        if (state.marks.has(z.id)) el.classList.add("cbm-zone-marked");
        el.addEventListener("click", function () { toggleZone(z.id, z.anatomy); });
        g.appendChild(el);
      });
    }

    function renderList() {
      list.textContent = "";
      var hasMarks = state.marks.size > 0;
      emptyMsg.style.display = hasMarks ? "none" : "";
      state.marks.forEach(function (m, zoneId) {
        var row = document.createElement("div");
        row.className = "cbm-mark-row";

        var rowLabel = document.createElement("div");
        rowLabel.className = "cbm-mark-label";
        rowLabel.textContent = m.anatomy || zoneId; // textContent only, never innerHTML
        row.appendChild(rowLabel);

        var noteInput = document.createElement("input");
        noteInput.type = "text";
        noteInput.className = "cbm-mark-note";
        noteInput.placeholder = "A quick note for this area (optional)";
        noteInput.value = m.note || "";
        noteInput.addEventListener("input", function () {
          m.note = noteInput.value;
        });
        row.appendChild(noteInput);

        var clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "cbm-mark-clear";
        clearBtn.textContent = "Clear this mark";
        clearBtn.addEventListener("click", function () { toggleZone(zoneId, m.anatomy); });
        row.appendChild(clearBtn);

        list.appendChild(row);
      });
    }

    function toggleZone(zoneId, anatomy) {
      if (state.marks.has(zoneId)) {
        state.marks.delete(zoneId);
      } else {
        state.marks.set(zoneId, { anatomy: anatomy || anatomyFor(zoneId), note: "" });
      }
      renderMap();
      renderList();
      clearMessages();
    }

    function applyPriorMarks() {
      if (!priorParsed || priorParsed.system !== state.system) return;
      state.marks = new Map();
      (priorParsed.marks || []).forEach(function (m) {
        if (!m || typeof m.zone !== "string") return;
        state.marks.set(m.zone, {
          anatomy: m.anatomy || anatomyFor(m.zone),
          note: typeof m.note === "string" ? m.note : ""
        });
      });
      state.note = typeof priorParsed.note === "string" ? priorParsed.note : "";
      overallNote.value = state.note;
      priorParsed = null; // applied once
    }

    function setView(view) {
      state.view = view;
      frontBtn.classList.toggle("cbm-active", view === "front");
      backBtn.classList.toggle("cbm-active", view === "back");
      renderMap();
    }

    function loadSystem(system) {
      select.value = system;
      state.system = system;
      clearMessages();

      fetch("/body-map/data?system=" + encodeURIComponent(system))
        .then(function (r) { return r.json(); })
        .then(function (data) {
          state.payload = data;
          var hasBack = !!(data.outlines && data.outlines.back);
          viewToggle.style.display = hasBack ? "" : "none";
          setView("front");
          applyPriorMarks();
          renderMap();
          renderList();
        })
        .catch(function () {
          msgLine.textContent = "We could not load this body map right now. Please try again.";
        });
    }

    // ---------------- events ----------------

    select.addEventListener("change", function () {
      var next = select.value;
      if (state.marks.size > 0) {
        var proceed = window.confirm(
          "Switching body systems will clear your current marks for this homework. Continue?"
        );
        if (!proceed) {
          select.value = state.system;
          return;
        }
      }
      state.marks = new Map();
      state.note = "";
      overallNote.value = "";
      resultLine.textContent = "";
      loadSystem(next);
    });

    frontBtn.addEventListener("click", function () { setView("front"); });
    backBtn.addEventListener("click", function () { setView("back"); });

    overallNote.addEventListener("input", function () {
      state.note = overallNote.value;
    });

    saveBtn.addEventListener("click", function () {
      clearMessages();
      resultLine.textContent = "";
      var noteEmpty = !(state.note && state.note.trim());
      if (state.marks.size === 0 && noteEmpty) {
        msgLine.textContent = "Please mark at least one area or add a note.";
        return;
      }

      var body2 = _serialize(state);
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving...";

      fetch(submitUrl + location.search, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body2)
      })
        .then(function (r) {
          return r.json().then(function (d) { return { ok: r.ok, d: d }; });
        })
        .then(function (res) {
          saveBtn.disabled = false;
          saveBtn.textContent = "Save my body map";
          if (res.d && res.d.ok) {
            var pieces = [];
            if (res.d.rating) pieces.push(res.d.rating);
            if (res.d.feedback) pieces.push(res.d.feedback);
            resultLine.textContent = pieces.length ? pieces.join(". ") : "Saved. Thank you.";
          } else {
            resultLine.textContent = "We could not save that just now. Please try again in a moment.";
          }
        })
        .catch(function () {
          saveBtn.disabled = false;
          saveBtn.textContent = "Save my body map";
          resultLine.textContent = "We could not save that just now. Please try again in a moment.";
        });
    });

    // ---------------- init ----------------
    loadSystem(initialSystem);
  }

  window.CourseBodyMap = {
    mount: mount,
    _serialize: _serialize
  };
})();
