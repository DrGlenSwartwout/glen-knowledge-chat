/* Knowledge Atlas module. Mount:
 *   <div id="rm-atlas"></div>
 *   <script src="/atlas.js" data-target="#rm-atlas" data-mode="compact"></script>
 * Or full page: <script src="/atlas.js" data-target="#root" data-mode="full"></script>
 */
(function () {
  "use strict";
  var script = document.currentScript;
  var origin = script.src.replace(/\/atlas\.js.*$/, "");
  var mountSel = script.getAttribute("data-target") || "#rm-atlas";
  var startMode = script.getAttribute("data-mode") || "compact";

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function Atlas(root, mode) {
    this.root = root; this.mode = mode;
    this.state = { concepts: [], byId: {}, hierarchy: {}, view: "map", selectedId: null };
    this.build();
    this.load();
  }

  Atlas.prototype.build = function () {
    this.root.className = "rm-atlas rm-atlas--" + this.mode;
    this.root.innerHTML = "";
    var bar = el("div", "rm-atlas__bar");
    var self = this;
    [["map", "◉ Map"], ["az", "⊞ A–Z"], ["hier", "⋔ Hierarchy"]].forEach(function (t) {
      var b = el("button", "rm-atlas__tab" + (t[0] === self.state.view ? " rm-atlas__tab--on" : ""), t[1]);
      b.onclick = function () { self.setView(t[0]); };
      b.dataset.view = t[0];
      bar.appendChild(b);
    });
    var exp = el("button", "rm-atlas__expand", this.mode === "full" ? "⤡" : "⤢");
    exp.onclick = function () { self.toggleExpand(); };
    bar.appendChild(exp);
    this.bar = bar;

    var body = el("div", "rm-atlas__body");
    this.viewEl = el("div", "rm-atlas__view");
    this.drawer = el("div", "rm-drawer", "<h4>Select a concept</h4>");
    body.appendChild(this.viewEl); body.appendChild(this.drawer);

    var chat = el("div", "rm-chat");
    this.answerEl = el("div", "rm-chat__answer");
    var input = el("input", "rm-chat__input");
    input.placeholder = "Ask — e.g. what helps night vision?";
    input.onkeydown = function (e) { if (e.key === "Enter") self.ask(input.value); };
    chat.appendChild(this.answerEl); chat.appendChild(input);

    this.root.appendChild(bar); this.root.appendChild(body); this.root.appendChild(chat);
  };

  Atlas.prototype.load = function () {
    var self = this;
    fetch(origin + "/atlas/data").then(function (r) { return r.json(); }).then(function (g) {
      self.state.concepts = g.concepts || [];
      self.state.hierarchy = g.hierarchy || {};
      self.state.byId = {};
      self.state.concepts.forEach(function (c) { self.state.byId[c.id] = c; });
      self.render();
    }).catch(function () { self.viewEl.innerHTML = "<p style='padding:16px'>Atlas is being built.</p>"; });
  };

  Atlas.prototype.setView = function (v) {
    this.state.view = v;
    [].forEach.call(this.bar.querySelectorAll(".rm-atlas__tab"), function (b) {
      b.classList.toggle("rm-atlas__tab--on", b.dataset.view === v);
    });
    this.render();
  };

  Atlas.prototype.render = function () {
    if (this.state.view === "map") this.renderMap();
    else if (this.state.view === "az") this.renderAZ();
    else this.renderHierarchy();
    this.renderDrawer();
  };

  Atlas.prototype.renderMap = function () {
    var self = this, W = 600, H = 360;
    var svg = '<svg class="rm-atlas__map" viewBox="0 0 ' + W + ' ' + H + '">';
    var pos = {};
    this.state.concepts.forEach(function (c) {
      pos[c.id] = { x: 20 + c.coords.x * (W - 40), y: 20 + c.coords.y * (H - 40) };
    });
    this.state.concepts.forEach(function (c) {
      (c.neighbors || []).forEach(function (nid) {
        if (pos[nid]) svg += '<line x1="' + pos[c.id].x + '" y1="' + pos[c.id].y +
          '" x2="' + pos[nid].x + '" y2="' + pos[nid].y + '" stroke="#21472d"/>';
      });
    });
    this.state.concepts.forEach(function (c) {
      var sel = c.id === self.state.selectedId ? " rm-node--sel" : "";
      svg += '<g class="rm-node' + sel + '" data-id="' + c.id + '">' +
        '<circle cx="' + pos[c.id].x + '" cy="' + pos[c.id].y + '" r="7" fill="#3d8a52"/>' +
        '<text x="' + (pos[c.id].x + 10) + '" y="' + (pos[c.id].y + 4) + '">' + c.label + '</text></g>';
    });
    svg += "</svg>";
    this.viewEl.innerHTML = svg;
    [].forEach.call(this.viewEl.querySelectorAll(".rm-node"), function (g) {
      g.onclick = function () { self.select(g.dataset.id); };
    });
  };

  Atlas.prototype.renderAZ = function () {
    var self = this, items = this.state.concepts.slice().sort(function (a, b) {
      return a.label.localeCompare(b.label);
    });
    var ul = el("ul", "rm-list"), letter = "";
    items.forEach(function (c) {
      var L = c.label[0].toUpperCase();
      if (L !== letter) { letter = L; ul.appendChild(el("li", "rm-list__letter", L)); }
      var li = el("li", c.id === self.state.selectedId ? "rm-list--sel" : "", c.label);
      li.onclick = function () { self.select(c.id); };
      ul.appendChild(li);
    });
    this.viewEl.innerHTML = ""; this.viewEl.appendChild(ul);
  };

  Atlas.prototype.renderHierarchy = function () {
    var self = this, ul = el("ul", "rm-list");
    Object.keys(this.state.hierarchy).sort().forEach(function (group) {
      ul.appendChild(el("li", "rm-list__letter", group));
      self.state.hierarchy[group].forEach(function (cid) {
        var c = self.state.byId[cid]; if (!c) return;
        var li = el("li", cid === self.state.selectedId ? "rm-list--sel" : "", "&nbsp;&nbsp;" + c.label);
        li.onclick = function () { self.select(cid); };
        ul.appendChild(li);
      });
    });
    this.viewEl.innerHTML = ""; this.viewEl.appendChild(ul);
  };

  Atlas.prototype.renderDrawer = function () {
    var c = this.state.byId[this.state.selectedId];
    if (!c) { this.drawer.innerHTML = "<h4>Select a concept</h4>"; return; }
    var icon = { video: "▦", article: "📄", product: "🧪" };
    var links = (c.links || []).map(function (l) {
      return '<a href="' + l.url + '" target="_blank" rel="noopener">' +
        (icon[l.type] || "•") + " " + (l.title || l.type) + "</a>";
    }).join("");
    this.drawer.innerHTML = "<h4>" + c.label + "</h4><div>" + (c.summary || "") + "</div>" +
      '<div style="margin-top:8px">' + (links || "<span>No links yet.</span>") + "</div>";
  };

  Atlas.prototype.select = function (id) {
    this.state.selectedId = id;
    this.render();
  };

  Atlas.prototype.highlight = function (ids) {
    var set = {}; ids.forEach(function (i) { set[i] = 1; });
    [].forEach.call(this.viewEl.querySelectorAll(".rm-node"), function (g) {
      g.classList.toggle("rm-node--hi", !!set[g.dataset.id]);
    });
    if (ids.length) this.select(ids[0]);
  };

  Atlas.prototype.ask = function (q) {
    if (!q || !q.trim()) return;
    var self = this;
    this.answerEl.textContent = "…";
    fetch(origin + "/atlas/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    }).then(function (r) { return r.json(); }).then(function (res) {
      self.renderAnswer(res.answer, res.concept_ids || []);
      self.highlight(res.highlight || []);
    }).catch(function () { self.answerEl.textContent = "See the highlighted concepts on the map."; });
  };

  Atlas.prototype.renderAnswer = function (text, ids) {
    var self = this;
    this.answerEl.innerHTML = "";
    this.answerEl.appendChild(document.createTextNode("Atlas: " + (text || "")));
    if (ids.length) {
      this.answerEl.appendChild(document.createTextNode("  ["));
      ids.forEach(function (id, i) {
        var c = self.state.byId[id]; if (!c) return;
        var span = el("span", "rm-term", c.label);
        span.onclick = function () { self.select(id); if (self.state.view !== "map") self.setView("map"); self.highlight([id]); };
        self.answerEl.appendChild(span);
        if (i < ids.length - 1) self.answerEl.appendChild(document.createTextNode(", "));
      });
      this.answerEl.appendChild(document.createTextNode("]"));
    }
  };

  Atlas.prototype.toggleExpand = function () {
    this.mode = this.mode === "full" ? "compact" : "full";
    this.build();           // rebuild chrome (expand glyph + class)
    this.render();
  };

  function mount() {
    var target = document.querySelector(mountSel);
    if (!target) { target = el("div"); script.parentNode.insertBefore(target, script.nextSibling); }
    new Atlas(target, startMode);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
