/* Journey navigation shell (1a + polish). Vanilla, self-contained, idempotent. */
(function () {
  if (window.__jshellBooted) return;
  window.__jshellBooted = true;
  var MODE = (window.__SHELL__ && window.__SHELL__.mode) || "funnel";
  var TRAIL_KEY = "jshell.trail";

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function isExternal(href) {
    if (!href) return false;
    if (/^(mailto:|tel:|#|javascript:)/i.test(href)) return false;
    try { return new URL(href, location.href).origin !== location.origin; }
    catch (e) { return false; }
  }

  // --- My Path (this-visit trail) ---
  function recordVisit() {
    var trail = [];
    try { trail = JSON.parse(localStorage.getItem(TRAIL_KEY) || "[]"); } catch (e) {}
    var here = { t: document.title || location.pathname, p: location.pathname + location.search };
    if (!trail.length || trail[trail.length - 1].p !== here.p) trail.push(here);
    if (trail.length > 50) trail = trail.slice(-50);
    try { localStorage.setItem(TRAIL_KEY, JSON.stringify(trail)); } catch (e) {}
    return trail;
  }

  // --- external links open in a new tab + get a marker ---
  function tagExternalLinks() {
    document.querySelectorAll("a[href]").forEach(function (a) {
      if (a.dataset.jshellExt) return;
      if (isExternal(a.getAttribute("href"))) {
        a.target = "_blank"; a.rel = "noopener noreferrer";
        a.dataset.jshellExt = "1";
        a.appendChild(el("span", "js-ext-mark", "↗"));
      }
    });
  }

  // --- push the site's own fixed/sticky top bars (theme toggle, page headers)
  //     below the ribbon so nothing tucks underneath it ---
  function deCollideFixed(shellH) {
    function bump(e2, force) {
      if (!e2 || e2.dataset.jshellOffset) return;
      if (e2.id === "journey-shell" || e2.closest("#journey-shell")) return;
      var cs = getComputedStyle(e2);
      if (cs.position !== "fixed" && cs.position !== "sticky") return;
      var r = e2.getBoundingClientRect();
      var ok = r.top < shellH && r.height > 0 && (force || (r.height < 140 && r.width > 36));
      if (ok) {
        e2.dataset.jshellOffset = "1";
        e2.style.top = (parseFloat(cs.top || "0") + shellH) + "px";
      }
    }
    document.querySelectorAll("body > *, header, nav, [class*='header'], [class*='navbar']").forEach(function (e2) { bump(e2, false); });
    // the site's own theme toggle is small + added on DOMContentLoaded — force it
    bump(document.getElementById("rm-theme-toggle"), true);
  }

  // --- ribbon scaffold ---
  function buildRibbon(trail) {
    var bar = el("div"); bar.id = "journey-shell";
    var home = el("button", "js-home", "🏠"); home.title = "Home";
    home.onclick = function () { location.href = "/"; };
    var back = el("button", "js-back", "←"); back.title = "Back";
    back.onclick = function () {
      if (document.referrer && new URL(document.referrer).origin === location.origin) history.back();
      else location.href = "/";
    };
    var mapBtn = el("button", "js-mapbtn", "🗺️"); mapBtn.title = "Open your journey map";
    var path = el("div", "js-path"); path.id = "js-path";
    var mypathBtn = el("button", "js-mypath-btn", "My Path");
    bar.appendChild(home); bar.appendChild(back); bar.appendChild(mapBtn); bar.appendChild(path);

    var mnav = null;
    if (MODE === "member") {
      mnav = el("div", "js-mnav",
        '<a href="/client-portal">Journal</a><a href="/coaching">Coaching</a>' +
        '<a href="/client-portal">Account</a>');
      bar.appendChild(mnav);
    }

    bar.appendChild(mypathBtn);
    document.body.appendChild(bar);
    document.body.classList.add("js-shell-on");

    var drawer = buildMyPath(trail);
    mypathBtn.onclick = function () { drawer.classList.toggle("open"); };
    return { path: path, mapBtn: mapBtn };
  }

  function buildMyPath(trail) {
    var d = el("div", "js-mypath");
    d.appendChild(el("h4", null, "My Path — this visit"));
    trail.slice().reverse().forEach(function (v) {
      var a = el("a", null, v.t); a.href = v.p; d.appendChild(a);
    });
    document.body.appendChild(d);
    return d;
  }

  // --- render the 4 lands from /begin/state journey_map ---
  function renderLands(pathEl, journey, mapCfg) {
    pathEl.innerHTML = "";
    var lands = (mapCfg && mapCfg.lands) || {};
    var cats = (mapCfg && mapCfg.categories) || {};
    var seenNext = false;
    journey.forEach(function (card, i) {
      if (i > 0) {
        var link = el("span", "js-trail-link" + (journey[i - 1].status === "done" ? " done" : ""), "—");
        pathEl.appendChild(link);
      }
      var meta = lands[card.key] || {};
      var icon = (cats[meta.category] || {}).icon || "•";
      var cls = "js-land";
      if (card.status === "done") cls += " done";
      else if (card.status === "next") { cls += " next"; seenNext = true; }
      else if (seenNext) cls += " fog";  // fog upcoming lands beyond the current next
      var land = el("div", cls,
        '<span class="js-icon">' + icon + '</span>' +
        '<span>' + (meta.name || card.label) + '</span>');
      land.title = (meta.intrigue || card.paren || "");
      land.onclick = function (e) { e.stopPropagation(); if (card.href) location.href = card.href; };
      pathEl.appendChild(land);
    });
    // compact "n / total" shown only on mobile (CSS-gated)
    var total = journey.length;
    var nextIdx = journey.map(function (c) { return c.status; }).indexOf("next");
    var current = nextIdx >= 0 ? nextIdx + 1 : total;
    pathEl.appendChild(el("span", "js-count", current + " / " + total));
  }

  // --- expand-to-map overlay: lands open to their pavilions ---
  function buildOverlay(journey, mapCfg) {
    var lands = (mapCfg && mapCfg.lands) || {};
    var cats = (mapCfg && mapCfg.categories) || {};
    var ov = el("div", "js-overlay");
    var close = el("button", "js-overlay-close", "×");
    close.onclick = function () { ov.classList.remove("open"); };
    var inner = el("div", "js-overlay-inner");
    var seenNext = false;
    journey.forEach(function (card) {
      var meta = lands[card.key] || {};
      var icon = (cats[meta.category] || {}).icon || "•";
      var fog = (card.status !== "done" && card.status !== "next" && seenNext);
      if (card.status === "next") seenNext = true;
      var box = el("div", "js-pav-land" + (fog ? " fog" : ""));
      box.appendChild(el("h3", null, icon + " " + (meta.name || card.label)));
      box.appendChild(el("div", "js-intrigue", meta.intrigue || card.paren || ""));
      (card.steps || []).forEach(function (s) {
        var a = el("a", "js-pav" + (s.done ? " done" : ""), s.label);
        a.href = card.href || "#";
        box.appendChild(a);
      });
      inner.appendChild(box);
    });
    ov.appendChild(close); ov.appendChild(inner);
    ov.onclick = function (e) { if (e.target === ov) ov.classList.remove("open"); };
    document.body.appendChild(ov);
    return ov;
  }

  function boot() {
    var trail = recordVisit();
    tagExternalLinks();
    var ui = buildRibbon(trail);
    deCollideFixed(52);
    // re-run after DOMContentLoaded/load so late-added fixed bars (the theme
    // toggle is appended on DOMContentLoaded) also get pushed below the ribbon
    window.addEventListener("load", function () { deCollideFixed(52); });
    setTimeout(function () { deCollideFixed(52); }, 400);
    Promise.all([
      fetch("/begin/state", { credentials: "same-origin" }).then(function (r) { return r.json(); }).catch(function () { return {}; }),
      fetch("/static/shell-map.json").then(function (r) { return r.json(); }).catch(function () { return {}; })
    ]).then(function (res) {
      var journey = (res[0] && res[0].journey_map) || [];
      if (journey.length) {
        renderLands(ui.path, journey, res[1]);
        var overlay = buildOverlay(journey, res[1]);
        ui.mapBtn.addEventListener("click", function () { overlay.classList.add("open"); });
      } else ui.path.appendChild(el("span", "js-land", "illtowell.com"));
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
