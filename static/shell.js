/* Journey navigation shell (1a). Vanilla, self-contained, idempotent. */
(function () {
  if (window.__jshellBooted) return;
  window.__jshellBooted = true;
  var MODE = (window.__SHELL__ && window.__SHELL__.mode) || "funnel";
  var TRAIL_KEY = "jshell.trail";
  var REWARDS = !!(window.__SHELL__ && window.__SHELL__.rewards1b);
  var REWARDS_GIFT = !!(window.__SHELL__ && window.__SHELL__.rewardsGift);

  // --- Light/Dark theme (shared mechanism: localStorage 'rm-theme' + <html data-theme>).
  //     The ribbon is theme-aware and owns its own toggle; theme-toggle.js suppresses its
  //     floating button when the ribbon (window.__SHELL__) is present. ---
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  function applyShellTheme() {
    try { applyTheme(localStorage.getItem("rm-theme")); } catch (e) {}
    // Inject the light-palette override only if no other owner already did (theme-toggle.js
    // or op-nav.js). Keep in sync with static/theme-toggle.js.
    if (!document.getElementById("rm-theme-style") && !document.getElementById("op-nav-theme-style")) {
      var s = document.createElement("style"); s.id = "rm-theme-style";
      s.textContent = ':root[data-theme="light"]{' +
        '--bg:#FBF8F3;--bg-2:#F4ECDE;--surface:#FFFFFF;--surface-2:#F4ECDE;--surface2:#F4ECDE;' +
        '--border:#E2D9C9;--rule:#E2D9C9;--hair:#E2D9C9;--cream:#1E2A2A;' +
        '--muted:#5F6B6B;--dim:#5F6B6B;--gray:#5F6B6B;--ink:#1E2A2A;--ink-2:#5F6B6B;--ink-3:#7A8585;' +
        '--gold:#B08A3E;--gold-soft:#EFE4C8;--green:#2D7A6A;--panel:#FFFFFF;--panel-2:#F4ECDE;' +
        '--text:#1E2A2A;--text-muted:#5F6B6B;--accent:#B08A3E;--accent2:#2D7A6A;}';
      (document.head || document.documentElement).appendChild(s);
    }
  }

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

    if (MODE === "member") {
      var mnav = el("div", "js-mnav",
        '<a href="/client-portal">Journal</a><a href="/coaching">Coaching</a>' +
        '<a href="/client-portal">Account</a>');
      var toggle = el("button", "js-maptoggle", "🗺️"); toggle.title = "Map / nav";
      bar.insertBefore(toggle, home);  // unobtrusive upper-left toggle
      bar.appendChild(mnav);
      toggle.onclick = function () { path.classList.toggle("js-hide"); mnav.classList.toggle("js-hide"); };
    }

    bar.appendChild(mypathBtn);

    if (REWARDS) {
      var orb = el("span", "js-orb"); orb.setAttribute("data-lit", "0"); orb.title = "Your biofield";
      bar.appendChild(orb);
      var walletBtn = el("button", "js-mypath-btn", "Wallet");
      bar.appendChild(walletBtn);
      var wp = el("div", "js-mypath"); wp.id = "js-wallet";
      wp.appendChild(el("h4", null, "Your offers"));
      var walletBody = el("div"); walletBody.id = "js-wallet-body"; wp.appendChild(walletBody);
      document.body.appendChild(wp);
      walletBtn.onclick = function () { wp.classList.toggle("open"); };
    }

    // Light/Dark toggle — the ribbon owns the theme control on public pages.
    var themeBtn = el("button", "js-theme", "☀"); themeBtn.title = "Light / Dark";
    function relabelTheme() {
      themeBtn.textContent = document.documentElement.getAttribute("data-theme") === "light" ? "☾" : "☀";
    }
    relabelTheme();
    themeBtn.onclick = function () {
      var next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      applyTheme(next);
      try { localStorage.setItem("rm-theme", next); } catch (e) {}
      relabelTheme();
    };
    bar.appendChild(themeBtn);
    window.addEventListener("storage", function (e) {
      if (e.key !== "rm-theme") return;
      applyTheme(e.newValue); relabelTheme();
    });

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
      land.onclick = function (e) {
        e.stopPropagation();
        if (!card.href) return;
        // External destinations (e.g. the E4L Listening Pool) open in a new tab;
        // internal ones navigate in place.
        if (isExternal(card.href)) window.open(card.href, "_blank", "noopener");
        else location.href = card.href;
      };
      pathEl.appendChild(land);
    });
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
      if (REWARDS && meta.featured) {
        var fb = el("div", "js-pav-featured",
          '<div class="js-fname">' + meta.featured.product_name + '</div>' +
          '<div class="js-fpower">' + meta.featured.healing_power + '</div>');
        var claim = el("button", "js-claim", "Claim 15% off");
        if (card.status !== "done") { claim.disabled = true; claim.textContent = "Complete this step to unlock"; }
        claim.onclick = function () { claimCoupon(card.key, claim); };
        fb.appendChild(claim);
        box.appendChild(fb);
      }
      if (REWARDS_GIFT && card.key === "give") {
        var gb = el("button", "js-claim", "Unlock gifting");
        gb.onclick = function () {
          gb.disabled = true; gb.textContent = "Activating…";
          fetch("/api/journey/activate-gifting", {method: "POST", credentials: "same-origin",
            headers: {"Content-Type": "application/json"}, body: "{}"})
            .then(function (r) { return r.json().then(function (j) { return {s: r.status, j: j}; }); })
            .then(function (res) {
              if (res.s === 200) { gb.textContent = "✓ Gifting unlocked"; refreshWallet(); }
              else if (res.j && res.j.needs === "email_tos") { location.href = "/begin/match"; }
              else { gb.disabled = false; gb.textContent = "Unlock gifting"; }
            }).catch(function () { gb.disabled = false; gb.textContent = "Unlock gifting"; });
        };
        box.appendChild(gb);
      }
      (card.steps || []).forEach(function (s) {
        var a = el("a", "js-pav" + (s.done ? " done" : ""), s.label);
        a.href = card.href || "#";
        if (isExternal(a.getAttribute("href"))) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
        box.appendChild(a);
      });
      inner.appendChild(box);
    });
    ov.appendChild(close); ov.appendChild(inner);
    ov.onclick = function (e) { if (e.target === ov) ov.classList.remove("open"); };
    document.body.appendChild(ov);
    return ov;
  }

  function claimCoupon(land, btn) {
    var orig = btn.textContent;
    btn.disabled = true; btn.textContent = "Claiming…";
    fetch("/api/journey/claim-coupon", {
      method: "POST", credentials: "same-origin",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({land: land})
    }).then(function (r) { return r.json().then(function (j) { return {s: r.status, j: j}; }); })
      .then(function (res) {
        if (res.s === 200) { btn.textContent = "✓ In your wallet"; refreshWallet(); }
        else if (res.j && res.j.needs === "email_tos") { btn.disabled = false; location.href = "/begin/match"; }
        else { btn.disabled = false; btn.textContent = orig; }
      }).catch(function () { btn.disabled = false; btn.textContent = "Claim 15% off"; });
  }

  function refreshWallet() {
    if (!REWARDS) return;
    fetch("/api/journey/wallet", {credentials: "same-origin"})
      .then(function (r) { return r.json(); })
      .then(function (j) {
        var coupons = (j && j.coupons) || [];
        var orb = document.querySelector("#journey-shell .js-orb");
        if (orb) orb.setAttribute("data-lit", String(Math.min(coupons.length, 3)));
        var panel = document.getElementById("js-wallet-body");
        if (panel) {
          panel.innerHTML = coupons.length ? "" : "<p class='js-fpower'>No offers yet — complete a step to earn one.</p>";
          coupons.forEach(function (c) {
            panel.appendChild(el("div", "js-wallet-coupon",
              "<b>15% off</b> " + c.product_slug +
              "<div class='js-exp'>expires " + (c.expires_at || "").slice(0, 10) + "</div>"));
          });
          var gifts = (j && j.gifts) || [];
          gifts.forEach(function (g) {
            var row = el("div", "js-wallet-coupon",
              "<b>Gift 15% off</b> " + g.product_slug + " <span class='js-exp'>to a friend</span>");
            var share = el("button", "js-claim", "Share");
            share.onclick = function () {
              var url = location.origin + g.share_url;
              (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject())
                .then(function () { share.textContent = "✓ Link copied"; })
                .catch(function () { share.textContent = url; });
            };
            row.appendChild(share);
            panel.appendChild(row);
          });
        }
      }).catch(function () {});
  }

  function boot() {
    // The journey ribbon is a CUSTOMER wayfinding bar. Suppress it inside an embedded
    // frame (e.g. the funnel chat iframe), and on internal ops pages — those carry the
    // GLEN·OPS bar (op-nav.js), which is their header and owns their theme toggle.
    // Without this, the ribbon stacks on top of op-nav as a duplicate header.
    try { if (window.top !== window.self) return; } catch (e) { return; }
    if (document.querySelector(".op-nav-bar")) return;
    applyShellTheme();
    var trail = recordVisit();
    tagExternalLinks();
    var ui = buildRibbon(trail);
    Promise.all([
      fetch("/begin/state", { credentials: "same-origin" }).then(function (r) { return r.json(); }).catch(function () { return {}; }),
      fetch("/static/shell-map.json").then(function (r) { return r.json(); }).catch(function () { return {}; })
    ]).then(function (res) {
      var journey = (res[0] && res[0].journey_map) || [];
      if (journey.length) {
        renderLands(ui.path, journey, res[1]);
        var overlay = buildOverlay(journey, res[1]);
        ui.mapBtn.addEventListener("click", function () { overlay.classList.add("open"); });
        refreshWallet();
      } else ui.path.appendChild(el("span", "js-land", "illtowell.com"));
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
