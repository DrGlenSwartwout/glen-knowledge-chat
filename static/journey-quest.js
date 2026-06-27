/* Journey Quest overlay — Task 3 scaffold (no audio yet). ES5-style IIFE.
   Exposes window.__JQUEST__ = { open, close, state }.
   Hooks .js-mapbtn so the map button opens the scene overlay when questEnabled. */
(function () {
  // Guard: only run when quest feature is enabled
  if (!(window.__SHELL__ && window.__SHELL__.questEnabled)) { return; }
  // Idempotent: never build twice
  if (window.__JQUEST__) { return; }

  var _state = { open: false };
  var _overlay = null;

  function isExternal(href) {
    if (!href) { return false; }
    if (/^(mailto:|tel:|#|javascript:)/i.test(href)) { return false; }
    try { return new URL(href, location.href).origin !== location.origin; }
    catch (e) { return false; }
  }

  function open() {
    if (!_overlay) { return; }
    _overlay.classList.add("open");
    _state.open = true;
  }

  function close() {
    if (!_overlay) { return; }
    _overlay.classList.remove("open");
    _state.open = false;
  }

  // Expose API immediately — before the async fetch resolves — so that
  // shell.js's Promise.all callback can check window.__JQUEST__ if it resolves last.
  window.__JQUEST__ = { open: open, close: close, state: _state };

  function buildOverlay(scene, journeyMap) {
    if (_overlay) { return; }

    // Build href map: home comes from scene.home.href; lands come from /begin/state journey_map
    var hrefMap = {};
    if (scene.home && scene.home.href) { hrefMap["home"] = scene.home.href; }
    (journeyMap || []).forEach(function (card) {
      if (card.key && card.href) { hrefMap[card.key] = card.href; }
    });

    // Overlay backdrop
    var ov = document.createElement("div");
    ov.className = "jq-overlay";

    // Close button
    var closeBtn = document.createElement("button");
    closeBtn.className = "jq-close-btn";
    closeBtn.setAttribute("aria-label", "Close journey map");
    closeBtn.textContent = "×";
    closeBtn.onclick = function () { close(); };

    // Stage (scene image container)
    var stage = document.createElement("div");
    stage.className = "jq-stage";

    // Scene background image
    var img = document.createElement("img");
    img.className = "jq-scene";
    img.src = scene.image;
    img.alt = "Your healing journey";
    stage.appendChild(img);

    // Hotspots — five invisible buttons positioned from scene.hotspots (% coords)
    var order = scene.order || Object.keys(scene.hotspots || {});
    order.forEach(function (key) {
      var spot = (scene.hotspots || {})[key];
      if (!spot) { return; }

      var btn = document.createElement("button");
      btn.className = "jq-hot";
      btn.setAttribute("data-key", key);
      btn.style.left = spot.x + "%";
      btn.style.top = spot.y + "%";
      btn.style.width = spot.w + "%";
      btn.style.height = spot.h + "%";

      var href = hrefMap[key] || "";
      btn.onclick = function () {
        if (!href) { return; }
        if (isExternal(href)) { window.open(href, "_blank", "noopener"); }
        else { location.href = href; }
      };

      stage.appendChild(btn);
    });

    // Close on backdrop click
    ov.onclick = function (e) { if (e.target === ov) { close(); } };

    ov.appendChild(closeBtn);
    ov.appendChild(stage);
    document.body.appendChild(ov);
    _overlay = ov;
  }

  // Fetch map data and (optionally) journey state for hrefs
  Promise.all([
    fetch("/static/shell-map.json")
      .then(function (r) { return r.json(); })
      .catch(function () { return {}; }),
    fetch("/begin/state", { credentials: "same-origin" })
      .then(function (r) { return r.json(); })
      .catch(function () { return {}; })
  ]).then(function (res) {
    var mapCfg = res[0];
    var stateData = res[1];
    var scene = mapCfg.scene;
    if (!scene) { return; }
    var journeyMap = (stateData && stateData.journey_map) || [];
    buildOverlay(scene, journeyMap);

    // Hook mapBtn: covers the race where shell.js's Promise.all resolved FIRST
    // and set the default pavilion listener before __JQUEST__ existed.
    // Using onclick (not addEventListener) so this cleanly overrides.
    var mb = document.querySelector(".js-mapbtn");
    if (mb) {
      mb.onclick = function () { open(); };
    }
  }).catch(function () {});
})();
