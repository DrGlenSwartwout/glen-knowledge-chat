/* Journey Quest overlay — Task 4: ordered progress + ribbon/hotspot lock states.
   ES5-style IIFE. Exposes window.__JQUEST__ = { open, close, state, curIdx, render }.
   Quest state persisted to localStorage["jquest.v1"].
   Ordering enforced: clicking a non-current hotspot is a gentle no-op.
   Hooks .js-mapbtn so the map button opens the scene overlay when questEnabled. */
(function () {
  // Guard: only run when quest feature is enabled
  if (!(window.__SHELL__ && window.__SHELL__.questEnabled)) { return; }
  // Idempotent: never build twice
  if (window.__JQUEST__) { return; }

  var LS = "jquest.v1";
  var _overlayOpen = false;
  var _overlay = null;
  var _order = [];   // populated from scene.order after fetch

  // --- Quest progress state ---
  // Per-key {found:bool, done:bool} for each stage in scene.order.
  // Also carries a paths array (Task 6 fills it).
  var _qs = { paths: [] };

  function loadQS() {
    try {
      var raw = localStorage.getItem(LS);
      var parsed = raw ? JSON.parse(raw) : null;
      if (parsed && typeof parsed === "object") {
        _qs = parsed;
        if (!_qs.paths) { _qs.paths = []; }
      }
    } catch (e) { /* malformed — start fresh */ }
  }

  function saveQS() {
    try { localStorage.setItem(LS, JSON.stringify(_qs)); } catch (e) {}
  }

  function ensureKeys(order) {
    var i;
    for (i = 0; i < order.length; i++) {
      if (!_qs[order[i]]) { _qs[order[i]] = { found: false, done: false }; }
    }
  }

  // curIdx: index of first key in _order whose done is false; -1 if all done
  function curIdx() {
    var i;
    for (i = 0; i < _order.length; i++) {
      if (!(_qs[_order[i]] && _qs[_order[i]].done)) { return i; }
    }
    return -1;
  }

  // Apply one of the three lock-state classes to an element
  function setLockClass(el, lockState) {
    el.classList.remove("jq-locked", "jq-current", "jq-unlocked");
    if (lockState === "locked")       { el.classList.add("jq-locked"); }
    else if (lockState === "current") { el.classList.add("jq-current"); }
    else                              { el.classList.add("jq-unlocked"); }
  }

  // render: sync lock classes to both overlay hotspots and ribbon land elements
  function render() {
    var cur = curIdx();
    var i, key, lockState, hot, land;
    for (i = 0; i < _order.length; i++) {
      key = _order[i];
      if (_qs[key] && _qs[key].done) {
        lockState = "unlocked";
      } else if (i === cur) {
        lockState = "current";
      } else {
        lockState = "locked";
      }

      // Overlay hotspot
      hot = document.querySelector(".jq-hot[data-key=\"" + key + "\"]");
      if (hot) { setLockClass(hot, lockState); }

      // Ribbon land — only the 4 trademark lands (scan/find/heal/give), not home.
      // Home lock state shows only on the overlay hotspot for v1.
      if (key !== "home") {
        land = document.querySelector("#js-path [data-key=\"" + key + "\"]");
        if (land) { setLockClass(land, lockState); }
      }
    }
  }

  // tryFind: called when a hotspot is clicked.
  // Only the current target advances; all others are a gentle no-op.
  function tryFind(key) {
    var cur = curIdx();
    var curKey = cur >= 0 ? _order[cur] : null;
    if (key !== curKey) { return; }   // not the current target — ignore
    // Mark found + done.
    // Task 6 will insert the "engage" second step between found and done.
    _qs[key].found = true;
    _qs[key].done = true;
    saveQS();
    render();
  }

  function isExternal(href) {
    if (!href) { return false; }
    if (/^(mailto:|tel:|#|javascript:)/i.test(href)) { return false; }
    try { return new URL(href, location.href).origin !== location.origin; }
    catch (e) { return false; }
  }

  function open() {
    if (!_overlay) { return; }
    _overlay.classList.add("open");
    _overlayOpen = true;
  }

  function close() {
    if (!_overlay) { return; }
    _overlay.classList.remove("open");
    _overlayOpen = false;
  }

  // Load persisted quest state before exposing the API, so callers see correct state
  loadQS();

  // Expose API immediately — before the async fetch resolves — so that
  // shell.js's Promise.all callback can check window.__JQUEST__ if it resolves last.
  // `render` is exposed so shell.js can re-apply lock classes after renderLands.
  window.__JQUEST__ = {
    open: open,
    close: close,
    state: _qs,      // live reference — mutations are visible to callers
    curIdx: curIdx,
    render: render
  };

  function buildOverlay(scene, journeyMap) {
    if (_overlay) { return; }

    // Set the canonical order and ensure all keys have state entries
    _order = scene.order || Object.keys(scene.hotspots || {});
    ensureKeys(_order);
    saveQS();

    // Build href map: home comes from scene.home.href; lands from /begin/state journey_map
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

    // Hotspots — positioned buttons from scene.hotspots (% coords)
    _order.forEach(function (key) {
      var spot = (scene.hotspots || {})[key];
      if (!spot) { return; }
      var btn = document.createElement("button");
      btn.className = "jq-hot";
      btn.setAttribute("data-key", key);
      btn.style.left = spot.x + "%";
      btn.style.top = spot.y + "%";
      btn.style.width = spot.w + "%";
      btn.style.height = spot.h + "%";
      btn.onclick = function () { tryFind(key); };
      stage.appendChild(btn);
    });

    // Close on backdrop click
    ov.onclick = function (e) { if (e.target === ov) { close(); } };

    ov.appendChild(closeBtn);
    ov.appendChild(stage);
    document.body.appendChild(ov);
    _overlay = ov;

    // Body class enables CSS scoping for ribbon land lock states
    document.body.classList.add("jq-quest-active");

    // Initial render: styles hotspots immediately; also styles ribbon lands if
    // shell.js's renderLands has already run (i.e., shell.js resolved first).
    render();
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

    // Hook mapBtn: covers the race where shell.js resolved FIRST
    // and set the default pavilion listener before __JQUEST__ existed.
    // Using onclick (not addEventListener) so this cleanly overrides.
    var mb = document.querySelector(".js-mapbtn");
    if (mb) { mb.onclick = function () { open(); }; }

    // Second render pass: catches any ribbon lands added by shell.js AFTER our
    // first render() call in buildOverlay (handles the shell-resolves-last race).
    render();
  }).catch(function () {});
})();
