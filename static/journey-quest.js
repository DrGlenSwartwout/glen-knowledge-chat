/* Journey Quest overlay — Tasks 4–6: ordered progress, lock states, two-step gating,
   entry flow, progressive coupon, and rewards.
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
  var _soundMap = {};  // key -> sound string, populated from scene.hotspots in buildOverlay
  // hrefMap/nameMap/subMap: built from scene.home + journeyMap cards;
  // consumed by engageDone (navigate to real land href) and showPop (display stage name).
  var _hrefMap = {};
  var _nameMap = {};
  var _subMap  = {};

  // DOM elements created in buildOverlay
  var _pop = null;
  var _popTimer = null;
  var _engage = null;
  var _engageKey = null;
  var _reward = null;
  var _rewardTimer = null;
  var _introCard = null;
  var _mobileHint = null;

  // --- Quest progress state ---
  // Per-key {found:bool, done:bool} for each stage in scene.order.
  // state.paths: distinct rails ever used (values: "hunt"|"video"|"chat").
  // state.entered: true after the user completes the intro gate.
  var _qs = { paths: [], entered: false };

  function loadQS() {
    try {
      var raw = localStorage.getItem(LS);
      var parsed = raw ? JSON.parse(raw) : null;
      if (parsed && typeof parsed === "object") {
        _qs = parsed;
        if (!_qs.paths)           { _qs.paths = []; }
        if (!_qs.entered)         { _qs.entered = false; }
      }
    } catch (e) { /* malformed -- start fresh */ }
  }

  function saveQS() {
    try { localStorage.setItem(LS, JSON.stringify(_qs)); } catch (e) {}
    _syncPush();
  }

  // --- Server sync (members only, best-effort) ---
  var _isMember = !!(window.__SHELL__ && window.__SHELL__.mode === "member");
  var _syncUrl = "/api/journey/quest-state";

  function _mergeServerState(serverState) {
    if (!serverState || typeof serverState !== "object") { return; }
    var keys = ["home", "scan", "find", "heal", "give"];
    var i, k;
    // OR: never downgrade a local true flag
    if (serverState.entered) { _qs.entered = true; }
    if (Array.isArray(serverState.paths)) {
      for (i = 0; i < serverState.paths.length; i++) {
        var p = serverState.paths[i];
        if (p && !arrayIncludes(_qs.paths, p)) { _qs.paths.push(p); }
      }
    }
    for (i = 0; i < keys.length; i++) {
      k = keys[i];
      if (serverState[k] && typeof serverState[k] === "object") {
        if (!_qs[k]) { _qs[k] = { found: false, done: false }; }
        if (serverState[k].found) { _qs[k].found = true; }
        if (serverState[k].done)  { _qs[k].done  = true; }
      }
    }
  }

  function _syncPull() {
    if (!_isMember) { return; }
    fetch(_syncUrl, { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d && d.ok && d.state) {
          _mergeServerState(d.state);
          try { localStorage.setItem(LS, JSON.stringify(_qs)); } catch (e) {}
          render();
        }
      })
      .catch(function () {});
  }

  function _syncPush() {
    if (!_isMember) { return; }
    fetch(_syncUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_qs)
    }).catch(function () {});
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

  // --- Coupon helpers ---
  function arrayIncludes(arr, val) {
    var i;
    for (i = 0; i < arr.length; i++) { if (arr[i] === val) { return true; } }
    return false;
  }

  function addPath(rail) {
    if (!arrayIncludes(_qs.paths, rail)) { _qs.paths.push(rail); }
  }

  function couponPct() {
    return Math.min(15, 5 * _qs.paths.length);
  }

  // Apply one of the three lock-state classes to an element
  function setLockClass(el, lockState) {
    el.classList.remove("jq-locked", "jq-current", "jq-unlocked");
    if (lockState === "locked")       { el.classList.add("jq-locked"); }
    else if (lockState === "current") { el.classList.add("jq-current"); }
    else                              { el.classList.add("jq-unlocked"); }
  }

  // Update the mobile hint to show the current target name
  function updateMobileHint() {
    if (!_mobileHint) { return; }
    var cur = curIdx();
    if (cur >= 0) {
      var key = _order[cur];
      _mobileHint.querySelector(".jq-hint-name").textContent = _nameMap[key] || key;
      _mobileHint.style.display = "";
    } else {
      _mobileHint.style.display = "none";
    }
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

      // Ribbon land -- only the 4 trademark lands (scan/find/heal/give), not home.
      // Home lock state shows only on the overlay hotspot for v1.
      if (key !== "home") {
        land = document.querySelector("#js-path [data-key=\"" + key + "\"]");
        if (land) { setLockClass(land, lockState); }
      }
    }
    // Sync approach soundscape to current target
    if (window.__JQAUDIO__) {
      try {
        var curKey = cur >= 0 ? _order[cur] : null;
        __JQAUDIO__.setTarget(curKey, curKey ? (_soundMap[curKey] || null) : null);
      } catch (e) {}
    }
    updateMobileHint();
  }

  // --- Pop card ---
  function showPop(key) {
    if (!_pop) { return; }
    var hs = document.querySelector(".jq-hot[data-key=\"" + key + "\"]");
    if (hs) {
      _pop.style.left = hs.style.left;
      _pop.style.top  = hs.style.top;
    }
    _pop.querySelector(".jq-pop-title").textContent = _nameMap[key] || key;
    _pop.querySelector(".jq-pop-sub").textContent   = _subMap[key]  || "You found it!";
    _pop.classList.add("on");
    if (_popTimer) { clearTimeout(_popTimer); }
    _popTimer = setTimeout(function () { if (_pop) { _pop.classList.remove("on"); } }, 2600);
  }

  // --- Engage panel ---
  function openEngage(key) {
    if (!_engage) { return; }
    _engageKey = key;
    _engage.querySelector(".jq-engage-title").textContent = _nameMap[key] || key;
    _engage.classList.add("on");
  }

  // engageDone: mark done, record rail, fire gate reward, navigate to real land href.
  // v1: clicking through marks the engage step done immediately (full rail-completion
  // detection -- waiting for the user to return from the page -- is a Phase 3 feature).
  function engageDone(key, rail) {
    if (!_engage) { return; }
    _engage.classList.remove("on");
    _engageKey = null;
    _qs[key].done = true;
    addPath(rail);
    saveQS();
    gateReward(key);
    render();
    // Navigate to this stage's real land href (external -> new tab, internal -> same tab).
    var href = _hrefMap[key] || null;
    if (href) {
      if (isExternal(href)) {
        window.open(href, "_blank", "noopener");
      } else {
        location.href = href;
      }
    }
  }

  // --- Reward toast ---
  function showReward(big, sub, pct, ms) {
    if (!_reward) { return; }
    _reward.querySelector(".jq-reward-big").textContent = big;
    _reward.querySelector(".jq-reward-sub").textContent = sub;
    var offEl = _reward.querySelector(".jq-reward-off");
    if (pct > 0) {
      offEl.textContent = pct + "% OFF unlocked";
      offEl.style.display = "inline-block";
    } else {
      offEl.style.display = "none";
    }
    _reward.classList.add("on");
    if (_rewardTimer) { clearTimeout(_rewardTimer); }
    _rewardTimer = setTimeout(function () { if (_reward) { _reward.classList.remove("on"); } }, ms || 2400);
  }

  // huntReward: fired on every first-find. Adds "hunt" rail, shows toast + arrival sound.
  function huntReward(key) {
    addPath("hunt");
    saveQS();
    var pct = couponPct();
    var name = _nameMap[key] || key;
    showReward(name + " found!", "A hidden link discovered — " + pct + "% off coupon unlocked.", pct, 2400);
    if (window.__JQAUDIO__) {
      try { __JQAUDIO__.arrival(key, _soundMap[key] || null); } catch (e) {}
    }
  }

  // gateReward: fired when a stage is fully done (engage step cleared). Finale on all 5 done.
  function gateReward(key) {
    var allDone = curIdx() === -1;
    var pct = couponPct();
    if (allDone) {
      showReward(
        "✨ Journey Unlocked! ✨",
        "Your coupon is maxed at 15% — and here is a second one to give away.",
        15,
        3600
      );
    } else {
      var name = _nameMap[key] || key;
      showReward(name + " opened", "A gate opens — your path lights up above.", pct, 1800);
    }
  }

  // --- Entry gate ---
  // Dismisses intro card, marks entered, reveals hunt, inits audio.
  function enterQuest() {
    if (_introCard) { _introCard.style.display = "none"; }
    _qs.entered = true;
    saveQS();
    if (window.__JQAUDIO__) { try { __JQAUDIO__.init(); } catch (e) {} }
    render();
  }

  // --- tryFind (two-step gating) ---
  // First click on current target: mark found, fire huntReward, show pop, open engage panel.
  // Second step (engageDone) marks done and navigates to the real land href.
  // Re-clicking an already-found (but not done) stage re-opens the engage panel.
  function tryFind(key) {
    var cur = curIdx();
    var curKey = cur >= 0 ? _order[cur] : null;
    if (key !== curKey) { return; }   // not the current target -- ignore

    if (_qs[key] && _qs[key].found) {
      // Already found -- re-open engage panel so user can complete the second step
      openEngage(key);
      return;
    }

    // First find: mark found (NOT done yet), fire hunt reward, show pop, then engage panel
    _qs[key].found = true;
    huntReward(key);
    showPop(key);
    setTimeout(function () { openEngage(key); }, 1300);
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
    _syncPull();
    if (!_qs.entered) {
      // Show intro gate; audio deferred until user explicitly enters
      if (_introCard) { _introCard.style.display = "flex"; }
    } else {
      // Returning user -- skip intro, init audio
      if (_introCard) { _introCard.style.display = "none"; }
      if (window.__JQAUDIO__) { try { __JQAUDIO__.init(); } catch (e) {} }
    }
  }

  function close() {
    if (!_overlay) { return; }
    _overlay.classList.remove("open");
    _overlayOpen = false;
    if (window.__JQAUDIO__) { window.__JQAUDIO__.stopAll(); }
  }

  // Load persisted quest state before exposing the API, so callers see correct state
  loadQS();

  // Expose API immediately -- before the async fetch resolves -- so that
  // shell.js's Promise.all callback can check window.__JQUEST__ if it resolves last.
  // `render` is exposed so shell.js can re-apply lock classes after renderLands.
  window.__JQUEST__ = {
    open: open,
    close: close,
    state: _qs,      // live reference -- mutations are visible to callers
    curIdx: curIdx,
    render: render
  };

  function buildOverlay(scene, journeyMap) {
    if (_overlay) { return; }

    // Set the canonical order and ensure all keys have state entries
    _order = scene.order || Object.keys(scene.hotspots || {});
    ensureKeys(_order);
    saveQS();

    // Populate sound map from scene.hotspots
    var _hs = scene.hotspots || {};
    var _sk;
    for (_sk in _hs) {
      if (_hs.hasOwnProperty(_sk) && _hs[_sk].sound) {
        _soundMap[_sk] = _hs[_sk].sound;
      }
    }

    // Build href/name/sub maps from scene.home + journeyMap cards.
    // hrefMap consumed by engageDone (navigate) and enterQuest; nameMap by showPop and engage panel.
    if (scene.home) {
      if (scene.home.href)  { _hrefMap["home"] = scene.home.href; }
      if (scene.home.name)  { _nameMap["home"] = scene.home.name; }
      if (scene.home.sub)   { _subMap["home"]  = scene.home.sub;  }
    }
    (journeyMap || []).forEach(function (card) {
      if (!card.key) { return; }
      if (card.href)  { _hrefMap[card.key] = card.href; }
      if (card.name)  { _nameMap[card.key] = card.name; }
      if (card.label && !_nameMap[card.key]) { _nameMap[card.key] = card.label; }
      if (card.sub)   { _subMap[card.key]  = card.sub; }
      if (card.description && !_subMap[card.key]) { _subMap[card.key] = card.description; }
    });
    // Also check scene.hotspots for name/sub (wins only if journeyMap didn't provide them)
    for (_sk in _hs) {
      if (_hs.hasOwnProperty(_sk)) {
        if (_hs[_sk].name && !_nameMap[_sk]) { _nameMap[_sk] = _hs[_sk].name; }
        if (_hs[_sk].sub  && !_subMap[_sk])  { _subMap[_sk]  = _hs[_sk].sub;  }
      }
    }

    // ---------- Build DOM ----------

    var ov = document.createElement("div");
    ov.className = "jq-overlay";

    // Close button
    var closeBtn = document.createElement("button");
    closeBtn.className = "jq-close-btn";
    closeBtn.setAttribute("aria-label", "Close journey map");
    closeBtn.textContent = "×";
    closeBtn.onclick = function () { close(); };
    ov.appendChild(closeBtn);

    // Intro card -- shown on first open when !entered; absolute-positioned to cover stage.
    // v1 simplification: the full "hide shell ribbon until first touch" entry is out of scope.
    // This card lives inside .jq-overlay (not a separate full-screen fixed element).
    var intro = document.createElement("div");
    intro.className = "jq-intro-card";
    intro.style.display = "none";

    var introH2 = document.createElement("h2");
    introH2.textContent = "Welcome. Your healing journey is hidden — come find it.";
    intro.appendChild(introH2);

    var introP = document.createElement("p");
    introP.textContent = "Do one thing first: watch the welcome video or say hello in the chat. Your journey map will appear, and the hunt begins.";
    intro.appendChild(introP);

    var introOpts = document.createElement("div");
    introOpts.className = "jq-intro-opts";

    var iVidBtn = document.createElement("button");
    var iVidEm = document.createElement("span");
    iVidEm.className = "jq-em";
    iVidEm.textContent = "▶";
    iVidBtn.appendChild(iVidEm);
    iVidBtn.appendChild(document.createTextNode(" Watch the video"));
    iVidBtn.onclick = function () { enterQuest(); };

    var iChatBtn = document.createElement("button");
    var iChatEm = document.createElement("span");
    iChatEm.className = "jq-em";
    iChatEm.textContent = "💬";
    iChatBtn.appendChild(iChatEm);
    iChatBtn.appendChild(document.createTextNode(" Open the chat"));
    iChatBtn.onclick = function () { enterQuest(); };

    introOpts.appendChild(iVidBtn);
    introOpts.appendChild(iChatBtn);
    intro.appendChild(introOpts);
    ov.appendChild(intro);
    _introCard = intro;

    // Stage (scene image container)
    var stage = document.createElement("div");
    stage.className = "jq-stage";

    // Scene background image
    var img = document.createElement("img");
    img.className = "jq-scene";
    img.src = scene.image;
    img.alt = "Your healing journey";
    stage.appendChild(img);

    // Hotspots -- positioned buttons from scene.hotspots (% coords)
    _order.forEach(function (key) {
      var spot = _hs[key];
      if (!spot) { return; }
      var btn = document.createElement("button");
      btn.className = "jq-hot";
      btn.setAttribute("data-key", key);
      btn.style.left   = spot.x + "%";
      btn.style.top    = spot.y + "%";
      btn.style.width  = spot.w + "%";
      btn.style.height = spot.h + "%";
      btn.onclick = function () { tryFind(key); };
      stage.appendChild(btn);
    });

    // Shared pop card (repositioned dynamically by showPop)
    var pop = document.createElement("div");
    pop.className = "jq-pop";
    var popTitle = document.createElement("div");
    popTitle.className = "jq-pop-title";
    var popSub = document.createElement("div");
    popSub.className = "jq-pop-sub";
    pop.appendChild(popTitle);
    pop.appendChild(popSub);
    stage.appendChild(pop);
    _pop = pop;

    // Engage panel (the second step -- centered overlay inside stage)
    var eng = document.createElement("div");
    eng.className = "jq-engage";

    var engTitle = document.createElement("h3");
    engTitle.className = "jq-engage-title";
    eng.appendChild(engTitle);

    var engP = document.createElement("p");
    engP.textContent = "You found it! Now open the gate — one more step:";
    eng.appendChild(engP);

    var engOpts = document.createElement("div");
    engOpts.className = "jq-engage-opts";

    var vidBtn = document.createElement("button");
    var vidEm = document.createElement("span");
    vidEm.className = "jq-em";
    vidEm.textContent = "▶";
    vidBtn.appendChild(vidEm);
    vidBtn.appendChild(document.createTextNode(" Watch the video"));
    vidBtn.onclick = function () {
      if (_engageKey) { engageDone(_engageKey, "video"); }
    };
    engOpts.appendChild(vidBtn);

    var chatBtn = document.createElement("button");
    var chatEm = document.createElement("span");
    chatEm.className = "jq-em";
    chatEm.textContent = "💬";
    chatBtn.appendChild(chatEm);
    chatBtn.appendChild(document.createTextNode(" Ask in the chat"));
    chatBtn.onclick = function () {
      if (_engageKey) { engageDone(_engageKey, "chat"); }
    };
    engOpts.appendChild(chatBtn);

    eng.appendChild(engOpts);
    stage.appendChild(eng);
    _engage = eng;

    // Mobile hint: shows current target name on touch devices or when muted (sound-off guarantee)
    var mh = document.createElement("div");
    mh.className = "jq-mobile-hint";
    var mhName = document.createElement("span");
    mhName.className = "jq-hint-name";
    mh.appendChild(mhName);
    var mhSep = document.createTextNode(" — ");
    mh.appendChild(mhSep);
    var mhLine = document.createElement("span");
    mhLine.className = "jq-hint-line";
    mhLine.textContent = "explore the scene to find it";
    mh.appendChild(mhLine);
    stage.appendChild(mh);
    _mobileHint = mh;

    // Mute toggle button
    var muteBtn = document.createElement("button");
    muteBtn.className = "jq-mute-btn";
    muteBtn.setAttribute("aria-label", "Toggle sound");
    muteBtn.setAttribute("title", "Toggle sound");
    function updateMuteBtn() {
      var muted = !!(window.__JQAUDIO__ && __JQAUDIO__.isMuted());
      muteBtn.textContent = muted ? "🔇" : "🔊";
      ov.classList.toggle("jq-soundoff", muted);
    }
    updateMuteBtn();
    muteBtn.onclick = function () {
      if (window.__JQAUDIO__) {
        try { __JQAUDIO__.toggleMute(); } catch (e) {}
      } else {
        // No audio engine loaded -- toggle via localStorage only
        try {
          var m = !!JSON.parse(localStorage.getItem("jquest.muted"));
          localStorage.setItem("jquest.muted", JSON.stringify(!m));
        } catch (ex) {}
      }
      updateMuteBtn();
    };
    stage.appendChild(muteBtn);

    // Proximity: mousemove on stage drives approach gain for the current hotspot
    stage.addEventListener("mousemove", function (e) {
      if (!window.__JQAUDIO__) { return; }
      var cur = curIdx();
      if (cur < 0) { try { __JQAUDIO__.proximity(0); } catch (ex) {} return; }
      var key = _order[cur];
      var spot = _hs[key];
      if (!spot) { return; }
      var r = stage.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width * 100;
      var py = (e.clientY - r.top) / r.height * 100;
      var dx = px - spot.x;
      var dy = py - spot.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      var p = Math.max(0, 1 - dist / 40);
      try { __JQAUDIO__.proximity(p); } catch (ex) {}
    });
    stage.addEventListener("mouseleave", function () {
      if (window.__JQAUDIO__) { try { __JQAUDIO__.proximity(0); } catch (e) {} }
    });

    // Close on backdrop click
    ov.onclick = function (e) { if (e.target === ov) { close(); } };

    ov.appendChild(stage);
    document.body.appendChild(ov);
    _overlay = ov;

    // Reward toast -- appended to body so it floats above the overlay
    var reward = document.createElement("div");
    reward.className = "jq-reward";
    var rewBig = document.createElement("div");
    rewBig.className = "jq-reward-big";
    var rewSub = document.createElement("div");
    rewSub.className = "jq-reward-sub";
    var rewOff = document.createElement("span");
    rewOff.className = "jq-reward-off";
    rewOff.style.display = "none";
    reward.appendChild(rewBig);
    reward.appendChild(rewSub);
    reward.appendChild(rewOff);
    document.body.appendChild(reward);
    _reward = reward;

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
    var mb = document.querySelector(".js-mapbtn");
    if (mb) { mb.onclick = function () { open(); }; }

    // Second render pass: catches any ribbon lands added by shell.js AFTER our
    // first render() call in buildOverlay (handles the shell-resolves-last race).
    render();
  }).catch(function () {});
}());
