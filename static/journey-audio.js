/* journey-audio.js — Task 5: approach/arrival audio engine for the journey quest.
   ES5-style IIFE. Exposes window.__JQAUDIO__ = { init, setTarget, proximity, arrival,
   stopAll, toggleMute, isMuted }.
   Synth engine ported from ~/Downloads/journey-ribbon-samples/landmarks/hunt-prototype.html.
   Asset-manifest override: fetch /static/journey/audio/manifest.json once; if it has a
   real URL for approach or arrival, play that via HTMLAudio instead of the synth.
   Missing file / empty manifest / missing key -> synth (never throws). */
(function () {
  "use strict";

  // ── State ────────────────────────────────────────────────────────────────────
  var AC = null;           // AudioContext (created on first init())
  var master = null;       // GainNode → destination
  var muteGain = null;     // GainNode between master and destination (controls mute)
  var approach = null;     // { setProx(p), stop() } for current approach soundscape
  var approachKey = null;  // key of the active approach (null = none)
  var curProx = 0;         // last proximity value passed to proximity()

  var manifest = null;     // loaded manifest object (null = not yet loaded)
  var manifestLoading = false;

  var LS_MUTED = "jquest.muted";
  var _muted = false;
  // Load persisted mute state eagerly
  try { _muted = !!JSON.parse(localStorage.getItem(LS_MUTED)); } catch (e) {}

  // ── Manifest ─────────────────────────────────────────────────────────────────
  function loadManifest(cb) {
    if (manifest !== null) { if (cb) { cb(); } return; }
    if (manifestLoading) { return; } // already in flight; cb won't wait, ok
    manifestLoading = true;
    try {
      var xhr = new XMLHttpRequest();
      xhr.open("GET", "/static/journey/audio/manifest.json", true);
      xhr.onload = function () {
        try { manifest = JSON.parse(xhr.responseText) || {}; } catch (e) { manifest = {}; }
        if (cb) { cb(); }
      };
      xhr.onerror = function () { manifest = {}; if (cb) { cb(); } };
      xhr.send();
    } catch (e) { manifest = {}; if (cb) { cb(); } }
  }

  function manifestUrl(key, kind) {
    // kind = "approach" | "arrival"
    if (!manifest) { return null; }
    var entry = manifest[key];
    if (!entry || typeof entry !== "object") { return null; }
    return entry[kind] || null;
  }

  // ── AudioContext init ────────────────────────────────────────────────────────
  function init() {
    if (AC) {
      if (AC.state === "suspended") { try { AC.resume(); } catch (e) {} }
      return;
    }
    try {
      AC = new (window.AudioContext || window.webkitAudioContext)();
      // Chain: master → muteGain → destination
      master = AC.createGain();
      master.gain.value = 0.9;
      muteGain = AC.createGain();
      muteGain.gain.value = _muted ? 0 : 1;
      master.connect(muteGain);
      muteGain.connect(AC.destination);
      // Start loading the manifest in the background
      loadManifest(null);
    } catch (e) { AC = null; }
  }

  // ── Synth primitives (ported from prototype) ─────────────────────────────────
  function loopNoise() {
    var len = Math.floor(AC.sampleRate * 2);
    var buf = AC.createBuffer(1, len, AC.sampleRate);
    var d = buf.getChannelData(0);
    for (var i = 0; i < len; i++) { d[i] = Math.random() * 2 - 1; }
    var src = AC.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    return src;
  }

  function burst(dur, vol, freq, type) {
    var t = AC.currentTime;
    var len = Math.floor(AC.sampleRate * dur);
    var buf = AC.createBuffer(1, len, AC.sampleRate);
    var d = buf.getChannelData(0);
    var i;
    for (i = 0; i < len; i++) { d[i] = Math.random() * 2 - 1; }
    var src = AC.createBufferSource();
    src.buffer = buf;
    var f = AC.createBiquadFilter();
    f.type = type || "lowpass";
    f.frequency.value = freq || 300;
    var g = AC.createGain();
    g.gain.setValueAtTime(vol, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f);
    f.connect(g);
    g.connect(master);
    src.start(t);
    src.stop(t + dur);
  }

  function tone(freq, dur, vol, type, when, glideTo, pan) {
    var t = AC.currentTime + (when || 0);
    var o = AC.createOscillator();
    var g = AC.createGain();
    o.type = type || "sine";
    o.frequency.setValueAtTime(freq, t);
    if (glideTo) { o.frequency.exponentialRampToValueAtTime(glideTo, t + dur); }
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    var node = g;
    if (pan && AC.createStereoPanner) {
      var p = AC.createStereoPanner();
      p.pan.setValueAtTime(-pan, t);
      p.pan.linearRampToValueAtTime(pan, t + dur);
      g.connect(p);
      node = p;
    }
    o.connect(g);
    node.connect(master);
    o.start(t);
    o.stop(t + dur + 0.05);
  }

  function beep(freq, dur, type, when, vol, glideTo, pan) {
    tone(freq, dur, vol == null ? 0.4 : vol, type || "sine", when || 0, glideTo, pan || 0);
  }

  function say(text, pitch, rate, vol) {
    if (_muted) { return; }
    if (!window.speechSynthesis) { return; }
    var u = new SpeechSynthesisUtterance(text);
    u.pitch = pitch == null ? 1 : pitch;
    u.rate  = rate  == null ? 0.95 : rate;
    u.volume = vol  == null ? 1 : vol;
    speechSynthesis.speak(u);
  }

  // ── Approach soundscapes (ported from prototype) ─────────────────────────────
  // sound key → approach soundscape builder; g is a GainNode connected to master.
  // Returns an object with { setProx(p), stop() }.
  function buildSynthApproach(sound, g) {
    var nodes = [];
    var timer = null;

    if (sound === "whisper") {
      // 172 Hz Tibetan bowl
      var o1 = AC.createOscillator(); o1.type = "sine"; o1.frequency.value = 172;
      var o2 = AC.createOscillator(); o2.type = "sine"; o2.frequency.value = 344;
      var g2 = AC.createGain(); g2.gain.value = 0.25;
      o2.connect(g2); g2.connect(g);
      o1.connect(g); o1.start(); o2.start();
      nodes = [o1, o2];
    } else if (sound === "creak") {
      // muffled talk / murmur behind the door
      var srcC = loopNoise();
      var fC = AC.createBiquadFilter(); fC.type = "lowpass"; fC.frequency.value = 360;
      srcC.connect(fC); fC.connect(g); srcC.start(); nodes = [srcC];
      timer = setInterval(function () {
        if (curProx > 0.04) { tone(170 + Math.random() * 130, 0.32, curProx * 0.12, "triangle"); }
      }, 620);
    } else if (sound === "chaching") {
      // succussion + mortar & pestle grind
      var srcCh = loopNoise();
      var fCh = AC.createBiquadFilter(); fCh.type = "bandpass"; fCh.frequency.value = 1600; fCh.Q.value = 0.5;
      var ggCh = AC.createGain(); ggCh.gain.value = 0.18;
      srcCh.connect(fCh); fCh.connect(ggCh); ggCh.connect(g); srcCh.start(); nodes = [srcCh];
      timer = setInterval(function () {
        if (curProx > 0.04) { burst(0.09, curProx * 0.55, 200, "lowpass"); }
      }, 1000);
    } else if (sound === "doppler") {
      // footsteps on the dirt path (timer-only, no persistent nodes)
      timer = setInterval(function () {
        if (curProx > 0.04) { burst(0.13, curProx * 0.55, 260, "lowpass"); }
      }, 520);
    } else if (sound === "oasis") {
      // breeze + birds
      var srcO = loopNoise();
      var fO = AC.createBiquadFilter(); fO.type = "bandpass"; fO.frequency.value = 700; fO.Q.value = 0.5;
      srcO.connect(fO); fO.connect(g); srcO.start(); nodes = [srcO];
      timer = setInterval(function () {
        if (curProx > 0.06 && Math.random() < 0.6) {
          tone(1800 + Math.random() * 800, 0.12, curProx * 0.35, "sine");
        }
      }, 680);
    }

    return {
      setProx: function (p) {
        g.gain.setTargetAtTime(0.05 + p * 0.5, AC.currentTime, 0.06);
      },
      stop: function () {
        if (timer) { clearInterval(timer); }
        nodes.forEach(function (n) { try { n.stop(); } catch (e) {} });
        try { g.disconnect(); } catch (e) {}
      }
    };
  }

  // HTMLAudio-based approach (asset manifest hit)
  function buildAssetApproach(url, g) {
    var el = null;
    try {
      el = new Audio(url);
      el.loop = true;
      el.volume = 0.05; // faint floor
      el.play().catch(function () {});
    } catch (e) { el = null; }
    return {
      setProx: function (p) {
        if (el) { el.volume = Math.max(0, Math.min(1, 0.05 + p * 0.5)); }
        // Also drive the GainNode for consistency (not strictly needed for asset path)
        g.gain.setTargetAtTime(0.05 + p * 0.5, AC.currentTime, 0.06);
      },
      stop: function () {
        if (el) { try { el.pause(); el.currentTime = 0; } catch (e) {} }
        try { g.disconnect(); } catch (e) {}
      }
    };
  }

  function startApproach(sound, key) {
    var g = AC.createGain();
    g.gain.value = 0;
    g.connect(master);
    var url = manifestUrl(key, "approach");
    if (url) {
      approach = buildAssetApproach(url, g);
    } else {
      approach = buildSynthApproach(sound, g);
    }
  }

  function stopApproach() {
    if (approach) { approach.stop(); approach = null; }
  }

  // ── Arrival cues (ported from prototype) ────────────────────────────────────
  function playSynthArrival(sound) {
    if (!AC) { return; }
    if (sound === "creak") {
      // Door opens, then a hearty welcome in Glendalf's voice
      var t = AC.currentTime;
      var o = AC.createOscillator();
      var g = AC.createGain();
      o.type = "sawtooth";
      o.frequency.setValueAtTime(70, t);
      o.frequency.linearRampToValueAtTime(150, t + 0.6);
      o.frequency.linearRampToValueAtTime(110, t + 0.9);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.18 * 0.9, t + 0.1);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 1.0);
      var f = AC.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = 900;
      o.connect(f); f.connect(g); g.connect(master);
      o.start(t); o.stop(t + 1.05);
      setTimeout(function () { say("Welcome, friend! Come in, come in!", 0.6, 0.95); }, 750);
    } else if (sound === "whisper") {
      // Counting 1 to 10 at the ear
      for (var n = 1; n <= 10; n++) {
        (function (num) {
          setTimeout(function () { say(String(num), 1.05, 0.95, 0.85); }, (num - 1) * 360);
        }(n));
      }
    } else if (sound === "chaching") {
      // The sound of relief — "Ahhhh!"
      tone(440, 0.5, 0.16, "sine", 0, 330);
      say("Ahhh", 0.9, 0.7, 0.9);
    } else if (sound === "doppler") {
      // "We will walk this path together" in Glendalf's voice
      tone(300, 0.7, 0.3, "sawtooth", 0, 1200, 0.9);
      setTimeout(function () { say("We will walk this path together.", 0.6, 0.92); }, 250);
    } else if (sound === "oasis") {
      // Breeze + birds, plus fountains and wind chimes on arrival
      burst(1.4, 0.06, 600, "bandpass");
      var birds = [0, 0.3, 0.6, 0.95];
      birds.forEach(function (d, i) { tone(1800 + Math.sin(i) * 500, 0.12, 0.16, "sine", d); });
      burst(1.6, 0.05, 400, "bandpass");
      var chimes = [1320, 1568, 1760, 2093];
      chimes.forEach(function (fr, i) { tone(fr, 0.7, 0.13, "sine", 0.2 + i * 0.18); });
    }
    // Fanfare common to all hunts
    [392, 523, 659].forEach(function (fr, k) { beep(fr, 0.22, "triangle", 0.5 + k * 0.1, 0.3); });
  }

  // ── Public API ───────────────────────────────────────────────────────────────
  function setTarget(key, sound) {
    if (!AC) { return; }
    if (key === approachKey) { return; }   // already active — no-op
    approachKey = key;
    stopApproach();
    if (!key || !sound) { return; }
    startApproach(sound, key);
  }

  function proximity(p) {
    curProx = p;
    if (approach) { approach.setProx(p); }
  }

  function arrival(key, sound) {
    if (!AC) { return; }
    var url = manifestUrl(key, "arrival");
    if (url) {
      try {
        var el = new Audio(url);
        el.volume = 0.9;
        el.play().catch(function () {});
      } catch (e) {}
      return;
    }
    if (sound) { playSynthArrival(sound); }
  }

  function stopAll() {
    stopApproach();
    approachKey = null;
    curProx = 0;
    try { if (window.speechSynthesis) { speechSynthesis.cancel(); } } catch (e) {}
  }

  function toggleMute() {
    _muted = !_muted;
    if (muteGain) { muteGain.gain.value = _muted ? 0 : 1; }
    try { if (window.speechSynthesis && _muted) { speechSynthesis.cancel(); } } catch (e) {}
    try { localStorage.setItem(LS_MUTED, JSON.stringify(_muted)); } catch (e) {}
    return _muted;
  }

  function isMuted() { return _muted; }

  window.__JQAUDIO__ = {
    init: init,
    setTarget: setTarget,
    proximity: proximity,
    arrival: arrival,
    stopAll: stopAll,
    toggleMute: toggleMute,
    isMuted: isMuted
  };
}());
