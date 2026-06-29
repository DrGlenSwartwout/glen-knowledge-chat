/* tts-output.js — per-message "Listen" button for chat replies.
 *
 * Usage: after a bot reply is finalized, call
 *   window.TTS.attach(containerEl, plainText)
 * It appends a small 🔊 Listen button to containerEl. On click it speaks the
 * reply via /chat/tts (ElevenLabs, Dr. Glen's voice). If that endpoint is
 * unavailable (503/429/network/not configured), it falls back to the browser's
 * built-in speechSynthesis voice so the feature always does something.
 *
 * Pass PLAIN text (e.g. element.innerText), not rendered HTML/markdown.
 * Mirrors the drop-in style of mic-input.js. No-op if the browser supports
 * neither Audio nor speechSynthesis.
 */
(function () {
  var hasAudio  = typeof window.Audio !== 'undefined';
  var hasSpeech = 'speechSynthesis' in window;
  if (!hasAudio && !hasSpeech) return;

  // ── styles (once) ──────────────────────────────────────────────────────────
  var css = ''
    + '.tts-btn{display:inline-flex;align-items:center;gap:5px;'
    +   'border:1px solid rgba(127,127,127,.35);border-radius:6px;'
    +   'background:transparent;color:inherit;font:inherit;font-size:12px;line-height:1;'
    +   'padding:5px 9px;margin-top:6px;cursor:pointer;opacity:.75;'
    +   'transition:opacity .15s,border-color .15s,color .15s;}'
    + '.tts-btn:hover{opacity:1;border-color:rgba(127,127,127,.6);}'
    + '.tts-btn.busy{opacity:.5;cursor:progress;}'
    + '.tts-btn.playing{opacity:1;border-color:currentColor;}'
    + '.msg-actions .tts-btn{margin-top:0;}';  // sit inline in index.html action bar
  var st = document.createElement('style');
  st.id = 'tts-output-styles';
  st.textContent = css;
  document.head.appendChild(st);

  // ── single active player across the whole page ─────────────────────────────
  var active = null;  // { stop: fn, btn: el }

  function stopActive() {
    if (active) { try { active.stop(); } catch (e) {} active = null; }
  }

  function setLabel(btn, state) {
    btn.classList.remove('busy', 'playing');
    if (state === 'loading')      { btn.classList.add('busy');    btn.innerHTML = '⏳ Loading…'; }
    else if (state === 'playing') { btn.classList.add('playing'); btn.innerHTML = '⏹ Stop'; }
    else                          {                               btn.innerHTML = '🔊 Listen'; }
  }

  // ── browser fallback voice ─────────────────────────────────────────────────
  function browserSpeak(btn, text) {
    if (!hasSpeech) { setLabel(btn, 'idle'); return; }
    try {
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0; u.pitch = 1.0;
      u.onend   = function () { setLabel(btn, 'idle'); active = null; };
      u.onerror = function () { setLabel(btn, 'idle'); active = null; };
      active = { btn: btn, stop: function () { window.speechSynthesis.cancel(); setLabel(btn, 'idle'); } };
      setLabel(btn, 'playing');
      window.speechSynthesis.speak(u);
    } catch (e) { setLabel(btn, 'idle'); active = null; }
  }

  // ── Glen's voice via /chat/tts, fall back on any error ─────────────────────
  function speak(btn, text) {
    setLabel(btn, 'loading');
    fetch('/chat/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    }).then(function (r) {
      if (!r.ok) throw new Error('status ' + r.status);
      return r.blob();
    }).then(function (blob) {
      if (!hasAudio) throw new Error('no audio');
      var url   = URL.createObjectURL(blob);
      var audio = new window.Audio(url);
      function cleanup() { URL.revokeObjectURL(url); }
      audio.onended = function () { cleanup(); setLabel(btn, 'idle'); active = null; };
      audio.onerror = function () { cleanup(); setLabel(btn, 'idle'); active = null; };
      active = { btn: btn, stop: function () { try { audio.pause(); } catch (e) {} cleanup(); setLabel(btn, 'idle'); } };
      setLabel(btn, 'playing');
      audio.play().catch(function () { cleanup(); active = null; browserSpeak(btn, text); });
    }).catch(function () {
      browserSpeak(btn, text);
    });
  }

  // ── public: attach a Listen button to a finalized bot message ──────────────
  function attach(container, text) {
    if (!container || container.__ttsAttached) return null;
    text = (text || '').replace(/\s+/g, ' ').trim();
    if (!text) return null;
    container.__ttsAttached = true;

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tts-btn';
    btn.title = 'Listen to this reply';
    setLabel(btn, 'idle');
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      if (active && active.btn === btn) { stopActive(); return; }  // toggle off
      stopActive();
      speak(btn, text);
    });
    container.appendChild(btn);
    return btn;
  }

  // ── public: attach a Listen button AND speak once now (auto voice-out) ─────
  function attachAndSpeak(container, text) {
    var btn = attach(container, text);
    if (btn) { stopActive(); speak(btn, (text || '').replace(/\s+/g, ' ').trim()); }
    return btn;
  }

  window.TTS = { attach: attach, attachAndSpeak: attachAndSpeak, stop: stopActive };
})();
