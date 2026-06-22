// Shared calendar alarm engine: Web Audio chime + spoken announcement + browser
// Notification, fired `leadMin` minutes before an event start. Tab must be open.
// Used by the workspace calendar (and available to the console calendar).
(function (w) {
  const timers = {};

  function playChime(title, leadMin) {
    leadMin = leadMin || 3;
    try {
      const ctx = new (w.AudioContext || w.webkitAudioContext)();
      const notes = [523.25, 659.25, 783.99, 1046.50]; // C5 E5 G5 C6
      notes.forEach((freq, i) => {
        const osc = ctx.createOscillator(), gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.type = 'sine'; osc.frequency.value = freq;
        const t = ctx.currentTime + i * 0.18;
        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(0.15, t + 0.04);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 1.1);
        osc.start(t); osc.stop(t + 1.1);
      });
    } catch (e) {}
    setTimeout(() => {
      if ('speechSynthesis' in w) {
        const m = new SpeechSynthesisUtterance(
          `${title} is starting in ${leadMin} minute${leadMin === 1 ? '' : 's'}.`);
        m.rate = 0.92;
        const voices = w.speechSynthesis.getVoices();
        const pref = voices.find(v => /samantha|karen|moira|fiona|daniel/i.test(v.name)) || voices[0];
        if (pref) m.voice = pref;
        w.speechSynthesis.speak(m);
      }
    }, 900);
  }

  function scheduleAlert(evId, isoStart, title, leadMin) {
    leadMin = leadMin || 3;
    cancelAlert(evId);
    if (!isoStart || !isoStart.includes('T')) return;
    const delay = new Date(isoStart).getTime() - leadMin * 60 * 1000 - Date.now();
    if (delay < 0) return; // already past
    timers[evId] = setTimeout(() => {
      playChime(title, leadMin);
      if (w.Notification && Notification.permission === 'granted') {
        const n = new Notification(`⏰ Starting in ${leadMin} minutes`,
          { body: title, requireInteraction: true });
        n.onclick = () => { w.focus(); n.close(); };
      }
      delete timers[evId];
    }, delay);
  }

  function cancelAlert(evId) {
    if (timers[evId]) { clearTimeout(timers[evId]); delete timers[evId]; }
  }

  w.CalAlarm = { playChime, scheduleAlert, cancelAlert };
})(window);
