// static/js/portal-onboarding.js
// My Onboarding tile: 3-phase progress (Be read / Match remedies / Accelerate healing).
// Consumes GET /api/portal/<token>/onboarding -> {enabled, status:{phases:[{key,title,steps:[{key,label,done,href,soon?}]}], member}}.
function renderOnboarding(status) {
  if (!status || !status.phases || !status.phases.length) return '';
  const phases = status.phases.map(function (ph) {
    const steps = (ph.steps || []).map(function (st) {
      const mark = st.done === true ? '✓' : (st.done === false ? '○' : '•');
      var label = escapeHtml(st.label);
      if (st.soon) label += ' (coming soon)';
      var text = st.href
        ? '<a href="' + st.href + '">' + label + '</a>'
        : label;
      return '<li class="ob-step"><span class="ob-mark">' + mark + '</span> ' + text + '</li>';
    }).join('');
    var extra = '';
    if (ph.key === 'match') {
      // Task 6: the compact glaucoma-pilot triage form, shown only until the
      // "history" step is done (i.e. no condition-sourced recs seeded yet).
      var historyStep = _stepByKey(ph.steps, 'history');
      if (historyStep && historyStep.done === false) {
        extra += renderTriageForm();
      }
      // Task 6 / P1.T3 fast-follow: surface status.member as a quiet inline
      // thread near Match/Heal -- not a checklist item.
      extra += renderMemberThread(status.member);
    }
    return '<div class="ob-phase"><h3>' + escapeHtml(ph.title) + '</h3><ul class="ob-steps">' + steps + '</ul>' + extra + '</div>';
  }).join('');
  return '<section class="portal-onboarding">' + phases + '</section>';
}

function _stepByKey(steps, key) {
  steps = steps || [];
  for (var i = 0; i < steps.length; i++) {
    if (steps[i] && steps[i].key === key) return steps[i];
  }
  return null;
}

// Compact glaucoma-pilot self-report (Plan 2 Task 6). Pure markup only --
// submit is wired via a delegated listener in the browser-init block below so
// this function (and renderOnboarding) stays side-effect-free / DOM-free and
// unit-testable under plain Node. All labels/text here are fixed copy (no
// server-provided strings), so nothing needs escaping.
function renderTriageForm() {
  return '' +
    '<form class="ob-triage-form" data-condition="glaucoma">' +
      '<p class="ob-triage-intro">A couple of quick questions about your eye pressure get you starter remedies today.</p>' +
      '<div class="ob-triage-row">' +
        '<label class="ob-triage-field">OD (right eye) IOP' +
          '<input type="number" step="0.1" min="0" max="60" name="iop_od" class="ob-iop-od" placeholder="e.g. 18"></label>' +
        '<label class="ob-triage-field">OS (left eye) IOP' +
          '<input type="number" step="0.1" min="0" max="60" name="iop_os" class="ob-iop-os" placeholder="e.g. 18"></label>' +
      '</div>' +
      '<label class="ob-triage-check">' +
        '<input type="checkbox" name="on_meds" class="ob-on-meds"> On IOP-lowering medication' +
      '</label>' +
      '<span class="ob-med-count">(if yes, how many? ' +
        '<input type="number" min="0" max="20" name="med_count" class="ob-med-count-input" style="width:3em"></span>' +
      '<label class="ob-triage-check">' +
        '<input type="checkbox" name="field_loss" class="ob-field-loss"> Peripheral vision loss' +
      '</label>' +
      '<button type="submit" class="ob-triage-submit">Show my starter remedies</button>' +
      '<p class="ob-triage-fallback">Not sure? Pick a category: ' +
        '<button type="button" class="ob-triage-cat" data-category="normal">Normal</button> ' +
        '<button type="button" class="ob-triage-cat" data-category="elevated">Elevated</button>' +
      '</p>' +
      '<p class="ob-triage-msg" aria-live="polite"></p>' +
    '</form>';
}

// Task 6 / P1.T3 fast-follow: status.member was computed but never rendered.
// A quiet inline thread, not a checklist entry.
function renderMemberThread(member) {
  if (member) {
    return '<p class="ob-member-thread ob-member-yes">Member ✓</p>';
  }
  return '<p class="ob-member-thread ob-member-no">' +
    '<a href="#offers" class="ob-upgrade-link">Upgrade to unlock deeper matching &amp; tools</a></p>';
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
if (typeof module !== 'undefined' && module.exports) { module.exports = { renderOnboarding: renderOnboarding }; }

// Browser: fetch + mount. Token is the last path segment of /portal/<token>.
// Hides (empties) the mount when the flag is off.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-onboarding-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    var token = m[1];

    function loadAndRender() {
      return fetch('/api/portal/' + token + '/onboarding')
        .then(function (r) { return r.ok ? r.json() : {enabled:false, status:null}; })
        .then(function (d) { mount.innerHTML = d.enabled ? renderOnboarding(d.status) : ''; })
        .catch(function () {});
    }

    function _num(v) {
      if (v === '' || v == null) return undefined;
      var n = parseFloat(v);
      return isNaN(n) ? undefined : n;
    }

    // Builds the JSON payload from the form's live DOM values and POSTs it.
    // CRITICAL: on_meds / field_loss go over the wire as real JSON booleans
    // (checkbox.checked), never strings -- the server does
    // bool(answers.get("on_meds")), so a string like "false" would be truthy.
    function submitTriage(form, extra) {
      var msg = form.querySelector('.ob-triage-msg');
      var payload = {condition: form.getAttribute('data-condition') || 'glaucoma'};
      var iopOd = _num((form.querySelector('.ob-iop-od') || {}).value);
      var iopOs = _num((form.querySelector('.ob-iop-os') || {}).value);
      if (iopOd !== undefined) payload.iop_od = iopOd;
      if (iopOs !== undefined) payload.iop_os = iopOs;
      var onMeds = !!(form.querySelector('.ob-on-meds') || {}).checked;
      var fieldLoss = !!(form.querySelector('.ob-field-loss') || {}).checked;
      payload.on_meds = onMeds;
      payload.field_loss = fieldLoss;
      if (onMeds) {
        var medCount = _num((form.querySelector('.ob-med-count-input') || {}).value);
        if (medCount !== undefined) payload.med_count = medCount;
      }
      if (extra) { for (var k in extra) { if (extra.hasOwnProperty(k)) payload[k] = extra[k]; } }
      if (msg) { msg.textContent = 'Saving…'; msg.className = 'ob-triage-msg'; }
      fetch('/api/portal/' + token + '/triage', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      })
        .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('bad response')); })
        .then(function () {
          if (msg) {
            msg.textContent = 'Thanks — your starter remedies are ready.';
            msg.className = 'ob-triage-msg ob-triage-ok';
          }
          // Brief pause so the success message is actually seen before the
          // tile re-renders (the history step flips to done and the form
          // disappears once the new status comes back).
          setTimeout(loadAndRender, 900);
        })
        .catch(function () {
          if (msg) {
            msg.textContent = 'Something went wrong — please try again.';
            msg.className = 'ob-triage-msg ob-triage-err';
          }
        });
    }

    // Delegated listeners: the tile's innerHTML is replaced wholesale on every
    // loadAndRender(), so listeners live on the stable mount node, not on the
    // (re-created) form elements themselves.
    mount.addEventListener('submit', function (e) {
      var form = e.target && e.target.closest ? e.target.closest('.ob-triage-form') : null;
      if (!form) return;
      e.preventDefault();
      submitTriage(form);
    });
    mount.addEventListener('click', function (e) {
      var btn = e.target && e.target.closest ? e.target.closest('.ob-triage-cat') : null;
      if (!btn) return;
      var form = btn.closest('.ob-triage-form');
      if (!form) return;
      submitTriage(form, {category: btn.getAttribute('data-category')});
    });

    loadAndRender();
  });
}
