// static/js/portal-onboarding.js
// My Onboarding tile: 3-phase progress (Be read / Match remedies / Accelerate healing).
// Consumes GET /api/portal/<token>/onboarding -> {enabled, status:{phases:[{key,title,steps:[{key,label,done,href,soon?}]}], member}}.
function renderOnboarding(status) {
  if (!status || !status.phases || !status.phases.length) return '';
  const phases = status.phases.map(function (ph) {
    const steps = (ph.steps || []).map(function (st) {
      const mark = st.done === true ? '✓' : (st.in_progress ? '◐' : (st.done === false ? '○' : '•'));
      const markClass = st.done === true ? 'ob-mark-done' : (st.in_progress ? 'ob-mark-progress' : (st.done === false ? 'ob-mark-open' : 'ob-mark-resource'));
      var label = escapeHtml(st.label);
      if (st.in_progress) label += ' (in progress)';
      if (st.soon) label += ' (coming soon)';
      var text = st.href
        ? '<a href="' + st.href + '">' + label + '</a>'
        : label;
      return '<li class="ob-step"><span class="ob-mark ' + markClass + '">' + mark + '</span> ' + text + '</li>';
    }).join('');
    var extra = '';
    if (ph.key === 'match') {
      // Ask about every condition with an authored protocol until at least one
      // condition-sourced recommendation has been saved.
      var historyStep = _stepByKey(ph.steps, 'history');
      if (historyStep && historyStep.done === false) {
        extra += renderTriageForm(status);
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

// Condition-history checklist with condition-specific follow-ups. Pure markup --
// submit is wired via a delegated listener in the browser-init block below so
// this function (and renderOnboarding) stays side-effect-free / DOM-free and
// unit-testable under plain Node. All labels/text here are fixed copy (no
// server-provided strings), so nothing needs escaping.
function renderTriageForm(status) {
  status = status || {};
  var conditionSection = '';
  if (!status.history_conditions_done) {
    conditionSection =
      '<div class="ob-history-section" data-history-section="conditions">' +
        '<p class="ob-triage-intro">Do you have any of these eye or vision issues? Check all that apply.</p>' +
        '<div class="ob-condition-list">' +
          _conditionChoice('glaucoma', 'Glaucoma') +
          _conditionChoice('cataract', 'Cataracts') +
          _conditionChoice('macular', 'Macular degeneration') +
          _conditionChoice('dry-eye', 'Dry eye') +
          _conditionChoice('retinitis-pigmentosa', 'Retinitis pigmentosa') +
          _conditionChoice('diabetic-retinopathy', 'Diabetic retinopathy') +
          _conditionChoice('vision-improvement', 'Reduced vision or vision you want to improve') +
          _conditionChoice('other', 'Other') +
        '</div>' +
        _glaucomaFollowup() + _cataractFollowup() +
        _macularFollowup() + _dryEyeFollowup() +
        '<div class="ob-condition-detail" data-condition-detail="other" hidden>' +
          '<p class="ob-followup-title">Tell us what else is going on</p>' +
          '<label class="ob-triage-field">Other eye or vision issue' +
            '<textarea name="other_condition" rows="3"></textarea></label>' +
        '</div>' +
      '</div>';
  }
  var productSection = '';
  if (!status.history_products_done) {
    productSection =
      '<div class="ob-history-section" data-history-section="products">' +
        '<p class="ob-followup-title">What are you currently taking?</p>' +
        '<p class="ob-triage-intro">Check each category that applies. Include the brand and product names so we can review them for analysis and possible supportive remedies.</p>' +
        _productHistoryChoice('prescriptions', 'Prescription medications') +
        _productHistoryChoice('otc', 'Over-the-counter drugs') +
        _productHistoryChoice('supplements', 'Supplements') +
      '</div>';
  }
  var extendedSection = '';
  if (!status.history_extended_done) {
    extendedSection =
      '<div class="ob-history-section" data-history-section="extended">' +
        '<p class="ob-followup-title">More about your health history</p>' +
        '<p class="ob-triage-intro">Check every category that applies and add the requested details.</p>' +
        _extendedHistoryChoice('surgeries', 'Hospitalizations or surgeries',
          'List each procedure or hospitalization, the reason, and your age at the time.') +
        _extendedHistoryChoice('physical_trauma', 'Physical trauma or injuries',
          'Describe what happened, body areas affected, your age, and any lasting effects.') +
        _extendedHistoryChoice('psychoemotional_trauma', 'Psychoemotional trauma or major stress',
          'Describe what happened, your approximate age or year, and any lasting effects.') +
        _extendedHistoryChoice('toxins', 'Toxin or environmental exposures',
          'List the substance, source, duration, approximate dates, and any reaction.') +
        _extendedHistoryChoice('vaccinations', 'COVID or other vaccinations',
          'List the vaccine, brand, date, and any reactions or changes afterward.') +
        _extendedHistoryChoice('family_history', 'Family health history',
          'List the parent or grandparent, condition, maternal or paternal side, and age at onset.') +
        _extendedHistoryChoice('diagnoses', 'Other current or past medical diagnoses',
          'List the diagnosis, whether it is current or past, and your age at onset.') +
        _extendedHistoryChoice('allergies', 'Food or environmental allergies or sensitivities',
          'List each food or environmental sensitivity and the reaction it causes.') +
        _extendedHistoryChoice('dental', 'Dental issues, amalgams, or root canals',
          'List the issue or procedure, tooth if known, reason, and approximate date.') +
        _extendedHistoryChoice('sleep', 'Trouble falling asleep, staying asleep, or waking frequently',
          'Describe the sleep pattern, frequency, and how long it has been happening.') +
      '</div>';
  }
  return '' +
    '<form class="ob-triage-form">' +
      conditionSection +
      productSection +
      extendedSection +
      '<button type="submit" class="ob-triage-submit">Show my starter remedies</button>' +
      '<p class="ob-triage-msg" aria-live="polite"></p>' +
    '</form>';
}

function _extendedHistoryChoice(kind, label, prompt) {
  return '<div class="ob-product-history">' +
    '<label class="ob-triage-check"><input type="checkbox" name="' + kind +
      '_yes" data-extended-kind="' + kind + '"> ' + label + '</label>' +
    '<label class="ob-triage-field ob-product-detail" data-extended-detail="' + kind +
      '" hidden>' + prompt +
      '<textarea name="' + kind + '_text" rows="4" placeholder="One event or item per line"></textarea></label>' +
  '</div>';
}

function _productHistoryChoice(kind, label) {
  return '<div class="ob-product-history">' +
    '<label class="ob-triage-check"><input type="checkbox" name="' + kind +
      '_yes" data-product-kind="' + kind + '"> ' + label + '</label>' +
    '<label class="ob-triage-field ob-product-detail" data-product-detail="' + kind +
      '" hidden>Brand and product names' +
      '<textarea name="' + kind + '_text" rows="3" placeholder="One product per line"></textarea></label>' +
  '</div>';
}

function _conditionChoice(value, label) {
  return '<label class="ob-condition-choice"><input type="checkbox" name="conditions" value="' +
    value + '"> ' + label + '</label>';
}

function _detailCheck(name, label) {
  return '<label class="ob-triage-check"><input type="checkbox" name="' +
    name + '"> ' + label + '</label>';
}

function _glaucomaFollowup() {
  return '<div class="ob-condition-detail" data-condition-detail="glaucoma" hidden>' +
    '<p class="ob-followup-title">A few details about your glaucoma</p>' +
    '<label class="ob-triage-field">What type or pressure pattern were you told you have?' +
      '<select name="category"><option value="">Not sure</option><option value="elevated">Elevated-pressure glaucoma</option><option value="normal">Normal-tension glaucoma</option></select></label>' +
    '<div class="ob-triage-row">' +
      '<label class="ob-triage-field">Right eye pressure (OD)<input type="number" step="0.1" min="0" max="60" name="iop_od" placeholder="e.g. 18"></label>' +
      '<label class="ob-triage-field">Left eye pressure (OS)<input type="number" step="0.1" min="0" max="60" name="iop_os" placeholder="e.g. 18"></label>' +
    '</div>' +
    _detailCheck('on_meds', 'I use eye-pressure-lowering medication') +
    '<label class="ob-triage-field ob-inline-field">If yes, how many medications?<input type="number" min="0" max="20" name="med_count"></label>' +
    _detailCheck('field_loss', 'I have peripheral vision loss') +
  '</div>';
}

function _cataractFollowup() {
  return '<div class="ob-condition-detail" data-condition-detail="cataract" hidden>' +
    '<p class="ob-followup-title">A few details about your cataracts</p>' +
    '<label class="ob-triage-field">What type were you told you have?' +
      '<select name="cataract_type"><option value="">Not sure</option><option value="senile">Age-related / nuclear</option><option value="psc">Posterior subcapsular (PSC)</option></select></label>' +
    '<label class="ob-triage-field ob-inline-field">Your age<input type="number" min="0" max="120" name="age"></label>' +
    _detailCheck('steroids', 'Current or past steroid use') +
    _detailCheck('diabetes', 'Diabetes') +
    _detailCheck('inflammation', 'Chronic inflammation') +
    _detailCheck('radiation', 'Radiation exposure or treatment') +
    _detailCheck('atopy', 'Allergies, asthma, or eczema') +
    _detailCheck('yellow_vision', 'Vision seems more yellow or less blue') +
  '</div>';
}

function _macularFollowup() {
  return '<div class="ob-condition-detail" data-condition-detail="macular" hidden>' +
    '<p class="ob-followup-title">A few details about your macular degeneration</p>' +
    '<label class="ob-triage-field">What type were you told you have?' +
      '<select name="amd_type"><option value="">Not sure</option><option value="dry">Dry</option><option value="wet">Wet</option></select></label>' +
    _detailCheck('injections', 'I receive eye injections for it') +
    _detailCheck('distortion', 'Straight lines look bent, wavy, or distorted') +
  '</div>';
}

function _dryEyeFollowup() {
  return '<div class="ob-condition-detail" data-condition-detail="dry-eye" hidden>' +
    '<p class="ob-followup-title">A few details about your dry eye</p>' +
    '<label class="ob-triage-field">Do your eyes make enough tears?' +
      '<select name="not_enough_tears"><option value="">Not sure</option><option value="false">Yes</option><option value="true">No</option></select></label>' +
    _detailCheck('sjogrens', 'I also have dry mouth or vaginal dryness') +
    _detailCheck('severe', 'My dry eye is severe') +
  '</div>';
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

// Consult nudge (Cataract/Macular triage task): when the triage response
// says consult_recommended (e.g. a wet-AMD resolve), the success message
// additionally invites a Biofield Analysis with Dr. Glen. Kept as a pure,
// DOM-free helper so it's unit-testable under plain Node like renderOnboarding.
function _triageSuccessMessage(consultRecommended) {
  var msg = 'Thanks — your starter remedies are ready.';
  if (consultRecommended) {
    msg += ' Given what you shared, a Biofield Analysis with Dr. Glen would help us take a closer look.';
  }
  return msg;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderOnboarding: renderOnboarding, _triageSuccessMessage: _triageSuccessMessage };
}

// Browser: fetch + mount. Token is the last path segment of /portal/<token>.
// Hides (empties) the mount when the flag is off.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-onboarding-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    var token = m[1];
    var draftKey = 'rm_portal_onboarding_draft:' + token;
    var draftTimer = null;

    function formValues(form) {
      var values = {};
      Array.prototype.forEach.call(form.querySelectorAll('input,select,textarea'), function (field) {
        if (!field.name) return;
        if (field.type === 'checkbox') {
          if (field.name === 'conditions') {
            if (!values.conditions) values.conditions = [];
            if (field.checked) values.conditions.push(field.value);
          } else {
            values[field.name] = field.checked;
          }
        } else {
          values[field.name] = field.value;
        }
      });
      return values;
    }

    function applyFormValues(form, values) {
      values = values || {};
      Array.prototype.forEach.call(form.querySelectorAll('input,select,textarea'), function (field) {
        if (!field.name) return;
        if (field.name === 'conditions') {
          field.checked = (values.conditions || []).indexOf(field.value) !== -1;
        } else if (field.type === 'checkbox') {
          if (values[field.name] !== undefined) field.checked = !!values[field.name];
        } else if (values[field.name] !== undefined && values[field.name] !== null) {
          field.value = values[field.name];
        }
      });
      Array.prototype.forEach.call(form.querySelectorAll('input[name="conditions"]'), function (field) {
        var detail = form.querySelector('[data-condition-detail="' + field.value + '"]');
        if (detail) detail.hidden = !field.checked;
      });
      Array.prototype.forEach.call(form.querySelectorAll('[data-product-kind]'), function (field) {
        var detail = form.querySelector('[data-product-detail="' + field.getAttribute('data-product-kind') + '"]');
        if (detail) detail.hidden = !field.checked;
      });
      Array.prototype.forEach.call(form.querySelectorAll('[data-extended-kind]'), function (field) {
        var detail = form.querySelector('[data-extended-detail="' + field.getAttribute('data-extended-kind') + '"]');
        if (detail) detail.hidden = !field.checked;
      });
    }

    function serverPrefill(status) {
      var prefill = (status && status.history_prefill) || {};
      var values = {};
      var selected = [];
      (prefill.conditions || []).forEach(function (item) {
        if (!item || !item.condition) return;
        selected.push(item.condition);
        Object.keys(item.answers || {}).forEach(function (key) {
          values[key] = item.answers[key];
        });
      });
      values.conditions = selected;
      Object.keys(prefill.products || {}).forEach(function (key) {
        if (key !== 'updated_at') values[key] = prefill.products[key];
      });
      Object.keys(prefill.extended || {}).forEach(function (key) {
        values[key] = prefill.extended[key];
      });
      return values;
    }

    function hydrateDraft(status) {
      var form = mount.querySelector('.ob-triage-form');
      if (!form) return;
      var values = serverPrefill(status);
      try {
        var saved = JSON.parse(localStorage.getItem(draftKey) || 'null');
        if (saved && saved.values) values = Object.assign(values, saved.values);
      } catch (_error) {}
      applyFormValues(form, values);
    }

    function loadAndRender() {
      return fetch('/api/portal/' + token + '/onboarding')
        .then(function (r) { return r.ok ? r.json() : {enabled:false, status:null}; })
        .then(function (d) {
          mount.innerHTML = d.enabled ? renderOnboarding(d.status) : '';
          if (d.enabled) hydrateDraft(d.status);
        })
        .catch(function () {});
    }
    window.refreshPortalOnboarding = loadAndRender;

    mount.addEventListener('click', function (event) {
      var link = event.target.closest && event.target.closest('a[href]');
      if (!link) return;
      var url;
      try { url = new URL(link.href, location.href); } catch (_error) { return; }
      if (url.origin !== location.origin || url.pathname !== location.pathname || !url.hash) return;
      event.preventDefault();
      if (location.hash !== url.hash) history.pushState(null, '', url.hash);
      if (typeof window.applyPortalHash === 'function') window.applyPortalHash();
    });

    function _num(v) {
      if (v === '' || v == null) return undefined;
      var n = parseFloat(v);
      return isNaN(n) ? undefined : n;
    }

    function conditionPayload(form, condition) {
      var detail = form.querySelector('[data-condition-detail="' + condition + '"]');
      var payload = {condition: condition};
      if (!detail) return payload;
      Array.prototype.forEach.call(detail.querySelectorAll('input,select,textarea'), function (field) {
        var value;
        if (field.type === 'checkbox') {
          value = field.checked;
        } else if (field.type === 'number') {
          value = _num(field.value);
        } else if (field.name === 'not_enough_tears') {
          value = field.value === '' ? undefined : field.value === 'true';
        } else {
          value = field.value || undefined;
        }
        if (value !== undefined) payload[field.name] = value;
      });
      return payload;
    }

    function productsPayload(form) {
      var payload = {};
      ['prescriptions', 'otc', 'supplements'].forEach(function (kind) {
        var yes = form.querySelector('input[name="' + kind + '_yes"]');
        var text = form.querySelector('textarea[name="' + kind + '_text"]');
        payload[kind + '_yes'] = !!(yes && yes.checked);
        payload[kind + '_text'] = text ? text.value.trim() : '';
      });
      return payload;
    }

    function extendedHistoryPayload(form) {
      var payload = {};
      [
        'surgeries', 'physical_trauma', 'psychoemotional_trauma', 'toxins',
        'vaccinations', 'family_history', 'diagnoses', 'allergies', 'dental',
        'sleep'
      ].forEach(function (kind) {
        var yes = form.querySelector('input[name="' + kind + '_yes"]');
        var text = form.querySelector('textarea[name="' + kind + '_text"]');
        payload[kind + '_yes'] = !!(yes && yes.checked);
        payload[kind + '_text'] = text ? text.value.trim() : '';
      });
      return payload;
    }

    function submitStarterRemedies(form, selected) {
      var payload = {
        conditions: selected.map(function (condition) {
          return conditionPayload(form, condition);
        }),
        products: productsPayload(form),
        extended: extendedHistoryPayload(form)
      };
      function send(attempt) {
        return fetch('/api/portal/' + token + '/starter-remedies', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        }).then(function (r) {
          return r.json().catch(function () { return {}; }).then(function (body) {
            if (r.status === 503 && attempt < 1) {
              return new Promise(function (resolve) {
                setTimeout(resolve, 800);
              }).then(function () { return send(attempt + 1); });
            }
            if (!r.ok) {
              throw new Error(body.error || 'We could not save your answers.');
            }
            return body;
          });
        });
      }
      return send(0);
    }

    // Delegated listeners: the tile's innerHTML is replaced wholesale on every
    // loadAndRender(), so listeners live on the stable mount node, not on the
    // (re-created) form elements themselves.
    mount.addEventListener('submit', function (e) {
      var form = e.target && e.target.closest ? e.target.closest('.ob-triage-form') : null;
      if (!form) return;
      e.preventDefault();
      var selected = Array.prototype.map.call(
        form.querySelectorAll('input[name="conditions"]:checked'),
        function (el) { return el.value; }
      );
      var msg = form.querySelector('.ob-triage-msg');
      var conditionsRequired = !!form.querySelector(
        '[data-history-section="conditions"]');
      if (conditionsRequired && !selected.length) {
        msg.textContent = 'Please check at least one issue.';
        msg.className = 'ob-triage-msg ob-triage-err';
        return;
      }
      var missingProduct = null;
      ['prescriptions', 'otc', 'supplements'].some(function (kind) {
        var yes = form.querySelector('input[name="' + kind + '_yes"]');
        var text = form.querySelector('textarea[name="' + kind + '_text"]');
        if (yes && yes.checked && (!text || !text.value.trim())) {
          missingProduct = kind;
          return true;
        }
        return false;
      });
      if (missingProduct) {
        msg.textContent = 'Please enter the brand and product names for each checked category.';
        msg.className = 'ob-triage-msg ob-triage-err';
        return;
      }
      var missingExtended = null;
      Array.prototype.some.call(
        form.querySelectorAll('input[data-extended-kind]'),
        function (yes) {
          var kind = yes.getAttribute('data-extended-kind');
          var text = form.querySelector('textarea[name="' + kind + '_text"]');
          if (yes.checked && (!text || !text.value.trim())) {
            missingExtended = kind;
            return true;
          }
          return false;
        }
      );
      if (missingExtended) {
        msg.textContent = 'Please add the requested details for each checked history category.';
        msg.className = 'ob-triage-msg ob-triage-err';
        return;
      }
      var otherText = form.querySelector(
        '[data-condition-detail="other"] textarea[name="other_condition"]');
      if (selected.indexOf('other') !== -1 &&
          (!otherText || !otherText.value.trim())) {
        msg.textContent = 'Please tell us what the other issue is.';
        msg.className = 'ob-triage-msg ob-triage-err';
        return;
      }
      msg.textContent = 'Saving…';
      msg.className = 'ob-triage-msg';
      submitStarterRemedies(form, selected).then(function (body) {
        msg.textContent = _triageSuccessMessage(
          !!(body && body.consult_recommended));
        msg.className = 'ob-triage-msg ob-triage-ok';
        try { localStorage.removeItem(draftKey); } catch (_error) {}
        setTimeout(loadAndRender, 900);
      }).catch(function (error) {
        msg.textContent = (error && error.message) ||
          'Something went wrong — please try again.';
        msg.className = 'ob-triage-msg ob-triage-err';
      });
    });
    mount.addEventListener('change', function (e) {
      var checkbox = e.target;
      if (!checkbox) return;
      if (checkbox.name === 'conditions') {
        var detail = mount.querySelector(
          '[data-condition-detail="' + checkbox.value + '"]');
        if (detail) detail.hidden = !checkbox.checked;
      }
      if (checkbox.getAttribute('data-product-kind')) {
        var productDetail = mount.querySelector(
          '[data-product-detail="' +
          checkbox.getAttribute('data-product-kind') + '"]');
        if (productDetail) productDetail.hidden = !checkbox.checked;
      }
      if (checkbox.getAttribute('data-extended-kind')) {
        var extendedDetail = mount.querySelector(
          '[data-extended-detail="' +
          checkbox.getAttribute('data-extended-kind') + '"]');
        if (extendedDetail) extendedDetail.hidden = !checkbox.checked;
      }
    });
    mount.addEventListener('input', function (e) {
      var form = e.target && e.target.closest ? e.target.closest('.ob-triage-form') : null;
      if (!form) return;
      clearTimeout(draftTimer);
      draftTimer = setTimeout(function () {
        try {
          localStorage.setItem(draftKey, JSON.stringify({
            saved_at: new Date().toISOString(),
            values: formValues(form)
          }));
        } catch (_error) {}
      }, 250);
    });

    loadAndRender();
  });
}
