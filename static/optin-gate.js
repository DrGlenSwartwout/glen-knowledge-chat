/* optin-gate.js — the soft consent gate (Tier-0 Visitor → Tier-1 Member).
 *
 * One shared component for every chat/order surface. The SERVER decides when a
 * turn is gated (it emits {"gate": true} in the SSE stream, or returns
 * {"need_optin": true} from checkout). This module only renders the soft
 * opt-in: first name + last name + email + a single Terms-of-Service checkbox.
 * Agreeing POSTs trigger:"tos" to /begin/unlock (which stamps tos_agreed_at),
 * then calls onAgree() so the caller can re-ask / retry.
 *
 * It is deliberately tiny and dependency-free so embed.html, begin-match.html,
 * and begin-buy.html can all `<script src="/optin-gate.js"></script>` it.
 */
(function () {
  if (window.OptinGate) return;

  var TERMS_URL = 'https://illtowell.com/terms';
  var _open = false;

  function _injectStyles() {
    if (document.getElementById('optin-gate-styles')) return;
    var css = ''
      + '.og-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99998;}'
      + '.og-card{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);'
      + 'width:min(420px,92vw);background:#15171c;color:#e9eaee;border:1px solid #2a2e37;'
      + 'border-radius:14px;padding:18px 18px 16px;z-index:99999;'
      + 'box-shadow:0 12px 40px rgba(0,0,0,.5);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;}'
      + '.og-card h4{margin:0 0 6px;font-size:15px;font-weight:600;}'
      + '.og-card p{margin:0 0 12px;color:#aab0bd;font-size:12.5px;}'
      + '.og-fields{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}'
      + '.og-fields input{flex:1 1 45%;min-width:120px;background:#0e1014;color:#e9eaee;'
      + 'border:1px solid #2a2e37;border-radius:8px;padding:8px 10px;font-size:13px;}'
      + '.og-fields input:focus{outline:none;border-color:#6ea8fe;}'
      + '.og-tos{display:flex;gap:8px;align-items:flex-start;margin-bottom:8px;font-size:12px;color:#cfd4de;}'
      + '.og-tos input{margin-top:2px;flex-shrink:0;}'
      + '.og-tos a{color:#6ea8fe;text-decoration:underline;}'
      + '.og-err{color:#e08a8a;font-size:11.5px;min-height:1em;margin-bottom:6px;}'
      + '.og-btn{width:100%;background:#6ea8fe;color:#0e1014;border:0;border-radius:8px;'
      + 'padding:10px;font-size:14px;font-weight:600;cursor:pointer;}'
      + '.og-btn:disabled{opacity:.45;cursor:not-allowed;}'
      + '.og-alt{margin-top:10px;text-align:center;font-size:11.5px;color:#8b92a1;}'
      + '.og-alt a{color:#8b92a1;text-decoration:underline;cursor:pointer;}'
      + '.og-ok{color:#8fd19e;font-size:13px;text-align:center;padding:6px 0;}';
    var s = document.createElement('style');
    s.id = 'optin-gate-styles';
    s.textContent = css;
    document.head.appendChild(s);
  }

  function _ls(k) { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } }
  function _lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

  function _close(card, backdrop) {
    _open = false;
    if (card && card.parentNode) card.parentNode.removeChild(card);
    if (backdrop && backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
  }

  /* Show the opt-in. opts: { base, onAgree } */
  function show(opts) {
    opts = opts || {};
    if (_open) return;                       // never stack two gates
    if (document.querySelector('.og-card')) return;
    _open = true;
    _injectStyles();
    var base = opts.base || '';

    var backdrop = document.createElement('div');
    backdrop.className = 'og-backdrop';

    var card = document.createElement('div');
    card.className = 'og-card';

    // Prefill from prior soft opt-in if we have it.
    var priorName = _ls('amg_name');
    var priorFirst = priorName.split(/\s+/)[0] || '';
    var priorLast = priorName.indexOf(' ') > -1 ? priorName.slice(priorName.indexOf(' ') + 1) : '';

    card.innerHTML =
      '<h4>One quick step for personal guidance</h4>'
      + '<p>This part is wellness education, not medical advice. Add your name and agree to '
      + 'the Terms and I can give you specific, individual guidance.</p>'
      + '<div class="og-fields">'
      + '  <input id="og-first" type="text" placeholder="First name" autocomplete="given-name">'
      + '  <input id="og-last" type="text" placeholder="Last name" autocomplete="family-name">'
      + '  <input id="og-email" type="email" placeholder="your@email.com" autocomplete="email" style="flex-basis:100%">'
      + '</div>'
      + '<label class="og-tos"><input id="og-tos" type="checkbox">'
      + '<span>I understand this is wellness coaching and education, not medical advice, and I agree to the '
      + '<a href="' + TERMS_URL + '" target="_blank" rel="noopener">Terms &amp; Conditions</a>.</span></label>'
      + '<div class="og-err" id="og-err"></div>'
      + '<button class="og-btn" id="og-btn">Agree &amp; continue</button>'
      + '<div class="og-alt">Already a member? <a id="og-signin">Sign in from this device</a></div>';

    document.body.appendChild(backdrop);
    document.body.appendChild(card);

    var first = card.querySelector('#og-first');
    var last = card.querySelector('#og-last');
    var email = card.querySelector('#og-email');
    var tos = card.querySelector('#og-tos');
    var err = card.querySelector('#og-err');
    var btn = card.querySelector('#og-btn');

    if (priorFirst) first.value = priorFirst;
    if (priorLast) last.value = priorLast;
    if (_ls('amg_email')) email.value = _ls('amg_email');
    setTimeout(function () { (first.value ? email : first).focus(); }, 30);

    btn.addEventListener('click', function () {
      var f = (first.value || '').trim();
      var l = (last.value || '').trim();
      var e = (email.value || '').trim();
      if (!f || !l) { err.textContent = 'Please enter your first and last name.'; return; }
      if (!e || e.indexOf('@') < 1) { err.textContent = 'Please enter a valid email.'; return; }
      if (!tos.checked) { err.textContent = 'Please check the box to agree to the Terms.'; return; }
      err.textContent = '';
      btn.disabled = true;
      btn.textContent = 'One moment…';

      fetch(base + '/begin/unlock', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trigger: 'tos', first_name: f, last_name: l, email: e, tos: true
        })
      }).then(function (r) { return r.ok ? r.json().catch(function () { return {}; }) : null; })
        .then(function () {
          _lsSet('amg_email', e);
          _lsSet('amg_name', (f + ' ' + l).trim());
          _lsSet('amg_tos_agreed', '1');
          card.innerHTML = '<div class="og-ok">You’re all set. One moment…</div>';
          setTimeout(function () {
            _close(card, backdrop);
            if (typeof opts.onAgree === 'function') opts.onAgree({ first: f, last: l, email: e });
          }, 600);
        })
        .catch(function () {
          btn.disabled = false;
          btn.textContent = 'Agree & continue';
          err.textContent = 'Connection hiccup. Please try again.';
        });
    });

    // Optional cross-device return: send a magic-link sign-in email.
    card.querySelector('#og-signin').addEventListener('click', function () {
      var e = (email.value || '').trim();
      if (!e || e.indexOf('@') < 1) { err.textContent = 'Enter your email above, then tap Sign in.'; email.focus(); return; }
      err.textContent = '';
      fetch(base + '/auth/magic-link/request', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: e, name: _ls('amg_name') })
      }).then(function (r) { return r.json().catch(function () { return {}; }); })
        .then(function (d) {
          card.innerHTML = '<div class="og-ok">' + (d && d.note ? d.note : 'Check your email for a sign-in link.') + '</div>';
          setTimeout(function () { _close(card, backdrop); }, 2200);
        })
        .catch(function () { err.textContent = 'Could not send the link. Please try again.'; });
    });
  }

  /* SSE convenience: call inside a stream parse loop. */
  function onSSE(evt, opts) {
    if (evt && evt.gate) { show(opts); return true; }
    return false;
  }

  /* Checkout convenience: call with the parsed checkout JSON. */
  function onCheckout(data, opts) {
    if (data && data.need_optin) { show(opts); return true; }
    return false;
  }

  window.OptinGate = { show: show, onSSE: onSSE, onCheckout: onCheckout };
})();
