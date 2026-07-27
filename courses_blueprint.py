from __future__ import annotations

import os
import sqlite3
import threading
import time as _time
from urllib.parse import urlparse

from flask import Blueprint, request, jsonify, redirect, make_response, render_template_string, abort

try:
    from markupsafe import escape
except ImportError:  # pragma: no cover - Flask always pulls markupsafe in
    from html import escape

from dashboard import courses_content as cc
from dashboard import courses_access as ca
from dashboard import courses_identity as cid
from dashboard import course_tokens
from dashboard import stripe_pay
from dashboard.courses_sanitize import sanitize_html

courses_bp = Blueprint("courses", __name__)
_write_lock = threading.Lock()

_RL_WINDOW_S = 3600
_RL_MAX_PER_IP = 10
_RL_MAX_PER_EMAIL = 3


def _rate_limited(cx, ip: str, email: str) -> bool:
    """sqlite-backed sliding window: caps intake attempts per-IP and per-email
    within _RL_WINDOW_S. sqlite-backed (not in-memory) so the limit survives
    across gunicorn worker processes."""
    cx.execute("CREATE TABLE IF NOT EXISTS course_intake_rl(k TEXT, ts REAL)")
    now = _time.time()
    cx.execute("DELETE FROM course_intake_rl WHERE ts < ?", (now - _RL_WINDOW_S,))
    ip_n = cx.execute("SELECT COUNT(*) FROM course_intake_rl WHERE k=?", (f"ip:{ip}",)).fetchone()[0]
    em_n = cx.execute("SELECT COUNT(*) FROM course_intake_rl WHERE k=?", (f"em:{email}",)).fetchone()[0]
    if ip_n >= _RL_MAX_PER_IP or em_n >= _RL_MAX_PER_EMAIL:
        cx.commit()
        return True
    cx.execute("INSERT INTO course_intake_rl(k, ts) VALUES(?,?)", (f"ip:{ip}", now))
    cx.execute("INSERT INTO course_intake_rl(k, ts) VALUES(?,?)", (f"em:{email}", now))
    cx.commit()
    return False


def _mentorship_host() -> str:
    mb = os.environ.get("MENTORSHIP_BASE_URL", "")
    return (urlparse(mb).hostname or "") if mb else ""


@courses_bp.before_request
def _gate_to_mentorship_host():
    host = _mentorship_host()
    req_host = (request.host or "").split(":")[0].lower()
    # Fail closed: if no mentorship host is configured, the blueprint serves nowhere,
    # so an unset MENTORSHIP_BASE_URL in prod cannot expose course routes on illtowell.com.
    if not host or req_host != host.lower():
        abort(404)


def _db_path():
    return os.path.join(os.environ.get("DATA_DIR", "."), "chat_log.db")


def _connect():
    from dashboard import db
    return db.connect(_db_path())


def _member_level():
    token = request.args.get("token") or request.cookies.get("mu_token")
    if not token:
        return 0
    cx = _connect()
    try:
        return cid.member_level_for(cx, token)
    finally:
        cx.close()


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · MentorshipU</title></head>
<body style="font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1c2b26">
{{ body|safe }}
</body></html>"""


# NOTE: `/learn` and `/learn/<course_slug>` are NOT registered on this blueprint.
# They collide with app.py's own /learn topic-page routes, and because the
# blueprint is registered first Werkzeug would let the blueprint win on every
# host, permanently shadowing illtowell's topic pages. Instead app.py's
# learn_index / learn_topic_page delegate here via _on_mentorship_host(). These
# stay plain module-level functions for that delegation.
_REGISTER_FORM = """
<div id="register" style="margin-top:2rem;padding:1.25rem;border:1px solid #cfe0da;border-radius:8px;background:#f6faf8">
  <h2 style="margin-top:0">Welcome. Register free</h2>
  <p>Leave your name and email below and we will send you an access link to unlock the member lessons. No cost, no pressure.</p>
  <form id="mu-register-form">
    <p><label>Name<br><input type="text" name="name" id="mu-name" style="width:100%;padding:.4rem"></label></p>
    <p><label>Email<br><input type="email" name="email" id="mu-email" required style="width:100%;padding:.4rem"></label></p>
    <p><label><input type="checkbox" name="tos_agreed" id="mu-tos" required> I agree to be contacted about my free access link.</label></p>
    <input type="text" name="company" id="mu-company" style="position:absolute;left:-9999px" tabindex="-1" autocomplete="off">
    <p><button type="submit">Register free</button></p>
  </form>
  <p id="mu-register-msg" style="display:none"></p>
</div>
<script>
(function () {
  var form = document.getElementById("mu-register-form");
  if (!form) return;
  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var msg = document.getElementById("mu-register-msg");
    fetch("/api/mentorship/intake/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: document.getElementById("mu-name").value,
        email: document.getElementById("mu-email").value,
        tos_agreed: document.getElementById("mu-tos").checked,
        company: document.getElementById("mu-company").value
      })
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data && data.ok) {
          form.style.display = "none";
          msg.textContent = "Check your email for your access link.";
          msg.style.display = "block";
        } else {
          msg.textContent = "Something did not go through. Please check your email address and try again.";
          msg.style.display = "block";
        }
      })
      .catch(function () {
        msg.textContent = "Something did not go through. Please check your email address and try again.";
        msg.style.display = "block";
      });
  });
})();
</script>
"""


def _enroll_panel(course):
    # The $99/mo membership button shows only when its price is configured
    # (STRIPE_MEMBERSHIP_PRICE_ID). At the Option-1 launch only the one-time cert is
    # live; the membership button stays hidden until Build #2 wires the per-module
    # drip model, then reappears automatically with no code change.
    membership_btn = ""
    if os.environ.get("STRIPE_MEMBERSHIP_PRICE_ID", "").strip():
        membership_btn = (' <button type="button" onclick="muCheckout(\'membership\')">'
                          'Join monthly, $99 a month</button>')
    return (
        '<div class="enroll">'
        '<p>Unlock all twelve certification modules and learn the full Accelerated Self Healing method '
        'at your own pace, for life.</p>'
        '<p>'
        '<button type="button" onclick="muCheckout(\'onetime\')">Get the full certification, $2,997</button>'
        + membership_btn +
        '</p>'
        '<script>function muCheckout(p){fetch("/api/courses/checkout",{method:"POST",'
        'headers:{"Content-Type":"application/json"},body:JSON.stringify({product:p})})'
        '.then(function(r){return r.json()}).then(function(d){'
        'if(d.url){location.href=d.url}else{alert(d.error||"Checkout is not available right now.")}})}'
        '</script></div>')


def learn_home():
    # Following an emailed link sets the member cookie, then redirects clean.
    token = request.args.get("token")
    if token:
        resp = make_response(redirect("/learn", code=302))
        resp.set_cookie("mu_token", token, httponly=True, samesite="Lax", secure=True, max_age=60 * 60 * 24 * 30)
        return resp
    level = _member_level()
    items = []
    for course in cc.list_courses():
        items.append(f'<li><a href="/learn/{course.slug}">{escape(course.title)}</a>: {escape(course.description)}</li>')
    cta = "" if level else '<p><a href="/learn#register">Register free to unlock member lessons</a></p>'
    form = "" if level else _REGISTER_FORM
    body = f"<h1>MentorshipU</h1><ul>{''.join(items)}</ul>{cta}{form}"
    return render_template_string(_PAGE, title="Courses", body=body)


def course_home(course_slug):
    level = _member_level()
    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return render_template_string(_PAGE, title="Not found", body="<h1>Course not found</h1>"), 404
    rows = []
    has_locked_paid = False
    for m in course.modules:
        rows.append(f"<h3>{escape(m.title)}</h3><ul>")
        for l in m.lessons:
            state = ca.lock_state(l.access, level)
            if state == "open":
                rows.append(f'<li><a href="/learn/{course.slug}/{m.slug}/{l.slug}">{escape(l.title)}</a></li>')
            elif state == "locked_register":
                rows.append(f'<li>{escape(l.title)} <a href="/learn#register">(register free)</a></li>')
            else:  # locked_upgrade
                rows.append(f"<li>{escape(l.title)} (certification module)</li>")
                has_locked_paid = True
        rows.append("</ul>")
    body = f'<p><a href="/learn">← All courses</a></p><h1>{escape(course.title)}</h1><p>{escape(course.description)}</p>{"".join(rows)}'
    if has_locked_paid:
        body += _enroll_panel(course)
    return render_template_string(_PAGE, title=course.title, body=body)


@courses_bp.route("/learn/<course_slug>/<module_slug>/<lesson_slug>")
def lesson_page(course_slug, module_slug, lesson_slug):
    level = _member_level()
    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return render_template_string(_PAGE, title="Not found", body="<h1>Not found</h1>"), 404
    lesson = None
    for m in course.modules:
        if m.slug == module_slug:
            for l in m.lessons:
                if l.slug == lesson_slug:
                    lesson = l
    if lesson is None:
        return render_template_string(_PAGE, title="Not found", body="<h1>Lesson not found</h1>"), 404
    if not ca.is_visible(lesson.access, level):
        state = ca.lock_state(lesson.access, level)
        head = (f'<p><a href="/learn/{course.slug}">← {escape(course.title)}</a></p>'
                f'<h1>{escape(lesson.title)}</h1>')
        if state == "locked_upgrade":
            body = head + _enroll_panel(course)
        else:
            body = head + ('<p>Register free to watch this lesson.</p>'
                            '<p><a href="/learn#register">Register</a></p>')
        return render_template_string(_PAGE, title=lesson.title, body=body), 403
    safe_body = sanitize_html(lesson.body_md)
    dls = "".join(
        f'<li><a href="{escape(d.get("url",""))}">{escape(d.get("label","Download"))}</a></li>'
        for d in lesson.downloads
    )
    dls = f"<h3>Resources</h3><ul>{dls}</ul>" if dls else ""
    body = (f'<p><a href="/learn/{course.slug}">← {escape(course.title)}</a></p>'
            f'<h1>{escape(lesson.title)}</h1><div>{safe_body}</div>{dls}')
    return render_template_string(_PAGE, title=lesson.title, body=body)


@courses_bp.route("/api/mentorship/intake/start", methods=["POST"])
def mentorship_intake_start():
    import app as appmod  # late import: only for the sender + base, never at module top
    data = request.get_json(silent=True) or {}
    if (data.get("company") or "").strip():  # honeypot
        return jsonify({"ok": True})
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    if "@" not in email or "." not in email or not data.get("tos_agreed"):
        return jsonify({"ok": False, "error": "invalid"}), 400
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr or "") or "").split(",")[0].strip()
    with _write_lock:
        cx = _connect()
        try:
            if _rate_limited(cx, ip, email):
                return jsonify({"ok": False, "error": "rate_limited"}), 429
            try:
                from dashboard import customers
                customers.find_or_create_by_email(cx, email=email, name=name)  # lead capture
            except Exception:
                appmod.app.logger.exception("mentorship lead capture failed")
            token = course_tokens.mint_course_token(cx, email, name)
        finally:
            cx.close()
    setup_url = f"{appmod.mentorship_base()}/learn?token={token}"
    try:
        appmod.send_mentorship_setup_link(email, name, setup_url)
    except Exception:
        appmod.app.logger.exception("mentorship setup link email failed")
    return jsonify({"ok": True})


def _stripe_active() -> bool:
    return os.environ.get("STRIPE_ACTIVE", "").strip().lower() in ("1", "true", "yes", "on")


_COURSE_PRICE_ENV = {"onetime": "STRIPE_CERT_PRICE_ID", "membership": "STRIPE_MEMBERSHIP_PRICE_ID",
                     "plan": "STRIPE_PLAN_PRICE_ID"}


@courses_bp.route("/api/courses/checkout", methods=["POST"])
def courses_checkout():
    import app as appmod  # late import: only for mentorship_base()
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip().lower()
    if product not in _COURSE_PRICE_ENV:
        return jsonify({"error": "unknown product"}), 400
    price_id = os.environ.get(_COURSE_PRICE_ENV[product], "").strip()
    if not (_stripe_active() and price_id):
        return jsonify({"error": "not available"}), 503

    email = (data.get("email") or "").strip().lower()
    if not email:
        token = request.args.get("token") or request.cookies.get("mu_token")
        if token:
            cx = _connect()
            try:
                email = (course_tokens.resolve_course_token(cx, token) or "").strip().lower()
            finally:
                cx.close()

    base = appmod.mentorship_base()
    mode = "payment" if product == "onetime" else "subscription"
    sub_kind = "course_plan" if product == "plan" else "course_membership"
    metadata = {"kind": "course_purchase", "email": email or None, "product": product}
    sub_md = {"kind": sub_kind, "email": email or None} if mode == "subscription" else None
    try:
        sess = stripe_pay.create_price_checkout_session(
            price_id, mode=mode, customer_email=(email or None), metadata=metadata,
            subscription_metadata=sub_md,
            success_url=f"{base}/learn/ash-certification?enrolled=1",
            cancel_url=f"{base}/learn/ash-certification")
    except Exception:
        appmod.app.logger.exception("courses checkout failed")
        return jsonify({"error": "checkout failed"}), 502
    return jsonify({"url": sess.get("url")})
