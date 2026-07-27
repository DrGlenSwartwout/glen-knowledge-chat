from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time as _time
from pathlib import Path
from urllib.parse import urlparse

from flask import (Blueprint, request, jsonify, redirect, make_response,
                    render_template_string, abort, send_from_directory)

try:
    from markupsafe import escape
except ImportError:  # pragma: no cover - Flask always pulls markupsafe in
    from html import escape

from dashboard import courses_content as cc
from dashboard import courses_access as ca
from dashboard import courses_identity as cid
from dashboard import course_tokens
from dashboard import module_certifications
from dashboard import stripe_pay
from dashboard import bodymap_homework
from dashboard.courses_sanitize import sanitize_html

courses_bp = Blueprint("courses", __name__)
_write_lock = threading.Lock()
_STATIC_DIR = Path(__file__).resolve().parent / "static"

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


def _resolve_learner_email():
    token = request.args.get("token") or request.cookies.get("mu_token")
    if not token:
        return ""
    cx = _connect()
    try:
        return (course_tokens.resolve_course_token(cx, token) or "").strip().lower()
    finally:
        cx.close()


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · MentorshipU</title></head>
<body style="font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1c2b26">
{{ body|safe }}
<script src="/ref-capture.js"></script>
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
    # (STRIPE_MEMBERSHIP_PRICE_ID); it's the live drip button once Build #2's price
    # is set, no code change needed. Same for the 12-month plan.
    membership_btn = ""
    if os.environ.get("STRIPE_MEMBERSHIP_PRICE_ID", "").strip():
        membership_btn = (' <button type="button" onclick="muCheckout(\'membership\')">'
                          'Join monthly, $99 a month, a new module unlocks every month</button>')
    plan_btn = ""
    if os.environ.get("STRIPE_PLAN_PRICE_ID", "").strip():
        plan_btn = (' <button type="button" onclick="muCheckout(\'plan\')">'
                    'Pay over 12 months, $297 a month, full access from day one</button>')
    return (
        '<div class="enroll">'
        '<p>Unlock all twelve certification modules and learn the full Accelerated Self Healing method '
        'at your own pace, for life.</p>'
        '<p>'
        '<button type="button" onclick="muCheckout(\'onetime\')">Pay in full, $2,997 (save $567)</button>'
        + plan_btn + membership_btn +
        '</p>'
        '<script>function muCheckout(p){fetch("/api/courses/checkout",{method:"POST",'
        'headers:{"Content-Type":"application/json"},'
        'body:JSON.stringify({product:p,ref:(window.getRef?getRef():"")})})'
        '.then(function(r){return r.json()}).then(function(d){'
        'if(d.url){location.href=d.url}else{alert(d.error||"Checkout is not available right now.")}})}'
        '</script></div>')


def _paid_module_open(cx, email, course_obj, module_slug) -> bool:
    """Per-module access for a `paid` module: full-cert holder, OR the module is
    completed (banked for life), OR unlocked and the learner's drip membership
    is currently active. Anonymous (email == "") is always False."""
    from dashboard import course_entitlements, course_module_unlocks, course_progress
    module = next((m for m in course_obj.modules if m.slug == module_slug), None)
    if module is None:
        return False
    lesson_slugs = [l.slug for l in module.lessons]
    full_cert = course_entitlements.paid_level_for(cx, email) == 2
    completed = course_progress.module_completed(cx, email, course_obj.slug, module_slug, lesson_slugs)
    unlocked = module_slug in course_module_unlocks.unlocked_modules(cx, email, course_obj.slug)
    drip = course_entitlements.drip_active(cx, email)
    return ca.module_access(full_cert=full_cert, completed=completed, unlocked=unlocked, drip_active=drip)


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
    email = _resolve_learner_email()
    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return render_template_string(_PAGE, title="Not found", body="<h1>Course not found</h1>"), 404
    rows = []
    has_locked_paid = False
    cx = _connect()
    try:
        paid_open_cache = {}
        for m in course.modules:
            rows.append(f"<h3>{escape(m.title)}</h3><ul>")
            for l in m.lessons:
                if l.access == "paid":
                    if m.slug not in paid_open_cache:
                        paid_open_cache[m.slug] = _paid_module_open(cx, email, course, m.slug)
                    state = "open" if paid_open_cache[m.slug] else "locked_upgrade"
                else:
                    state = ca.lock_state(l.access, level)
                if state == "open":
                    rows.append(f'<li><a href="/learn/{course.slug}/{m.slug}/{l.slug}">{escape(l.title)}</a></li>')
                elif state == "locked_register":
                    rows.append(f'<li>{escape(l.title)} <a href="/learn#register">(register free)</a></li>')
                else:  # locked_upgrade
                    rows.append(f"<li>{escape(l.title)} (certification module)</li>")
                    has_locked_paid = True
            rows.append("</ul>")
    finally:
        cx.close()
    body = f'<p><a href="/learn">← All courses</a></p><h1>{escape(course.title)}</h1><p>{escape(course.description)}</p>{"".join(rows)}'
    if has_locked_paid:
        body += _enroll_panel(course)
    return render_template_string(_PAGE, title=course.title, body=body)


@courses_bp.route("/course-bodymap.js")
def course_bodymap_js():
    # Serves the Body-Map homework widget (static/course-bodymap.js). Host-gated
    # by the blueprint's before_request like every other courses_bp route.
    resp = send_from_directory(_STATIC_DIR, "course-bodymap.js")
    resp.headers["Content-Type"] = "application/javascript"
    return resp


# module slug -> the interactive homework tool it uses. Modules not listed here
# get the plain free-text textarea + muHw() path.
_MODULE_TOOL = {"02-body": "bodymap"}


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

    email = _resolve_learner_email()
    cx = _connect()
    try:
        if lesson.access == "paid":
            visible = _paid_module_open(cx, email, course, module_slug)
            locked_state = "locked_upgrade"
        else:
            visible = ca.is_visible(lesson.access, level)
            locked_state = ca.lock_state(lesson.access, level)
        if not visible:
            head = (f'<p><a href="/learn/{course.slug}">← {escape(course.title)}</a></p>'
                    f'<h1>{escape(lesson.title)}</h1>')
            if locked_state == "locked_upgrade":
                body = head + _enroll_panel(course)
            else:
                body = head + ('<p>Register free to watch this lesson.</p>'
                                '<p><a href="/learn#register">Register</a></p>')
            return render_template_string(_PAGE, title=lesson.title, body=body), 403

        prior = None
        if email:
            from dashboard import course_progress as cp
            prior = cp.homework(cx, email, course.slug, module_slug)
    finally:
        cx.close()
    safe_body = sanitize_html(lesson.body_md)
    dls = "".join(
        f'<li><a href="{escape(d.get("url",""))}">{escape(d.get("label","Download"))}</a></li>'
        for d in lesson.downloads
    )
    dls = f"<h3>Resources</h3><ul>{dls}</ul>" if dls else ""
    body = (f'<p><a href="/learn/{course.slug}">← {escape(course.title)}</a></p>'
            f'<h1>{escape(lesson.title)}</h1><div>{safe_body}</div>{dls}')
    watch_js = (
        f'<p><button type="button" id="mu-watched" onclick="muWatched()">Mark watched</button> '
        f'<span id="mu-watched-ok"></span></p>'
        f'<script>'
        f'function muWatched(){{fetch("/api/courses/{course.slug}/{module_slug}/{lesson_slug}/watched"'
        f'+location.search,{{method:"POST"}}).then(function(r){{if(r.ok)'
        f'document.getElementById("mu-watched-ok").textContent="Marked watched.";}});}}'
        # YouTube IFrame API: auto-mark when the embedded video ends.
        f'(function(){{var t=document.createElement("script");t.src="https://www.youtube.com/iframe_api";'
        f'document.head.appendChild(t);window.onYouTubeIframeAPIReady=function(){{'
        f'var f=document.querySelector("iframe[src*=\\"youtube\\"]");if(!f)return;'
        f'try{{new YT.Player(f,{{events:{{onStateChange:function(e){{if(e.data===0)muWatched();}}}}}});}}catch(_){{}}}};}})();'
        f'</script>')
    body += watch_js
    assignment = escape(_HOMEWORK_ASSIGNMENT.get(module_slug, _HOMEWORK_DEFAULT))
    prior_html = ""
    if prior and prior.get("ai_feedback"):
        prior_html = (f'<p class="mu-fb"><b>Feedback:</b> {escape(prior["ai_feedback"])}'
                      + (f' <i>({escape(prior["ai_rating"])})</i>' if prior.get("ai_rating") else "") + '</p>')
    if _MODULE_TOOL.get(module_slug) == "bodymap":
        mount_opts = {
            "system": "organs",
            "priorPayload": (prior or {}).get("payload") or "",
            "submitUrl": f"/api/courses/{course.slug}/{module_slug}/homework",
        }
        # Safely embed server-derived text in an inline <script>: JSON-encode,
        # then neutralize </script>/HTML-comment breakout sequences (matches the
        # `_safe = json.dumps(...).replace(...)` convention used elsewhere in app.py).
        mount_json = (json.dumps(mount_opts).replace("<", "\\u003c")
                      .replace(">", "\\u003e").replace("&", "\\u0026"))
        hw_html = (
            f'<div class="mu-hw"><h3>Homework</h3><p>{assignment}</p>'
            f'<div id="mu-hw-bodymap"></div>'
            f'{prior_html}'
            f'<script src="/course-bodymap.js"></script>'
            f'<script>CourseBodyMap.mount(document.getElementById("mu-hw-bodymap"), {mount_json});</script>'
            f'</div>')
    else:
        hw_html = (
            f'<div class="mu-hw"><h3>Homework</h3><p>{assignment}</p>'
            f'<textarea id="mu-hw" rows="6" style="width:100%">{escape((prior or {}).get("payload") or "")}</textarea>'
            f'<p><button type="button" onclick="muHw()">Submit homework</button> <span id="mu-hw-ok"></span></p>'
            f'{prior_html}'
            f'<script>function muHw(){{fetch("/api/courses/{course.slug}/{module_slug}/homework"+location.search,'
            f'{{method:"POST",headers:{{"Content-Type":"application/json"}},'
            f'body:JSON.stringify({{payload:document.getElementById("mu-hw").value}})}})'
            f'.then(function(r){{return r.json()}}).then(function(d){{'
            f'document.getElementById("mu-hw-ok").textContent=d.ok?("Submitted."+(d.feedback?" "+d.feedback:"")):(d.error||"Error");}});}}'
            f'</script></div>')
    body += hw_html

    cert_price_set = bool(os.environ.get("STRIPE_MODULE_CERT_PRICE_ID", "").strip())
    import app as _appmod  # late import: the certifiable-module set
    cert_certifiable = (course_slug != _appmod._CERT_COURSE
                        or module_slug in _appmod._CERT_REQUIRED_MODULES)
    if email and cert_price_set and cert_certifiable:
        cert_module = next((m for m in course.modules if m.slug == module_slug), None)
        cert_lesson_slugs = [l.slug for l in cert_module.lessons] if cert_module else []
        cx2 = _connect()
        try:
            from dashboard import course_progress as cp
            cert_completed = cp.module_completed(cx2, email, course_slug, module_slug, cert_lesson_slugs)
            cert_status = (module_certifications.status_for(cx2, email, course_slug, module_slug)
                           if cert_completed else None)
        finally:
            cx2.close()
        if cert_completed:
            if cert_status == "approved":
                body += '<div class="mu-cert"><p><b>Module certified.</b></p></div>'
            elif cert_status == "pending":
                body += '<div class="mu-cert"><p>Certification submitted. It is pending review.</p></div>'
            else:
                body += (
                    '<div class="mu-cert"><h3>Certify this module</h3>'
                    '<p>Certification for this module is $200.</p>'
                    '<button type="button" onclick="muCertify()">Certify this module, $200</button> '
                    '<span id="mu-cert-ok"></span>'
                    f'<script>function muCertify(){{fetch("/api/courses/{course_slug}/{module_slug}/certify"'
                    f'+location.search,{{method:"POST",headers:{{"Content-Type":"application/json"}},'
                    f'body:JSON.stringify({{ref:(window.getRef?getRef():"")}})}})'
                    f'.then(function(r){{return r.json()}}).then(function(d){{if(d.url){{location.href=d.url}}'
                    f'else{{document.getElementById("mu-cert-ok").textContent=(d.error||"Not available right now.");}}}});}}'
                    f'</script></div>'
                )
    return render_template_string(_PAGE, title=lesson.title, body=body)


# module slug -> the homework assignment prompt shown to the learner + given to the AI.
# Body uses the interactive Body Map tool (see _MODULE_TOOL); the rest fill in on
# the drip cadence (Family: chart family health challenges; etc.).
_HOMEWORK_ASSIGNMENT = {
    "02-body": "On the body map below, mark the areas where you feel concern, tension, "
               "symptoms, or a history of trouble. Add a short note on any area that needs "
               "one, then add an overall reflection on what you are noticing.",
}
_HOMEWORK_DEFAULT = "Your top takeaways from this module and one concrete action you will apply."


@courses_bp.route("/api/courses/<course_slug>/<module_slug>/homework", methods=["POST"])
def courses_submit_homework(course_slug, module_slug):
    from dashboard import course_progress as cp
    from dashboard import homework_analysis
    email = _resolve_learner_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    payload = ((request.get_json(silent=True) or {}).get("payload") or "").strip()
    if not payload:
        return jsonify({"error": "empty"}), 400
    parsed_bodymap = bodymap_homework.parse_marks(payload)
    if parsed_bodymap is not None and not bodymap_homework.has_content(parsed_bodymap):
        return jsonify({"error": "empty"}), 400
    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return jsonify({"error": "not found"}), 404
    module = next((m for m in course.modules if m.slug == module_slug), None)
    if module is None:
        return jsonify({"error": "not found"}), 404
    assignment = _HOMEWORK_ASSIGNMENT.get(module_slug, _HOMEWORK_DEFAULT)
    cx = _connect()
    try:
        # Paywall gate: a module is "paid" if any of its lessons is. A learner
        # without current access (full cert, banked completion, or active drip
        # unlock) cannot bank a completion by submitting homework for free.
        is_paid_module = any(l.access == "paid" for l in module.lessons)
        if is_paid_module and not _paid_module_open(cx, email, course, module_slug):
            return jsonify({"error": "forbidden"}), 403
        fb = homework_analysis.analyze(module_slug, assignment, payload)  # advisory, never raises
        cp.record_homework(cx, email, course_slug, module_slug, payload,
                           ai_rating=fb.get("rating") or None, ai_feedback=fb.get("feedback") or None)
    finally:
        cx.close()
    return jsonify({"ok": True, "rating": fb.get("rating", ""), "feedback": fb.get("feedback", "")})


@courses_bp.route("/api/courses/<course_slug>/<module_slug>/<lesson_slug>/watched", methods=["POST"])
def courses_mark_watched(course_slug, module_slug, lesson_slug):
    from dashboard import course_progress as cp
    email = _resolve_learner_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return jsonify({"error": "not found"}), 404
    lesson = None
    for m in course.modules:
        if m.slug == module_slug:
            for l in m.lessons:
                if l.slug == lesson_slug:
                    lesson = l
    if lesson is None:
        return jsonify({"error": "not found"}), 404
    cx = _connect()
    try:
        # Paywall gate: a free registered learner cannot forge "watched" on a
        # paid lesson they never unlocked and bank the module's completion.
        if lesson.access == "paid" and not _paid_module_open(cx, email, course, module_slug):
            return jsonify({"error": "forbidden"}), 403
        cp.mark_watched(cx, email, course_slug, module_slug, lesson_slug)
    finally:
        cx.close()
    return jsonify({"ok": True})


@courses_bp.route("/api/courses/<course_slug>/unlock-next", methods=["POST"])
def courses_unlock_next(course_slug):
    """Self-select which paid module unlocks next on the learner's next drip
    charge (consumed once by _drip_unlock_next in app.py). Token-gated."""
    from dashboard import course_module_unlocks as cmu
    email = _resolve_learner_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401
    module = ((request.get_json(silent=True) or {}).get("module") or "").strip()
    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return jsonify({"error": "not found"}), 404
    paid_modules = {m.slug for m in course.modules if any(l.access == "paid" for l in m.lessons)}
    if module not in paid_modules:
        return jsonify({"error": "unknown module"}), 400
    cx = _connect()
    try:
        cmu.set_unlock_pref(cx, email, course_slug, module)
    finally:
        cx.close()
    return jsonify({"ok": True})


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

    # Ambassador ref passthrough (rm_ref cookie doesn't cross to this host, so
    # the client resends it in the POST body). Crediting lands in a later task
    # — here we only stamp it into Stripe metadata for the session (all modes)
    # and the subscription (membership/plan), so it survives to the webhook.
    ref = (data.get("ref") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ref):
        ref = ""
    if ref:
        metadata["ref"] = ref
        if sub_md is not None:
            sub_md["ref"] = ref
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


@courses_bp.route("/api/courses/<course_slug>/<module_slug>/certify", methods=["POST"])
def courses_certify_module(course_slug, module_slug):
    """Checkout for a $200 module certification, gated on the learner having
    already completed the module (all lessons watched + homework submitted).
    Lands PENDING via the Stripe webhook fulfiller (record_purchase); this
    route only creates the Checkout Session."""
    import app as appmod  # late import: only for mentorship_base()
    from dashboard import course_progress as cp

    email = _resolve_learner_email()
    if not email:
        return jsonify({"error": "unauthorized"}), 401

    price_id = os.environ.get("STRIPE_MODULE_CERT_PRICE_ID", "").strip()
    if not (_stripe_active() and price_id):
        return jsonify({"error": "not available"}), 503

    try:
        course = cc.load_course(course_slug)
    except FileNotFoundError:
        return jsonify({"error": "not found"}), 404
    module = next((m for m in course.modules if m.slug == module_slug), None)
    if module is None:
        return jsonify({"error": "not found"}), 404
    # On the real certification course, only the 12 certifiable modules count
    # toward the credential; refuse to sell a dead-end certification for a
    # non-required module (e.g. the intro).
    if course_slug == appmod._CERT_COURSE and module_slug not in appmod._CERT_REQUIRED_MODULES:
        return jsonify({"error": "not certifiable"}), 404
    lesson_slugs = [l.slug for l in module.lessons]

    cx = _connect()
    try:
        if not cp.module_completed(cx, email, course_slug, module_slug, lesson_slugs):
            return jsonify({"error": "module not completed"}), 403
        st = module_certifications.status_for(cx, email, course_slug, module_slug)
    finally:
        cx.close()
    if st in ("pending", "approved"):
        return jsonify({"error": f"already {st}"}), 409

    data = request.get_json(silent=True) or {}
    ref = (data.get("ref") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ref):
        ref = ""
    metadata = {"kind": "module_certification", "email": email,
                "course": course_slug, "module": module_slug}
    if ref:
        metadata["ref"] = ref

    base = appmod.mentorship_base()
    try:
        sess = stripe_pay.create_price_checkout_session(
            price_id, mode="payment", customer_email=email,
            metadata=metadata,
            success_url=f"{base}/learn/{course_slug}/{module_slug}?certified=1",
            cancel_url=f"{base}/learn/{course_slug}/{module_slug}")
    except Exception:
        appmod.app.logger.exception("module certification checkout failed")
        return jsonify({"error": "checkout failed"}), 502
    return jsonify({"url": sess.get("url")})
