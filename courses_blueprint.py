from __future__ import annotations

import os
import sqlite3
import threading
import time as _time
from urllib.parse import urlparse

from flask import Blueprint, request, jsonify, redirect, make_response, render_template_string, abort

from dashboard import courses_content as cc
from dashboard import courses_access as ca
from dashboard import courses_identity as cid
from dashboard import course_tokens

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
    return sqlite3.connect(_db_path())


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


def learn_home():
    # Following an emailed link sets the member cookie, then redirects clean.
    token = request.args.get("token")
    if token:
        resp = make_response(redirect("/learn", code=302))
        resp.set_cookie("mu_token", token, httponly=True, samesite="Lax", max_age=60 * 60 * 24 * 30)
        return resp
    level = _member_level()
    items = []
    for course in cc.list_courses():
        items.append(f'<li><a href="/learn/{course.slug}">{course.title}</a>: {course.description}</li>')
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
    for m in course.modules:
        rows.append(f"<h3>{m.title}</h3><ul>")
        for l in m.lessons:
            state = ca.lock_state(l.access, level)
            if state == "open":
                rows.append(f'<li><a href="/learn/{course.slug}/{m.slug}/{l.slug}">{l.title}</a></li>')
            elif state == "locked_register":
                rows.append(f'<li>{l.title} <a href="/learn#register">(register free)</a></li>')
            else:
                rows.append(f"<li>{l.title} (members-only, upgrade coming soon)</li>")
        rows.append("</ul>")
    body = f'<p><a href="/learn">← All courses</a></p><h1>{course.title}</h1><p>{course.description}</p>{"".join(rows)}'
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
        msg = "Register free to watch this lesson." if state == "locked_register" else "This lesson is for paid members."
        body = f'<p><a href="/learn/{course.slug}">← {course.title}</a></p><h1>{lesson.title}</h1><p>{msg}</p><p><a href="/learn#register">Register</a></p>'
        return render_template_string(_PAGE, title=lesson.title, body=body), 403
    embed = ""
    if lesson.rumble_id:
        embed = (f'<div style="position:relative;padding-bottom:56.25%"><iframe '
                 f'src="https://rumble.com/embed/{lesson.rumble_id}/" frameborder="0" allowfullscreen '
                 f'style="position:absolute;width:100%;height:100%"></iframe></div>')
    dls = "".join(f'<li><a href="{d.get("url","")}">{d.get("label","Download")}</a></li>' for d in lesson.downloads)
    dls = f"<h3>Resources</h3><ul>{dls}</ul>" if dls else ""
    body = (f'<p><a href="/learn/{course.slug}">← {course.title}</a></p>'
            f'<h1>{lesson.title}</h1>{embed}<div>{cc.render_body(lesson.body_md)}</div>{dls}')
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
