from __future__ import annotations

import os
import sqlite3
import threading

from flask import Blueprint, request, jsonify, redirect, make_response, render_template_string

from dashboard import courses_content as cc
from dashboard import courses_access as ca
from dashboard import courses_identity as cid
from dashboard import course_tokens

courses_bp = Blueprint("courses", __name__)
_write_lock = threading.Lock()


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
<title>{{ title }} — MentorshipU</title></head>
<body style="font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1c2b26">
{{ body|safe }}
</body></html>"""


@courses_bp.route("/learn")
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
        items.append(f'<li><a href="/learn/{course.slug}">{course.title}</a> — {course.description}</li>')
    cta = "" if level else '<p><a href="/learn/register">Register free to unlock member lessons</a></p>'
    body = f"<h1>MentorshipU</h1><ul>{''.join(items)}</ul>{cta}"
    return render_template_string(_PAGE, title="Courses", body=body)


@courses_bp.route("/learn/<course_slug>")
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
                rows.append(f'<li>{l.title} <a href="/learn/register">(register free)</a></li>')
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
        body = f'<p><a href="/learn/{course.slug}">← {course.title}</a></p><h1>{lesson.title}</h1><p>{msg}</p><p><a href="/learn/register">Register</a></p>'
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
    with _write_lock:
        cx = _connect()
        try:
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
