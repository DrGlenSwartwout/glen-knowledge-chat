"""HTML rendering for the local Biofield Analysis viewer (Glen's Mac only).

Pure string builders so they're unit-testable without Flask. ALL dynamic values
(remedy names, timing, client names) come from FileMaker free-text fields and are
HTML-escaped.
"""
from html import escape as _e

_STYLE = """
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#2a2f3a;--fg:#e8ebf0;--muted:#9aa3b2;--accent:#c9a23a;--ok:#3fb968}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1040px;margin:0 auto;padding:22px}
 a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
 h1{font-size:21px;margin:0 0 2px} h2{font-size:15px;color:var(--muted);margin:22px 0 8px;
   text-transform:uppercase;letter-spacing:.04em}
 .sub{color:var(--muted);margin:0 0 16px}
 table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
   border-radius:10px;overflow:hidden}
 th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:14px}
 th{color:var(--muted);font-weight:600;background:#13161c}
 tr:last-child td{border-bottom:0}
 .lyr{color:var(--accent);font-weight:700;white-space:nowrap}
 .slot{font-weight:600;color:var(--accent);white-space:nowrap;width:130px}
 .food{color:var(--muted);font-size:12px}
 input[type=search]{background:#0c0e12;color:var(--fg);border:1px solid var(--line);
   border-radius:8px;padding:8px 10px;width:280px;font:inherit}
 .pill{display:inline-block;background:#0c0e12;border:1px solid var(--line);border-radius:999px;
   padding:1px 8px;font-size:12px;color:var(--muted)}
</style>
"""


def _page(title, body):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title>{_STYLE}</head>"
            f"<body><div class=wrap>{body}</div></body></html>")


def render_report_html(report):
    c = report.get("client") or {}
    name = _e(c.get("name") or "(unknown)")
    email = _e(c.get("email") or "")
    date = _e(report.get("date") or "")
    head = (f"<p><a href='/'>&larr; All tests</a></p>"
            f"<h1>{name}</h1>"
            f"<p class=sub>{email} &nbsp;&middot;&nbsp; {date} "
            f"&nbsp;&middot;&nbsp; test {_e(report.get('test_id') or '')}</p>")

    # Causal chain table
    rows = ""
    for l in report.get("layers") or []:
        ln = l.get("layer")
        rows += (
            "<tr>"
            f"<td class=lyr>{_e(str(ln)) if ln is not None else '&middot;'}</td>"
            f"<td>{_e(l.get('head') or '')}</td>"
            f"<td>{_e(l.get('most_affected') or '')}</td>"
            f"<td>{_e(l.get('remedy') or '')}</td>"
            f"<td>{_e(l.get('dosage') or '')}</td>"
            f"<td>{_e(l.get('frequency') or '')}</td>"
            f"<td>{_e(l.get('timing') or '')}</td>"
            "</tr>")
    chain = ("<h2>Causal Chain Report</h2>"
             "<table><tr><th>Layer</th><th>Head of Chain</th><th>Most Affected</th>"
             "<th>Remedy</th><th>Dosage</th><th>Frequency</th><th>Timing</th></tr>"
             f"{rows}</table>")

    # Schedule grid
    sched = report.get("schedule") or {}
    entries = sched.get("entries") or []
    placed = [e for e in entries if not e.get("as_directed")]
    srows = ""
    for slot in sched.get("slots") or []:
        here = [e for e in placed if slot in (e.get("slots") or [])]
        if not here:
            continue
        cells = "; ".join(
            f"{_e(e.get('name') or '')} <span class=food>({_e(e.get('dosage') or '')}"
            + (f", {_e(e.get('food'))}" if e.get('food') else "") + ")</span>"
            for e in here)
        srows += f"<tr><td class=slot>{_e(slot)}</td><td>{cells}</td></tr>"
    asdir = [e for e in entries if e.get("as_directed")]
    if asdir:
        cells = "; ".join(
            f"{_e(e.get('name') or '')} <span class=food>({_e(e.get('timing') or 'as directed')})</span>"
            for e in asdir)
        srows += f"<tr><td class=slot>As directed</td><td>{cells}</td></tr>"
    schedule = ("<h2>Remedy Schedule</h2>"
                "<table><tr><th>When</th><th>Take</th></tr>" + srows + "</table>")

    return _page(f"{name} — Biofield Analysis", head + chain + schedule)


def render_list_html(tests, q=""):
    rows = ""
    for t in tests or []:
        rows += (
            "<tr>"
            f"<td><a href='/test/{_e(str(t.get('test_id') or ''))}'>{_e(t.get('name') or '(unknown)')}</a></td>"
            f"<td>{_e(t.get('email') or '')}</td>"
            f"<td>{_e(t.get('date') or '')}</td>"
            f"<td><span class=pill>{_e(str(t.get('layer_count') or 0))}</span></td>"
            "</tr>")
    body = (
        "<h1>Biofield Analysis</h1>"
        "<p class=sub>Causal Chain Reports from your FileMaker data (local).</p>"
        "<form method=get><input type=search name=q placeholder='Search name or email' "
        f"value='{_e(q or '')}' autofocus></form>"
        "<h2>Tests</h2>"
        "<table><tr><th>Client</th><th>Email</th><th>Date</th><th>Remedies</th></tr>"
        f"{rows}</table>")
    return _page("Biofield Analysis", body)
