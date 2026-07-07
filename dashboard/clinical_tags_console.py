"""Clinical Tags review console — a local page to confirm/reject the tagger's suggested tags
(the `client_clinical_tags` ledger in e4l.db). Confirm → status='active', confirmed_by='glen';
reject → status='retired'. Read-only chrome reused from biofield_report_html (_page/_e)."""
import datetime as _dt

from dashboard.biofield_report_html import _e, _page

TABLE = "client_clinical_tags"


def _has_table(cx):
    return cx.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone() is not None


def review_queue(cx):
    """Clients that have suggested tags awaiting confirm, most first."""
    if not _has_table(cx):
        return []
    rows = cx.execute(
        f"SELECT t.client_id, c.name, COUNT(*) FROM {TABLE} t "
        f"JOIN e4l_clients c ON c.client_id=t.client_id "
        f"WHERE t.status='suggested' GROUP BY t.client_id ORDER BY COUNT(*) DESC, c.name").fetchall()
    return [{"client_id": r[0], "name": r[1] or f"client {r[0]}", "n": r[2]} for r in rows]


def client_tags(cx, client_id):
    row = cx.execute("SELECT name FROM e4l_clients WHERE client_id=?", (client_id,)).fetchone()
    out = {"name": (row[0] if row else None) or f"client {client_id}",
           "client_id": client_id, "suggested": [], "active": []}
    if not _has_table(cx):
        return out
    for axis, tag, status, conf, source, ev, cb in cx.execute(
            f"SELECT axis,tag,status,confidence,source,evidence,confirmed_by FROM {TABLE} "
            f"WHERE client_id=? AND status IN ('suggested','active') ORDER BY axis,tag", (client_id,)):
        rec = {"axis": axis, "tag": tag, "confidence": conf, "source": source, "evidence": ev, "confirmed_by": cb}
        out["suggested" if status == "suggested" else "active"].append(rec)
    return out


def _set_status(cx, client_id, tags, new_status, extra_sql):
    today = _dt.date.today().isoformat()
    n = 0
    for tag in tags:
        cur = cx.execute(
            f"UPDATE {TABLE} SET status=?, {extra_sql} WHERE client_id=? AND tag=? AND status='suggested'",
            (new_status, today, client_id, tag))
        n += cur.rowcount
    cx.commit()
    return n


def confirm(cx, client_id, tags):
    """Suggested → active, confirmed_by='glen'. Only affects currently-suggested tags."""
    return _set_status(cx, client_id, tags, "active", "confirmed_by='glen', last_seen=?")


def reject(cx, client_id, tags):
    """Suggested → retired (kept for trajectory, not deleted)."""
    return _set_status(cx, client_id, tags, "retired", "retired_at=?")


# --------------------------------------------------------------------------- HTML
def render_queue_html(queue):
    if queue:
        rows = "".join(
            f"<tr><td><a href='/clinical-tags/{q['client_id']}'>{_e(q['name'])}</a></td>"
            f"<td><span class=pill>{q['n']}</span></td></tr>" for q in queue)
    else:
        rows = "<tr><td colspan=2 class=food>Nothing to review — all caught up.</td></tr>"
    body = ("<h1>Clinical Tags — review queue</h1>"
            "<p class=sub>Clients with suggested tags awaiting your confirm.</p>"
            "<table><tr><th>Client</th><th>Suggested</th></tr>" + rows + "</table>")
    return _page("Clinical Tags", body)


def render_client_html(data):
    def srow(r):
        conf = "" if r["confidence"] is None else f"{r['confidence']:.2f}"
        return (f"<tr><td><input type=checkbox name=tags value='{_e(r['tag'])}'></td>"
                f"<td><code>{_e(r['tag'])}</code></td><td>{_e(r['axis'])}</td><td>{conf}</td>"
                f"<td class=food>{_e((r.get('source') or ''))} · {_e((r.get('evidence') or ''))}</td></tr>")
    sugg = data["suggested"]
    if sugg:
        form = (
            "<form method=post>"
            "<table><tr><th><input type=checkbox onclick=\"for(c of "
            "document.querySelectorAll('input[name=tags]'))c.checked=this.checked\"></th>"
            "<th>Tag</th><th>Axis</th><th>Conf</th><th>Source · evidence</th></tr>"
            + "".join(srow(r) for r in sugg) + "</table>"
            "<button class=btn name=action value=confirm type=submit>&#10003; Confirm selected</button> "
            "<button class=btn name=action value=reject type=submit>&#10005; Reject selected</button>"
            "</form>")
    else:
        form = "<p class=food>No suggested tags — all confirmed, or none yet.</p>"
    active = data["active"]
    arows = "".join(
        f"<tr><td><code>{_e(r['tag'])}</code></td><td>{_e(r['axis'])}</td>"
        f"<td>{'&#10003; glen' if r.get('confirmed_by') else 'auto'}</td></tr>" for r in active
    ) or "<tr><td colspan=3 class=food>none</td></tr>"
    body = ("<p><a href='/clinical-tags'>&larr; review queue</a></p>"
            f"<h1>{_e(data['name'])}</h1>"
            "<h2>Suggested — confirm or reject</h2>" + form +
            "<h2>Active</h2><table><tr><th>Tag</th><th>Axis</th><th>Confirmed</th></tr>" + arows + "</table>")
    return _page("Clinical Tags — " + data["name"], body)
