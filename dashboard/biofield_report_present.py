"""Clean, client-facing presentation of a Biofield report — no editing chrome.
Shared renderer for the print/PDF (and later portal) skins. Pure function:
takes the report dict + narrative text, returns a complete print-styled HTML doc.
Section order is schedule-forward (the printed sheet ships with the bottles)."""

WORDMARK = "Accelerated Self Healing™"
FOOTER = "In wellness, Dr. Glen & Rae · illtowell.com"


def _e(s):
    return (str("" if s is None else s)
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


_CSS = """
  :root { --ink:#1a1a1a; --muted:#6b6b6b; --gold:#9a7a1f; --hair:#e6e1d5; }
  * { box-sizing: border-box; }
  body { font: 14px/1.5 Georgia, 'Times New Roman', serif; color: var(--ink);
         max-width: 760px; margin: 0 auto; padding: 28px; }
  .masthead { text-align: center; border-bottom: 2px solid var(--gold); padding-bottom: 12px; margin-bottom: 18px; }
  .masthead .wordmark { font-weight: 700; letter-spacing: .04em; font-size: 20px; }
  .masthead .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
  h2 { color: var(--gold); font-size: 16px; border-bottom: 1px solid var(--hair); padding-bottom: 3px; margin: 22px 0 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 4px 7px; border-bottom: 1px solid var(--hair); vertical-align: top; }
  th { background: #faf7ef; }
  .slot { white-space: nowrap; font-weight: 600; }
  .food { color: var(--muted); }
  .narrative p { margin: 0 0 9px; }
  .footer { margin-top: 26px; border-top: 1px solid var(--hair); padding-top: 8px;
            text-align: center; color: var(--muted); font-size: 12px; }
  @media print {
    body { padding: 0; max-width: none; }
    h2 { page-break-after: avoid; }
    tr, .narrative p { page-break-inside: avoid; }
  }
"""


def _masthead(report):
    c = report.get("client") or {}
    sub = " · ".join(x for x in (_e(c.get("name")), _e(report.get("date"))) if x)
    return (f'<div class="masthead"><div class="wordmark">{_e(WORDMARK)}</div>'
            f'<div class="sub">Biofield Analysis · {sub}</div></div>')


def _schedule(report):
    sched = report.get("schedule") or {}
    entries = sched.get("entries") or []
    placed = [e for e in entries if not e.get("as_directed")]
    rows = ""
    for slot in sched.get("slots") or []:
        here = [e for e in placed if slot in (e.get("slots") or [])]
        if not here:
            continue
        cells = "; ".join(
            f"{_e(e.get('name'))} <span class=food>({_e(e.get('dosage'))}"
            + (f", {_e(e.get('food'))}" if e.get("food") else "") + ")</span>"
            for e in here)
        rows += f"<tr><td class=slot>{_e(slot)}</td><td>{cells}</td></tr>"
    asdir = [e for e in entries if e.get("as_directed")]
    if asdir:
        cells = "; ".join(
            f"{_e(e.get('name'))} <span class=food>({_e(e.get('timing') or 'as directed')})</span>"
            for e in asdir)
        rows += f"<tr><td class=slot>As directed</td><td>{cells}</td></tr>"
    return ("<h2>Remedy Schedule</h2>"
            "<table><tr><th>When</th><th>Take</th></tr>" + rows + "</table>")


def _narrative(narrative):
    paras = "".join(f"<p>{_e(p.strip())}</p>" for p in (narrative or "").split("\n") if p.strip())
    return f'<h2>Narrative</h2><div class="narrative">{paras}</div>'


def _chain(report):
    from dashboard.biofield_report_html import render_chain_table
    # Grouped by layer (Head/Tail span their remedies) to match the editor + viewer.
    return "<h2>Causal Chain</h2>" + render_chain_table(report.get("layers") or [])


def render_present(report, narrative=""):
    body = (_masthead(report) + _schedule(report) + _narrative(narrative)
            + _chain(report) + f'<div class="footer">{_e(FOOTER)}</div>')
    return (f"<!doctype html><html><head><meta charset=utf-8>"
            f"<title>Biofield Analysis</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")
