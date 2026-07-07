"""Local-only Biofield Analysis viewer (runs on Glen's Mac).

Reads the FileMaker snapshot in chat_log.db (`fmp_snap_*` tables) and serves the
layer-ordered Causal Chain Report + remedy schedule in the browser. All patient
data stays on this machine — DO NOT deploy this to a server.

Setup / refresh:
  1. Open the New App "Remedy Match.fmp12" in FileMaker Pro 18 Advanced.
  2. Extract:  ~/.venvs/fmp-ingest/bin/python3 \
       "$HOME/AI-Training/02 Skills/fmp-applescript-extract.py" \
       --database 'Remedy Match.fmp12' --source newapp \
       --tables clients,client_biofield_test,client_active_main_stress,\
client_active_stress_affects,client_causal_chain,client_causal_chain_sec_affects,\
client_nosode,client_remedy,client_foods,products,products_phases,products_systems,products_items
  3. Load + serve:  python3 biofield_local_app.py --load

Then open http://127.0.0.1:8011
"""
import argparse
import datetime
import os
import sqlite3

from flask import Flask, Response, redirect, request, send_from_directory

from dashboard import biofield_fee
from dashboard.biofield_report import causal_chain_report, list_tests
from dashboard.biofield_report_html import (
    render_author_html, render_e4l_panel, render_fee_panel, render_list_html,
    render_report_html, render_stress_panel, render_suggest_panel)
from dashboard.biofield_e4l import (
    _db_path as _e4l_db_path, fetch_live as _fetch_live,
    scan_context as _scan_context, search_clients as _search_clients)
from dashboard.biofield_narrative import (
    generate_narrative, generate_video_script, get_narrative, get_notes,
    get_video_script, save_narrative, save_notes, save_video_script)
from dashboard.biofield_authoring import (
    add_chain_row, authored_report, confirm_all, confirm_row, create_test,
    delete_chain_row, delete_test, list_authored, remedy_catalog, remedy_dosing,
    resolve_remedy_name, resolve_stress_name, stress_suggestions, stress_vocab,
    update_chain_row, update_header)
from dashboard.biofield_dimensions import (
    DEPTH_KEY, dimension_values, seed_dimensions, tag as dim_tag)
from dashboard.biofield_interpret import interpret_transcript
from dashboard.biofield_report_present import render_present
from dashboard.biofield_report_pdf import report_pdf_bytes, save_report_pdf

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biofield-audio")


def _layer_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _safe_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def openai_complete(system, user):
    """Default narrative LLM. Needs OPENAI_API_KEY in env (run via `doppler run`)."""
    import openai
    model = os.environ.get("BIOFIELD_NARRATIVE_MODEL", "gpt-4o")
    resp = openai.OpenAI().chat.completions.create(
        model=model,
        temperature=0.4,  # grounded, less flowery
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content


def openai_json(system, user):
    """Deterministic JSON completion for the transcript interpreter (temp 0, JSON mode)."""
    import openai
    model = os.environ.get("BIOFIELD_NARRATIVE_MODEL", "gpt-4o")
    resp = openai.OpenAI().chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content


def elevenlabs_tts(text):
    """Render text to mp3 bytes in Glen's cloned voice. Needs ELEVENLABS_API_KEY."""
    import requests
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "jFxSqMckq2I4mET3C5QC")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                 "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        timeout=180)
    r.raise_for_status()
    return r.content


def deepgram_temp_key():
    """Mint a short-lived (1h) Deepgram key so the long-lived key never reaches the
    browser. Needs DEEPGRAM_API_KEY in env (run via `doppler run`)."""
    import requests
    h = {"Authorization": "Token " + os.environ["DEEPGRAM_API_KEY"]}
    pid = requests.get("https://api.deepgram.com/v1/projects", headers=h,
                       timeout=20).json()["projects"][0]["project_id"]
    r = requests.post(f"https://api.deepgram.com/v1/projects/{pid}/keys", headers=h, timeout=20,
                      json={"comment": "biofield-live-session", "scopes": ["usage:write"],
                            "time_to_live_in_seconds": 3600})
    r.raise_for_status()
    return r.json()["key"]


def deepgram_browser_token():
    """Token for the browser's Deepgram socket. Prefer a short-lived key; fall back to
    the env key if this key lacks key-management scope (fine: localhost-only tool, the
    token only ever reaches Glen's own browser)."""
    try:
        return deepgram_temp_key()
    except Exception:
        return os.environ["DEEPGRAM_API_KEY"]


def _default_fetch_profile(email):
    """Best-effort: pull a client's consolidated profile from the prod People hub
    (same endpoint e4l-reveal-push.py:fetch_history uses). Returns {} on any failure."""
    import json as _json
    import urllib.parse
    import urllib.request
    email = (email or "").strip()
    if not email:
        return {}
    try:
        key = os.environ["CONSOLE_SECRET"]
        base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
        url = (f"{base}/api/people?key=" + urllib.parse.quote(key)
               + "&q=" + urllib.parse.quote(email))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        people = _json.load(urllib.request.urlopen(req, timeout=20)).get("people", [])
        return next(
            (p for p in people if (p.get("email") or "").lower() == email.lower()), {})
    except Exception:
        return {}


def _default_fetch_recent_comms(email):
    """Best-effort: pull a client's windowed recent comms from the prod endpoint.
    Returns {} on any failure (incl. missing CONSOLE_SECRET -> no network call)."""
    import json as _json
    import urllib.parse
    import urllib.request
    email = (email or "").strip()
    if not email:
        return {}
    try:
        key = os.environ["CONSOLE_SECRET"]
        base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
        url = (f"{base}/api/people/recent-comms?key=" + urllib.parse.quote(key)
               + "&q=" + urllib.parse.quote(email))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        return _json.load(urllib.request.urlopen(req, timeout=20)) or {}
    except Exception:
        return {}

DEFAULT_DB = os.environ.get(
    "BIOFIELD_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.db"))


def create_app(db_path=DEFAULT_DB, complete=None, tts=None, deepgram_token=None,
               interpret_complete=None, scan_lookup=None, client_search=None,
               fetch_runner=None, fetch_profile=None, fetch_recent_comms=None,
               e4l_db=None, fee_get=None, fee_set=None, fee_clear=None):
    app = Flask(__name__)
    # The clinical-tags ledger lives in the SEPARATE local e4l.db (not the app's chat_log.db).
    e4l_db = e4l_db or _e4l_db_path()
    complete = complete or openai_complete
    tts = tts or elevenlabs_tts
    deepgram_token = deepgram_token or deepgram_browser_token
    interpret_complete = interpret_complete or openai_json
    # E4L scan pull: as soon as the client (email) is known, surface their most recent
    # voice scan. Default reads ~/AI-Training/e4l.db read-only as of today's date;
    # injectable for tests. Never raises -> intake is never blocked.
    scan_lookup = scan_lookup or (
        lambda email: _scan_context(email, datetime.date.today().isoformat()))
    # Name picker + on-demand live fetch (the local mirror lags, so selecting a client
    # can pull their newest scan straight from the live E4L portal). Both injectable.
    client_search = client_search or (lambda q: _search_clients(q))
    fetch_runner = fetch_runner  # None -> fetch_live uses the real scraper+parser
    fetch_profile = fetch_profile or _default_fetch_profile
    fetch_recent_comms = fetch_recent_comms or _default_fetch_recent_comms
    fee_get = fee_get or biofield_fee.default_fee_get
    fee_set = fee_set or biofield_fee.default_fee_set
    fee_clear = fee_clear or biofield_fee.default_fee_clear

    def _report_for(cx, test_id):
        return (authored_report(cx, test_id) if str(test_id).startswith("a")
                else causal_chain_report(cx, test_id))

    def _e4l(cx, test_id):
        """Scan context + rendered panel for a test's stored client email."""
        rep = _report_for(cx, test_id)
        ctx = scan_lookup((rep.get("client") or {}).get("email") or "")
        return ctx, rep

    def _mine_profile(cx, test_id):
        """Mine the client's consolidated People-hub profile into tag stresses.
        Best-effort: returns {"added": n} on success, {"added": 0, "error": ...} on
        any failure. Never raises."""
        from dashboard.biofield_interpret import interpret_stresses
        from dashboard.biofield_profile import mine_profile_stresses
        from dashboard import biofield_stress as _st
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"added": 0, "error": "No client selected yet"}
        try:
            profile = fetch_profile(email) or {}
            labels = mine_profile_stresses(
                profile, lambda t: interpret_stresses(t, interpret_complete))
            added = sum(1 for label in labels
                        if _st.add_stress(cx, test_id, label, source="tag"))
        except Exception as e:
            return {"added": 0, "error": str(e)[:200]}
        return {"added": added}

    def _mine_comms(cx, test_id):
        """Mine the client's recent communications into comm stresses. Best-effort."""
        from dashboard.biofield_interpret import interpret_stresses
        from dashboard.biofield_comms import comms_to_text
        from dashboard import biofield_stress as _st
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"added": 0, "error": "No client selected yet"}
        try:
            ctx = fetch_recent_comms(email) or {}
            text = comms_to_text(ctx)
            labels = interpret_stresses(text, interpret_complete) if text.strip() else []
            added = sum(1 for label in labels if _st.add_stress(cx, test_id, label, source="comm"))
        except Exception as e:
            return {"added": 0, "error": str(e)[:200]}
        return {"added": added}

    def _seed_stresses(cx, test_id, *, force=False, layers=None):
        """Synthesize reveal layers + seed the stress coverage map for this test.
        The ONLY early return is the no-email guard — nothing to mine/seed without
        one.  Scan-seeding is conditional: runs only when a scan is found AND (force
        OR no scan stresses exist yet) — same idempotency as before.  If *layers* is
        supplied they are used directly, skipping the synthesis network call (pass
        from a route that already ran synthesize_reveal_layers).  Synthesis errors
        are swallowed so the intake flow is never blocked.
        Profile mining (source='tag') always runs after the scan block, but at most
        once per session: skipped when tag stresses already exist for this test.
        This means profile mining fires even when no E4L scan is present."""
        from dashboard import biofield_reveal_import as _ri
        from dashboard import biofield_stress as _st
        import datetime as _dt
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return  # only early return: nothing to mine/seed without an email
        _st.init_stress_tables(cx)
        ctx = scan_lookup(email)
        if ctx.get("found"):
            # Scan-seeding: skip if already seeded (unless force)
            if force or not cx.execute(
                    "SELECT 1 FROM biofield_auth_stress WHERE test_id=? AND source='scan' LIMIT 1",
                    (int(str(test_id).lstrip("a") or 0),)).fetchone():
                if layers is not None:
                    coverage = _ri.build_coverage(layers)
                    _st.seed_from_scan(cx, test_id, ctx.get("findings") or [], coverage)
                else:
                    try:
                        res = _ri.synthesize_reveal_layers(email, today=_dt.date.today().isoformat())
                    except Exception:
                        pass  # synthesis failure: skip scan seeding, fall through to profile mining
                    else:
                        coverage = _ri.build_coverage(res.get("layers") or [])
                        _st.seed_from_scan(cx, test_id, ctx.get("findings") or [], coverage)
        # Always-on: mine profile stresses (best-effort, at most once per session).
        # Guard: skip when tag stresses already exist to avoid a redundant HTTP fetch
        # on every header-save.  The explicit /mine-profile route calls _mine_profile
        # directly (unguarded) for on-demand re-mining.
        if not cx.execute(
                "SELECT 1 FROM biofield_auth_stress WHERE test_id=? AND source='tag' LIMIT 1",
                (int(str(test_id).lstrip("a") or 0),)).fetchone():
            try:
                _mine_profile(cx, test_id)
            except Exception:
                pass
        if not cx.execute(
                "SELECT 1 FROM biofield_auth_stress WHERE test_id=? AND source='comm' LIMIT 1",
                (int(str(test_id).lstrip("a") or 0),)).fetchone():
            try:
                _mine_comms(cx, test_id)
            except Exception:
                pass

    with sqlite3.connect(db_path) as _cx:
        seed_dimensions(_cx)

    # Same console key as the rest of Glen's console (when CONSOLE_SECRET is set, e.g.
    # under `doppler run`). The launcher passes ?key=; we cookie it so same-origin
    # fetches stay authed. No CONSOLE_SECRET -> open (e.g. bare local dev / tests).
    _secret = os.environ.get("CONSOLE_SECRET", "")

    @app.before_request
    def _console_gate():
        if not _secret:
            return None
        key = (request.args.get("key", "") or request.cookies.get("rm_biofield_key", "")
               or request.headers.get("X-Console-Key", ""))
        if key != _secret:
            return Response("Unauthorized — open this from the console 'Biofield Intake' link.",
                            status=401, mimetype="text/plain")
        return None

    @app.after_request
    def _console_cookie(resp):
        if _secret and request.args.get("key", "") == _secret:
            resp.set_cookie("rm_biofield_key", _secret, httponly=True, samesite="Lax")
        return resp

    @app.route("/")
    def index():
        q = request.args.get("q", "")
        with sqlite3.connect(db_path) as cx:
            return Response(render_list_html(list_tests(cx, q), q, list_authored(cx)),
                            mimetype="text/html")

    @app.route("/clinical-tags")
    def clinical_tags_queue():
        from dashboard import clinical_tags_console as _ctc
        with sqlite3.connect(e4l_db) as cx:
            return Response(_ctc.render_queue_html(_ctc.review_queue(cx)), mimetype="text/html")

    @app.route("/clinical-tags/<int:client_id>", methods=["GET", "POST"])
    def clinical_tags_client(client_id):
        from dashboard import clinical_tags_console as _ctc
        with sqlite3.connect(e4l_db) as cx:
            if request.method == "POST":
                tags = request.form.getlist("tags")
                action = request.form.get("action")
                if tags and action == "confirm":
                    _ctc.confirm(cx, client_id, tags)
                elif tags and action == "reject":
                    _ctc.reject(cx, client_id, tags)
                return redirect(f"/clinical-tags/{client_id}")
            return Response(_ctc.render_client_html(_ctc.client_tags(cx, client_id)),
                            mimetype="text/html")

    @app.route("/test/<test_id>")
    def report(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = (authored_report(cx, test_id) if str(test_id).startswith("a")
                   else causal_chain_report(cx, test_id))
            notes, narrative = get_notes(cx, test_id), get_narrative(cx, test_id)
            vscript = get_video_script(cx, test_id)
            stresses = None
            try:
                chain_rows = [{"head": l.get("head"), "remedy": l.get("remedy")}
                              for l in (rep.get("layers") or [])]
                stresses = _st.list_stresses(cx, test_id, chain_rows)
            except Exception:
                stresses = None
        return Response(render_report_html(rep, notes, narrative, vscript, stresses=stresses),
                        mimetype="text/html")

    @app.route("/test/<test_id>/report")
    def report_present(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        return Response(render_present(rep, narrative), mimetype="text/html")

    @app.route("/test/<test_id>/report.pdf")
    def report_present_pdf(test_id):
        import os
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        html = render_present(rep, narrative)
        reports_dir = os.environ.get("BIOFIELD_REPORTS_DIR",
                                     os.path.join(os.path.expanduser("~"), "biofield-reports"))
        date = (rep.get("date") or "").replace("/", "-") or "undated"
        out = os.path.join(reports_dir, f"report_{test_id}_{date}.pdf")
        try:
            save_report_pdf(html, out)          # keep a local copy to print/ship
            data = open(out, "rb").read()
        except Exception as e:
            return Response(f"PDF generation failed: {e}", status=500)
        return Response(data, mimetype="application/pdf", headers={
            "Content-Disposition": f'inline; filename="biofield-{test_id}.pdf"'})

    # --- Authoring (Increment 4a) ---
    @app.route("/author/new", methods=["POST"])
    def author_new():
        with sqlite3.connect(db_path) as cx:
            tid = create_test(cx, "", "", "")
        return redirect(f"/author/{tid}")

    @app.route("/author/<test_id>")
    def author_edit(test_id):
        from dashboard.biofield_report_html import group_layers
        from dashboard.biofield_stress import list_stresses
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
            dv = dimension_values(cx, DEPTH_KEY)
            transcript = get_notes(cx, test_id)
            # Stresses each layer's remedies cover, keyed by card layer number so the
            # cards can show them inline (chain_rows grouped to match the head cards).
            groups = group_layers(rep.get("layers") or [])
            chain_rows = [{"layer": g["layer"], "head": g["head"], "remedy": r.get("remedy")}
                          for g in groups for r in g["rows"]]
            sdata = list_stresses(cx, test_id, chain_rows)
            covered = {L["layer"]: L["stresses"] for L in sdata.get("by_layer") or []}
            narrative = get_narrative(cx, test_id)
            c_email = ((rep.get("client") or {}).get("email") or "").strip()
        fstate = biofield_fee.build_fee_state(c_email, fee_get)
        return Response(render_author_html(rep, dv, transcript, covered_by_layer=covered,
                                           narrative=narrative, fee_state=fstate),
                        mimetype="text/html")

    @app.route("/author/<test_id>/depth", methods=["POST"])
    def author_depth(test_id):
        d = request.get_json(silent=True) or {}
        side = d.get("side")
        if side not in ("stress", "remedy") or not d.get("rid"):
            return {"error": "bad params"}
        with sqlite3.connect(db_path) as cx:
            dim_tag(cx, "auth_" + side, d.get("rid"), DEPTH_KEY, d.get("rank"))
        return {"ok": True}

    @app.route("/author/<test_id>/header", methods=["POST"])
    def author_header(test_id):
        d = request.get_json(silent=True) or {}
        with sqlite3.connect(db_path) as cx:
            update_header(cx, test_id, name=d.get("name"), email=d.get("email"),
                          date=d.get("date"))
            ctx, _ = _e4l(cx, test_id)  # client now known -> pull recent E4L scan
            _seed_stresses(cx, test_id)  # synthesize + seed stress coverage if scan found
        return {"ok": True, "e4l": ctx, "html": render_e4l_panel(ctx)}

    @app.route("/author/<test_id>/fee", methods=["POST"])
    def author_fee(test_id):
        d = request.get_json(silent=True) or {}
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"ok": False, "error": "Add a client email in the header first."}, 400
        try:
            cents = biofield_fee.dollars_to_cents(d.get("dollars"))
        except (ValueError, TypeError):
            return {"ok": False, "error": "Enter a valid non-negative amount."}, 400
        fee_set(email, cents, (d.get("note") or "").strip())
        state = biofield_fee.build_fee_state(email, fee_get)
        return {"ok": True, "html": render_fee_panel(state)}

    @app.route("/author/<test_id>/fee/clear", methods=["POST"])
    def author_fee_clear(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"ok": False, "error": "Add a client email in the header first."}, 400
        fee_clear(email)
        state = biofield_fee.build_fee_state(email, fee_get)
        return {"ok": True, "html": render_fee_panel(state)}

    @app.route("/author/<test_id>/e4l")
    def author_e4l(test_id):
        with sqlite3.connect(db_path) as cx:
            ctx, _ = _e4l(cx, test_id)
        return {"e4l": ctx, "html": render_e4l_panel(ctx)}

    @app.route("/api/e4l/clients")
    def api_e4l_clients():
        return {"clients": client_search(request.args.get("q", ""))}

    @app.route("/author/<test_id>/e4l/refresh", methods=["POST"])
    def author_e4l_refresh(test_id):
        """Pull this client's newest scan from the LIVE E4L portal, then re-read the
        panel. Synchronous (localhost + threaded server -> no gateway timeout); the
        browser shows a spinner while it runs (~15-20s for the Playwright login)."""
        body = request.get_json(silent=True) or {}
        client_id = body.get("client_id")
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
        client = rep.get("client") or {}
        email = client.get("email") or ""
        if client_id is None and not email:
            return {"ok": False, "error": "no client selected yet",
                    "newer": False, "e4l": scan_lookup(""),
                    "html": render_e4l_panel(scan_lookup(""))}
        before = scan_lookup(email)
        res = _fetch_live(client_id=client_id, name=client.get("name"), runner=fetch_runner)
        after = scan_lookup(email)
        newer = bool(after.get("found") and (
            not before.get("found")
            or (after.get("scan_date") or "") > (before.get("scan_date") or "")))
        return {"ok": bool(res.get("ok")), "error": res.get("error"),
                "newer": newer, "e4l": after, "html": render_e4l_panel(after)}

    @app.route("/author/<test_id>/e4l/import-reveal", methods=["POST"])
    def author_import_reveal(test_id):
        """Import the client's recent (<7d) E4L reveal layers + remedies as
        needs-review causal-chain rows. Appends only after an explicit force when the
        session already has rows. Synthesis runs in-process (PHI stays local)."""
        import datetime as _dt
        from dashboard import biofield_reveal_import as _ri
        force = bool((request.get_json(silent=True) or {}).get("force"))
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            email = ((rep.get("client") or {}).get("email") or "").strip()
            if not email:
                return {"ok": False, "reason": "No client selected yet"}
            try:
                res = _ri.synthesize_reveal_layers(email, today=_dt.date.today().isoformat())
            except Exception as e:
                return {"ok": False, "reason": f"Reveal synthesis failed: {e}"}
            if not res.get("found"):
                return {"ok": False, "reason": "No E4L scan on file"}
            if not res.get("fresh"):
                return {"ok": False,
                        "reason": f"Latest scan is {res.get('days_ago')} days old "
                                  "— refresh to import"}
            existing = len(rep.get("layers") or [])
            if existing and not force:
                return {"ok": False, "needs_confirm": True, "existing": existing}
            imported = _ri.import_layers_to_test(cx, test_id, res.get("layers") or [])
            # Pass already-synthesized layers so the pipeline runs only once per import
            _seed_stresses(cx, test_id, force=True, layers=res.get("layers") or [])
        return {"ok": True, "imported": imported}

    @app.route("/author/<test_id>/stresses")
    def author_stresses(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            chain_rows = [{"layer": l.get("layer"), "head": l.get("head"),
                           "remedy": l.get("remedy")}
                          for l in (rep.get("layers") or [])]
            data = _st.list_stresses(cx, test_id, chain_rows)
        return {"data": data, "html": render_stress_panel(data)}

    def _chain_rows_for(rep):
        return [{"layer": l.get("layer"), "head": l.get("head"), "remedy": l.get("remedy")}
                for l in (rep.get("layers") or [])]

    def _suggest_payload(data):
        return {"ok": True, "picks": data["picks"], "uncovered": data["uncovered"],
                "source": data["source"], "pattern_key": data["pattern_key"],
                "has_pattern": data["has_pattern"], "html": render_suggest_panel(data)}

    def _append_layers(cx, test_id, rems):
        """Append each remedy as a new causal-chain layer at the bottom (existing
        layers untouched); head = the stresses that remedy covers. Returns count."""
        from dashboard import biofield_stress as _st
        rems = [(r or "").strip() for r in (rems or []) if (r or "").strip()]
        if not rems:
            return 0
        tnum = int(str(test_id).lstrip("a") or 0)
        rep = _report_for(cx, test_id)
        data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep))
        cover_by = {p["remedy"]: (p.get("covers") or []) for p in data["picks"]}
        nxt = int(cx.execute("SELECT COALESCE(MAX(layer),0) FROM biofield_auth_chain "
                             "WHERE test_id=?", (tnum,)).fetchone()[0] or 0)
        added = 0
        for r in rems:
            nxt += 1
            head = ", ".join(cover_by.get(r, []))[:200]
            add_chain_row(cx, test_id, nxt, head, "", r, "", "", "", confirmed=1, origin="live")
            added += 1
        return added

    @app.route("/author/<test_id>/suggest-remedies")
    def author_suggest_remedies(test_id):
        from dashboard import biofield_stress as _st
        force = request.args.get("force") == "computed"
        only_saved = request.args.get("only_persisted") == "1"
        with sqlite3.connect(db_path) as cx:
            # Init-time restore: render only if a set was previously preserved for
            # this test, else stay hidden (empty html).
            if only_saved and _st.get_saved_remedy_set(cx, test_id) is None:
                return {"ok": True, "html": "", "picks": [], "uncovered": [],
                        "source": None, "pattern_key": "", "has_pattern": False}
            rep = _report_for(cx, test_id)
            data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep), force_computed=force)
        return _suggest_payload(data)

    @app.route("/author/<test_id>/remedy-set/suggest", methods=["POST"])
    def author_remedy_set_suggest(test_id):
        # Resolve + PERSIST, so the suggested list survives the reloads a live
        # biofield recording triggers.
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep))
            _st.save_remedy_set(cx, test_id, data["remedies"])
            data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep))
        return _suggest_payload(data)

    @app.route("/author/<test_id>/remedy-set/add-one", methods=["POST"])
    def author_remedy_set_add_one(test_id):
        rem = (request.get_json(silent=True) or {}).get("remedy") or ""
        with sqlite3.connect(db_path) as cx:
            added = _append_layers(cx, test_id, [rem])
        return {"ok": True, "added": added}

    @app.route("/author/<test_id>/remedy-set", methods=["POST"])
    def author_remedy_set_save(test_id):
        from dashboard import biofield_stress as _st
        rems = (request.get_json(silent=True) or {}).get("remedies") or []
        with sqlite3.connect(db_path) as cx:
            _st.save_remedy_set(cx, test_id, rems)
            rep = _report_for(cx, test_id)
            data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep))
        return _suggest_payload(data)

    @app.route("/author/<test_id>/remedy-set/recompute", methods=["POST"])
    def author_remedy_set_recompute(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            _st.clear_remedy_set(cx, test_id)
            rep = _report_for(cx, test_id)
            data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep), force_computed=True)
            _st.save_remedy_set(cx, test_id, data["remedies"])  # preserve the recompute
            data = _st.resolve_remedy_set(cx, test_id, _chain_rows_for(rep))
        return _suggest_payload(data)

    @app.route("/author/<test_id>/remedy-set/save-pattern", methods=["POST"])
    def author_remedy_set_save_pattern(test_id):
        from dashboard import biofield_stress as _st
        rems = (request.get_json(silent=True) or {}).get("remedies") or []
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            res = _st.save_pattern_set(cx, test_id, _chain_rows_for(rep), rems)
        return res

    @app.route("/author/<test_id>/remedy-set/apply-to-chain", methods=["POST"])
    def author_remedy_set_apply(test_id):
        rems = (request.get_json(silent=True) or {}).get("remedies") or []
        with sqlite3.connect(db_path) as cx:
            added = _append_layers(cx, test_id, rems)
        return {"ok": True, "added": added}

    @app.route("/author/<test_id>/stress/<int:sid>/balance", methods=["POST"])
    def author_stress_balance(test_id, sid):
        from dashboard import biofield_stress as _st
        value = bool((request.get_json(silent=True) or {}).get("value"))
        with sqlite3.connect(db_path) as cx:
            _st.set_manual_balanced(cx, test_id, sid, value)
        return {"ok": True}

    @app.route("/author/<test_id>/row", methods=["POST"])
    def author_row_add(test_id):
        d = request.get_json(silent=True) or {}
        with sqlite3.connect(db_path) as cx:
            rid = add_chain_row(cx, test_id, _layer_int(d.get("layer")), d.get("head", ""),
                                d.get("most_affected", ""), d.get("remedy", ""),
                                d.get("dosage", ""), d.get("frequency", ""), d.get("timing", ""))
        return {"ok": True, "rid": rid}

    @app.route("/author/<test_id>/row/<int:rid>", methods=["POST"])
    def author_row_save(test_id, rid):
        d = request.get_json(silent=True) or {}
        fields = {}
        for k in ("layer", "head", "most_affected", "remedy", "dosage", "frequency", "timing"):
            if k in d:
                fields[k] = _layer_int(d[k]) if k == "layer" else d[k]
        if "layer" in fields:
            new_layer = fields.pop("layer")
            from dashboard.biofield_authoring import reorder_chain
            with sqlite3.connect(db_path) as cx:
                if fields:
                    update_chain_row(cx, rid, **fields)
                reorder_chain(cx, test_id, rid, new_layer)
            return {"ok": True}
        with sqlite3.connect(db_path) as cx:
            update_chain_row(cx, rid, **fields)
        return {"ok": True}

    @app.route("/author/<test_id>/row/<int:rid>/delete", methods=["POST"])
    def author_row_delete(test_id, rid):
        with sqlite3.connect(db_path) as cx:
            delete_chain_row(cx, rid)
        return {"ok": True}

    @app.route("/author/<test_id>/reorder-layers", methods=["POST"])
    def author_reorder_layers(test_id):
        order = (request.get_json(silent=True) or {}).get("order") or []
        from dashboard.biofield_authoring import set_layer_order
        with sqlite3.connect(db_path) as cx:
            set_layer_order(cx, test_id, order)
        return {"ok": True}

    @app.route("/author/<test_id>/stress/<int:sid>/cover", methods=["POST"])
    def author_stress_cover(test_id, sid):
        rids = (request.get_json(silent=True) or {}).get("rids") or []
        from dashboard.biofield_stress import cover_stress
        with sqlite3.connect(db_path) as cx:
            code = cover_stress(cx, test_id, sid, rids)
        return {"ok": code is not None, "code": code}

    @app.route("/api/catalog")
    def api_catalog():
        with sqlite3.connect(db_path) as cx:
            return {"catalog": remedy_catalog(cx, request.args.get("q", ""),
                                              _safe_int(request.args.get("limit"), 20))}

    @app.route("/api/dosing")
    def api_dosing():
        with sqlite3.connect(db_path) as cx:
            return remedy_dosing(cx, request.args.get("name", ""))

    @app.route("/api/vocab")
    def api_vocab():
        with sqlite3.connect(db_path) as cx:
            return {"vocab": stress_vocab(cx, request.args.get("q", ""),
                                          _safe_int(request.args.get("limit"), 20))}

    @app.route("/api/suggest")
    def api_suggest():
        with sqlite3.connect(db_path) as cx:
            return {"suggestions": stress_suggestions(cx, request.args.get("stress", ""))}

    @app.route("/api/deepgram-token")
    def api_deepgram_token():
        try:
            return {"key": deepgram_token()}
        except Exception as e:  # no key / network / Deepgram error
            return {"error": str(e)[:200]}

    @app.route("/author/<test_id>/session", methods=["POST"])
    def author_session(test_id):
        txt = ((request.get_json(silent=True) or {}).get("transcript") or "").strip()
        if not txt:
            return {"ok": True, "skipped": "empty"}
        with sqlite3.connect(db_path) as cx:
            save_notes(cx, test_id, txt)  # box holds the full transcript -> replace
        return {"ok": True}

    @app.route("/author/<test_id>/mine-profile", methods=["POST"])
    def author_mine_profile(test_id):
        with sqlite3.connect(db_path) as cx:
            return _mine_profile(cx, test_id)

    @app.route("/author/<test_id>/mine-comms", methods=["POST"])
    def author_mine_comms(test_id):
        with sqlite3.connect(db_path) as cx:
            return _mine_comms(cx, test_id)

    @app.route("/author/<test_id>/capture-stresses", methods=["POST"])
    def author_capture_stresses(test_id):
        from dashboard.biofield_interpret import interpret_stresses
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            transcript = get_notes(cx, test_id)
            if not transcript.strip():
                return {"added": 0, "error": "no transcript yet -- record a session first"}
            try:
                labels = interpret_stresses(transcript, interpret_complete)
            except Exception as e:
                return {"added": 0, "error": str(e)[:200]}
            added = sum(1 for label in labels if _st.add_voice_stress(cx, test_id, label))
        return {"added": added}

    @app.route("/author/<test_id>/interpret", methods=["POST"])
    def author_interpret(test_id):
        with sqlite3.connect(db_path) as cx:
            transcript = get_notes(cx, test_id)
            if not transcript.strip():
                return {"added": 0, "error": "no transcript yet -- record a session first"}
            try:
                result = interpret_transcript(transcript, interpret_complete)
            except Exception as e:
                return {"error": str(e)[:200]}
            added = 0
            for l in result.get("layers", []):
                remedy = resolve_remedy_name(cx, l["remedy"])  # auto-correct + title-case ASR mangles
                head = resolve_stress_name(cx, l["head"])      # capitalize/match stress names too
                most_affected = resolve_stress_name(cx, l["most_affected"])
                dosage, frequency, timing = l.get("dosage", ""), l.get("frequency", ""), l.get("timing", "")
                if not (dosage or frequency or timing):  # no spoken dose -> catalog minimum
                    d = remedy_dosing(cx, remedy)
                    dosage, frequency, timing = d["dosage"], d["frequency"], d["timing"]
                add_chain_row(cx, test_id, l.get("layer"), head, most_affected,
                              remedy, dosage, frequency, timing, confirmed=0)  # voice -> unconfirmed
                added += 1
        return {"added": added, "header": result.get("header", "")}

    @app.route("/author/<test_id>/delete", methods=["POST"])
    def author_delete(test_id):
        with sqlite3.connect(db_path) as cx:
            delete_test(cx, test_id)
        return {"ok": True}

    @app.route("/author/<test_id>/confirm-all", methods=["POST"])
    def author_confirm_all(test_id):
        with sqlite3.connect(db_path) as cx:
            confirm_all(cx, test_id)
        return {"ok": True}

    @app.route("/author/<test_id>/row/<int:rid>/confirm", methods=["POST"])
    def author_row_confirm(test_id, rid):
        with sqlite3.connect(db_path) as cx:
            confirm_row(cx, rid)
        return {"ok": True}

    @app.route("/test/<test_id>/notes", methods=["POST"])
    def notes_save(test_id):
        with sqlite3.connect(db_path) as cx:
            save_notes(cx, test_id, (request.get_json(silent=True) or {}).get("notes", ""))
        return {"ok": True}

    @app.route("/test/<test_id>/narrative", methods=["POST"])
    def narrative_save(test_id):
        with sqlite3.connect(db_path) as cx:
            save_narrative(cx, test_id, (request.get_json(silent=True) or {}).get("narrative", ""))
        return {"ok": True}

    @app.route("/test/<test_id>/generate", methods=["POST"])
    def narrative_generate(test_id):
        notes = (request.get_json(silent=True) or {}).get("notes", "")
        with sqlite3.connect(db_path) as cx:
            save_notes(cx, test_id, notes)
            ctx, rep = _e4l(cx, test_id)  # authored or FMP report + recent E4L scan
            prof = {}
            try:
                prof = fetch_profile(((rep.get("client") or {}).get("email") or "").strip()) or {}
            except Exception:
                prof = {}
            try:
                text = generate_narrative(rep, notes, complete, scan=ctx, profile=prof)
            except Exception as e:  # no API key / network / model error
                return {"error": str(e)[:200]}
            save_narrative(cx, test_id, text)
        return {"narrative": text}

    @app.route("/test/<test_id>/video-generate", methods=["POST"])
    def video_generate(test_id):
        notes = (request.get_json(silent=True) or {}).get("notes", "")
        with sqlite3.connect(db_path) as cx:
            ctx, rep = _e4l(cx, test_id)
            try:
                script = generate_video_script(rep, notes, complete, scan=ctx)
            except Exception as e:
                return {"error": str(e)[:200]}
            save_video_script(cx, test_id, script)
        return {"script": script}

    @app.route("/test/<test_id>/video-script", methods=["POST"])
    def video_script_save(test_id):
        with sqlite3.connect(db_path) as cx:
            save_video_script(cx, test_id, (request.get_json(silent=True) or {}).get("script", ""))
        return {"ok": True}

    @app.route("/test/<test_id>/audio", methods=["POST"])
    def make_audio(test_id):
        with sqlite3.connect(db_path) as cx:
            script = get_video_script(cx, test_id)
        if not (script or "").strip():
            return {"error": "no script yet -- generate or write one first"}
        try:
            audio = tts(script)
        except Exception as e:
            return {"error": str(e)[:200]}
        os.makedirs(AUDIO_DIR, exist_ok=True)
        fname = f"test_{test_id}.mp3"
        with open(os.path.join(AUDIO_DIR, fname), "wb") as f:
            f.write(audio)
        # Auto-attach: if this client's report is already published, refresh their
        # portal so the new audio appears without a manual re-publish (best-effort,
        # send=False so no re-email; reuses the original special price).
        attached = False
        sp = _published_special(test_id)
        if sp is not None:
            try:
                res = _do_publish(test_id, int(sp or 0), send=False)
                attached = not res.get("unresolved")
            except Exception as e:
                print(f"[audio] portal auto-attach failed: {e}", flush=True)
        return {"url": f"/audio/{fname}", "bytes": len(audio), "portal_attached": attached}

    @app.route("/audio/<path:fname>")
    def serve_audio(fname):
        return send_from_directory(AUDIO_DIR, fname, mimetype="audio/mpeg")

    def _pub_init(cx):
        cx.execute("CREATE TABLE IF NOT EXISTS biofield_portal_published("
                   "test_id TEXT PRIMARY KEY, special_price_cents INTEGER, updated_at TEXT)")

    def _mark_published(test_id, special):
        with sqlite3.connect(db_path) as cx:
            _pub_init(cx)
            cx.execute("INSERT INTO biofield_portal_published(test_id,special_price_cents,updated_at) "
                       "VALUES(?,?,?) ON CONFLICT(test_id) DO UPDATE SET "
                       "special_price_cents=excluded.special_price_cents, updated_at=excluded.updated_at",
                       (test_id, int(special or 0), datetime.datetime.utcnow().isoformat()))
            cx.commit()

    def _published_special(test_id):
        with sqlite3.connect(db_path) as cx:
            _pub_init(cx)
            r = cx.execute("SELECT special_price_cents FROM biofield_portal_published "
                           "WHERE test_id=?", (test_id,)).fetchone()
        return r[0] if r else None

    def _do_publish(test_id, special, send):
        """Build + upload PDF/audio assets + upsert the portal for a test. Returns the
        upsert response dict (or {"unresolved": [...]}); raises on hard errors."""
        from dashboard import biofield_portal_publish as _bpp
        with sqlite3.connect(db_path) as cx:
            pre = _bpp.build_portal_content(cx, test_id, special_price_cents=special)
            if pre["unresolved"]:
                return {"unresolved": pre["unresolved"]}
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        base = os.environ.get("PORTAL_PUBLISH_BASE_URL", "")
        key = os.environ.get("CONSOLE_SECRET", "")
        if not base:
            raise RuntimeError("PORTAL_PUBLISH_BASE_URL not set")
        pdf_bytes = report_pdf_bytes(render_present(rep, narrative))
        pdf_url = _bpp.upload_asset(pdf_bytes, _bpp._asset_name("pdf"), base_url=base, console_key=key)
        audio_url = None
        audio_path = os.path.join(AUDIO_DIR, f"test_{test_id}.mp3")
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as af:
                audio_url = _bpp.upload_asset(af.read(), _bpp._asset_name("mp3"),
                                              base_url=base, console_key=key)
        with sqlite3.connect(db_path) as cx:
            payload = _bpp.build_portal_content(cx, test_id, special_price_cents=special,
                                                audio_url=audio_url, report_pdf_url=pdf_url)
        return _bpp.publish_to_portal(payload, base_url=base, console_key=key, send=send)

    @app.route("/test/<test_id>/publish-portal", methods=["POST"])
    def publish_portal(test_id):
        body = request.get_json(silent=True) or {}
        try:
            special = int(body.get("special_price_cents") or 0)
        except (TypeError, ValueError):
            return {"ok": False, "error": "special_price_cents must be an integer"}, 400
        try:
            res = _do_publish(test_id, special, send=True)
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}, 502
        if res.get("unresolved"):
            return {"ok": False, "unresolved": res["unresolved"]}, 409
        _mark_published(test_id, special)   # remember for auto-attach on later audio
        return {"ok": True, "url": res.get("url", ""),
                "updated": bool(res.get("updated")), "note": res.get("note", ""),
                "emailed": bool(res.get("emailed")), "unresolved": []}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local Biofield Analysis viewer")
    ap.add_argument("--load", action="store_true",
                    help="(re)load the snapshot from --export-dir before serving")
    ap.add_argument("--export-dir", default="/tmp/fmp-export/newapp")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    if args.load:
        from dashboard.biofield_fmp_snapshot import snapshot_csv_dir
        counts = snapshot_csv_dir(args.export_dir, DEFAULT_DB)
        print("loaded snapshot:", {k: counts[k] for k in sorted(counts)})
    print(f"Biofield Analysis viewer -> http://127.0.0.1:{args.port}  (local only; Ctrl-C to stop)")
    create_app().run(host="127.0.0.1", port=args.port, debug=False)
