"""The authoring editor page: header inputs + editable chain rows + add-row."""
from dashboard.biofield_report_html import render_author_html, render_list_html


def _report():
    return {
        "test_id": "a1", "client": {"name": "Jane Doe", "email": "jane@x.com"},
        "date": "2026-06-23",
        "layers": [
            {"layer": 1, "head": "Night", "most_affected": "Night", "remedy": "TMG",
             "dosage": "1 scoop", "frequency": "daily", "timing": "at night", "rid": 5},
        ],
        "schedule": {"slots": [], "entries": []},
    }


def test_author_page_shows_transcript_saved_date():
    html = render_author_html(_report(), transcript_updated="2026-07-10T22:14:03Z")
    assert "Last saved: Jul 10, 2026 · 12:14 PM HST" in html
    assert "id=sessSaved" in html


def test_author_page_omits_saved_date_when_never_saved():
    html = render_author_html(_report())
    assert "id=sessSaved class=food></div>" in html   # empty placeholder, filled live after save
    assert "HST</div>" not in html                     # no server-rendered date


def test_author_page_has_header_rows_and_endpoints():
    html = render_author_html(_report())
    assert "Jane Doe" in html and "2026-06-23" in html       # header prefilled
    assert "TMG" in html and "Night" in html                 # existing row prefilled
    assert "/author/a1/header" in html                       # save header
    assert "/author/a1/row" in html                          # add/save rows
    assert "Add remedy" in html and "Add layer" in html      # per-layer + new-layer add
    assert "/test/a1" in html                                # link to the read-only report
    assert "fillDose" in html                                # remedy auto-fills dosing on change


def test_author_chain_is_cards_readable_and_reorderable():
    html = render_author_html(_report(), [], "")
    assert "class=lcard" in html and "data-gid=g0" in html   # a layer card
    assert ">Head</label>" in html and ">Tail</label>" in html  # head/tail on line 1
    assert "onclick=\"xpand(this)\"" in html and "function xpand" in html  # expand full head/tail
    assert "list=catalog" in html                            # remedy keeps its catalog autocomplete
    assert "draggable=true" in html and "class=grip" in html # drag handle
    assert "function drop" in html and "reorder-layers" in html  # drag persists new order
    assert "data-gid=gnew" in html                           # trailing "new layer" card
    # depth toggle still present, hidden by default (dcol)
    assert "id=depthbtn" in html and "function toggleDepth" in html
    assert "class=dcol" in html and "function restoreDepth" in html


def test_author_has_left_layer_rail_reorderable():
    rep = {"test_id": "a1", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [
               {"layer": 1, "head": "Lymph", "most_affected": "G", "remedy": "R1", "rid": 5, "confirmed": 1},
               {"layer": 2, "head": "Neural", "most_affected": "C", "remedy": "R2", "rid": 6, "confirmed": 1},
               {"layer": 3, "head": "Neural", "most_affected": "C", "remedy": "R3", "rid": 7, "confirmed": 1}],
           "schedule": {"slots": [], "entries": []}}
    html = render_author_html(rep, [], "")
    assert "class=chainlayout" in html and "id=layerrail" in html
    assert html.count("class=railitem") == 2                 # one chip per layer (Neural merged)
    assert 'data-rids="6,7"' in html                         # merged layer carries both remedies
    assert "onclick=\"focusCard('g1')\"" in html             # click scrolls to its card
    assert "draggable=true" in html and "persistOrder(box)" in html  # rail + cards share reorder


def test_save_buttons_track_saved_and_dirty_state():
    rep = {"test_id": "a1", "client": {"name": "J", "email": ""}, "date": "",
           "layers": [{"layer": 1, "head": "H", "most_affected": "T", "remedy": "R",
                       "rid": 5, "confirmed": 1}], "schedule": {"slots": [], "entries": []}}
    html = render_author_html(rep, [], "")
    # save buttons are tagged for the saved/dirty toggle
    assert "class='btn savebtn' data-dirty=Update" in html          # remedy Save
    assert "savebtn' data-dirty='Update layer'" in html             # layer Save
    # editing a field marks its button dirty; saving turns it green + keeps it
    assert 'oninput="dirtyRow(this)"' in html and 'oninput="dirtyLayer(this)"' in html
    assert "function setSaved" in html and "function markDirty" in html


def test_editor_has_narrative_generate_and_prefill():
    rep = {"test_id": "a4", "client": {"name": "Agnes", "email": ""}, "date": "",
           "layers": [{"layer": 1, "head": "H", "remedy": "R", "rid": 5, "confirmed": 1}],
           "schedule": {"slots": [], "entries": []}}
    html = render_author_html(rep, [], "", narrative="Aloha Agnes, draft narrative.")
    assert "<h2>Narrative</h2>" in html
    assert "onclick=genNarr()" in html and "function genNarr" in html   # generate in editor
    assert "id=narrEd" in html and "Aloha Agnes, draft narrative." in html  # prefilled + editable
    assert "/test/a4/generate" in html                                  # wired to the generate route
    assert "id=narrSaveBtn" in html and "function saveNarrEd" in html   # save draft


def test_author_page_escapes_free_text():
    rep = _report()
    rep["client"]["name"] = "<script>x</script>"
    html = render_author_html(rep)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_author_page_has_session_recording_ui():
    html = render_author_html(_report())
    assert "Record" in html
    assert "/api/deepgram-token" in html      # browser fetches a short-lived token
    assert "/author/a1/session" in html        # transcript save endpoint
    assert "sessText" in html                  # live transcript box


def test_author_page_delete_confirm_and_unconfirmed_highlight():
    rep = _report()
    rep["layers"][0]["confirmed"] = 0          # a voice-added row
    html = render_author_html(rep)
    assert "delTest()" in html and "confirmAll()" in html
    assert "rline unconf" in html              # unconfirmed remedy line highlighted for Rae
    assert "confirmRow('5')" in html           # per-row confirm (rid 5)


def test_author_page_prefills_saved_transcript():
    html = render_author_html(_report(), transcript="BSI 21 phase 2 toxicity head and tail")
    assert "BSI 21 phase 2 toxicity head and tail" in html   # persists across refresh


def test_list_html_shows_authored_and_new_button():
    html = render_list_html(
        tests=[{"test_id": "10", "name": "Lewis", "email": "l@x.com",
                "date": "2026-06-11", "layer_count": 19}],
        authored=[{"test_id": "a1", "name": "Jane Doe", "email": "j@x.com",
                   "date": "2026-06-23", "layer_count": 1, "authored": True}])
    assert "Jane Doe" in html and "/author/a1" in html       # authored test -> editor
    assert "/author/new" in html                             # New test action
    assert "Lewis" in html                                   # FMP tests still listed
