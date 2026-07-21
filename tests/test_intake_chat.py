from dashboard import intake_chat as ic


def test_form_outline_lists_fields_and_scale_meanings():
    txt = ic.form_outline()
    assert "terrain" in txt and "commitment" in txt
    assert "health_concerns" in txt and "concern" in txt          # table columns
    assert "Stress or Hormonal Imbalance" in txt                  # scale label surfaced
    assert "Key Dimensions of the Clinical Theory of Everything" in txt


def test_build_system_injects_voice_and_name_and_prefill_rule():
    sp = ic.build_system("VOICE-OF-GLEN", "Francis")
    assert "VOICE-OF-GLEN" in sp
    assert "Francis" in sp
    for f in ("first_name", "last_name", "email"):
        assert f in sp                                            # do-not-ask list present


def test_parse_reply_plain_json():
    say, upd, done = ic.parse_reply('{"say":"Hi","updates":{"terrain":5},"done":false}')
    assert say == "Hi" and upd == {"terrain": 5} and done is False


def test_parse_reply_fenced_json():
    say, upd, done = ic.parse_reply('```json\n{"say":"Done","updates":{},"done":true}\n```')
    assert say == "Done" and done is True and upd == {}


def test_parse_reply_prose_wrapped_json():
    say, upd, _ = ic.parse_reply('Sure!\n{"say":"ok","updates":{"sleep":"Yes"}}')
    assert upd == {"sleep": "Yes"} and say == "ok"


def test_parse_reply_garbage_falls_back_to_say():
    say, upd, done = ic.parse_reply("just talking, no json")
    assert say == "just talking, no json" and upd == {} and done is False
