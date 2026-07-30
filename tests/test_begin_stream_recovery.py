from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEGIN = (ROOT / "static" / "begin.html").read_text()
FIRESIDE = (ROOT / "static" / "begin-fireside.html").read_text()


def test_begin_chat_aborts_stalled_stream_and_restores_retry_message():
    assert "var controller = new AbortController();" in BEGIN
    assert "}, 20000);" in BEGIN
    assert "That reply took too long. Please try sending your message again." in BEGIN


def test_fireside_stops_murmur_when_text_finishes_not_when_tts_finishes():
    finish = FIRESIDE.split("function finishReply(text, hook)", 1)[1]
    before_tts = finish.split("fetch('/chat/tts'", 1)[0]
    assert "replyStarted = true;" in before_tts
    assert "stopMurmur();" in before_tts


def test_fireside_bounds_agent_and_tts_waits():
    assert "activeReplyController = new AbortController();" in FIRESIDE
    assert "if (activeReplyController) activeReplyController.abort();" in FIRESIDE
    assert "var ttsController = new AbortController();" in FIRESIDE
    assert "ttsController.abort()" in FIRESIDE
