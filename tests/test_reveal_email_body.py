import importlib


def _reveal_email_body_fn(tmp_path, monkeypatch):
    # Point DATA_DIR at a fresh tmp dir and reload so LOG_DB (which honors the env
    # DATA_DIR) resolves under tmp_path -- import-time _init_shortlink_cache() needs a
    # writable dir, and each test stays isolated from the real persistent DB.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as _app
    importlib.reload(_app)
    return _app._reveal_email_body


ORIGINAL = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            "https://x/begin/biofield/tok\n\nIn wellness,\nDr. Glen and Rae\n")

def test_body_without_portal_is_byte_identical_to_original(tmp_path, monkeypatch):
    _reveal_email_body = _reveal_email_body_fn(tmp_path, monkeypatch)
    assert _reveal_email_body("https://x/begin/biofield/tok") == ORIGINAL
    assert _reveal_email_body("https://x/begin/biofield/tok", None) == ORIGINAL

def test_body_with_portal_adds_one_portal_paragraph_and_keeps_reveal(tmp_path, monkeypatch):
    _reveal_email_body = _reveal_email_body_fn(tmp_path, monkeypatch)
    body = _reveal_email_body("https://x/begin/biofield/tok", "https://x/portal/ptok")
    assert "https://x/begin/biofield/tok" in body          # funnel link kept
    assert "https://x/portal/ptok" in body                 # portal link added
    assert body.endswith("In wellness,\nDr. Glen and Rae\n")
    assert body.count("https://x/portal/ptok") == 1
