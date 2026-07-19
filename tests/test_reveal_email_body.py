from app import _reveal_email_body

ORIGINAL = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            "https://x/begin/biofield/tok\n\nIn wellness,\nDr. Glen and Rae\n")

def test_body_without_portal_is_byte_identical_to_original():
    assert _reveal_email_body("https://x/begin/biofield/tok") == ORIGINAL
    assert _reveal_email_body("https://x/begin/biofield/tok", None) == ORIGINAL

def test_body_with_portal_adds_one_portal_paragraph_and_keeps_reveal():
    body = _reveal_email_body("https://x/begin/biofield/tok", "https://x/portal/ptok")
    assert "https://x/begin/biofield/tok" in body          # funnel link kept
    assert "https://x/portal/ptok" in body                 # portal link added
    assert body.endswith("In wellness,\nDr. Glen and Rae\n")
    assert body.count("https://x/portal/ptok") == 1
