def test_claim_sign_is_normalized_and_stable():
    import app
    sig = app._portal_claim_sign("Person@X.com ")
    assert sig == app._portal_claim_sign("person@x.com")  # trim + lowercase
    assert len(sig) == 40


def test_claim_url_carries_email_and_signature():
    import app
    url = app._portal_claim_url("person@x.com")
    assert "/portal/claim?e=person%40x.com&s=" + app._portal_claim_sign("person@x.com") in url


def test_claim_signature_is_email_bound():
    import app
    # a signature for one email must not validate another (no forging arbitrary claims)
    assert app._portal_claim_sign("a@x.com") != app._portal_claim_sign("b@x.com")
