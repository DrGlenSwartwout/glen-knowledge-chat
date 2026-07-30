from pathlib import Path


HTML = (
    Path(__file__).resolve().parents[1] / "static" / "client-portal.html"
).read_text(encoding="utf-8")


def test_client_name_header_is_at_top_of_portal():
    identity = HTML.index('id="portal-client-identity"')
    onboarding = HTML.index('id="portal-onboarding-mount"')
    app = HTML.index('id="app"')

    assert identity < onboarding < app
    assert 'id="portal-client-name"' in HTML
    assert "Healing Oasis client portal" in HTML


def test_client_name_header_uses_account_name_without_email_fallback():
    assert (
        'const portalClientName = d.name || (v && v.account && v.account.name) || "";'
        in HTML
    )
    assert 'identityName.textContent = safeClientName;' in HTML
    assert 'identity.hidden = !safeClientName;' in HTML
    assert '/[@]/.test(portalClientName)' in HTML
