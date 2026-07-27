from unittest import mock

import app as appmod


def test_membership_page_lists_individual_household_and_biofield_benefits():
    appmod.app.config["TESTING"] = True
    with mock.patch.object(appmod, "MEMBERSHIP_PRODUCTS_ENABLED", True):
        response = appmod.app.test_client().get("/membership")
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Individual membership" in text
    assert "Household membership" in text
    assert "$1,497/yr" in text
    assert "Automated Biofield Analysis" in text
    assert "Included with membership" in text
    assert "Personal Causal Biofield Analysis" in text
    assert "$200 for active members" in text
    assert "non-member price is $300" in text
    assert "not a personal consultation or a Causal Biofield Analysis" in text
    assert "apply a $99 credit automatically" in text
    assert "/family-plan/checkout" in text
