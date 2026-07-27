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
    assert "Biofield Analyses for $200" in text
    assert "apply a $99 credit automatically" in text
    assert "/family-plan/checkout" in text
