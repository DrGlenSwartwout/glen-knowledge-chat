"""TDD tests for scripts/refresh_ingredients_from_fmp.py"""
import json, sys, os, importlib, types, pytest

# ---------------------------------------------------------------------------
# Helpers to import the module under test (stubs populate_bottle_types if
# the real one is not yet present in this checkout path)
# ---------------------------------------------------------------------------

def _import_module():
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import refresh_ingredients_from_fmp as m
    return m


# ---------------------------------------------------------------------------
# Unit tests — ingredient parsing
# ---------------------------------------------------------------------------

class TestParseIngredientLine:
    """Tests for _parse_ingredient_line (or equivalent internal logic)."""

    def _module(self):
        return _import_module()

    def test_name_after_dash(self):
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "100mg - R-Lipoic Acid", "zc_mg": "100", "qty": "100", "unit_measurement": "mg"}
        )
        assert result is not None
        assert result["name"] == "R-Lipoic Acid"

    def test_dose_from_zc_mg_integer(self):
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "100mg - Benfotiamine", "zc_mg": "100", "qty": "100", "unit_measurement": "mg"}
        )
        assert result["dose"] == "100 mg"

    def test_dose_no_trailing_zero(self):
        """1.0 mg should become '1 mg', not '1.0 mg'."""
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "1mg - Methylcobalamin", "zc_mg": "1", "qty": "1", "unit_measurement": "mg"}
        )
        assert result["dose"] == "1 mg"

    def test_dose_point_seven_preserved(self):
        """.7 mg (non-integer, <1) must keep the decimal."""
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": ".7mg - Huperzine A", "zc_mg": ".7", "qty": ".7", "unit_measurement": "mg"}
        )
        assert result["dose"] == ".7 mg"

    def test_dose_1_3_preserved(self):
        """1.3 mg must keep the decimal."""
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "1.3mg - Vitamin D3: Cholecalciferol", "zc_mg": "1.3", "qty": "1.3", "unit_measurement": "mg"}
        )
        assert result["dose"] == "1.3 mg"

    def test_dose_fallback_to_qty_unit(self):
        """When zc_mg is 0 or empty, fall back to qty + unit_measurement."""
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "200mcg - 5-MTHF", "zc_mg": "0", "qty": "200", "unit_measurement": "mcg"}
        )
        assert result["dose"] == "200 mcg"

    def test_blank_name_skipped(self):
        """'1ea. - ' → name is empty → None."""
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "1ea. - ", "zc_mg": "0", "qty": "1", "unit_measurement": "ea."}
        )
        assert result is None

    def test_plantcaps_skipped(self):
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "1ea. - Plantcaps®", "zc_mg": "0", "qty": "1", "unit_measurement": "ea."}
        )
        assert result is None

    def test_capsule_in_name_skipped(self):
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "1ea. - 00 Capsule filler", "zc_mg": "0", "qty": "1", "unit_measurement": "ea."}
        )
        assert result is None

    def test_pullulan_skipped(self):
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "30 - Pullulan vegi caps", "zc_mg": "0", "qty": "30", "unit_measurement": "ea."}
        )
        assert result is None

    def test_bottle_skipped(self):
        m = self._module()
        result = m._parse_ingredient_line(
            {"zc_raw_display": "1ea. - Wide Neck Bottle 100mL", "zc_mg": "0", "qty": "1", "unit_measurement": "ea."}
        )
        assert result is None


# ---------------------------------------------------------------------------
# Integration-style tests using tiny in-memory data
# ---------------------------------------------------------------------------

PRODUCTS_DATA = {
    "products": {
        "nerve-pulse": {
            "name": "Nerve Pulse",
            "price_cents": 6997,
            "ingredients": [{"name": "Benfotiamine", "dose": "100 mg"}],
            "ingredients_source": "store-recovery-2026-06-08",
        },
        "mystery-herb": {
            "name": "Mystery Herb",
            "price_cents": 4997,
            "ingredients": [{"name": "Unknown", "dose": ""}],
            "ingredients_source": "store-recovery-2026-06-08",
        },
        "no-fmp-product": {
            "name": "Completely Unknown Formula",
            "price_cents": 2997,
        },
    }
}

FMP_PRODUCTS = [
    {"id_pk": "100", "product_name": "Nerve Pulse"},
    {"id_pk": "200", "product_name": "Mystery Herb Capsules"},  # suffix-strip match
]

FMP_ITEMS = {
    "100": [
        {"zc_raw_display": "100mg - R-Lipoic Acid", "zc_mg": "100", "qty": "100", "unit_measurement": "mg"},
        {"zc_raw_display": "100mg - Benfotiamine", "zc_mg": "100", "qty": "100", "unit_measurement": "mg"},
        {"zc_raw_display": "1ea. - Plantcaps®", "zc_mg": "0", "qty": "1", "unit_measurement": "ea."},
        {"zc_raw_display": "1ea. - ", "zc_mg": "0", "qty": "1", "unit_measurement": "ea."},
    ],
    "200": [
        {"zc_raw_display": "50mg - Ashwagandha", "zc_mg": "50", "qty": "50", "unit_measurement": "mg"},
    ],
}


class TestBuildIngredients:
    def _module(self):
        return _import_module()

    def test_product_with_recipe_gets_updated(self):
        m = self._module()
        fmp_index = m._build_fmp_index(FMP_PRODUCTS)
        staged, review = m._resolve_updates(PRODUCTS_DATA["products"], fmp_index, FMP_ITEMS)
        assert "nerve-pulse" in staged
        ingr = staged["nerve-pulse"]
        names = [i["name"] for i in ingr]
        assert "R-Lipoic Acid" in names
        assert "Benfotiamine" in names
        # packaging lines filtered out
        assert not any("Plantcaps" in n for n in names)

    def test_packaging_lines_filtered(self):
        m = self._module()
        fmp_index = m._build_fmp_index(FMP_PRODUCTS)
        staged, _ = m._resolve_updates(PRODUCTS_DATA["products"], fmp_index, FMP_ITEMS)
        ingr = staged["nerve-pulse"]
        assert all(i["name"] != "" for i in ingr)

    def test_no_fmp_match_left_unchanged(self):
        m = self._module()
        fmp_index = m._build_fmp_index(FMP_PRODUCTS)
        staged, review = m._resolve_updates(PRODUCTS_DATA["products"], fmp_index, FMP_ITEMS)
        assert "no-fmp-product" not in staged
        slugs_in_review = [r["slug"] for r in review]
        assert "no-fmp-product" in slugs_in_review

    def test_suffix_strip_match_resolves(self):
        """'Mystery Herb' should match FMP 'Mystery Herb Capsules' via suffix-strip."""
        m = self._module()
        fmp_index = m._build_fmp_index(FMP_PRODUCTS)
        staged, _ = m._resolve_updates(PRODUCTS_DATA["products"], fmp_index, FMP_ITEMS)
        assert "mystery-herb" in staged

    def test_fuzzy_match_goes_to_review(self):
        """A fuzzy-only match must appear in review, not silently accepted."""
        m = self._module()
        # Add a product that needs fuzzy matching (close but not exact/suffix)
        products = {
            "alpha-lipoc-acid": {
                "name": "Alpha Lipoc Acid",  # deliberate typo, close to "Alpha Lipoic Acid"
                "price_cents": 4997,
            }
        }
        fmp_products = [{"id_pk": "999", "product_name": "Alpha Lipoic Acid"}]
        fmp_items = {"999": [
            {"zc_raw_display": "100mg - Alpha Lipoic Acid", "zc_mg": "100", "qty": "100", "unit_measurement": "mg"},
        ]}
        fmp_index = m._build_fmp_index(fmp_products)
        staged, review = m._resolve_updates(products, fmp_index, fmp_items)
        # fuzzy match → in review
        review_slugs = [r["slug"] for r in review]
        assert "alpha-lipoc-acid" in review_slugs

    def test_synergy_syntropy_alias(self):
        """'syntropy' and 'synergy' should resolve to the same FMP key."""
        m = self._module()
        products = {
            "msm-syntropy": {
                "name": "MSM Syntropy",
                "price_cents": 4997,
            }
        }
        fmp_products = [{"id_pk": "777", "product_name": "MSM Synergy"}]
        fmp_items = {"777": [
            {"zc_raw_display": "500mg - MSM", "zc_mg": "500", "qty": "500", "unit_measurement": "mg"},
        ]}
        fmp_index = m._build_fmp_index(fmp_products)
        staged, review = m._resolve_updates(products, fmp_index, fmp_items)
        assert "msm-syntropy" in staged
