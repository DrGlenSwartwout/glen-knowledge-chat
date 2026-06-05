# Products Enrichment — Match Report (Stage A + A.1)

Generated: 2026-06-05  |  Source: `scripts/enrich_products.py`  |  Stage B (GK) run

## Summary

| Metric | Count |
|--------|-------|
| Total slugs | 321 |
| FMP-new matched (high conf) | 77 |
| FMP-new matched (medium conf) | 14 |
| FMP-new matched (low conf) | 3 |
| **FMP-new total (non-empty)** | **94** |
| FMP-matched-but-empty -> T33 ingredients (high) | 30 |
| FMP-matched-but-empty -> T33 ingredients (medium) | 9 |
| FMP-matched-but-empty -> T33 ingredients (low) | 2 |
| **FMP-empty -> T33 total (source=fmp_new_empty_t33)** | **41** |
| FMP-empty AND T33 also empty (no ingredients) | 77 |
| T33 fallback, no FMP match (high conf) | 14 |
| T33 fallback, no FMP match (medium conf) | 3 |
| T33 fallback, no FMP match (low conf) | 2 |
| **T33-only fallback total** | **19** |
| No match (no FMP, no T33) | 90 |
| Slugs matching a Functional Formulation FMP type | 138 |
| FMP Functional Formulation products available | 178 |

## Final ingredients_source breakdown

| ingredients_source | Count |
|--------------------|-------|
| fmp_new | 94 |
| t33 | 52 |
| gk | 13 |
| none | 162 |

## Low-Confidence Matches (needs Glen review)

These matched but with a fuzzy score below 0.85. Verify the FMP/T33 name is actually the same product before applying.

| Slug | Name | Pinecone Title | Best FMP/T33 Match | Score | Source |
|------|------|----------------|-------------------|-------|--------|
| magnesium-taurate | Magnesium Taurate | Magnesium Taurate | Magnesium Glycinate | 0.72 | fmp_new |
| humic-acid | Humic Acid | Humic Acid | Fulvic Acid Complex | 0.76 | fmp_old_t33 |
| dry-eye-relief-program | Dry Eye Relief Program | Dry Eye Relief Program | (T33) Moisturize | 0.78 | fmp_new_empty_t33 |
| zinc-taste-test | Zinc Taste Test | Zinc Taste Test | (T33) Chromium Taste Test | 0.79 | fmp_new_empty_t33 |
| c15-syntropy-pentadecanoic-acid | C15 Syntropy: Pentadecanoic Acid | C15 Syntropy: Pentadecanoic Acid | C15 Syntropy: Pentadecanoic Essential Fatty Acid | 0.79 | fmp_old_t33 |
| pms-relief | PMS Relief | PMS Relief | (T33) OcuFlow Daytime | 0.80 | fmp_new_empty_t33 |
| macular-wellness-program | Macular Wellness Program | Macular Wellness Program | Macular Wellness Crocin | 0.81 | fmp_new |
| glucose-tolerance-program | Glucose Tolerance Program | Glucose Tolerance Program | Glucose Tolerance | 0.81 | fmp_new |

## Unmatched Slugs (no FMP or T33 match)

These produced no match above the 0.72 cutoff. They may be Essences, Infoceuticals, or products not yet in FMP/T33. Ingredients will need manual entry.

| Slug | Name | Pinecone Title | Best FMP Guess |
|------|------|----------------|----------------|
| esr | Emotional Stress Release (MB5) | MB 5 | Sumac Bran 50:1 (0.42) |
| juglans-nigra | Juglans nigra | Juglans nigra | Pregnancy underwear (0.52) |
| clear-the-way | 5 Ways "Clear the Way" Naturally Transforms Tissue Healing | 5 Ways &quot;Clear the Way&quot; Naturally Transforms Tissue Healing | MSM Pure & Natural Lotion  (0.43) |
| electrolyte-mineral-manna | Electrolyte Mineral Manna | Electrolyte Mineral Manna | Spirit Mineral Iron in Terrain Restore (0.51) |
| bath-salt | Bath Salt | Bath Salt | Breast Support (0.57) |
| mb4-cch | MB 4 | MB 4 | B17 Max (0.35) |
| mb1-bsh | MB 1 | MB 1 | B17 Syntropy (0.57) |
| mb2-cmh | MB 2 | MB 2 | B17 Max (0.35) |
| mb8-love | MB 8 | MB 8 | B17 Max (0.35) |
| et2-immune-2 | ET2 Immu 2: Immunity 2 | ET2 Immu 2: Immunity 2 | ET2 Imu-2 Immune Energetic Terrain Infoceutical (0.53) |
| mb3-cbh | MB 3 | MB 3 | WholOmega 30 gelcaps (0.38) |
| et1-immune-1 | ET1 Immu 1: Immunity 1 | ET1 Immu 1: Immunity 1 | ET1 Imu-1 Immune Energetic Terrain Infoceutical (0.53) |
| lapachol | Lapachol | Lapachol | Lymph Flow (0.44) |
| et3-immune-3 | ET3 Immu 3: Immunity 3 | ET3 Immu 3: Immunity 3 | ET3 Imu-3 Immune Energetic Terrain Infoceutical (0.53) |
| gut-terrain-program | Gut Terrain Program | Gut Terrain Program | Terrain Restore Drops (0.65) |
| transresveratrol | Trans-Resveratrol | Trans-Resveratrol | Terrain Restore Drops (0.56) |
| centrophenoxine | Centrophenoxine | Centrophenoxine | N-Acetyl L-Carnosine (0.51) |
| et4-nerve | ET4 Nerve: Nervous System | ET4 Nerve: Nervous System | Nerve Pulse (0.51) |
| 5mthf | 5-MTHF | 5-MTHF | Eye Homeopathic Complex (0.38) |
| licorice-omnipotent | Licorice Omnipotent | Licorice Omnipotent | Microbiome (0.48) |
| ursolic-acid | Ursolic Acid 50% | Ursolic Acid 50% | Microwater, Acid (0.47) |
| et7-cfs | ET7 CFS: Chronic Fatigue Syndrome | ET7 CFS: Chronic Fatigue Syndrome | Kidney Failure - Chronic ebook (0.51) |
| wash--rinse | Wash & Rinse | Wash &amp; Rinse | Rise & Shine (0.64) |
| tmg-syntropy-powder-trimethylglycine | TMG Powder (Trimethylglycine) | TMG Powder (Trimethylglycine) | Magnesium Glycinate (0.48) |
| cds-water-purifier--activator | CDS Water Purifier & Activator | CDS Water Purifier &amp; Activator | CDS Activator
MMS (0.53) |
| phosphatides | Phosphatides | Phosphatides | Prostate Powder (0.60) |
| brain-program | Brain Program | Brain Program | Crystalline Lens Program (0.65) |
| ed10-skin-driver | ED10 Skin Driver | ED10 Skin Driver | ED10 Skin Energetic Driver Infoceutical (0.58) |
| c3g | C3G | C3G | TMG Syntropy Powder (0.33) |
| et8-neuro | ET8 Neuron  ET8 | ET8 Neuron  ET8 | Neuro+ Eye Drops (0.61) |
| et6-cfi | ET6 CFI: Colds Flu Immunity | ET6 CFI: Colds Flu Immunity | Shields Up - Cold/Flu Formula (0.49) |
| gingerol | Gingerol | Gingerol | Indigo (0.57) |
| et5-bsv | ET5 BSV: Broad Spectrum Virus | ET5 BSV: Broad Spectrum Virus | Holy Grail Full Spectrum ORMUS 3C (0.49) |
| tetrahydrocurcumin | Tetrahydrocurcumin | Tetrahydrocurcumin | Three Treasures (0.44) |
| reverse-aging-program | Reverse Aging Program | Reverse Aging Program | Reverse AGE (0.62) |
| ed6-heart-driver | ED6 Heart Driver | ED6 Heart Driver | ED6 Heart Energetic Driver Infoceutical (0.58) |
| ed1-source-driver | ED1 Source Driver | ED1 Source Driver | ED1 Source Energetic Driver Infoceutical (0.60) |
| iop-program | IOP Program | IOP Program | Immuno Spray (0.52) |
| dead-sea-salt | Dead Sea Salt | Dead Sea Salt | Heart Health (0.56) |
| ei3-mucous-membranessmall-intestine-meridian | EI3 Mucous Membranes/Small Intestine Meridian | EI3 Mucous Membranes/Small Intestine Meridian | EI3 Mucosae/Small Intestine Meridian Energetic Integrator Infoceutical (0.61) |
| baicalein | Baicalein | Baicalein | Brain Cleanse (0.64) |
| methylselenocysteine | Methylselenocysteine | Methylselenocysteine | N-Acetyl L-Carnosine (0.55) |
| ginsengosides | Ginsengosides | Ginsengosides | Invoice (0.48) |
| inositol | Inositol | Inositol | Indigo (0.53) |
| ei7-bloodgall-bladder-meridian | EI7 Blood/Gall Bladder Meridian | EI7 Blood/Gall Bladder Meridian | EI7 Blood/Gallbladder Meridian Energetic Integrator Infoceutical (0.63) |
| coluracetam | Coluracetam | Coluracetam | Source (0.59) |
| apigenin | Apigenin | Apigenin | AngiogenX (0.59) |
| zinc-ascorbate | Zinc Ascorbate | Zinc Ascorbate | Zinc Taste Test Drops (0.62) |
| ed9-muscle-driver | ED9 Muscle Driver | ED9 Muscle Driver | ED9 Muscle Energetic Driver Infoceutical (0.60) |
| salvianolic-acid | Salvianolic Acid | Salvianolic Acid | Svetinorm (Liver Peptide) (0.51) |
| ed8-stomach-driver | ED8 Stomach Driver | ED8 Stomach Driver | ED8 Stomach Energetic Driver Infoceutical (0.61) |
| rlipoate | R-Lipoate | R-Lipoate | Prostate Powder (0.59) |
| ed11-liver-driver | ED11 Liver Driver Pancreas) | ED11 Liver Driver Pancreas) | ED11 Liver Energetic Driver Infoceutical (0.58) |
| vinpocetine | Vinpocetine | Vinpocetine | Invoice (0.56) |
| ed2-imprinter-driver | ED2 Imprinter Driver | ED2 Imprinter Driver | ED2 Heart Imprinter Energetic Driver Infoceutical (0.58) |
| magnesium-acetyltaurate | Magnesium Acetyl-Taurate | Magnesium Acetyl-Taurate | Magnesium Glycinate (0.65) |
| dalpha-tocopherol-succinate | d-Alpha Tocopherol Succinate | d-Alpha Tocopherol Succinate | Spirit Mineral Copper in Terrain Restore (0.49) |
| ei9-thyroidtriple-warmer-meridian | EI9 Thyroid/Triple Warmer Meridian | EI9 Thyroid/Triple Warmer Meridian | EI9 Thyroid/Triple Burner Meridian Energetic Integrator Infoceutical (0.61) |
| asiaticosides | Asiaticosides | Asiaticosides | Pesticides ebook (0.55) |
| ed5-circulation-driver | ED5 Circulation Driver | ED5 Circulation Driver | ED5 Circulation Energetic Driver Infoceutical (0.66) |
| cholecalciferol | Cholecalciferol | Cholecalciferol | Crucifer Complex Powder (0.52) |
| ed4-nervous-system-driver | ED4 Nervous System Driver | ED4 Nervous System Driver | ED4 Nerve Energetic Driver Infoceutical (0.53) |
| viscum-album | Viscum album | Viscum album | Vitamin A Syntropy (0.48) |
| astragalus-membranaceus | Astragalus membranaceus | Astragalus membranaceus | Glucose Tolerance (0.55) |
| lipase | Lipase | Lipase | Lipid Cleanse (0.63) |
| methyl-cobalamin | Methyl Cobalamin | Methyl Cobalamin | Hydrolyzed Collagen Powder (0.51) |
| cocos-nucifera | Cocos nucifera | Cocos nucifera | Crucifer Complex Powder (0.64) |
| ed3-cell-driver | ED3 Cell Driver | ED3 Cell Driver | ED3 Cell Energetic Driver Infoceutical (0.57) |
| ed12-kidney-driver | ED12 Kidney Driver | ED12 Kidney Driver | ED12 Kidney Energetic Driver Infoceutical (0.61) |
| honokiol | Honokiol | Honokiol | Neem Oil Roll On (0.42) |
| ed7-lung-driver | ED7 Lung Driver | ED7 Lung Driver | ED7 Lung Energetic Driver Infoceutical (0.57) |
| serrapeptase | Serrapeptase | Serrapeptase | Cerluten (Brain Peptide) (0.54) |
| polyphenols-camellia-sinensis | Polyphenols (Camellia sinensis) | Polyphenols (Camellia sinensis) | Little Whelks Shell Essence in Terrain Restore (0.47) |
| neutral-protease | Neutral Protease | Neutral Protease | Neuroprotect (0.64) |
| ed16-bone-driver | ED16 Bone Driver | ED16 Bone Driver | ED16 Bone Energetic Driver Infoceutical (0.58) |
| piperine | Piperine | Piperine | Spike Shield (0.50) |
| ei10-circulationheart-protector-meridian | EI10 Circulation/Heart Protector Meridian | EI10 Circulation/Heart Protector Meridian | EI10 Circulation/Pericardium Meridian Energetic Integrator Infoceutical (0.58) |
| ascorbyl-palmitate | Ascorbyl Palmitate | Ascorbyl Palmitate | Color & Light ebook (0.43) |
| ei11-bone-marrowstomach-meridian | EI11 Bone Marrow/Stomach Meridian | EI11 Bone Marrow/Stomach Meridian | Stamakort (Stomach Peptide) (0.52) |
| proanthocyanidin-pinus-pinaster- | Proanthocyanidin (Pinus pinaster) | Proanthocyanidin (Pinus pinaster) | Globe Artichoke Cynarin (Cynara scolymus) in Terrain Restore (0.46) |
| cotinus-coggygria | Cotinus coggygria | Cotinus coggygria | Crystalline Lens Program (0.49) |
| ed15-pancreas-driver- | ED15 Pancreas Driver | ED15 Pancreas Driver | ED15 Pancreas Energetic Driver Infoceutical (0.63) |
| polysaccharides-aloe-barbadensis- | Polysaccharides (Aloe barbadensis) | Polysaccharides (Aloe barbadensis) | Aloe (Aloe barbadensis) tincture in Terrain Restore (0.58) |
| adenosyl-cobalamin | Adenosyl Cobalamin | Adenosyl Cobalamin | Adrenal Syntropy Powder (0.48) |
| ed14-spleenthymus-driver | ED14 Spleen/Thymus Driver | ED14 Spleen/Thymus Driver | ED14 Spleen Energetic Driver Infoceutical (0.61) |
| ginkgo-biloba-extract | Ginkgo Biloba Extract | Ginkgo Biloba Extract | Q-Link Gold Tag (0.50) |
| ed13-immunity-driver | ED13 Immunity Driver | ED13 Immunity Driver | ED13 Immunity Energetic Driver Infoceutical (0.63) |
| anthocyanidins-ribes-nigrum | Anthocyanidins (Ribes nigrum) | Anthocyanidins (Ribes nigrum) | Moisture Eyes Night Drops  (0.45) |
| ei2-chest-heartlung-meridian | EI2 Chest: Heart/Lung Meridian | EI2 Chest: Heart/Lung Meridian | Chelohart (Heart Peptide) (0.54) |
| c15-pentadecanoic-acid | C15: Pentadecanoic Acid | C15: Pentadecanoic Acid | Obie One Angelic Essence in Terrain Restore (0.45) |
