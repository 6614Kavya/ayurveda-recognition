"""
VedaVision — Module 3 Backend: Species Display Metadata
==========================================================
The classifier only ever outputs a species CODE (the label strings it
was trained on, e.g. "ranawara", "wal_kollu") plus a confidence score —
it has no notion of Sinhala names, traditional uses, or which diseases
a species treats. Those are static reference facts, not something a
vision model predicts, so they're looked up here after prediction.

TODO: replace the placeholder sinhala_name/uses/diseases_treated values
below with the verified entries from the project's species reference
sheet (the same source used for the interim report / hospital
collaboration) before this goes into any real deployment or demo.
Species codes below are taken from the current model's training labels
(model_training.py look-alike clusters) — if you add/rename a species
and retrain, update this dict to match the new label strings exactly,
or the API will fall back to showing the raw code.
"""

SPECIES_METADATA: dict[str, dict] = {
    "thunpath_kurundu":     {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "kasthuri_dehi":        {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "beli":                 {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "wal_kollu":             {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "kattakumanjal":        {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "kalawal":              {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "kathurupila":          {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "nil_awariya":          {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "wal_bilin":            {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "maha_undupiyaliya":    {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "ranawara":             {"sinhala_name": "", "uses": "", "diseases_treated": []},
    "siymbala":             {"sinhala_name": "", "uses": "", "diseases_treated": []},
}


def get_species_display(species_code: str) -> dict:
    meta = SPECIES_METADATA.get(species_code, {})
    return {
        "plant_name": species_code,
        "sinhala_name": meta.get("sinhala_name", ""),
        "uses": meta.get("uses", ""),
        "diseases_treated": meta.get("diseases_treated", []),
    }
