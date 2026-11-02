"""Project paths. Everything lives under the project root, never in the user's home."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CLINICAL_ML_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
WAREHOUSE = DATA / "warehouse" / "omop.duckdb"
VOCAB_DIR = DATA / "vocab"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

DIABETES_DIR = RAW / "diabetes130"
FHIR_DEMO_DIR = RAW / "mimic_iv_fhir_demo"   # PhysioNet project root (SHA256SUMS.txt)
FHIR_DIR = FHIR_DEMO_DIR / "fhir"            # the NDJSON resources

# One OMOP CDM schema per source, as in an OHDSI network: the same analysis SQL runs
# against each. Vocabulary is shared; cohorts, labels and features go to `results`.
CDM_SCHEMAS = {"d130": "cdm_d130", "mimic": "cdm_mimic"}
VOCAB = "vocab"
RESULTS = "results"

SEED = 20261102
