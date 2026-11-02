"""OMOP vocabulary: well-known standard concept IDs, source-code normalisation, Athena
loading, and the source -> standard ("Maps to") lookup used by every ETL.

Without the Athena files in data/vocab, vocab tables are empty and every clinical
concept_id becomes 0 (source values are always kept). Re-running the ETL after the
files are added fills them in.
"""

import duckdb

from clinical_ml import config

# Standard concepts that are stable across vocabulary releases and needed even without
# Athena (gender, race, ethnicity, visit and record-provenance types).
MALE, FEMALE = 8507, 8532
WHITE, BLACK, ASIAN, AIAN, NHPI = 8527, 8516, 8515, 8657, 8557
HISPANIC, NOT_HISPANIC = 38003563, 38003564
INPATIENT_VISIT, ER_VISIT = 9201, 9203
EHR = 32817  # Type Concept "EHR"

# OMB race / ethnicity codes used by the US Core extensions (urn:oid:2.16.840.1.113883.6.238)
OMB_RACE = {"2106-3": WHITE, "2054-5": BLACK, "2028-9": ASIAN, "1002-5": AIAN, "2076-8": NHPI}
OMB_ETHNICITY = {"2135-2": HISPANIC, "2186-5": NOT_HISPANIC}


def icd9cm(code: str) -> str:
    """Normalise an ICD-9-CM diagnosis code to Athena's dotted form.

    Handles codes with dots stripped (MIMIC: '4280' -> '428.0', 'V462' -> 'V46.2',
    'E8790' -> 'E879.0') and with leading zeros stripped (Diabetes-130: '8' -> '008',
    '41.9' -> '041.9').
    """
    code = code.strip().upper()
    if "." in code:
        head, tail = code.split(".", 1)
    elif code.startswith("E"):
        head, tail = code[:4], code[4:]
    else:
        head, tail = code[:3], code[3:]
    if head.isdigit():
        head = head.zfill(3)
    return f"{head}.{tail}" if tail else head


def icd10cm(code: str) -> str:
    code = code.strip().upper().replace(".", "")
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def icd9proc(code: str) -> str:
    code = code.strip().replace(".", "")
    return f"{code[:2]}.{code[2:]}" if len(code) > 2 else code


def load_athena(con: duckdb.DuckDBPyConnection) -> dict:
    """Load Athena CSVs (tab-delimited, YYYYMMDD dates) from data/vocab if present."""
    loaded = {}
    specs = {
        "concept": "CONCEPT.csv",
        "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
        "vocabulary": "VOCABULARY.csv",
    }
    for table, fname in specs.items():
        path = config.VOCAB_DIR / fname
        if not path.exists():
            continue
        cols = [r[0] for r in con.execute(f"DESCRIBE {config.VOCAB}.{table}").fetchall()]
        select = ", ".join(
            f"strptime({c}, '%Y%m%d')::DATE" if c.endswith("_date") else c for c in cols
        )
        con.execute(f"DELETE FROM {config.VOCAB}.{table}")
        con.execute(
            f"INSERT INTO {config.VOCAB}.{table} SELECT {select} FROM read_csv(?, delim='\t', "
            "header=true, quote='', all_varchar=true)",
            [str(path)],
        )
        loaded[table] = con.execute(f"SELECT count(*) FROM {config.VOCAB}.{table}").fetchone()[0]
    return loaded


def build_code_map(con: duckdb.DuckDBPyConnection, staged: str) -> str:
    """Create a temp lookup (source_vocabulary_id, source_code) -> (source_concept_id,
    concept_id) for the distinct codes in `staged`, and return its name.

    Unmapped codes get 0. A source code that "Maps to" several standard concepts yields
    several rows, which is the OMOP convention (one record per standard concept).
    The pseudo-vocabulary 'RxNorm-ingredient-name' matches RxNorm ingredients by name,
    for sources (Diabetes-130) that record drugs by name only.
    """
    v = config.VOCAB
    name = f"{staged}_code_map"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {name} AS
        WITH codes AS (SELECT DISTINCT source_vocabulary_id, source_code FROM {staged}),
        maps_to AS (
            SELECT cr.concept_id_1, std.concept_id
            FROM {v}.concept_relationship cr
            JOIN {v}.concept std ON std.concept_id = cr.concept_id_2
             AND std.standard_concept = 'S' AND std.invalid_reason IS NULL
            WHERE cr.relationship_id = 'Maps to' AND cr.invalid_reason IS NULL
        ),
        by_code AS (
            SELECT c.source_vocabulary_id, c.source_code,
                   src.concept_id AS source_concept_id, m.concept_id
            FROM codes c
            LEFT JOIN {v}.concept src
              ON src.vocabulary_id = c.source_vocabulary_id AND src.concept_code = c.source_code
            LEFT JOIN maps_to m ON m.concept_id_1 = src.concept_id
            WHERE c.source_vocabulary_id <> 'RxNorm-ingredient-name'
        ),
        by_name AS (
            SELECT c.source_vocabulary_id, c.source_code,
                   ing.concept_id AS source_concept_id, ing.concept_id
            FROM codes c
            LEFT JOIN {v}.concept ing
              ON ing.vocabulary_id = 'RxNorm' AND ing.concept_class_id = 'Ingredient'
             AND ing.standard_concept = 'S' AND lower(ing.concept_name) = lower(c.source_code)
            WHERE c.source_vocabulary_id = 'RxNorm-ingredient-name'
        )
        SELECT source_vocabulary_id, source_code,
               coalesce(source_concept_id, 0) AS source_concept_id,
               coalesce(concept_id, 0) AS concept_id
        FROM (SELECT * FROM by_code UNION ALL SELECT * FROM by_name)
    """)
    return name
