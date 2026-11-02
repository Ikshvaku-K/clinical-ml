"""Diabetes 130-US Hospitals (Strack et al., 2014) -> OMOP CDM v5.4 (schema cdm_d130).

One source row is one inpatient encounter. Mapping decisions (see EXPLAIN.md 3.3):

* Date shift. The source has no dates. Each patient's encounters are ordered by
  encounter_id and placed 365 days apart from an anchor of 2000-01-01; the visit ends
  after time_in_hospital days. Dates are a storage convention only, never features.
* Race/ethnicity. OMOP separates them; the source puts "Hispanic" in race, so it maps
  to ethnicity = Hispanic with race unknown. Ethnicity is otherwise unknown (0).
* year_of_birth = 2000 - midpoint of the age band at the first encounter; the band
  itself is kept per encounter in `observation`.
* Expired dispositions (11, 19, 20, 21) also create a `death` record.
* The readmission outcome is not a clinical fact: it goes to results.d130_label.
"""

import duckdb

from clinical_ml import config
from clinical_ml.omop import vocab

SCHEMA = config.CDM_SCHEMAS["d130"]
ANCHOR = "DATE '2000-01-01'"
EXPIRED = (11, 19, 20, 21)

DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
# Encounter-level facts with no better OMOP home than `observation` (name, column).
NUMERIC_OBSERVATIONS = [
    ("Inpatient visits in prior year", "number_inpatient"),
    ("Emergency visits in prior year", "number_emergency"),
    ("Outpatient visits in prior year", "number_outpatient"),
    ("Lab tests during encounter", "num_lab_procedures"),
    ("Procedures during encounter", "num_procedures"),
    ("Distinct medications during encounter", "num_medications"),
    ("Diagnoses entered for encounter", "number_diagnoses"),
]
# Lab categories: the source gives no LOINC codes, so these are manual ETL mappings.
MEASUREMENTS = [
    ("A1Cresult", "4548-4", "HbA1c", "None"),
    ("max_glu_serum", "2345-7", "Serum glucose (max)", "None"),
    ("weight", "29463-7", "Body weight", "?"),
]


def _register_udf(con: duckdb.DuckDBPyConnection, name: str, fn) -> None:
    try:
        con.remove_function(name)
    except duckdb.InvalidInputException:
        pass
    con.create_function(name, fn, side_effects=False)


def stage(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS stage")
    con.execute(
        "CREATE OR REPLACE TABLE stage.d130_raw AS SELECT * FROM read_csv(?, all_varchar=true)",
        [str(config.DIABETES_DIR / "diabetic_data.csv")],
    )
    # IDS_mapping.csv is three tables stacked, separated by blank ',' rows.
    rows, kind = [], None
    for line in (config.DIABETES_DIR / "IDS_mapping.csv").read_text().splitlines():
        if not line.strip(",").strip():
            kind = None
            continue
        key, _, desc = line.partition(",")
        if kind is None:
            kind = key
            continue
        rows.append((kind, int(key), desc.strip().strip('"').strip()))
    con.execute(
        "CREATE OR REPLACE TABLE stage.d130_ids (kind VARCHAR, id INTEGER, description VARCHAR)")
    con.executemany("INSERT INTO stage.d130_ids VALUES (?, ?, ?)", rows)
    con.execute("""
        CREATE OR REPLACE TABLE stage.d130_enc AS
        SELECT r.*,
               encounter_id::INTEGER AS visit_id,
               patient_nbr::INTEGER AS pid,
               row_number() OVER (PARTITION BY patient_nbr ORDER BY encounter_id::INTEGER) AS k,
               ({anchor} + INTERVAL ((k - 1) * 365) DAY)::DATE AS start_date,
               (start_date + INTERVAL (time_in_hospital::INTEGER) DAY)::DATE AS end_date
        FROM stage.d130_raw r
    """.replace("{anchor}", ANCHOR))


def load(con: duckdb.DuckDBPyConnection) -> None:
    s = SCHEMA
    con.execute(f"""
        INSERT INTO {s}.cdm_source BY NAME SELECT
            'Diabetes 130-US Hospitals 1999-2008' AS cdm_source_name,
            'D130' AS cdm_source_abbreviation,
            'UCI Machine Learning Repository (CC BY 4.0)' AS cdm_holder,
            'Strack et al. 2014, BioMed Res Int, doi:10.1155/2014/781670' AS source_description,
            'https://archive.ics.uci.edu/dataset/296' AS source_documentation_reference,
            'clinical_ml.etl.diabetes130' AS cdm_etl_reference,
            DATE '2014-05-03' AS source_release_date,
            current_date AS cdm_release_date,
            'v5.4' AS cdm_version, 0 AS cdm_version_concept_id,
            coalesce((SELECT vocabulary_version FROM {config.VOCAB}.vocabulary
                      WHERE vocabulary_id = 'None'), 'none loaded') AS vocabulary_version
    """)

    con.execute(f"""
        INSERT INTO {s}.person BY NAME
        SELECT pid AS person_id,
               CASE gender WHEN 'Male' THEN {vocab.MALE} WHEN 'Female' THEN {vocab.FEMALE}
                    ELSE 0 END AS gender_concept_id,
               2000 - (regexp_extract(age, '\\[(\\d+)-', 1)::INTEGER + 5) AS year_of_birth,
               CASE race WHEN 'Caucasian' THEN {vocab.WHITE}
                         WHEN 'AfricanAmerican' THEN {vocab.BLACK}
                         WHEN 'Asian' THEN {vocab.ASIAN} ELSE 0 END AS race_concept_id,
               CASE race WHEN 'Hispanic' THEN {vocab.HISPANIC} ELSE 0 END AS ethnicity_concept_id,
               patient_nbr AS person_source_value,
               gender AS gender_source_value,
               race AS race_source_value,
               CASE race WHEN 'Hispanic' THEN 'Hispanic' END AS ethnicity_source_value
        FROM stage.d130_enc WHERE k = 1
    """)

    con.execute(f"""
        INSERT INTO {s}.observation_period BY NAME
        SELECT row_number() OVER (ORDER BY pid) AS observation_period_id, pid AS person_id,
               min(start_date) AS observation_period_start_date,
               max(end_date) AS observation_period_end_date,
               {vocab.EHR} AS period_type_concept_id
        FROM stage.d130_enc GROUP BY pid
    """)

    con.execute(f"""
        INSERT INTO {s}.visit_occurrence BY NAME
        SELECT e.visit_id AS visit_occurrence_id, e.pid AS person_id,
               {vocab.INPATIENT_VISIT} AS visit_concept_id,
               e.start_date AS visit_start_date, e.end_date AS visit_end_date,
               {vocab.EHR} AS visit_type_concept_id,
               e.encounter_id AS visit_source_value,
               0 AS admitted_from_concept_id,
               left(src.description, 50) AS admitted_from_source_value,
               0 AS discharged_to_concept_id,
               left(dis.description, 50) AS discharged_to_source_value,
               lag(e.visit_id) OVER (PARTITION BY e.pid ORDER BY e.k)
                   AS preceding_visit_occurrence_id
        FROM stage.d130_enc e
        LEFT JOIN stage.d130_ids src
          ON src.kind = 'admission_source_id' AND src.id = e.admission_source_id::INTEGER
        LEFT JOIN stage.d130_ids dis
          ON dis.kind = 'discharge_disposition_id' AND dis.id = e.discharge_disposition_id::INTEGER
    """)

    con.execute(f"""
        INSERT INTO {s}.death BY NAME
        SELECT pid AS person_id, end_date AS death_date, {vocab.EHR} AS death_type_concept_id
        FROM stage.d130_enc
        WHERE discharge_disposition_id::INTEGER IN {EXPIRED}
        QUALIFY row_number() OVER (PARTITION BY pid ORDER BY k) = 1
    """)

    # Conditions: diag_1 is the primary diagnosis, diag_2/3 secondary.
    _register_udf(con, "icd9cm", vocab.icd9cm)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE d130_cond AS
        SELECT visit_id, pid, start_date, end_date, pos, code AS raw_code,
               'ICD9CM' AS source_vocabulary_id, icd9cm(code) AS source_code
        FROM (UNPIVOT (SELECT visit_id, pid, start_date, end_date, diag_1, diag_2, diag_3
                       FROM stage.d130_enc)
              ON diag_1, diag_2, diag_3 INTO NAME pos VALUE code)
        WHERE code <> '?'
    """)
    cmap = vocab.build_code_map(con, "d130_cond")
    con.execute(f"""
        INSERT INTO {s}.condition_occurrence BY NAME
        SELECT row_number() OVER (ORDER BY c.visit_id, c.pos, m.concept_id)
                   AS condition_occurrence_id,
               c.pid AS person_id, m.concept_id AS condition_concept_id,
               c.start_date AS condition_start_date, c.end_date AS condition_end_date,
               {vocab.EHR} AS condition_type_concept_id, 0 AS condition_status_concept_id,
               c.visit_id AS visit_occurrence_id, c.source_code AS condition_source_value,
               m.source_concept_id AS condition_source_concept_id,
               CASE c.pos WHEN 'diag_1' THEN 'primary' ELSE 'secondary' END
                   AS condition_status_source_value
        FROM d130_cond c JOIN {cmap} m USING (source_vocabulary_id, source_code)
    """)

    # Drugs: one row per ingredient given; combination products split into ingredients.
    cols = ", ".join(f'"{c}"' for c in DRUG_COLUMNS)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE d130_drug AS
        SELECT visit_id, pid, start_date, end_date, product, status,
               unnest(string_split(product, '-')) AS source_code,
               'RxNorm-ingredient-name' AS source_vocabulary_id
        FROM (UNPIVOT (SELECT visit_id, pid, start_date, end_date, {cols} FROM stage.d130_enc)
              ON {cols} INTO NAME product VALUE status)
        WHERE status <> 'No'
    """)
    dmap = vocab.build_code_map(con, "d130_drug")
    con.execute(f"""
        INSERT INTO {s}.drug_exposure BY NAME
        SELECT row_number() OVER (ORDER BY d.visit_id, d.product, d.source_code)
                   AS drug_exposure_id,
               d.pid AS person_id, m.concept_id AS drug_concept_id,
               d.start_date AS drug_exposure_start_date, d.end_date AS drug_exposure_end_date,
               {vocab.EHR} AS drug_type_concept_id, d.visit_id AS visit_occurrence_id,
               d.product AS drug_source_value, m.source_concept_id AS drug_source_concept_id
        FROM d130_drug d JOIN {dmap} m USING (source_vocabulary_id, source_code)
    """)

    # Measurements: categorical results stored as source values, never as fake numbers.
    meas = " UNION ALL ".join(
        f"SELECT visit_id, pid, start_date, '{col}' AS col, '{name}' AS name, "
        f"'LOINC' AS source_vocabulary_id, '{loinc}' AS source_code, \"{col}\" AS value "
        f"FROM stage.d130_enc WHERE \"{col}\" <> '{missing}'"
        for col, loinc, name, missing in MEASUREMENTS
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE d130_meas AS {meas}")
    mmap = vocab.build_code_map(con, "d130_meas")
    con.execute(f"""
        INSERT INTO {s}.measurement BY NAME
        SELECT row_number() OVER (ORDER BY x.visit_id, x.col) AS measurement_id,
               x.pid AS person_id, m.concept_id AS measurement_concept_id,
               x.start_date AS measurement_date, {vocab.EHR} AS measurement_type_concept_id,
               x.visit_id AS visit_occurrence_id, x.name AS measurement_source_value,
               m.source_concept_id AS measurement_source_concept_id,
               x.value AS value_source_value
        FROM d130_meas x JOIN {mmap} m USING (source_vocabulary_id, source_code)
    """)

    # Observations: utilisation counts, age band, admission type, specialty, dose changes.
    numeric = " UNION ALL ".join(
        f"SELECT visit_id, pid, start_date, '{name}' AS name, {col}::DOUBLE AS num, "
        f"NULL AS str FROM stage.d130_enc"
        for name, col in NUMERIC_OBSERVATIONS
    )
    dose = " UNION ALL ".join(
        f"SELECT visit_id, pid, start_date, '{d} dose change' AS name, NULL AS num, "
        f"\"{d}\" AS str FROM stage.d130_enc WHERE \"{d}\" <> 'No'"
        for d in DRUG_COLUMNS
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE d130_obs AS
        {numeric} UNION ALL {dose}
        UNION ALL SELECT visit_id, pid, start_date, 'Age group', NULL, age FROM stage.d130_enc
        UNION ALL SELECT e.visit_id, e.pid, e.start_date, 'Admission type', NULL, i.description
            FROM stage.d130_enc e JOIN stage.d130_ids i
              ON i.kind = 'admission_type_id' AND i.id = e.admission_type_id::INTEGER
        UNION ALL SELECT visit_id, pid, start_date, 'Admitting physician specialty', NULL,
                         medical_specialty
            FROM stage.d130_enc WHERE medical_specialty <> '?'
    """)
    con.execute(f"""
        INSERT INTO {s}.observation BY NAME
        SELECT row_number() OVER (ORDER BY visit_id, name) AS observation_id,
               pid AS person_id, 0 AS observation_concept_id, start_date AS observation_date,
               {vocab.EHR} AS observation_type_concept_id, num AS value_as_number,
               str AS value_as_string, visit_id AS visit_occurrence_id,
               name AS observation_source_value
        FROM d130_obs
    """)

    con.execute(f"""
        INSERT INTO {s}.payer_plan_period BY NAME
        SELECT row_number() OVER (ORDER BY visit_id) AS payer_plan_period_id,
               pid AS person_id, start_date AS payer_plan_period_start_date,
               end_date AS payer_plan_period_end_date, 0 AS payer_concept_id,
               payer_code AS payer_source_value
        FROM stage.d130_enc WHERE payer_code <> '?'
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {config.RESULTS}.d130_label AS
        SELECT visit_id AS visit_occurrence_id, readmitted AS readmitted_source_value,
               readmitted = '<30' AS readmit_30d
        FROM stage.d130_enc
    """)


def run(con: duckdb.DuckDBPyConnection) -> dict:
    stage(con)
    load(con)
    return {
        t: con.execute(f"SELECT count(*) FROM {SCHEMA}.{t}").fetchone()[0]
        for t in ("person", "observation_period", "visit_occurrence", "condition_occurrence",
                  "drug_exposure", "measurement", "observation", "payer_plan_period", "death")
    }
