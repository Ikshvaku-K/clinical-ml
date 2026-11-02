"""Model features for the Diabetes-130 cohort, built from the OMOP tables (not the raw CSV).

Race, ethnicity and payer are deliberately NOT features; they are written to a separate
audit table used only for the fairness analysis (M3). Placeholder dates are never used.
"""

import duckdb
import pandas as pd

from clinical_ml import config

# Primary/secondary diagnosis groups as in Strack et al. (2014), on ICD-9-CM.
DIAG_GROUPS = """
CASE
  WHEN code LIKE '250%' THEN 'diabetes'
  WHEN code LIKE 'V%' OR code LIKE 'E%' THEN 'other'
  WHEN num BETWEEN 390 AND 459 OR num = 785 THEN 'circulatory'
  WHEN num BETWEEN 460 AND 519 OR num = 786 THEN 'respiratory'
  WHEN num BETWEEN 520 AND 579 OR num = 787 THEN 'digestive'
  WHEN num BETWEEN 800 AND 999 THEN 'injury'
  WHEN num BETWEEN 710 AND 739 THEN 'musculoskeletal'
  WHEN num BETWEEN 580 AND 629 OR num = 788 THEN 'genitourinary'
  WHEN num BETWEEN 140 AND 239 THEN 'neoplasms'
  ELSE 'other'
END
"""
GROUPS = ["circulatory", "respiratory", "digestive", "diabetes", "injury", "musculoskeletal",
          "genitourinary", "neoplasms", "other"]

NUMERIC_OBS = {
    "Inpatient visits in prior year": "prior_inpatient",
    "Emergency visits in prior year": "prior_emergency",
    "Outpatient visits in prior year": "prior_outpatient",
    "Lab tests during encounter": "n_lab_tests",
    "Procedures during encounter": "n_procedures",
    "Distinct medications during encounter": "n_medications",
    "Diagnoses entered for encounter": "n_diagnoses",
}
# Drugs given to >= 1% of encounters get their own dose-status feature.
MIN_DRUG_PREVALENCE = 0.01

CATEGORICAL = ["sex", "age_band", "admission_type", "admission_source", "discharge_disposition",
               "specialty", "primary_dx_group", "a1c", "glucose"]


def build(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    s, r = config.CDM_SCHEMAS["d130"], config.RESULTS
    num_cols = ",\n".join(
        f"max(value_as_number) FILTER (WHERE observation_source_value = '{k}') AS {v}"
        for k, v in NUMERIC_OBS.items()
    )
    sec_cols = ",\n".join(f"bool_or(grp = '{g}') AS sec_dx_{g}" for g in GROUPS)
    n_visits = con.execute(f"SELECT count(*) FROM {r}.d130_cohort").fetchone()[0]
    drugs = [d for (d,) in con.execute(f"""
        SELECT drug_source_value FROM {s}.drug_exposure GROUP BY 1
        HAVING count(DISTINCT visit_occurrence_id) >= {MIN_DRUG_PREVALENCE * n_visits}
        ORDER BY 1""").fetchall()]
    drug_cols = ",\n".join(
        f"coalesce(max(value_as_string) FILTER (WHERE observation_source_value = "
        f"'{d} dose change'), 'No') AS drug_{d.replace('-', '_')}"
        for d in drugs
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE {r}.d130_features AS
        WITH obs AS (
            SELECT visit_occurrence_id,
                   {num_cols},
                   max(value_as_string) FILTER (WHERE observation_source_value = 'Age group')
                       AS age_band,
                   max(value_as_string) FILTER (WHERE observation_source_value = 'Admission type')
                       AS admission_type,
                   max(value_as_string)
                       FILTER (WHERE observation_source_value = 'Admitting physician specialty')
                       AS specialty,
                   count(*) FILTER (WHERE observation_source_value LIKE '% dose change'
                                    AND value_as_string IN ('Up', 'Down')) AS n_dose_changes,
                   {drug_cols}
            FROM {s}.observation GROUP BY 1
        ),
        dx AS (
            SELECT visit_occurrence_id, condition_status_source_value AS status,
                   {DIAG_GROUPS.replace('code', 'condition_source_value')
                                .replace('num', 'try_cast(split_part(condition_source_value, '
                                         "'.', 1) AS INTEGER)")} AS grp
            FROM {s}.condition_occurrence
        ),
        dxw AS (
            SELECT visit_occurrence_id,
                   max(grp) FILTER (WHERE status = 'primary') AS primary_dx_group,
                   {sec_cols.replace("grp =", "status = 'secondary' AND grp =")}
            FROM dx GROUP BY 1
        ),
        meas AS (
            SELECT visit_occurrence_id,
                   max(value_source_value) FILTER (WHERE measurement_source_value = 'HbA1c')
                       AS a1c,
                   max(value_source_value)
                       FILTER (WHERE measurement_source_value = 'Serum glucose (max)') AS glucose
            FROM {s}.measurement GROUP BY 1
        ),
        drug AS (
            SELECT visit_occurrence_id, count(DISTINCT drug_source_value) AS n_diabetes_drugs,
                   bool_or(drug_source_value = 'insulin') AS on_insulin
            FROM {s}.drug_exposure GROUP BY 1
        )
        SELECT c.visit_occurrence_id, c.person_id, c.readmit_30d,
               CASE p.gender_concept_id WHEN 8507 THEN 'male' WHEN 8532 THEN 'female'
                    ELSE 'unknown' END AS sex,
               obs.age_band, obs.admission_type,
               v.admitted_from_source_value AS admission_source,
               v.discharged_to_source_value AS discharge_disposition,
               coalesce(obs.specialty, 'missing') AS specialty,
               (v.visit_end_date - v.visit_start_date) AS length_of_stay,
               {", ".join(f"obs.{c}" for c in NUMERIC_OBS.values())},
               obs.n_dose_changes,
               {", ".join(f"obs.drug_{d.replace('-', '_')}" for d in drugs)},
               coalesce(drug.n_diabetes_drugs, 0) AS n_diabetes_drugs,
               coalesce(drug.on_insulin, false) AS on_insulin,
               coalesce(dxw.primary_dx_group, 'missing') AS primary_dx_group,
               {", ".join(f"coalesce(dxw.sec_dx_{g}, false) AS sec_dx_{g}" for g in GROUPS)},
               coalesce(meas.a1c, 'not measured') AS a1c,
               coalesce(meas.glucose, 'not measured') AS glucose
        FROM {r}.d130_cohort c
        JOIN {s}.person p USING (person_id)
        JOIN {s}.visit_occurrence v USING (visit_occurrence_id)
        LEFT JOIN obs USING (visit_occurrence_id)
        LEFT JOIN dxw USING (visit_occurrence_id)
        LEFT JOIN meas USING (visit_occurrence_id)
        LEFT JOIN drug USING (visit_occurrence_id)
        WHERE c.eligible
    """)
    # Audit-only attributes, joined back at evaluation time.
    con.execute(f"""
        CREATE OR REPLACE TABLE {r}.d130_audit AS
        SELECT c.visit_occurrence_id,
               coalesce(nullif(p.race_source_value, '?'), 'Unknown') AS race,
               CASE p.ethnicity_concept_id WHEN 38003563 THEN 'Hispanic' ELSE 'Not recorded'
                    END AS ethnicity,
               CASE p.gender_concept_id WHEN 8507 THEN 'male' WHEN 8532 THEN 'female'
                    ELSE 'unknown' END AS sex,
               f.age_band,
               coalesce(pp.payer_source_value, 'missing') AS payer
        FROM {r}.d130_cohort c
        JOIN {s}.person p USING (person_id)
        JOIN {r}.d130_features f USING (visit_occurrence_id)
        LEFT JOIN (SELECT person_id, payer_plan_period_start_date, payer_source_value
                   FROM {s}.payer_plan_period) pp
          ON pp.person_id = c.person_id AND pp.payer_plan_period_start_date = c.visit_start_date
    """)
    return con.execute(f"SELECT * FROM {r}.d130_features ORDER BY visit_occurrence_id").df()
