"""MIMIC-IV Clinical Database Demo on FHIR (R4 NDJSON) -> OMOP CDM v5.4 (schema cdm_mimic).

FHIR resources are flattened in Python (references, extensions, multiple codings),
then loaded with SQL. OMOP keys reuse MIMIC's own integer identifiers
(subject_id, hadm_id, ED/ICU stay_id), so every row traces back to the source.

  Patient                  -> person (+ death)
  Encounter (hospital)     -> visit_occurrence, Inpatient Visit
  Encounter (ED)           -> visit_occurrence, Emergency Room Visit
  Encounter (ICU)          -> visit_detail under its hospital visit
  Condition(+ED)           -> condition_occurrence (ICD-9-CM / ICD-10-CM)
  Procedure(+ED, +ICU)     -> procedure_occurrence (ICD-9-Proc, ICD-10-PCS, SNOMED, MIMIC items)
  Observation labs         -> measurement (linked to the admission whose period contains it)
  Observation ED vitals    -> measurement (LOINC; blood-pressure panels split into components)
  Observation chartevents  -> measurement if numeric, observation if text
  MedicationRequest        -> drug_exposure (NDC where available)
"""

import gzip
import json
from collections.abc import Iterator

import duckdb
import pandas as pd

from clinical_ml import config
from clinical_ml.omop import vocab

SCHEMA = config.CDM_SCHEMAS["mimic"]

# FHIR code system (last path segment) -> (OMOP vocabulary_id, normaliser)
SYSTEMS = {
    "mimic-diagnosis-icd9": ("ICD9CM", vocab.icd9cm),
    "mimic-diagnosis-icd10": ("ICD10CM", vocab.icd10cm),
    "mimic-procedure-icd9": ("ICD9Proc", vocab.icd9proc),
    "mimic-procedure-icd10": ("ICD10PCS", str.strip),
    "loinc.org": ("LOINC", str.strip),
    "sct": ("SNOMED", str.strip),
    "mimic-medication-ndc": ("NDC", str.strip),
}


def _read(name: str) -> Iterator[dict]:
    with gzip.open(config.FHIR_DIR / f"Mimic{name}.ndjson.gz", "rt") as f:
        for line in f:
            yield json.loads(line)


def _ref(r: dict, key: str) -> str | None:
    ref = r.get(key)
    if isinstance(ref, list):
        ref = ref[0] if ref else None
    return ref["reference"].split("/", 1)[1] if ref else None


def _code(concept: dict | None) -> dict:
    """First coding of a CodeableConcept as (vocabulary, normalised code, raw code, display)."""
    coding = (concept or {}).get("coding", [{}])[0]
    system = coding.get("system", "").rstrip("/").rsplit("/", 1)[-1]
    raw = coding.get("code")
    vocab_id, norm = SYSTEMS.get(system, (f"MIMIC:{system}", str.strip))
    return {"vocab": vocab_id, "code": norm(raw) if raw else None, "raw_code": raw,
            "display": coding.get("display")}


def _ts(s: str | None) -> str | None:
    # MIMIC timestamps carry a constant offset; OMOP datetimes are local and naive.
    return s[:19].replace("T", " ") if s else None


def _ext(r: dict, url_suffix: str) -> dict:
    for e in r.get("extension", []):
        if e["url"].endswith(url_suffix):
            return {x["url"]: x for x in e.get("extension", [])}
    return {}


def flatten() -> dict[str, pd.DataFrame]:
    patients = []
    for r in _read("Patient"):
        race, eth = _ext(r, "us-core-race"), _ext(r, "us-core-ethnicity")
        patients.append({
            "fhir_id": r["id"], "subject_id": int(r["identifier"][0]["value"]),
            "gender": r.get("gender"), "birth_date": r.get("birthDate"),
            "race_code": race.get("ombCategory", {}).get("valueCoding", {}).get("code"),
            "race_text": race.get("text", {}).get("valueString"),
            "eth_code": eth.get("ombCategory", {}).get("valueCoding", {}).get("code"),
            "eth_text": eth.get("text", {}).get("valueString"),
            "deceased": _ts(r.get("deceasedDateTime")),
        })

    encounters = []
    for kind, name in (("hosp", "Encounter"), ("ed", "EncounterED"), ("icu", "EncounterICU")):
        for r in _read(name):
            hosp = r.get("hospitalization", {})
            encounters.append({
                "fhir_id": r["id"], "kind": kind, "patient": _ref(r, "subject"),
                "source_id": int(r["identifier"][0]["value"]),
                "start": _ts(r["period"].get("start")), "end": _ts(r["period"].get("end")),
                "class_code": r.get("class", {}).get("code"),
                "admit_source": _code(hosp.get("admitSource"))["raw_code"],
                "discharge_disp": _code(hosp.get("dischargeDisposition"))["raw_code"],
                "part_of": _ref(r, "partOf"),
            })

    conditions = [
        {"patient": _ref(r, "subject"), "encounter": _ref(r, "encounter"), **_code(r["code"])}
        for name in ("Condition", "ConditionED") for r in _read(name)
    ]

    procedures = []
    for name in ("Procedure", "ProcedureED", "ProcedureICU"):
        for r in _read(name):
            period = r.get("performedPeriod", {})
            procedures.append({
                "patient": _ref(r, "subject"), "encounter": _ref(r, "encounter"),
                "start": _ts(r.get("performedDateTime") or period.get("start")),
                "end": _ts(period.get("end")), **_code(r["code"]),
            })

    measurements = []
    for kind, name in (("lab", "ObservationLabevents"), ("ed_vital", "ObservationVitalSignsED"),
                       ("chart", "ObservationChartevents")):
        for r in _read(name):
            base = {"kind": kind, "patient": _ref(r, "subject"),
                    "encounter": _ref(r, "encounter"), "dt": _ts(r.get("effectiveDateTime"))}
            # Panels (e.g. blood pressure) carry their values in components.
            parts = r.get("component") or [r]
            for p in parts:
                q = p.get("valueQuantity", {})
                rng = (r.get("referenceRange") or [{}])[0]
                measurements.append({
                    **base, **_code(p["code"]),
                    "value_num": q.get("value"), "unit": q.get("unit"),
                    "value_str": p.get("valueString"),
                    "range_low": rng.get("low", {}).get("value"),
                    "range_high": rng.get("high", {}).get("value"),
                })

    meds = {}
    for r in _read("Medication"):
        meds[r["id"]] = _code(r.get("code"))
    for r in _read("MedicationMix"):
        meds[r["id"]] = {"vocab": "MIMIC:medication-mix", "code": r["identifier"][0]["value"],
                         "raw_code": None, "display": None}
    med_requests = []
    for r in _read("MedicationRequest"):
        if "medicationReference" in r:
            med = meds.get(_ref(r, "medicationReference"), {})
        else:
            med = _code(r.get("medicationCodeableConcept"))
        validity = r.get("dispenseRequest", {}).get("validityPeriod", {})
        dosage = (r.get("dosageInstruction") or [{}])[0]
        dose = (dosage.get("doseAndRate") or [{}])[0].get("doseQuantity", {})
        med_requests.append({
            "patient": _ref(r, "subject"), "encounter": _ref(r, "encounter"),
            "start": _ts(validity.get("start") or r.get("authoredOn")),
            "end": _ts(validity.get("end")),
            "vocab": med.get("vocab"), "code": med.get("code"),
            "route": _code(dosage.get("route"))["raw_code"],
            "dose": dose.get("value"), "dose_unit": dose.get("unit"),
        })

    return {name: pd.DataFrame(rows) for name, rows in {
        "patients": patients, "encounters": encounters, "conditions": conditions,
        "procedures": procedures, "measurements": measurements, "med_requests": med_requests,
    }.items()}


def stage(con: duckdb.DuckDBPyConnection) -> dict:
    con.execute("CREATE SCHEMA IF NOT EXISTS stage")
    frames = flatten()
    for name, df in frames.items():
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE stage.mimic_{name} AS SELECT * FROM _df")
        con.unregister("_df")
    # Resolve FHIR encounter ids to OMOP visit keys once.
    con.execute("""
        CREATE OR REPLACE TABLE stage.mimic_enc AS
        SELECT e.*, p.subject_id AS person_id,
               CASE e.kind WHEN 'icu' THEN parent.source_id ELSE e.source_id END
                   AS visit_occurrence_id,
               CASE e.kind WHEN 'icu' THEN e.source_id END AS visit_detail_id
        FROM stage.mimic_encounters e
        JOIN stage.mimic_patients p ON p.fhir_id = e.patient
        LEFT JOIN stage.mimic_encounters parent ON parent.fhir_id = e.part_of
    """)
    return {k: len(v) for k, v in frames.items()}


def load(con: duckdb.DuckDBPyConnection) -> None:
    s, R = SCHEMA, "stage.mimic"
    con.execute(f"""
        INSERT INTO {s}.cdm_source BY NAME SELECT
            'MIMIC-IV Clinical Database Demo on FHIR v2.1.0' AS cdm_source_name,
            'MIMIC-FHIR-DEMO' AS cdm_source_abbreviation,
            'PhysioNet (ODbL v1.0)' AS cdm_holder,
            '100-patient open demo of MIMIC-IV and MIMIC-IV-ED as FHIR R4' AS source_description,
            'https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/'
                AS source_documentation_reference,
            'clinical_ml.etl.fhir_mimic' AS cdm_etl_reference,
            DATE '2025-08-27' AS source_release_date, current_date AS cdm_release_date,
            'v5.4' AS cdm_version, 0 AS cdm_version_concept_id,
            coalesce((SELECT vocabulary_version FROM {config.VOCAB}.vocabulary
                      WHERE vocabulary_id = 'None'), 'none loaded') AS vocabulary_version
    """)

    race = " ".join(f"WHEN '{k}' THEN {v}" for k, v in vocab.OMB_RACE.items())
    eth = " ".join(f"WHEN '{k}' THEN {v}" for k, v in vocab.OMB_ETHNICITY.items())
    con.execute(f"""
        INSERT INTO {s}.person BY NAME
        SELECT subject_id AS person_id,
               CASE gender WHEN 'male' THEN {vocab.MALE} WHEN 'female' THEN {vocab.FEMALE}
                    ELSE 0 END AS gender_concept_id,
               year(birth_date::DATE) AS year_of_birth,
               month(birth_date::DATE) AS month_of_birth, day(birth_date::DATE) AS day_of_birth,
               CASE race_code {race} ELSE 0 END AS race_concept_id,
               CASE eth_code {eth} ELSE 0 END AS ethnicity_concept_id,
               subject_id::VARCHAR AS person_source_value, gender AS gender_source_value,
               left(race_text, 50) AS race_source_value,
               left(eth_text, 50) AS ethnicity_source_value
        FROM {R}_patients
    """)
    con.execute(f"""
        INSERT INTO {s}.death BY NAME
        SELECT subject_id AS person_id, deceased::DATE AS death_date,
               deceased::TIMESTAMP AS death_datetime, {vocab.EHR} AS death_type_concept_id
        FROM {R}_patients WHERE deceased IS NOT NULL
    """)

    con.execute(f"""
        INSERT INTO {s}.visit_occurrence BY NAME
        SELECT visit_occurrence_id, person_id,
               CASE kind WHEN 'hosp' THEN {vocab.INPATIENT_VISIT} ELSE {vocab.ER_VISIT} END
                   AS visit_concept_id,
               start::DATE AS visit_start_date, start::TIMESTAMP AS visit_start_datetime,
               "end"::DATE AS visit_end_date, "end"::TIMESTAMP AS visit_end_datetime,
               {vocab.EHR} AS visit_type_concept_id,
               kind || ':' || class_code AS visit_source_value,
               0 AS admitted_from_concept_id, admit_source AS admitted_from_source_value,
               0 AS discharged_to_concept_id, discharge_disp AS discharged_to_source_value
        FROM {R}_enc WHERE kind IN ('hosp', 'ed')
    """)
    con.execute(f"""
        INSERT INTO {s}.visit_detail BY NAME
        SELECT visit_detail_id, person_id, 0 AS visit_detail_concept_id,
               start::DATE AS visit_detail_start_date,
               start::TIMESTAMP AS visit_detail_start_datetime,
               "end"::DATE AS visit_detail_end_date, "end"::TIMESTAMP AS visit_detail_end_datetime,
               {vocab.EHR} AS visit_detail_type_concept_id, 'ICU stay' AS visit_detail_source_value,
               visit_occurrence_id
        FROM {R}_enc WHERE kind = 'icu'
    """)
    con.execute(f"""
        INSERT INTO {s}.observation_period BY NAME
        SELECT row_number() OVER (ORDER BY person_id) AS observation_period_id, person_id,
               min(start::DATE) AS observation_period_start_date,
               max("end"::DATE) AS observation_period_end_date,
               {vocab.EHR} AS period_type_concept_id
        FROM {R}_enc GROUP BY person_id
    """)

    # Conditions are dated at the start of the encounter that recorded them.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mimic_cond AS
        SELECT e.person_id, e.visit_occurrence_id, e.start::DATE AS start_date,
               e."end"::DATE AS end_date, c.vocab AS source_vocabulary_id, c.code AS source_code
        FROM {R}_conditions c JOIN {R}_enc e ON e.fhir_id = c.encounter
    """)
    m = vocab.build_code_map(con, "mimic_cond")
    con.execute(f"""
        INSERT INTO {s}.condition_occurrence BY NAME
        SELECT row_number() OVER (ORDER BY c.visit_occurrence_id, c.source_code, m.concept_id)
                   AS condition_occurrence_id,
               c.person_id, m.concept_id AS condition_concept_id,
               c.start_date AS condition_start_date, c.end_date AS condition_end_date,
               {vocab.EHR} AS condition_type_concept_id, c.visit_occurrence_id,
               c.source_code AS condition_source_value,
               m.source_concept_id AS condition_source_concept_id
        FROM mimic_cond c JOIN {m} m USING (source_vocabulary_id, source_code)
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mimic_proc AS
        SELECT e.person_id, e.visit_occurrence_id, e.visit_detail_id,
               p.start::TIMESTAMP AS start_dt, p."end"::TIMESTAMP AS end_dt,
               p.vocab AS source_vocabulary_id, p.code AS source_code
        FROM {R}_procedures p JOIN {R}_enc e ON e.fhir_id = p.encounter
    """)
    m = vocab.build_code_map(con, "mimic_proc")
    con.execute(f"""
        INSERT INTO {s}.procedure_occurrence BY NAME
        SELECT row_number() OVER (ORDER BY p.person_id, p.start_dt, p.source_code, m.concept_id)
                   AS procedure_occurrence_id,
               p.person_id, m.concept_id AS procedure_concept_id,
               p.start_dt::DATE AS procedure_date, p.start_dt AS procedure_datetime,
               p.end_dt::DATE AS procedure_end_date, p.end_dt AS procedure_end_datetime,
               {vocab.EHR} AS procedure_type_concept_id, p.visit_occurrence_id, p.visit_detail_id,
               left(p.source_code, 50) AS procedure_source_value,
               m.source_concept_id AS procedure_source_concept_id
        FROM mimic_proc p JOIN {m} m USING (source_vocabulary_id, source_code)
    """)

    # Labs carry no encounter reference: link to the admission whose period contains them.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mimic_meas AS
        SELECT x.*, p.subject_id AS person_id, x.dt::TIMESTAMP AS ts,
               coalesce(e.visit_occurrence_id, hosp.visit_occurrence_id) AS visit_occurrence_id,
               e.visit_detail_id,
               x.vocab AS source_vocabulary_id, x.code AS source_code
        FROM {R}_measurements x
        JOIN {R}_patients p ON p.fhir_id = x.patient
        LEFT JOIN {R}_enc e ON e.fhir_id = x.encounter
        LEFT JOIN {R}_enc hosp
          ON x.encounter IS NULL AND hosp.kind = 'hosp' AND hosp.patient = x.patient
         AND x.dt::TIMESTAMP BETWEEN hosp.start::TIMESTAMP AND hosp."end"::TIMESTAMP
    """)
    m = vocab.build_code_map(con, "mimic_meas")
    con.execute(f"""
        INSERT INTO {s}.measurement BY NAME
        SELECT row_number() OVER (ORDER BY x.person_id, x.ts, x.source_code, m.concept_id)
                   AS measurement_id,
               x.person_id, m.concept_id AS measurement_concept_id,
               x.ts::DATE AS measurement_date, x.ts AS measurement_datetime,
               {vocab.EHR} AS measurement_type_concept_id,
               x.value_num AS value_as_number, x.range_low, x.range_high,
               x.visit_occurrence_id, x.visit_detail_id,
               left(x.source_code, 50) AS measurement_source_value,
               m.source_concept_id AS measurement_source_concept_id,
               x.unit AS unit_source_value, left(x.value_str, 50) AS value_source_value
        FROM mimic_meas x JOIN {m} m USING (source_vocabulary_id, source_code)
        WHERE x.kind <> 'chart' OR x.value_num IS NOT NULL
    """)
    con.execute(f"""
        INSERT INTO {s}.observation BY NAME
        SELECT row_number() OVER (ORDER BY x.person_id, x.ts, x.source_code)
                   AS observation_id,
               x.person_id, 0 AS observation_concept_id, x.ts::DATE AS observation_date,
               x.ts AS observation_datetime, {vocab.EHR} AS observation_type_concept_id,
               x.value_str AS value_as_string, x.visit_occurrence_id, x.visit_detail_id,
               left(x.source_code, 50) AS observation_source_value
        FROM mimic_meas x WHERE x.kind = 'chart' AND x.value_num IS NULL
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mimic_drug AS
        SELECT e.person_id, e.visit_occurrence_id, r.start::TIMESTAMP AS start_dt,
               coalesce(r."end"::TIMESTAMP, r.start::TIMESTAMP) AS end_dt, r.route, r.dose,
               r.dose_unit, coalesce(r.vocab, 'MIMIC:unknown') AS source_vocabulary_id,
               coalesce(r.code, 'unknown') AS source_code
        FROM {R}_med_requests r JOIN {R}_enc e ON e.fhir_id = r.encounter
    """)
    m = vocab.build_code_map(con, "mimic_drug")
    con.execute(f"""
        INSERT INTO {s}.drug_exposure BY NAME
        SELECT row_number() OVER (ORDER BY d.person_id, d.start_dt, d.source_code, m.concept_id)
                   AS drug_exposure_id,
               d.person_id, m.concept_id AS drug_concept_id,
               d.start_dt::DATE AS drug_exposure_start_date,
               d.start_dt AS drug_exposure_start_datetime,
               d.end_dt::DATE AS drug_exposure_end_date, d.end_dt AS drug_exposure_end_datetime,
               {vocab.EHR} AS drug_type_concept_id, d.visit_occurrence_id,
               left(d.source_code, 50) AS drug_source_value,
               m.source_concept_id AS drug_source_concept_id,
               d.route AS route_source_value, d.dose_unit AS dose_unit_source_value
        FROM mimic_drug d JOIN {m} m USING (source_vocabulary_id, source_code)
    """)


def run(con: duckdb.DuckDBPyConnection) -> dict:
    staged = stage(con)
    load(con)
    counts = {
        t: con.execute(f"SELECT count(*) FROM {SCHEMA}.{t}").fetchone()[0]
        for t in ("person", "death", "visit_occurrence", "visit_detail", "observation_period",
                  "condition_occurrence", "procedure_occurrence", "measurement", "observation",
                  "drug_exposure")
    }
    return {"staged": staged, "cdm": counts}
