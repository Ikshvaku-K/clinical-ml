"""Hermetic fixtures: tiny hand-written sources (no real patient rows) and an in-memory
DuckDB with fresh OMOP schemas, so tests and CI never need the downloads."""

import csv
import gzip
import json

import duckdb
import pytest

from clinical_ml import config
from clinical_ml.etl.diabetes130 import DRUG_COLUMNS
from clinical_ml.omop import schema

D130_HEADER = (
    ["encounter_id", "patient_nbr", "race", "gender", "age", "weight", "admission_type_id",
     "discharge_disposition_id", "admission_source_id", "time_in_hospital", "payer_code",
     "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
     "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
     "number_diagnoses", "max_glu_serum", "A1Cresult"]
    + DRUG_COLUMNS + ["change", "diabetesMed", "readmitted"]
)


def _d130_row(enc, pid, race="Caucasian", gender="Female", age="[70-80)", disp=1,
              diags=("428", "250.01", "8"), insulin="Steady", a1c="None", readmitted="NO",
              payer="MC"):
    row = dict.fromkeys(D130_HEADER, "No")
    row.update({
        "encounter_id": enc, "patient_nbr": pid, "race": race, "gender": gender, "age": age,
        "weight": "?", "admission_type_id": 1, "discharge_disposition_id": disp,
        "admission_source_id": 7, "time_in_hospital": 3, "payer_code": payer,
        "medical_specialty": "?", "num_lab_procedures": 40, "num_procedures": 1,
        "num_medications": 12, "number_outpatient": 0, "number_emergency": 1,
        "number_inpatient": 2, "diag_1": diags[0], "diag_2": diags[1], "diag_3": diags[2],
        "number_diagnoses": 7, "max_glu_serum": "None", "A1Cresult": a1c,
        "insulin": insulin, "glyburide-metformin": "Up", "change": "Ch", "diabetesMed": "Yes",
        "readmitted": readmitted,
    })
    return row


@pytest.fixture
def d130_dir(tmp_path, monkeypatch):
    d = tmp_path / "d130"
    d.mkdir()
    rows = [
        _d130_row(10, 1, readmitted="<30", a1c=">8"),
        _d130_row(20, 1, readmitted="NO", diags=("V57", "41.9", "?")),
        _d130_row(30, 2, race="Hispanic", gender="Male", age="[50-60)", readmitted=">30",
                  payer="?"),
        _d130_row(40, 3, race="AfricanAmerican", disp=11),          # expired
        _d130_row(50, 4, race="?", disp=13, insulin="No"),          # hospice
    ]
    with open(d / "diabetic_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=D130_HEADER)
        w.writeheader()
        w.writerows(rows)
    (d / "IDS_mapping.csv").write_text(
        "admission_type_id,description\n1,Emergency\n,\n"
        "discharge_disposition_id,description\n1,Discharged to home\n11,Expired\n"
        "13,Hospice / home\n,\n"
        "admission_source_id,description\n7, Emergency Room\n"
    )
    monkeypatch.setattr(config, "DIABETES_DIR", d)
    return d


def _ndjson(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _enc(fid, pid, ident, start, end, cls="EMER", disp="HOME", part_of=None):
    r = {"id": fid, "resourceType": "Encounter", "class": {"code": cls},
         "subject": {"reference": f"Patient/{pid}"},
         "identifier": [{"value": str(ident)}],
         "period": {"start": start, "end": end},
         "hospitalization": {"dischargeDisposition": {"coding": [{"code": disp}]}}}
    if part_of:
        r["partOf"] = {"reference": f"Encounter/{part_of}"}
    return r


@pytest.fixture
def fhir_dir(tmp_path, monkeypatch):
    d = tmp_path / "fhir"
    d.mkdir()
    race_ext = [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                 "extension": [{"url": "ombCategory", "valueCoding": {"code": "2054-5"}},
                               {"url": "text", "valueString": "Black or African American"}]}]
    files = {
        "Patient": [
            {"id": "pa", "gender": "female", "birthDate": "2100-01-02",
             "identifier": [{"value": "10000001"}], "extension": race_ext},
            {"id": "pb", "gender": "male", "birthDate": "2090-05-06",
             "identifier": [{"value": "10000002"}],
             "deceasedDateTime": "2150-03-04T10:00:00-05:00"},
        ],
        "Encounter": [
            _enc("h1", "pa", 20000001, "2150-01-01T08:00:00-05:00", "2150-01-05T12:00:00-05:00"),
            _enc("h2", "pa", 20000002, "2150-01-20T08:00:00-05:00", "2150-01-22T12:00:00-05:00"),
            _enc("h3", "pb", 20000003, "2150-03-01T08:00:00-05:00", "2150-03-04T10:00:00-05:00",
                 disp="DIED"),
        ],
        "EncounterED": [_enc("e1", "pa", 30000001, "2150-01-01T06:00:00-05:00",
                             "2150-01-01T08:00:00-05:00", disp="ADMITTED", part_of="h1")],
        "EncounterICU": [_enc("i1", "pb", 30000002, "2150-03-02T00:00:00-05:00",
                              "2150-03-03T00:00:00-05:00", cls="ACUTE", part_of="h3")],
        "Condition": [
            {"id": "c1", "subject": {"reference": "Patient/pa"},
             "encounter": {"reference": "Encounter/h1"},
             "code": {"coding": [{"system": "http://x/CodeSystem/mimic-diagnosis-icd9",
                                  "code": "4280"}]}},
            {"id": "c2", "subject": {"reference": "Patient/pb"},
             "encounter": {"reference": "Encounter/h3"},
             "code": {"coding": [{"system": "http://x/CodeSystem/mimic-diagnosis-icd10",
                                  "code": "I5033"}]}},
        ],
        "ConditionED": [],
        "Procedure": [{"id": "p1", "subject": {"reference": "Patient/pa"},
                       "encounter": {"reference": "Encounter/h1"},
                       "performedDateTime": "2150-01-02T00:00:00-05:00",
                       "code": {"coding": [{"system": "http://x/CodeSystem/mimic-procedure-icd9",
                                            "code": "5491"}]}}],
        "ProcedureED": [],
        "ProcedureICU": [],
        "ObservationLabevents": [
            {"id": "l1", "subject": {"reference": "Patient/pa"},
             "effectiveDateTime": "2150-01-03T09:00:00-05:00",
             "code": {"coding": [{"system": "http://x/CodeSystem/mimic-d-labitems",
                                  "code": "50912"}]},
             "valueQuantity": {"value": 1.4, "unit": "mg/dL"},
             "referenceRange": [{"low": {"value": 0.5}, "high": {"value": 1.2}}]},
            {"id": "l2", "subject": {"reference": "Patient/pa"},
             "effectiveDateTime": "2150-02-15T09:00:00-05:00",   # outpatient: no admission
             "code": {"coding": [{"system": "http://x/CodeSystem/mimic-d-labitems",
                                  "code": "50912"}]},
             "valueQuantity": {"value": 1.1, "unit": "mg/dL"}},
        ],
        "ObservationVitalSignsED": [
            {"id": "v1", "subject": {"reference": "Patient/pa"},
             "encounter": {"reference": "Encounter/e1"},
             "effectiveDateTime": "2150-01-01T06:30:00-05:00",
             "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
             "component": [
                 {"code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]},
                  "valueQuantity": {"value": 120, "unit": "mm[Hg]"}},
                 {"code": {"coding": [{"system": "http://loinc.org", "code": "8462-4"}]},
                  "valueQuantity": {"value": 80, "unit": "mm[Hg]"}}]},
        ],
        "ObservationChartevents": [
            {"id": "ch1", "subject": {"reference": "Patient/pb"},
             "encounter": {"reference": "Encounter/i1"},
             "effectiveDateTime": "2150-03-02T01:00:00-05:00",
             "code": {"coding": [{"system": "http://x/CodeSystem/mimic-chartevents-d-items",
                                  "code": "220045"}]},
             "valueQuantity": {"value": 88, "unit": "bpm"}},
            {"id": "ch2", "subject": {"reference": "Patient/pb"},
             "encounter": {"reference": "Encounter/i1"},
             "effectiveDateTime": "2150-03-02T01:00:00-05:00",
             "code": {"coding": [{"system": "http://x/CodeSystem/mimic-chartevents-d-items",
                                  "code": "220048"}]},
             "valueString": "SR (Sinus Rhythm)"},
        ],
        "Medication": [{"id": "m1", "code": {"coding": [
            {"system": "http://x/CodeSystem/mimic-medication-ndc", "code": "51079030020"}]}}],
        "MedicationMix": [],
        "MedicationRequest": [
            {"id": "r1", "subject": {"reference": "Patient/pa"},
             "encounter": {"reference": "Encounter/h1"},
             "authoredOn": "2150-01-02T08:00:00-05:00",
             "medicationReference": {"reference": "Medication/m1"}},
            {"id": "r2", "subject": {"reference": "Patient/pb"},
             "encounter": {"reference": "Encounter/h3"},
             "authoredOn": "2150-03-02T08:00:00-05:00",
             "medicationCodeableConcept": {"coding": [
                 {"system": "http://x/CodeSystem/mimic-medication-formulary-drug-cd",
                  "code": "HEPA5I"}]}},
        ],
    }
    for name, rows in files.items():
        _ndjson(d / f"Mimic{name}.ndjson.gz", rows)
    monkeypatch.setattr(config, "FHIR_DIR", d)
    return d


@pytest.fixture
def con():
    c = duckdb.connect()
    schema.create_schema(c, config.VOCAB, vocab=True)
    c.execute(f"CREATE SCHEMA {config.RESULTS}")
    for s in config.CDM_SCHEMAS.values():
        schema.create_schema(c, s, vocab=False)
    yield c
    c.close()
