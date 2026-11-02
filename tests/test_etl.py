from clinical_ml import config
from clinical_ml.cohort import readmission
from clinical_ml.etl import diabetes130, fhir_mimic
from clinical_ml.omop import vocab

D, M = config.CDM_SCHEMAS["d130"], config.CDM_SCHEMAS["mimic"]


def q(con, sql):
    return con.execute(sql).fetchall()


def test_diabetes130_etl(con, d130_dir):
    counts = diabetes130.run(con)
    assert counts["person"] == 4 and counts["visit_occurrence"] == 5
    # 5 encounters x 3 diagnoses, minus one '?'
    assert counts["condition_occurrence"] == 14
    assert counts["death"] == 1
    # Hispanic goes to ethnicity, not race
    assert q(con, f"SELECT race_concept_id, ethnicity_concept_id FROM {D}.person "
                  "WHERE person_id = 2") == [(0, vocab.HISPANIC)]
    # leading zeros restored, dotted form kept
    codes = {r[0] for r in q(con, f"SELECT condition_source_value FROM {D}.condition_occurrence")}
    assert {"008", "041.9", "250.01", "V57"} <= codes
    # combination product split into ingredients, both tied to the source column
    assert q(con, f"""SELECT count(*) FROM {D}.drug_exposure
                      WHERE drug_source_value = 'glyburide-metformin'""") == [(10,)]
    # a second encounter is placed after the first, and dates never overlap
    assert q(con, f"""SELECT count(*) FROM {D}.visit_occurrence a JOIN {D}.visit_occurrence b
                      ON a.person_id = b.person_id AND a.visit_occurrence_id < b.visit_occurrence_id
                      WHERE b.visit_start_date <= a.visit_end_date""") == [(0,)]


def test_diabetes130_cohort(con, d130_dir):
    diabetes130.run(con)
    out = readmission.build(con, "d130")
    assert out["excluded_death_or_hospice"] == 2      # expired + hospice
    assert out["eligible"] == 3 and out["readmit_30d"] == 1


def test_fhir_etl(con, fhir_dir):
    out = fhir_mimic.run(con)["cdm"]
    assert out["person"] == 2 and out["death"] == 1
    assert out["visit_occurrence"] == 4 and out["visit_detail"] == 1
    # BP panel split into 2 components; chart text goes to observation
    assert out["measurement"] == 2 + 2 + 1 and out["observation"] == 1
    # inpatient lab linked by time window; outpatient lab left unlinked
    linked = q(con, f"""SELECT measurement_date::VARCHAR, visit_occurrence_id FROM {M}.measurement
                        WHERE measurement_source_value = '50912' ORDER BY 1""")
    assert linked == [("2150-01-03", 20000001), ("2150-02-15", None)]
    assert q(con, f"SELECT visit_occurrence_id FROM {M}.visit_detail") == [(20000003,)]
    assert {r[0] for r in q(con, f"SELECT condition_source_value FROM {M}.condition_occurrence")} \
        == {"428.0", "I50.33"}


def test_fhir_cohort_derives_readmission(con, fhir_dir):
    fhir_mimic.run(con)
    out = readmission.build(con, "mimic")
    # h1 -> h2 is 15 days (readmission); h2 has no follow-up; h3 died in hospital
    assert out == {"index_visits": 3, "excluded_death_or_hospice": 1,
                   "excluded_after_recorded_death": 0, "eligible": 2,
                   "eligible_persons": 1, "readmit_30d": 1, "readmit_30d_rate": 0.5}
