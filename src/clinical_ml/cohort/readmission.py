"""30-day readmission cohort, written once over OMOP and run against each CDM schema.

Index event: an inpatient visit. Excluded: discharges where the patient died or went to
hospice (they cannot be readmitted). Outcome: an inpatient readmission within 30 days.

How the outcome is obtained differs by source, and the code says so:
* MIMIC has real (shifted) dates, so readmission is derived from consecutive visits.
* Diabetes-130 has no dates; its outcome comes from the source label
  (results.d130_label), not from the placeholder dates.
"""

import duckdb

from clinical_ml import config
from clinical_ml.omop import vocab

INDEX_SQL = """
CREATE OR REPLACE TABLE {results}.{src}_cohort AS
WITH ip AS (
    SELECT v.*, lead(visit_start_date) OVER (PARTITION BY person_id
                                           ORDER BY visit_start_date, visit_occurrence_id)
               AS next_ip_start
    FROM {cdm}.visit_occurrence v
    WHERE visit_concept_id = {inpatient}
)
SELECT ip.visit_occurrence_id, ip.person_id, ip.visit_start_date, ip.visit_end_date,
       ip.discharged_to_source_value,
       coalesce(d.death_date BETWEEN ip.visit_start_date AND ip.visit_end_date, false)
         OR coalesce(regexp_matches(lower(ip.discharged_to_source_value),
                                    'expired|died|hospice'), false)
           AS excluded_death_or_hospice,
       coalesce(ip.visit_start_date > d.death_date, false) AS excluded_after_recorded_death,
       coalesce(ip.next_ip_start - ip.visit_end_date BETWEEN 0 AND 30, false)
           AS readmit_30d_derived
FROM ip
LEFT JOIN {cdm}.death d USING (person_id)
"""


def build(con: duckdb.DuckDBPyConnection, src: str) -> dict:
    cdm = config.CDM_SCHEMAS[src]
    con.execute(INDEX_SQL.format(results=config.RESULTS, src=src, cdm=cdm,
                                 inpatient=vocab.INPATIENT_VISIT))
    t = f"{config.RESULTS}.{src}_cohort"
    if src == "d130":
        con.execute(f"""
            CREATE OR REPLACE TABLE {t} AS
            SELECT c.*, l.readmit_30d FROM {t} c JOIN {config.RESULTS}.d130_label l
            USING (visit_occurrence_id)
        """)
    else:
        con.execute(f"ALTER TABLE {t} ADD COLUMN readmit_30d BOOLEAN")
        con.execute(f"UPDATE {t} SET readmit_30d = readmit_30d_derived")
    con.execute(f"""
        ALTER TABLE {t} ADD COLUMN eligible BOOLEAN;
        UPDATE {t} SET eligible = NOT excluded_death_or_hospice
                                  AND NOT excluded_after_recorded_death;
    """)
    total, died, after_death, eligible, positives, persons = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE excluded_death_or_hospice),
               count(*) FILTER (WHERE excluded_after_recorded_death
                                AND NOT excluded_death_or_hospice),
               count(*) FILTER (WHERE eligible),
               count(*) FILTER (WHERE eligible AND readmit_30d),
               count(DISTINCT person_id) FILTER (WHERE eligible)
        FROM {t}
    """).fetchone()
    assert total == died + after_death + eligible
    return {"index_visits": total, "excluded_death_or_hospice": died,
            "excluded_after_recorded_death": after_death, "eligible": eligible,
            "eligible_persons": persons, "readmit_30d": positives,
            "readmit_30d_rate": round(positives / eligible, 4) if eligible else None}
