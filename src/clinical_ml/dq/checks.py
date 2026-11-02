"""Data-quality checks on each OMOP CDM schema, in the spirit of the OHDSI Data Quality
Dashboard: conformance (keys resolve), plausibility (dates make sense) and completeness
(vocabulary coverage). NOT NULL and primary keys are already enforced by the DDL.

`error` checks must have zero failing rows or the pipeline stops; `info` checks are
reported only (e.g. concept coverage is legitimately 0% until Athena is loaded).
"""

import json
from dataclasses import asdict, dataclass

import duckdb

from clinical_ml import config

CLINICAL = {
    "condition_occurrence": ("condition_concept_id", "condition_source_value",
                             "condition_start_date"),
    "drug_exposure": ("drug_concept_id", "drug_source_value", "drug_exposure_start_date"),
    "procedure_occurrence": ("procedure_concept_id", "procedure_source_value", "procedure_date"),
    "measurement": ("measurement_concept_id", "measurement_source_value", "measurement_date"),
    "observation": ("observation_concept_id", "observation_source_value", "observation_date"),
}


@dataclass
class Check:
    name: str
    severity: str
    failing: int
    total: int

    @property
    def passed(self) -> bool:
        return self.severity != "error" or self.failing == 0


def _count(con, sql: str) -> int:
    return con.execute(sql).fetchone()[0]


def run_schema(con: duckdb.DuckDBPyConnection, s: str) -> dict:
    checks: list[Check] = []
    coverage = {}

    def add(name, failing_sql, total_sql, severity="error"):
        checks.append(Check(name, severity, _count(con, failing_sql), _count(con, total_sql)))

    tables = [t for t in CLINICAL if _count(con, f"SELECT count(*) FROM {s}.{t}") > 0]
    for t in tables + ["visit_occurrence", "observation_period", "death", "visit_detail"]:
        add(f"{t}.person_id exists in person",
            f"SELECT count(*) FROM {s}.{t} x ANTI JOIN {s}.person p USING (person_id)",
            f"SELECT count(*) FROM {s}.{t}")
    for t in tables + ["visit_detail"]:
        add(f"{t}.visit_occurrence_id exists in visit_occurrence",
            f"""SELECT count(*) FROM {s}.{t} x WHERE x.visit_occurrence_id IS NOT NULL
                AND x.visit_occurrence_id NOT IN (SELECT visit_occurrence_id
                                                  FROM {s}.visit_occurrence)""",
            f"SELECT count(*) FROM {s}.{t}")
    add("visit end >= visit start",
        f"SELECT count(*) FROM {s}.visit_occurrence WHERE visit_end_date < visit_start_date",
        f"SELECT count(*) FROM {s}.visit_occurrence")
    add("drug end >= drug start",
        f"""SELECT count(*) FROM {s}.drug_exposure
            WHERE drug_exposure_end_date < drug_exposure_start_date""",
        f"SELECT count(*) FROM {s}.drug_exposure")
    add("visits inside the person's observation period",
        f"""SELECT count(*) FROM {s}.visit_occurrence v JOIN {s}.observation_period o
            USING (person_id) WHERE v.visit_start_date < o.observation_period_start_date
            OR v.visit_end_date > o.observation_period_end_date""",
        f"SELECT count(*) FROM {s}.visit_occurrence")
    add("one observation period per person",
        f"""SELECT count(*) FROM (SELECT person_id FROM {s}.observation_period
            GROUP BY 1 HAVING count(*) > 1)""",
        f"SELECT count(*) FROM {s}.person")
    add("year_of_birth plausible (1850 to visit year)",
        f"""SELECT count(*) FROM {s}.person p WHERE year_of_birth < 1850 OR year_of_birth >
            (SELECT max(year(visit_end_date)) FROM {s}.visit_occurrence v
             WHERE v.person_id = p.person_id)""",
        f"SELECT count(*) FROM {s}.person")
    add("no clinical visits start after death",
        f"""SELECT count(*) FROM {s}.visit_occurrence v JOIN {s}.death d USING (person_id)
            WHERE v.visit_start_date > d.death_date""",
        f"SELECT count(*) FROM {s}.visit_occurrence", severity="info")

    for t in tables:
        concept, source, _ = CLINICAL[t]
        rows, mapped = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {concept} <> 0) FROM {s}.{t}"
        ).fetchone()
        codes, mapped_codes = con.execute(
            f"""SELECT count(DISTINCT {source}),
                       count(DISTINCT {source}) FILTER (WHERE {concept} <> 0)
                FROM {s}.{t}"""
        ).fetchone()
        coverage[t] = {"rows": rows, "rows_mapped_pct": round(100 * mapped / rows, 2),
                       "distinct_source_codes": codes,
                       "codes_mapped_pct": round(100 * mapped_codes / codes, 2) if codes else None}
        checks.append(Check(f"{t} standard-concept coverage", "info", rows - mapped, rows))

    return {"schema": s, "checks": [asdict(c) | {"passed": c.passed} for c in checks],
            "concept_coverage": coverage,
            "passed": all(c.passed for c in checks)}


def source_reconciliation(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Source rows in vs OMOP rows out, with the expected difference explained."""
    s = config.CDM_SCHEMAS["d130"]
    q = lambda sql: _count(con, sql)  # noqa: E731
    out = [
        ("d130 encounters -> visit_occurrence", q("SELECT count(*) FROM stage.d130_raw"),
         q(f"SELECT count(*) FROM {s}.visit_occurrence")),
        ("d130 distinct patients -> person",
         q("SELECT count(DISTINCT patient_nbr) FROM stage.d130_raw"),
         q(f"SELECT count(*) FROM {s}.person")),
        ("d130 non-missing diag_1..3 -> condition_occurrence",
         q("""SELECT count(*) FILTER (WHERE diag_1 <> '?') + count(*) FILTER (WHERE diag_2 <> '?')
                   + count(*) FILTER (WHERE diag_3 <> '?') FROM stage.d130_raw"""),
         q(f"SELECT count(*) FROM {s}.condition_occurrence")),
        ("d130 LOS within source inclusion (1-14 days)",
         q("SELECT count(*) FROM stage.d130_raw"),
         q(f"""SELECT count(*) FROM {s}.visit_occurrence
               WHERE visit_end_date - visit_start_date BETWEEN 1 AND 14""")),
    ]
    m = config.CDM_SCHEMAS["mimic"]
    out += [
        ("mimic Patient -> person", q("SELECT count(*) FROM stage.mimic_patients"),
         q(f"SELECT count(*) FROM {m}.person")),
        ("mimic hospital+ED Encounter -> visit_occurrence",
         q("SELECT count(*) FROM stage.mimic_encounters WHERE kind IN ('hosp', 'ed')"),
         q(f"SELECT count(*) FROM {m}.visit_occurrence")),
        ("mimic Condition -> condition_occurrence",
         q("SELECT count(*) FROM stage.mimic_conditions"),
         q(f"SELECT count(*) FROM {m}.condition_occurrence")),
        ("mimic Observation -> measurement + observation",
         q("SELECT count(*) FROM stage.mimic_measurements"),
         q(f"""SELECT (SELECT count(*) FROM {m}.measurement)
                    + (SELECT count(*) FROM {m}.observation)""")),
    ]
    rows = [{"check": n, "source": a, "omop": b, "passed": a == b} for n, a, b in out]
    rows.append({
        "check": "mimic labs linked to a visit (by time window)",
        "source": q(f"SELECT count(*) FROM {m}.measurement WHERE measurement_source_value IN "
                    "(SELECT code FROM stage.mimic_measurements WHERE kind = 'lab')"),
        "omop": q(f"""SELECT count(*) FROM {m}.measurement WHERE visit_occurrence_id IS NOT NULL
                      AND measurement_source_value IN (SELECT code FROM stage.mimic_measurements
                                                       WHERE kind = 'lab')"""),
        "passed": True,  # informational: outpatient labs legitimately have no admission
    })
    return rows


def run_all() -> dict:
    from clinical_ml.omop import schema

    with schema.connect() as con:
        report = {"schemas": [run_schema(con, s) for s in config.CDM_SCHEMAS.values()],
                  "reconciliation": source_reconciliation(con)}
    report["passed"] = all(r["passed"] for r in report["schemas"]) and all(
        r["passed"] for r in report["reconciliation"])
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "dq.json").write_text(json.dumps(report, indent=2))
    return report
