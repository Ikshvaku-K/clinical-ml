"""Airflow DAG: public sources -> OMOP CDM (DuckDB) -> DQ gates -> cohort -> baselines
-> calibration, conformal sets and the capacity threshold.

Airflow only orchestrates. It runs in its own environment (.venv-airflow) because its
dependency pins clash with MLflow's; every task shells out to the project's CLI in
.venv, so the logic stays testable without Airflow.

Tasks are sequential on purpose: DuckDB allows one writer process at a time.
Run locally with:  make dag-test
"""

from datetime import datetime
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

ROOT = Path(__file__).resolve().parents[1]
CLI = f"{ROOT}/.venv/bin/python -I -m clinical_ml.cli"
ENV = {
    "CLINICAL_ML_ROOT": str(ROOT),
    "MLFLOW_TRACKING_URI": f"sqlite:///{ROOT}/mlflow.db",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}

with DAG(
    dag_id="omop_readmission",
    description="FHIR + EHR extract -> OMOP CDM v5.4 -> 30-day readmission baselines",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["omop", "fhir", "readmission"],
) as dag:

    def step(task_id: str, args: str, **kw) -> BashOperator:
        return BashOperator(task_id=task_id, bash_command=f"{CLI} {args}", env=ENV,
                            append_env=True, cwd=str(ROOT), **kw)

    download = step("download_and_verify_sources", "download")
    vocab = step("load_vocabulary", "build-omop --source vocab")
    d130 = step("etl_diabetes130_to_omop", "build-omop --source d130")
    mimic = step("etl_fhir_to_omop", "build-omop --source mimic")
    dq = step("data_quality_gates", "dq")
    cohort = step("build_cohort_and_features", "cohort")
    train = step("train_baselines", "train")
    evaluate = step("calibrate_conformal_threshold", "evaluate")
    report = step("render_readme_results", "report")

    download >> vocab >> d130 >> mimic >> dq >> cohort >> train >> evaluate >> report
