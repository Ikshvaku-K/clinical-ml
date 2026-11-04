# Everything this project installs or writes stays inside this folder:
#   .tools/python      uv-managed CPython (not the system or conda Python)
#   .tools/uv-cache    package cache
#   .venv              project environment (ETL, models)
#   .venv-airflow      Airflow orchestrator only (kept apart: its pins clash with MLflow)
#   .airflow           AIRFLOW_HOME (sqlite metadata DB, logs)
#   data/, mlruns/, mlflow.db, reports/
# Delete the folder and nothing is left behind.

export UV_CACHE_DIR          := $(CURDIR)/.tools/uv-cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export UV_PYTHON_PREFERENCE  := only-managed
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv
export AIRFLOW_HOME          := $(CURDIR)/.airflow
export AIRFLOW__CORE__LOAD_EXAMPLES := False
export AIRFLOW__CORE__DAGS_FOLDER   := $(CURDIR)/dags
export CLINICAL_ML_ROOT      := $(CURDIR)
export MLFLOW_TRACKING_URI   := sqlite:///$(CURDIR)/mlflow.db
export MLFLOW_DISABLE_AGENT_HINT := 1

PYVER    := 3.12
AIRFLOW  := 3.3.2
# -I: isolated mode, so downloaded data directories can never shadow imports
PY       := $(CURDIR)/.venv/bin/python -I
CLI      := $(PY) -m clinical_ml.cli

.PHONY: setup setup-airflow setup-benchmark setup-rag data omop dq cohort train evaluate audit benchmark report all dag-test test lint clean

setup:
	uv python install $(PYVER)
	uv sync --extra dev --python $(PYVER)

setup-airflow:
	uv venv --python $(PYVER) .venv-airflow
	uv pip install --python .venv-airflow/bin/python "apache-airflow==$(AIRFLOW)" \
	  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW)/constraints-$(PYVER).txt"
	.venv-airflow/bin/airflow db migrate

# Phase 2a client side (BM25, embedded Postgres + pgvector, LangChain). The vLLM servers
# run on the GPU pod in their own env: see src/clinical_ml/rag/models.py.
setup-rag: setup
	uv sync --extra dev --extra rag --python $(PYVER)

# Linux only (CI, Docker or the GPU node): XGBoost's macOS wheel needs Homebrew's libomp
setup-benchmark: setup
	uv sync --extra dev --extra benchmark --python $(PYVER)

data:
	$(CLI) download

omop:
	$(CLI) build-omop

dq:
	$(CLI) dq

cohort:
	$(CLI) cohort

train:
	$(CLI) train

evaluate:
	$(CLI) evaluate

audit:
	$(CLI) audit

# Linux only: `make setup-benchmark` first; DEVICE=cuda on the GPU node
benchmark:
	$(CLI) benchmark $(if $(DEVICE),--device $(DEVICE))

report:
	$(CLI) report

all: data omop dq cohort train evaluate audit report

dag-test:
	.venv-airflow/bin/airflow dags test omop_readmission

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests dags

clean:
	rm -rf data/warehouse reports/*.json reports/figures
