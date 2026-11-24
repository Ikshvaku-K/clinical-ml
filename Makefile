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

.PHONY: setup setup-airflow setup-benchmark setup-rag setup-gpu setup-nori nori-export nori-predict nori-choose nori-evaluate xgboost-predictions nori-compare-xgboost curve-nori curve-trees curve-evaluate nori-full-predict nori-full-choose nori-full-evaluate setup-tabicl variant-predict variants-evaluate candidate-predict candidate-choose candidate-evaluate data omop dq cohort train evaluate audit benchmark report all dag-test test lint clean

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

# GPU host only: vLLM servers plus LoRA/DPO training, in one env apart from .venv (vLLM
# pins its own torch). The project is installed without deps for its config and data code.
GPU_PKGS := "vllm==0.32.0" "trl==1.17.1" "peft==0.21.2" "datasets==5.1.0" \
            "accelerate==1.15.0" "rouge-score==0.1.2"
setup-gpu:
	uv venv --python $(PYVER) --allow-existing .venv-vllm
	uv pip install --python .venv-vllm/bin/python $(GPU_PKGS)
	uv pip install --python .venv-vllm/bin/python --no-deps -e .

# Linux only (CI, Docker or the GPU node): XGBoost's macOS wheel needs Homebrew's libomp
setup-benchmark: setup
	uv sync --extra dev --extra benchmark --python $(PYVER)

# M2c: Nori (tabular foundation model) in its own env: it pins torch and huggingface-hub < 2,
# which clash with .venv-vllm. Weights go to .tools/hf at the revision pinned in
# src/clinical_ml/models/nori.py. Runs on CUDA, Apple MPS or CPU.
NORI_PKGS := "synthefy-nori==0.23.0"
NORI      := $(CURDIR)/.venv-nori/bin/python -I -m clinical_ml.models.nori
setup-nori:
	uv venv --python $(PYVER) --allow-existing .venv-nori
	uv pip install --python .venv-nori/bin/python $(NORI_PKGS)
	uv pip install --python .venv-nori/bin/python --no-deps -e .

# M2e: TabICL v2 in its own .venv-tabicl (torch pinned to .venv-nori's, so it is not
# downloaded twice); its checkpoint is fetched by pinned revision into .tools/hf.
TABICL_PKGS := "tabicl==2.2.0" "torch==2.13.0" pandas pyarrow
setup-tabicl:
	uv venv --python $(PYVER) --allow-existing .venv-tabicl
	uv pip install --python .venv-tabicl/bin/python $(TABICL_PKGS)
	uv pip install --python .venv-tabicl/bin/python --no-deps -e .

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

# Linux: M2b's XGBoost refit from data/nori/features.parquet, keeping its predictions (M2c)
xgboost-predictions:
	$(CLI) benchmark --rerun-xgboost $(if $(DEVICE),--device $(DEVICE))

# M2c, in order: export, predict STEP=cv, predict STEP=calibration, choose, predict
# STEP=test (refused before choose, and refused a second time), evaluate. STEP=probe
# measures speed and memory on train rows only (CONTEXT= / QUERY= subsample them).
nori-export:
	$(CLI) nori export

nori-predict:
	$(NORI) predict $(STEP) $(if $(CONTEXT),--context $(CONTEXT)) $(if $(QUERY),--query $(QUERY))

nori-choose:
	$(CLI) nori choose

nori-evaluate:
	$(CLI) nori evaluate

nori-compare-xgboost:
	$(CLI) nori compare-xgboost

# Learning curve on train (exploratory): curve-nori in .venv-nori (needs nori-export and
# the M2c CV run), curve-trees (MODELS= a subset of logistic_regression
# hist_gradient_boosting xgboost; XGBoost needs Linux, DEVICE=cuda), then curve-evaluate.
curve-nori:
	$(CURDIR)/.venv-nori/bin/python -I -m clinical_ml.models.curve nori

curve-trees:
	$(CLI) curve trees $(if $(MODELS),--models $(MODELS)) $(if $(DEVICE),--device $(DEVICE))

curve-evaluate:
	$(CLI) curve evaluate

# M2d, in order: nori-full-predict STEP=cv, nori-full-choose (the CV gate); only if it
# passed: STEP=calibration, nori-full-choose (recalibration), STEP=test (once),
# nori-full-evaluate. Needs the M2c scores and the learning curve.
nori-full-predict:
	$(CURDIR)/.venv-nori/bin/python -I -m clinical_ml.models.nori_full predict $(STEP)

nori-full-choose:
	$(CLI) nori-full choose

nori-full-evaluate:
	$(CLI) nori-full evaluate

# M2e (train CV only): VARIANT= tabicl | cluster_route | onehot | nori-30m | nori-100m, on
# the GPU (TabICL in .venv-tabicl, the rest in .venv-nori), then variants-evaluate.
variant-predict:
	$(CURDIR)/$(if $(filter tabicl,$(VARIANT)),.venv-tabicl,.venv-nori)/bin/python -I -m clinical_ml.models.tfm_variants predict $(VARIANT)

variants-evaluate:
	$(CLI) variants evaluate

# The M2e test candidate, only if the test rule picked one: candidate-predict
# STEP=calibration, candidate-choose, candidate-predict STEP=test (once), candidate-evaluate.
CANDIDATE_VENV = $(shell $(PY) -c "import json; c = json.load(open('reports/tfm_variants.json'))['test_rule']['test_candidate'] or ''; print('.venv-tabicl' if c.startswith('tabicl') else '.venv-nori')" 2>/dev/null)
candidate-predict:
	$(CURDIR)/$(CANDIDATE_VENV)/bin/python -I -m clinical_ml.models.tfm_variants candidate $(STEP)

candidate-choose:
	$(CLI) variants choose-candidate

candidate-evaluate:
	$(CLI) variants evaluate-candidate

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
