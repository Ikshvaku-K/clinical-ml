"""Command-line entry points. Each pipeline step is one subcommand, so Airflow,
the Makefile and CI all run exactly the same code."""

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def download(args) -> None:
    from clinical_ml.sources import download_all

    m = download_all()
    _print({k: {kk: vv for kk, vv in v.items() if kk != "sha256"} for k, v in m.items()})


def build_omop(args) -> None:
    from clinical_ml.omop import build, schema

    steps = {None: ["vocab", "d130", "mimic"]}.get(args.source, [args.source])
    with schema.connect() as con:
        for s in steps:
            out = build.init_vocab(con) if s == "vocab" else build.build_source(con, s)
            _print({s: out})


def dq(args) -> None:
    from clinical_ml.dq.checks import run_all

    report = run_all()
    for s in report["schemas"]:
        failed = [c["name"] for c in s["checks"] if not c["passed"]]
        print(f"{s['schema']}: {'PASS' if s['passed'] else 'FAIL'} {failed or ''}")
    for r in report["reconciliation"]:
        print(f"  {'ok ' if r['passed'] else 'BAD'} {r['check']}: {r['source']} -> {r['omop']}")
    if not report["passed"]:
        sys.exit("data-quality checks failed")


def cohort(args) -> None:
    from clinical_ml import config
    from clinical_ml.cohort import features, readmission
    from clinical_ml.omop import schema

    with schema.connect() as con:
        out = {s: readmission.build(con, s) for s in ("d130", "mimic")}
        _print(out)
        config.REPORTS.mkdir(parents=True, exist_ok=True)
        (config.REPORTS / "cohorts.json").write_text(json.dumps(out, indent=2))
        df = features.build(con)
        print(f"d130 features: {df.shape[0]} encounters x {df.shape[1] - 3} features")


def train(args) -> None:
    from clinical_ml.models.train import run

    s = run()
    _print({"split_sizes": s["split_sizes"],
            "test": {m: {k: v[k] for k in ("auroc", "auprc", "brier")}
                     for m, v in s["test"].items() if m in s["models"]}})


def evaluate(args) -> None:
    from clinical_ml.eval.m2 import run

    out = run()
    summary = {"primary_model_for_m3": out["primary_model_for_m3"], "run_id": out["run_id"]}
    for m, v in out["models"].items():
        summary[m] = {
            "recalibration": v["recalibration"]["selected"],
            "test_ece_recalibrated": v["calibration_test"]["recalibrated"]["ece"],
            "conformal_coverage": v["conformal_test"]["marginal"]["test"]["coverage"],
            "conformal_coverage_readmitted":
                v["conformal_test"]["marginal"]["test"]["coverage_readmitted"],
            "threshold": v["operating_point"]["threshold"],
            "ppv": v["operating_point"]["test"]["ppv"],
            "sensitivity": v["operating_point"]["test"]["sensitivity"],
        }
    _print(summary)


def audit(args) -> None:
    from clinical_ml.eval.m3 import run

    out = run()
    fair = out["fairness"]["attributes"]
    _print({"run_id": out["run_id"], "shap_checks": out["shap"]["checks"],
            "top_features": [g["label"] for g in out["shap"]["global"][:5]],
            "gaps": {a: {m: v["gaps"][m] for m in ("sensitivity", "observed_over_expected")}
                     for a, v in fair.items()}})


def rag(args) -> None:
    if args.step == "serve":
        from clinical_ml.rag.models import serve

        serve(args.role)
    from clinical_ml.rag import data, pipeline

    if args.step in ("generate", "judge"):
        from clinical_ml.rag import generate, judge

        if args.step == "generate":
            out = generate.run(args.split)
            _print({"split": args.split, "rows": len(out),
                    "abstained": int(out["abstain"].sum()),
                    "errors": int(out["error"].notna().sum())})
            return
        judge.run_judge(args.split)
        _print(judge.summarize(args.split))
        return
    steps = {"data": data.run, "index": pipeline.index, "retrieve": pipeline.retrieve_all}
    out = steps[args.step]()
    if args.step == "retrieve":
        out = {"selected": out["selected_for_generation"],
               "hnsw_vs_exact_overlap@20": out["hnsw_vs_exact_overlap@20"],
               "dev": {c: v["overall"] for c, v in out["splits"]["dev"]["configs"].items()}}
    _print(out)


def benchmark(args) -> None:
    from clinical_ml.models.benchmark import run, summary

    _print(summary(run(device=args.device)))


def report(args) -> None:
    from clinical_ml.report import update_readme

    update_readme()
    print("README.md results blocks (and MODEL_CARD.md, if present) updated")


def main(argv=None):
    p = argparse.ArgumentParser(prog="clinical-ml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download sources and verify checksums").set_defaults(
        fn=download)
    b = sub.add_parser("build-omop", help="(re)build OMOP schemas and run the ETLs")
    b.add_argument("--source", choices=["vocab", "d130", "mimic"])
    b.set_defaults(fn=build_omop)
    sub.add_parser("dq", help="data-quality checks; exits non-zero on failure").set_defaults(
        fn=dq)
    sub.add_parser("cohort", help="readmission cohorts + d130 features").set_defaults(fn=cohort)
    sub.add_parser("train", help="M1 baselines, MLflow-tracked").set_defaults(fn=train)
    sub.add_parser("evaluate", help="M2: calibration, conformal, threshold, decision curve"
                   ).set_defaults(fn=evaluate)
    sub.add_parser("audit", help="M3: TreeSHAP explanations and subgroup fairness audit"
                   ).set_defaults(fn=audit)
    rg = sub.add_parser("rag", help="Phase 2a clinical-note RAG (GPU pod)")
    rg.add_argument("step", choices=["serve", "data", "index", "retrieve", "generate",
                                     "judge"])
    rg.add_argument("--split", choices=["dev", "test"], default="dev")
    rg.add_argument("role", nargs="?", choices=["embed", "rerank", "generate", "judge"],
                    help="for serve: which vLLM server to start")
    rg.set_defaults(fn=rag)
    bm = sub.add_parser("benchmark", help="LR vs HistGBM vs XGBoost in one run (Linux only)")
    bm.add_argument("--device", choices=["cpu", "cuda"])
    bm.set_defaults(fn=benchmark)
    sub.add_parser("report", help="write README results from reports/*.json").set_defaults(
        fn=report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
