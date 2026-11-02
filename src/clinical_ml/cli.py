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


def report(args) -> None:
    from clinical_ml.report import update_readme

    update_readme()
    print("README.md results block updated")


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
    sub.add_parser("report", help="write README results from reports/*.json").set_defaults(
        fn=report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
