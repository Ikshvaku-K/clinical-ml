"""M2e (exploratory, train CV only; added after M2d): other ways to run a tabular foundation
model on this data, each scored on the M2c folds and compared, on the same patient
resamples, with the learning curve.

Fixed before any variant was scored:
* rows: the learning curve's nested seeded order in each fold, so a size of n means the
  same n rows that the trained models and Nori 1.5 got; the query is always the whole
  held-out fold;
* sizes: each variant is tried at 1,000, 2,500, 5,000 and 10,000 rows (TabICL also at 20,000
  and the whole pool), smallest first, and stops at the first size that does not fit the
  8 GB GPU in every fold at the M2c placement setting (out of memory, or more rows than its
  element budget allows);
* variants:
  - nori-30m, nori-100m: the larger Nori checkpoints (pinned revisions) with their default
    inference config; score P(y = 1) as in M2c;
  - onehot: Nori 1.5 with one-hot instead of ordinal categoricals, otherwise as M2c;
  - cluster_route: Nori 1.5 over the whole pool with Nori's own large-table policy, whose
    window follows from the same budget. The policy returns point predictions only, so
    the score is the predictive mean (tied with P(y = 1) in M2c's CV);
  - tabicl: the TabICL v2 classifier (pinned checkpoint) with its defaults; score
    P(y = 1);
* test: at most one variant may reach the test split: the one with the highest CV AUROC,
  and only if it beats XGBoost on every row (the learning curve's `xgboost@all`). That
  would follow M2d's steps; nothing here reads test.

  predict   .venv-nori (Nori variants) or .venv-tabicl (TabICL) -> data/variants/<name>.parquet
  evaluate  .venv -> reports/tfm_variants.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from clinical_ml import config
from clinical_ml.models import curve, nori

OUT = config.DATA / "variants"
REPORT = config.REPORTS / "tfm_variants.json"
TABICL = {"package": "2.2.0", "repo": "jingang/TabICL",
          "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
          "file": "tabicl-classifier-v2-20260212.ckpt"}
VARIANTS = {
    "tabicl": {"family": "tabicl", "sizes": curve.TREE_SIZES,
               "label": "TabICL v2 classifier"},
    "cluster_route": {"family": "nori", "model": nori.MODEL, "revision": nori.REVISION,
                      "policy": "cluster_route", "sizes": (None,),
                      "label": "Nori 1.5, its cluster_route policy over every row"},
    "onehot": {"family": "nori", "model": nori.MODEL, "revision": nori.REVISION,
               "encoding": "onehot", "sizes": curve.NORI_SIZES,
               "label": "Nori 1.5, one-hot categoricals"},
    "nori-30m": {"family": "nori", "model": "nori-30m",
                 "revision": "63c9f7facf9fb32c37ce3fc2fba331d524696318",
                 "sizes": curve.NORI_SIZES, "label": "Nori-30M"},
    "nori-100m": {"family": "nori", "model": "nori-100m",
                  "revision": "467ae9704fb0ab328c0d2f850602c27123928b48",
                  "sizes": curve.NORI_SIZES, "label": "Nori-100M"},
}


class NotFit(Exception):
    """This size does not fit the GPU at the protocol's settings."""


# ---- predict (.venv-nori / .venv-tabicl) ------------------------------------------------

def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def nori_scorer(spec: dict, ctx: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, dict]:
    import torch
    from synthefy_nori import ContextTooLargeError, NoriRegressor
    from synthefy_nori.hf import download_checkpoint

    X_ctx, X_q = ctx.drop(columns=nori.META), query.drop(columns=nori.META)
    kw = {"model": spec["model"], "categorical_columns": nori.categoricals(X_ctx),
          "categorical_encoding": spec.get("encoding", "ordinal"),
          "memory_policy": nori.memory_policy(),
          "model_path": download_checkpoint(model=spec["model"], revision=spec["revision"])}
    if spec.get("policy"):
        kw |= {"large_context_policy": spec["policy"],
               "large_context_threshold": nori.CONTEXT_ROWS}
    t0 = time.time()
    try:
        reg = NoriRegressor(**kw)
        reg.fit(X_ctx, ctx["readmit_30d"].to_numpy(dtype=float))
        if spec.get("policy"):
            p, score = np.asarray(reg.predict(X_q), dtype=float), "mean"
        else:
            dist = reg.predict(X_q, output_type="full")
            p, score = nori.p_one(dist["quantiles"], dist["taus"]), "p1"
    except (torch.OutOfMemoryError, ContextTooLargeError) as e:
        raise NotFit(f"{type(e).__name__}: {str(e)[:300]}") from e
    info = {"seconds": round(time.time() - t0, 1), "score": score,
            "device": str(getattr(reg, "device_", "unknown")),
            "memory_report": json.loads(json.dumps(getattr(reg, "memory_report_", None),
                                                   default=str)),
            "large_context_report": json.loads(json.dumps(
                getattr(reg, "large_context_report_", None), default=str))}
    return p, info


def tabicl_scorer(spec: dict, ctx: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, dict]:
    import torch
    from huggingface_hub import hf_hub_download
    from tabicl import TabICLClassifier

    path = hf_hub_download(TABICL["repo"], TABICL["file"], revision=TABICL["revision"])
    X_ctx, X_q = ctx.drop(columns=nori.META), query.drop(columns=nori.META)
    t0 = time.time()
    try:
        clf = TabICLClassifier(model_path=path, allow_auto_download=False,
                               checkpoint_version=TABICL["file"], random_state=config.SEED,
                               device="cuda" if torch.cuda.is_available() else None)
        clf.fit(X_ctx, ctx["readmit_30d"].to_numpy(dtype=int))
        p = clf.predict_proba(X_q)[:, list(clf.classes_).index(1)]
    except torch.OutOfMemoryError as e:
        raise NotFit(f"OutOfMemoryError: {str(e)[:300]}") from e
    return np.asarray(p, dtype=float), {"seconds": round(time.time() - t0, 1), "score": "p1",
                                        "device": str(getattr(clf, "device_", "unknown"))}


def predict(name: str, scorer=None) -> dict:
    """Score one variant on every CV fold, size by size, smallest first; stop at the first
    size that does not fit in every fold. Finished sizes are kept, so a rerun resumes."""
    spec = VARIANTS[name]
    scorer = scorer or {"nori": nori_scorer, "tabicl": tabicl_scorer}[spec["family"]]
    parts = OUT / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tr = curve.train_rows()
    folds = list(curve.folds(tr))
    calls, done = [], []
    for n in spec["sizes"]:
        size = curve.size_label(n)
        path = parts / f"{name}@{size}.parquet"
        if path.exists():
            done.append(size)
            calls += json.loads(path.with_suffix(".json").read_text())
            continue
        attempt = path.with_suffix(".attempt")
        if attempt.exists():          # an earlier run died here (memory guard or crash)
            calls.append({"size": size, "status": "does not fit",
                          "error": "Killed: the run stopped at this size on an earlier "
                                   "attempt (" + attempt.read_text().strip() + ")"})
            break
        attempt.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        rows, size_calls = [], []
        try:
            for k, order, query in folds:
                ctx = curve.rows(order, n)
                p, info = scorer(spec, ctx, query)
                rows.append(pd.DataFrame({"visit_occurrence_id": query["visit_occurrence_id"]
                                          .to_numpy(), "p": p}).assign(fold=k, size=size))
                size_calls.append({"fold": k, "size": size, "rows": len(ctx),
                                   "rows_sha256": curve.ids_sha256(ctx), **info})
                print(json.dumps({c: v for c, v in size_calls[-1].items()
                                  if not c.endswith("_report")}, default=str), flush=True)
                _free_gpu()
        except NotFit as e:
            calls.append({"size": size, "status": "does not fit", "error": str(e)})
            print(json.dumps(calls[-1]), flush=True)
            attempt.unlink()
            _free_gpu()
            break
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps(size_calls, default=str))
        attempt.unlink()
        calls += size_calls
        done.append(size)
    scores = [pd.read_parquet(parts / f"{name}@{s}.parquet") for s in done]
    log = {"variant": name, "spec": {k: v for k, v in spec.items() if k != "sizes"},
           "sizes_tried": [curve.size_label(n) for n in spec["sizes"]], "sizes_done": done,
           "versions": _versions(spec["family"]), "memory_policy": nori.memory_policy(),
           "calls": calls}
    if scores:
        pd.concat(scores, ignore_index=True).to_parquet(OUT / f"{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(log, indent=2, default=str))
    return log


def _versions(family: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    pkgs = ("torch", "synthefy-nori" if family == "nori" else "tabicl")
    out = {"python": sys.version.split()[0]}
    for pkg in pkgs:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


# ---- evaluate (.venv) -------------------------------------------------------------------

def _columns(log: dict) -> list[str]:
    return [f"{log['variant']}@{s}" for s in log["sizes_done"]]


def evaluate(n_boot: int = curve.N_BOOT) -> dict:
    """CV AUROC of every finished variant and size, paired with the learning curve's
    Nori 1.5 and XGBoost at the same size (and XGBoost on every row)."""
    from clinical_ml.eval.metrics import bootstrap_fn

    logs = {m: json.loads((OUT / f"{m}.json").read_text()) for m in VARIANTS
            if (OUT / f"{m}.json").exists()}
    wide = curve.load_scores()
    cols = []
    for m, log in logs.items():
        if not log["sizes_done"]:
            continue
        s = pd.read_parquet(OUT / f"{m}.parquet")
        for size, g in s.groupby("size", sort=False):
            col = g.set_index("visit_occurrence_id")["p"].rename(f"{m}@{size}")
            assert len(col) == len(wide) and col.index.is_unique, f"{col.name}: rows differ"
            wide = wide.join(col, on="visit_occurrence_id", validate="1:1")
            cols.append(col.name)
    pairs = []
    for c in cols:
        size = c.split("@")[1]
        refs = [f"nori@{size}", f"xgboost@{size}", "xgboost@all"]
        if c.startswith("cluster_route@"):
            refs = [f"nori@{nori.CONTEXT_ROWS}", "xgboost@all"]
        pairs += [(c, r) for r in dict.fromkeys(refs) if r in wide.columns]
    refs = sorted({r for _, r in pairs})

    def stats(readmit_30d, fold, **p):
        res = {c: curve.fold_mean_auroc(readmit_30d, fold, p[c]) for c in (*cols, *refs)}
        return res | {f"{a} - {b}": res[a] - res[b] for a, b in pairs}

    arrays = {"readmit_30d": wide["readmit_30d"].to_numpy(), "fold": wide["fold"].to_numpy(),
              **{c: wide[c].to_numpy() for c in (*cols, *refs)}}
    res = bootstrap_fn(stats, arrays, wide["person_id"].to_numpy(), n_boot=n_boot,
                       seed=config.SEED)
    folds = {c: [curve.auroc(g["readmit_30d"].to_numpy(), g[c].to_numpy())
                 for _, g in wide.groupby("fold")] for c in (*cols, *refs)}
    for c in (*cols, *refs):
        res[c]["folds"] = folds[c]
    for a, b in pairs:
        d = [x - y for x, y in zip(folds[a], folds[b], strict=True)]
        res[f"{a} - {b}"] |= {"folds": d, "folds_ahead": sum(x > 0 for x in d)}
    best = max(cols, key=lambda c: res[c]["value"]) if cols else None
    xgb = res["xgboost@all"]["value"]
    mean_rows = {}
    for m, log in logs.items():
        for size in log["sizes_done"]:
            n = [c["rows"] for c in log["calls"] if c.get("size") == size and "rows" in c]
            if n:
                mean_rows[f"{m}@{size}"] = float(np.mean(n))
    out = {"results": res, "columns": cols, "mean_rows": mean_rows,
           "variants": {m: {"label": VARIANTS[m]["label"], "sizes_tried": log["sizes_tried"],
                            "sizes_done": log["sizes_done"], "versions": log["versions"],
                            "not_fit": [c for c in log["calls"]
                                        if c.get("status") == "does not fit"],
                            "seconds": sum(c.get("seconds", 0) for c in log["calls"]),
                            "devices": sorted({c["device"] for c in log["calls"]
                                               if "device" in c})}
                        for m, log in logs.items()},
           "test_rule": {"best": best, "best_cv_auroc": res[best]["value"] if best else None,
                         "xgboost_all": xgb,
                         "test_candidate": best if best and res[best]["value"] > xgb else None}}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2, default=str))
    return out



# ---- the test candidate, if any (M2d's steps) -------------------------------------------

CANDIDATE = config.REPORTS / "tfm_candidate_choices.json"
CANDIDATE_REPORT = config.REPORTS / "benchmark_tfm_candidate.json"


def candidate() -> tuple[str, int | None]:
    """The variant the test rule picked, and its size (None: every train row)."""
    rule = json.loads(REPORT.read_text())["test_rule"]
    if not rule["test_candidate"]:
        raise SystemExit("no variant passed the test rule: the test split stays unread")
    name, size = rule["test_candidate"].split("@")
    return name, None if size == "all" else int(size)


def predict_split(step: str, scorer=None) -> dict:
    """The candidate on the calibration or test split, with train as context (all of it,
    or the first n rows of the learning curve's order). Test once, after the recalibration
    has been chosen."""
    name, n = candidate()
    if step == "test":
        if (OUT / "candidate_test.parquet").exists():
            raise SystemExit("test scores already exist: the test split is scored once")
        if not CANDIDATE.exists():
            raise SystemExit("choose the recalibration (after calibration) before test")
    spec = VARIANTS[name]
    scorer = scorer or {"nori": nori_scorer, "tabicl": tabicl_scorer}[spec["family"]]
    df = pd.read_parquet(nori.OUT / "features.parquet")
    tr = df[df["split"] == "train"]
    ctx, query = curve.rows(curve.row_order(tr), n), df[df["split"] == step]
    p, info = scorer(spec, ctx, query)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"visit_occurrence_id": query["visit_occurrence_id"].to_numpy(), "p": p}
                 ).to_parquet(OUT / f"candidate_{step}.parquet", index=False)
    log = {"candidate": f"{name}@{curve.size_label(n)}", "step": step, "context_rows": len(ctx),
           "context_ids_sha256": curve.ids_sha256(ctx), "query_rows": len(query),
           "versions": _versions(spec["family"]), **info}
    (OUT / f"candidate_{step}.json").write_text(json.dumps(log, indent=2, default=str))
    return log


def _candidate_frame(step: str) -> pd.DataFrame:
    meta = pd.read_parquet(nori.OUT / "features.parquet", columns=nori.META)
    s = pd.read_parquet(OUT / f"candidate_{step}.parquet").rename(columns={"p": "p1"})
    out = meta.merge(s, on="visit_occurrence_id", validate="1:1")
    assert len(out) == len(s)
    return out.sort_values("visit_occurrence_id", ignore_index=True)


def choose_candidate() -> dict:
    from clinical_ml.eval import calibration

    cal = _candidate_frame("calibration")
    p, _ = nori.to_probability(cal["p1"].to_numpy())
    method, cv_brier = calibration.select_method(p, cal["readmit_30d"].to_numpy(),
                                                 cal["person_id"].to_numpy())
    out = {"candidate": "@".join(map(str, candidate())).replace("None", "all"),
           "recalibration": {"selected": method, "cv_brier_on_calibration_split": cv_brier}}
    CANDIDATE.write_text(json.dumps(out, indent=2))
    return out


def evaluate_candidate() -> dict:
    """Test metrics, once: M2c's comparison with the candidate as the score, then paired
    differences against XGBoost and the M2c Nori on the same resamples."""
    from clinical_ml.eval.metrics import bootstrap
    from clinical_ml.models.benchmark import XGB_PREDICTIONS

    choices = json.loads(CANDIDATE.read_text())
    m1, xgb = nori._m1_predictions(), pd.read_parquet(XGB_PREDICTIONS)

    def frame(step):
        f = _candidate_frame(step).merge(m1[m1["split"] == step].drop(columns="split"),
                                         on="visit_occurrence_id", validate="1:1")
        return f.merge(nori._scores(step)[["visit_occurrence_id", "p1"]]
                       .rename(columns={"p1": "p_nori_10k"}), on="visit_occurrence_id",
                       validate="1:1")

    cal, test = frame("calibration"), frame("test")
    test = test.merge(xgb[xgb["split"] == "test"].drop(columns="split"),
                      on="visit_occurrence_id", validate="1:1")
    assert len(test) == len(_candidate_frame("test")), "test rows do not line up"
    out = nori.compare({"score": "p1", "recalibration": choices["recalibration"]}, cal, test)
    y, g = test["readmit_30d"].to_numpy(), test["person_id"].to_numpy()
    p, _ = nori.to_probability(test["p1"].to_numpy())
    for base in ("xgboost", "nori_10k"):
        pair = bootstrap(y, {base: test[f"p_{base}"].to_numpy(), "nori": p}, g,
                         n_boot=nori.N_BOOT, seed=config.SEED)
        out["discrimination"][f"nori_minus_{base}_auroc"] = pair[f"nori_minus_{base}_auroc"]
        out["discrimination"].setdefault(base, pair[base])
    m2c = json.loads((config.REPORTS / "benchmark_nori.json").read_text())["discrimination"]
    for m in nori.M1_MODELS:
        assert out["discrimination"][m] == m2c[m], "not the M1 resamples"
    assert out["discrimination"]["nori_10k"] == m2c["nori"], "not the M2c Nori scores"
    out["candidate"] = choices["candidate"]
    out["runs"] = {s: json.loads((OUT / f"candidate_{s}.json").read_text())
                   for s in ("calibration", "test")}
    CANDIDATE_REPORT.write_text(json.dumps(out, indent=2, default=str))
    return out

if __name__ == "__main__":   # python -m clinical_ml.models.tfm_variants predict <variant>
    import argparse

    from clinical_ml.rag.models import use_project_hf_cache

    a = argparse.ArgumentParser(prog="clinical_ml.models.tfm_variants")
    a.add_argument("cmd", choices=["predict", "candidate"])
    a.add_argument("target", help="a variant for predict; calibration or test for candidate")
    args = a.parse_args()
    use_project_hf_cache()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.cmd == "candidate":
        if args.target not in ("calibration", "test"):
            raise SystemExit("candidate takes calibration or test")
        print(json.dumps(predict_split(args.target), indent=2, default=str))
        raise SystemExit(0)
    if args.target not in VARIANTS:
        raise SystemExit(f"unknown variant {args.target}; one of {list(VARIANTS)}")
    log = predict(args.target)
    print(json.dumps({k: v for k, v in log.items() if k != "calls"}, indent=2, default=str))
