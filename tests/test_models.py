import numpy as np
import pandas as pd

from clinical_ml.eval.metrics import bootstrap
from clinical_ml.models.train import SPLIT_FRACTIONS, patient_split


def test_patient_split_is_disjoint_and_sized():
    rng = np.random.default_rng(0)
    persons = pd.Series(rng.integers(0, 2000, 6000))
    split = patient_split(persons, seed=1)
    assert pd.DataFrame({"p": persons, "s": split}).groupby("p")["s"].nunique().max() == 1
    share = split.groupby(persons).first().value_counts(normalize=True)
    for name, frac in SPLIT_FRACTIONS.items():
        assert abs(share[name] - frac) < 0.01
    assert split.equals(patient_split(persons, seed=1))  # deterministic


def test_bootstrap_cis_bracket_point_estimates():
    rng = np.random.default_rng(0)
    n = 3000
    groups = rng.integers(0, 1500, n)
    y = rng.integers(0, 2, n)
    good = np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1)
    noise = rng.random(n)
    out = bootstrap(y, {"noise": noise, "good": good}, groups, n_boot=200, seed=0)
    for m in ("noise", "good"):
        lo, hi = out[m]["auroc"]["ci95"]
        assert lo <= out[m]["auroc"]["value"] <= hi
    assert out["noise"]["auroc"]["ci95"][0] < 0.5 < out["noise"]["auroc"]["ci95"][1]
    assert out["good_minus_noise_auroc"]["ci95"][0] > 0
    assert out["bootstrap"]["n_patients"] == len(np.unique(groups))
