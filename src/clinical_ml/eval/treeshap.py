"""Exact path-dependent TreeSHAP for scikit-learn's HistGradientBoostingClassifier,
including its native categorical splits.

Why not shap.TreeExplainer: shap 0.54 reads HistGradientBoosting trees as if every split
were a numeric threshold, so categorical (bitset) splits are ignored and its values no
longer add up to the model output (M3 measures the gap on the real model). This module
ports shap's own algorithm (Lundberg et al. 2020, Algorithm 2; shap's cext/tree_shap.h)
and vectorises it over rows: the recursion visits every node whatever x is, so only the
"hot branch" indicator and the path weights depend on the row. Covers are the node
training counts, as in shap.

Checked in tests/test_m3.py: identical to shap.TreeExplainer on numeric-only models, and
exactly additive (expected value + sum of SHAP values == decision_function) with
categorical splits. The routing rule mirrors sklearn's _predictor.pyx and is asserted
against decision_function at run time, because it relies on private attributes.
"""

import numpy as np
import pandas as pd

ATOL = 1e-8


def _encode(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The matrix the trees actually see, and for each of its columns the index of the
    original column (HGB's preprocessor puts categorical columns first)."""
    if getattr(model, "_preprocessor", None) is None:
        return np.asarray(X, dtype=float), np.arange(X.shape[1])
    cat = np.asarray(model.is_categorical_)
    order = np.concatenate([np.flatnonzero(cat), np.flatnonzero(~cat)])
    return np.asarray(model._preprocessor.transform(X), dtype=float), order


def _in_bitset(bitsets: np.ndarray, row: int, v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64)
    return ((bitsets[row, v // 32] >> (v % 32).astype(np.uint32)) & 1).astype(bool)


def _goes_left(node, predictor, x: np.ndarray, known, f_idx_map) -> np.ndarray:
    """sklearn's _predict_one_from_raw_data, vectorised over rows."""
    missing = np.isnan(x)
    if not node["is_categorical"]:
        left = x <= node["num_threshold"]
    else:
        v = np.where(missing | (x < 0), 0, x)
        in_left = _in_bitset(predictor.raw_left_cat_bitsets, node["bitset_idx"], v)
        is_known = _in_bitset(known, f_idx_map[node["feature_idx"]], v)
        missing = missing | (x < 0) | (~in_left & ~is_known)
        left = in_left
    return np.where(missing, bool(node["missing_go_to_left"]), left)


def _trees(model):
    assert model.n_trees_per_iteration_ == 1, "binary classification only"
    return [row[0] for row in model._predictors]


def _known(model):
    if getattr(model, "_preprocessor", None) is None:
        return np.zeros((1, 8), dtype=np.uint32), np.zeros(1, dtype=np.uint32)
    return model._bin_mapper.make_known_categories_bitsets()


def raw_predict(model, X: pd.DataFrame) -> np.ndarray:
    """Log-odds by walking the trees with the routing rule used for SHAP."""
    Z, _ = _encode(model, X)
    known, f_idx_map = _known(model)
    out = np.full(len(Z), float(np.ravel(model._baseline_prediction)[0]))
    for pred in _trees(model):
        nodes, idx = pred.nodes, np.zeros(len(Z), dtype=np.int64)
        for _ in range(int(nodes["depth"].max()) + 1):
            for j in np.unique(idx):
                node = nodes[j]
                if node["is_leaf"]:
                    continue
                rows = idx == j
                left = _goes_left(node, pred, Z[rows, node["feature_idx"]], known, f_idx_map)
                idx[rows] = np.where(left, node["left"], node["right"])
        out += nodes["value"][idx]
    return out


class _Path:
    """shap's PathElement array. Feature and zero fraction are the same for every row;
    one fraction and permutation weight are per-row vectors."""

    def __init__(self, n: int):
        self.n, self.feat, self.zero, self.one, self.w = n, [], [], [], []

    def copy(self) -> "_Path":
        p = _Path(self.n)
        p.feat, p.zero, p.one, p.w = list(self.feat), list(self.zero), list(self.one), list(
            self.w)
        return p

    def extend(self, zero: float, one: np.ndarray, feat: int) -> None:
        d = len(self.feat)
        self.feat.append(feat)
        self.zero.append(zero)
        self.one.append(one)
        self.w.append(np.ones(self.n) if d == 0 else np.zeros(self.n))
        for i in range(d - 1, -1, -1):
            self.w[i + 1] = self.w[i + 1] + one * self.w[i] * (i + 1) / (d + 1)
            self.w[i] = zero * self.w[i] * (d - i) / (d + 1)

    def unwind(self, k: int) -> None:
        d, one, zero = len(self.feat) - 1, self.one[k], self.zero[k]
        hot, safe = one != 0, np.where(one != 0, one, 1.0)
        nxt = self.w[d]
        for i in range(d - 1, -1, -1):
            w_hot = nxt * (d + 1) / ((i + 1) * safe)
            nxt = np.where(hot, self.w[i] - w_hot * zero * (d - i) / (d + 1), nxt)
            self.w[i] = np.where(hot, w_hot, self.w[i] * (d + 1) / (zero * (d - i)))
        for lst in (self.feat, self.zero, self.one):
            lst.pop(k)
        self.w.pop(d)

    def unwound_sum(self, k: int) -> np.ndarray:
        d, one, zero = len(self.feat) - 1, self.one[k], self.zero[k]
        hot, safe = one != 0, np.where(one != 0, one, 1.0)
        nxt, total_hot, total_cold = self.w[d], 0.0, 0.0
        for i in range(d - 1, -1, -1):
            tmp = nxt / ((i + 1) * safe)
            total_hot = total_hot + tmp
            nxt = self.w[i] - tmp * zero * (d - i)
            total_cold = total_cold + self.w[i] / (zero * (d - i))
        return np.where(hot, total_hot, total_cold) * (d + 1)


def _tree_shap(pred, Z, phi, known, f_idx_map) -> None:
    nodes = pred.nodes
    count = nodes["count"].astype(float)

    def recurse(j: int, parent: _Path, zero: float, one: np.ndarray, feat: int) -> None:
        path = parent.copy()
        path.extend(zero, one, feat)
        node = nodes[j]
        if node["is_leaf"]:
            for i in range(1, len(path.feat)):
                w = path.unwound_sum(i)
                phi[:, path.feat[i]] += w * (path.one[i] - path.zero[i]) * node["value"]
            return
        f = int(node["feature_idx"])
        left = _goes_left(node, pred, Z[:, f], known, f_idx_map)
        in_zero, in_one = 1.0, np.ones(len(Z))
        if f in path.feat:
            k = path.feat.index(f)
            in_zero, in_one = path.zero[k], path.one[k]
            path.unwind(k)
        cover = count[j]
        recurse(int(node["left"]), path, count[node["left"]] / cover * in_zero,
                in_one * left, f)
        recurse(int(node["right"]), path, count[node["right"]] / cover * in_zero,
                in_one * ~left, f)

    recurse(0, _Path(len(Z)), 1.0, np.ones(len(Z)), -1)


def expected_value(model) -> float:
    """Cover-weighted mean log-odds over the training data (shap's base value)."""
    ev = float(np.ravel(model._baseline_prediction)[0])
    for pred in _trees(model):
        leaves = pred.nodes[pred.nodes["is_leaf"].astype(bool)]
        ev += float((leaves["value"] * leaves["count"]).sum() / pred.nodes["count"][0])
    return ev


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values on the log-odds scale, columns in X's order, plus the expected value.
    Raises if the explanation does not add up to the model's decision_function."""
    Z, order = _encode(model, X)
    known, f_idx_map = _known(model)
    phi_enc = np.zeros(Z.shape)
    for pred in _trees(model):
        _tree_shap(pred, Z, phi_enc, known, f_idx_map)
    phi = np.zeros(Z.shape)
    phi[:, order] = phi_enc
    ev = expected_value(model)
    gap = np.max(np.abs(ev + phi.sum(1) - model.decision_function(X)))
    assert gap < ATOL, f"TreeSHAP not additive (max error {gap:.2e})"
    return phi, ev
