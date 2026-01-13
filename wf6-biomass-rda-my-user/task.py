from typing import Dict
import numpy as np
import pandas as pd
import warnings
from typing import Any
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from typing import Optional
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib
import sklearn
import scipy
import statsmodels
import re
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import csv
from sklearn.linear_model import LinearRegression

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--clean_path', action='store', type=str, required=True, dest='clean_path')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

clean_path = args.clean_path.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'

warnings.filterwarnings("ignore")

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"matplotlib: {matplotlib.__version__}")  # <-- qui
print(f"scikit-learn: {sklearn.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"statsmodels: {statsmodels.__version__}")


CONFIG: Dict[str, Any] = {

    "transformY": {
        "enable": True,
        "method": "center",
        "pseudocount": 0.0,
        "negative_handling": "clip_zero",  # 'raise' | 'clip_zero' | 'shift_min_to_zero'
        "robust_center": True,
        "robust_scale": True,
        "robust_quantile_range": (25.0, 75.0)
    },

    "Xscaling": {
        "enable": False,
        "method": "none",
        "pseudocount": 0.0,             # used for log1p/log10p global
        "negative_handling": "clip_zero",
        "transforms": {
        }
    },

    "rda": {
        "n_components": 2,
        "variance_reference": "Y_total",      # 'Y_total' | 'Y_hat'
        "inertia_method": "svd"               # 'svd' (vegan) | 'var' (current)
    },

    "biplot": {
        "scaling": "scaling2",                # 'none' | 'scaling1' | 'scaling2'
        "convention": "vegan",                # 'vegan' | 'current'
        "biotic_arrow_factor": 1.8,
        "biotic_label_factor": 1.85,
        "abiotic_scale": 4.0,
        "biotic_arrow_enlarge_factor": 20.,
        "abiotic_arrow_enlarge_factor": 3.0 
    },

    "corr": {
        "method": "pearson",                  # 'pearson' | 'spearman' | 'regression'
        "center_X": True,
        "scale_X": False,
        "n_axes": 2
    },

    "vif": {
        "enable": True,
        "threshold": 5.0
    },

    "vegan": {
        "compute_unconstrained": True,
        "save_inertia_table": True,
        "compute_R2_adj": True,

        "include_factors_in_model": True,
        "factor_drop_first": False,

        "partial_rda": {
            "enable": False,
            "covariates_to_condition": []
        },

        "fit": {
            "use_intercept": True,
            "method": "qr"
        },

        "permutation_tests": {
            "enable": True,
            "n_permutations": 999,
            "seed": 0,
            "test_global": True,
            "test_axes": True,
            "test_terms": False,
        },

        "scores_type": "species"
    },

    "plot": {
        "figsize": (8, 8),
        "dpi": 300,
        "title": "RDA Biplot\nSamples, Biotic variable (blue), Abiotic variable (red)"
    },

    "n_species_col_name": "N Species"
}


def read_csv_auto(path):
    with open(path, 'rb') as f:
        sample = f.read(8192).decode('utf-8-sig', errors='ignore')
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',;')
        sep = dialect.delimiter
    except Exception:
        sep = ',' if sample.count(',') >= sample.count(';') else ';'
    dot_nums   = len(re.findall(r'\d+\.\d+', sample))
    comma_nums = len(re.findall(r'\d+,\d+', sample))
    decimal = ',' if (sep == ';' and comma_nums > dot_nums) else '.'
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.replace('\uFEFF','', regex=True)
    return df



def unified_transform_matrix(M: np.ndarray, method: str, opts: Dict[str, Any]) -> np.ndarray:
    if method is None:
        method = 'none'
    method = str(method).lower()

    M = np.asarray(M, dtype=float)

    if method == 'none':
        return M

    if method == 'center':
        col_means = np.nanmean(M, axis=0, keepdims=True)
        return M - col_means

    if method in ('standard', 'zscore'):
        col_means = np.nanmean(M, axis=0, keepdims=True)
        col_stds  = np.nanstd(M, axis=0, ddof=0, keepdims=True)
        col_stds[col_stds == 0] = 1.0
        return (M - col_means) / col_stds

    if method in ('standard_sample', 'zscore_sample'):
        col_means = np.nanmean(M, axis=0, keepdims=True)
        col_stds  = np.nanstd(M, axis=0, ddof=1, keepdims=True)
        col_stds[col_stds == 0] = 1.0
        return (M - col_means) / col_stds

    if method == 'robust':
        center = bool(opts.get("robust_center", True))
        scale  = bool(opts.get("robust_scale", True))
        qrange = tuple(opts.get("robust_quantile_range", (25.0, 75.0)))
        return RobustScaler(with_centering=center, with_scaling=scale, quantile_range=qrange).fit_transform(M)

    if method in ('log1p', 'log10p', 'hellinger'):
        if np.nanmin(M) < 0:
            nh = opts.get("negative_handling", "clip_zero")
            if nh == 'raise':
                raise ValueError("The matrix contains negative values that are not allowed by the transformation.")
            elif nh == 'clip_zero':
                M = np.where(M < 0, 0.0, M)
            elif nh == 'shift_min_to_zero':
                M = M - np.nanmin(M)

        pseudocount = float(opts.get("pseudocount", 0.0))

        if method == 'log1p':
            return np.nan_to_num(np.log1p(M + pseudocount), nan=0.0)

        if method == 'log10p':
            return np.nan_to_num(np.log10(1.0 + M + pseudocount), nan=0.0)

        if method == 'hellinger':
            row_sums = np.nansum(M, axis=1, keepdims=True)
            row_sums[row_sums == 0] = np.nan
            rel = M / row_sums
            return np.sqrt(np.nan_to_num(rel, nan=0.0))

    raise ValueError(f"Unrecognized transformation method: {method}")

def transform_response_param(Y: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    if not cfg.get("enable", True) or str(cfg.get("method", 'none')).lower() == 'none':
        return Y.to_numpy(dtype=float)
    return unified_transform_matrix(Y.to_numpy(dtype=float), cfg.get("method", 'none'), cfg)

def transform_covariates(X: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if not cfg.get("enable", True):
        return X.copy()

    X2 = X.copy()
    transforms = cfg.get("transforms", {}) or {}
    for col, ops in transforms.items():
        if col not in X2.columns:
            continue
        v = pd.to_numeric(X2[col], errors='coerce')

        if ops.get('log1p', False):
            nh = cfg.get("negative_handling", "clip_zero")
            vv = v.to_numpy(dtype=float)
            if np.nanmin(vv) < 0:
                if nh == 'raise':
                    raise ValueError(f"Column {col}: negative values ​​not allowed for log1p.")
                elif nh == 'clip_zero':
                    vv = np.where(vv < 0, 0.0, vv)
                elif nh == 'shift_min_to_zero':
                    vv = vv - np.nanmin(vv)
            pseudocount = float(cfg.get("pseudocount", 0.0))
            v = np.log1p(vv + pseudocount)

        if ops.get('log10p', False):
            nh = cfg.get("negative_handling", "clip_zero")
            vv = v.to_numpy(dtype=float)
            if np.nanmin(vv) < 0:
                if nh == 'raise':
                    raise ValueError(f"Column {col}: negative values ​​not allowed for log10p.")
                elif nh == 'clip_zero':
                    vv = np.where(vv < 0, 0.0, vv)
                elif nh == 'shift_min_to_zero':
                    vv = vv - np.nanmin(vv)
            pseudocount = float(cfg.get("pseudocount", 0.0))
            v = np.log10(1.0 + vv + pseudocount)

        deg = int(ops.get('poly', 1) or 1)
        if deg > 1:
            pf = PolynomialFeatures(degree=deg, include_bias=False)
            poly = pf.fit_transform(np.array(v).reshape(-1, 1))
            names = [f"{col}^{p}" for p in range(1, deg + 1)]
            X2 = X2.drop(columns=[col]).join(pd.DataFrame(poly, index=X2.index, columns=names))
        else:
            X2[col] = v

    return X2

def scale_covariates(X: pd.DataFrame, cfg: Dict[str, Any]):
    if not cfg.get("enable", True):
        return X.to_numpy(dtype=float), None

    method = str(cfg.get("method", cfg.get("scaler", "zscore")).lower())

    if method in ('none',):
        return X.to_numpy(dtype=float), None

    if method == 'center':
        Xt = unified_transform_matrix(X.to_numpy(dtype=float), 'center', cfg)
        return Xt, None

    if method in ('standard', 'zscore'):
        Xt = unified_transform_matrix(X.to_numpy(dtype=float), 'standard', cfg)
        return Xt, None

    if method in ('standard_sample', 'zscore_sample'):
        Xt = unified_transform_matrix(X.to_numpy(dtype=float), 'standard_sample', cfg)
        return Xt, None

    if method == 'robust':
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        Xt = scaler.fit_transform(X)
        return Xt, scaler

    if method in ('log1p', 'log10p', 'hellinger'):
        Xt = unified_transform_matrix(X.to_numpy(dtype=float), method, cfg)
        return Xt, None

    raise ValueError(f"Unrecognized method/scaler X: {method}")


def add_factors_to_X(X_scaled_df: pd.DataFrame,
                     factors_for_plot: Dict[str, pd.Series],
                     cfg_vegan: Dict[str, Any]) -> pd.DataFrame:
    if not cfg_vegan.get("include_factors_in_model", False):
        return X_scaled_df

    X2 = X_scaled_df.copy()
    drop_first = bool(cfg_vegan.get("factor_drop_first", True))

    for fcol, fser in (factors_for_plot or {}).items():
        fser = fser.loc[X2.index]
        dummies = pd.get_dummies(fser.astype(str), prefix=fcol, drop_first=drop_first)
        X2 = pd.concat([X2, dummies], axis=1)

    return X2

def constrained_fitted_values(Yt: np.ndarray, Xs: np.ndarray, fit_cfg: Dict[str, Any]) -> np.ndarray:
    use_intercept = bool(fit_cfg.get("use_intercept", False))
    method = str(fit_cfg.get("method", "pinv")).lower()

    X2 = Xs.copy()
    if use_intercept:
        X2 = np.column_stack([np.ones((X2.shape[0], 1)), X2])

    if method == "qr":
        Q, _ = np.linalg.qr(X2, mode='reduced')
        Hx = Q @ Q.T
        return Hx @ Yt
    else:
        return X2 @ np.linalg.pinv(X2) @ Yt

def residualize_on_covariates(Yt: np.ndarray, Xs: np.ndarray, C: np.ndarray, fit_cfg: Dict[str, Any]):
    if C.shape[1] == 0:
        return Yt, Xs

    use_intercept = bool(fit_cfg.get("use_intercept", False))
    method = str(fit_cfg.get("method", "pinv")).lower()

    C2 = C.copy()
    if use_intercept:
        C2 = np.column_stack([np.ones((C2.shape[0], 1)), C2])

    if method == "qr":
        Q, _ = np.linalg.qr(C2, mode='reduced')
        Hc = Q @ Q.T
    else:
        Hc = C2 @ np.linalg.pinv(C2)

    M = np.eye(C2.shape[0]) - Hc
    return M @ Yt, M @ Xs

def pca_on_matrix(M: np.ndarray, k: int, inertia_method: str = "var"):
    inertia_method = str(inertia_method or "var").lower()
    M = np.asarray(M, dtype=float)
    n, p = M.shape
    k_eff = min(k, n, p)

    if inertia_method == "svd":
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U_k = U[:, :k_eff]
        S_k = S[:k_eff]
        Vt_k = Vt[:k_eff, :]
        scores = U_k * S_k
        loadings = Vt_k.T
        eigvals = (S_k ** 2) / (n - 1) if n > 1 else (S_k ** 2)
        return scores, loadings, eigvals, None

    pca = PCA(n_components=k_eff)
    scores = pca.fit_transform(M)
    loadings = pca.components_.T
    eigvals = pca.explained_variance_
    return scores, loadings, eigvals, pca

def total_inertia_from_method(M: np.ndarray, inertia_method: str = "var") -> float:
    inertia_method = str(inertia_method or "var").lower()
    M = np.asarray(M, dtype=float)
    n = M.shape[0]

    if inertia_method == "svd":
        S = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        return float(np.sum((S ** 2) / (n - 1)) if n > 1 else np.sum(S ** 2))
    else:
        return float(np.var(M, axis=0, ddof=1).sum())

def compute_R2_and_adj(eig_constrained: np.ndarray,
                       total_inertia: float,
                       n_samples: int,
                       n_X: int):
    constrained_inertia = float(np.sum(eig_constrained))
    R2 = constrained_inertia / total_inertia if total_inertia > 0 else np.nan

    if n_samples - n_X - 1 <= 0:
        R2adj = np.nan
    else:
        R2adj = 1 - (1 - R2) * (n_samples - 1) / (n_samples - n_X - 1)

    return R2, R2adj, total_inertia, constrained_inertia

def permutation_anova_rda(Yt: np.ndarray,
                          Xs_final: np.ndarray,
                          conf: Dict[str, Any],
                          fit_cfg: Dict[str, Any],
                          n_perm: int = 999,
                          seed: int = 0,
                          test_global: bool = True,
                          test_axes: bool = False,
                          test_terms: bool = False,
                          term_names: Optional[list] = None):

    rng = np.random.default_rng(seed)
    n, pY = Yt.shape
    pX = Xs_final.shape[1]
    k = int(conf["rda"]["n_components"]) or 2
    k_eff = min(k, pY)
    inertia_method = conf.get("rda", {}).get("inertia_method", "var")

    Yhat_obs = constrained_fitted_values(Yt, Xs_final, fit_cfg)

    if inertia_method == "svd":
        S_obs = np.linalg.svd(Yhat_obs, full_matrices=False, compute_uv=False)
        eig_obs = (S_obs[:k_eff] ** 2) / (n - 1) if n > 1 else (S_obs[:k_eff] ** 2)
        total_var = total_inertia_from_method(Yt, "svd")
    else:
        eig_obs = PCA(n_components=k_eff).fit(Yhat_obs).explained_variance_
        total_var = total_inertia_from_method(Yt, "var")

    con_var_obs = float(np.sum(eig_obs))

    df1 = pX
    df2 = n - pX - 1 if (n - pX - 1) > 0 else np.nan
    F_obs = (con_var_obs/df1) / ((total_var - con_var_obs)/df2) if np.isfinite(df2) else np.nan

    F_perm = []
    axes_perm = []
    terms_perm = {name: [] for name in (term_names or [])}

    for _ in range(n_perm):
        idx = rng.permutation(n)
        Yp = Yt[idx, :]
        Yhat_p = constrained_fitted_values(Yp, Xs_final, fit_cfg)

        if inertia_method == "svd":
            S_p = np.linalg.svd(Yhat_p, full_matrices=False, compute_uv=False)
            eig_p = (S_p[:k_eff] ** 2) / (n - 1) if n > 1 else (S_p[:k_eff] ** 2)
        else:
            eig_p = PCA(n_components=k_eff).fit(Yhat_p).explained_variance_

        con_var_p = float(np.sum(eig_p))

        Fp = (con_var_p/df1) / ((total_var - con_var_p)/df2) if np.isfinite(df2) else np.nan
        F_perm.append(Fp)
        axes_perm.append(eig_p)

        if test_terms and term_names is not None:
            for j, name in enumerate(term_names):
                X_drop = np.delete(Xs_final, j, axis=1)
                Yhat_drop = constrained_fitted_values(Yp, X_drop, fit_cfg)

                if inertia_method == "svd":
                    S_drop = np.linalg.svd(Yhat_drop, full_matrices=False, compute_uv=False)
                    eig_drop = (S_drop[:k_eff] ** 2) / (n - 1) if n > 1 else (S_drop[:k_eff] ** 2)
                else:
                    eig_drop = PCA(n_components=k_eff).fit(Yhat_drop).explained_variance_

                con_drop = float(np.sum(eig_drop))
                terms_perm[name].append(con_var_p - con_drop)

    results = {}

    if test_global:
        F_perm = np.asarray(F_perm)
        pval = (np.sum(F_perm >= F_obs) + 1) / (n_perm + 1)
        results["global"] = {"F_obs": float(F_obs), "p_value": float(pval)}

    if test_axes:
        axes_perm = np.asarray(axes_perm)
        pvals_axes = []
        for a, eig_a in enumerate(eig_obs[:k_eff]):
            perm_a = axes_perm[:, a]
            pval_a = (np.sum(perm_a >= eig_a) + 1) / (n_perm + 1)
            pvals_axes.append(pval_a)
        results["axes"] = {f"RDA{a+1}": float(p) for a, p in enumerate(pvals_axes)}

    if test_terms and term_names is not None:
        term_pvals = {}
        for j, name in enumerate(term_names):
            X_drop = np.delete(Xs_final, j, axis=1)
            Yhat_drop = constrained_fitted_values(Yt, X_drop, fit_cfg)

            if inertia_method == "svd":
                S_drop_obs = np.linalg.svd(Yhat_drop, full_matrices=False, compute_uv=False)
                eig_drop_obs = (S_drop_obs[:k_eff] ** 2) / (n - 1) if n > 1 else (S_drop_obs[:k_eff] ** 2)
            else:
                eig_drop_obs = PCA(n_components=k_eff).fit(Yhat_drop).explained_variance_

            delta_obs = con_var_obs - float(np.sum(eig_drop_obs))

            perm_deltas = np.asarray(terms_perm[name])
            pval_t = (np.sum(perm_deltas >= delta_obs) + 1) / (n_perm + 1)
            term_pvals[name] = float(pval_t)
        results["terms"] = term_pvals

    return results


def apply_biplot_scaling_param(scores: np.ndarray, loadings: np.ndarray, eigvals: np.ndarray, cfg: Dict[str, Any]):
    mode = cfg.get("scaling", "none")
    convention = cfg.get("convention", "current")

    if mode == 'none':
        return scores, loadings

    lam_sqrt = np.sqrt(eigvals)

    if convention == 'vegan':
        if mode == 'scaling1':
            return scores, loadings
        if mode == 'scaling2':
            return scores / lam_sqrt, loadings * lam_sqrt
        raise ValueError(f"biplot_scaling='{mode}' not recognized.")
    elif convention == 'current':
        if mode == 'scaling1':
            return scores / lam_sqrt, loadings * lam_sqrt
        if mode == 'scaling2':
            return scores, loadings * lam_sqrt
        raise ValueError(f"biplot_scaling='{mode}' not recognized.")
    else:
        raise ValueError(f"biplot_convention='{convention}' not recognized ('vegan'|'current').")


def compute_env_associations(scores_df: pd.DataFrame,
                             X: pd.DataFrame,
                             corr_cfg: Dict[str, Any]) -> pd.DataFrame:
    axes = [c for c in scores_df.columns[: int(corr_cfg.get("n_axes", 2)) ]]
    S = scores_df[axes].to_numpy()

    Xc = X.copy()
    if corr_cfg.get("center_X", True):
        Xc = Xc - Xc.mean(axis=0)
    if corr_cfg.get("scale_X", False):
        std = Xc.std(axis=0, ddof=0).replace(0, 1.0)
        Xc = Xc / std
    Xc = Xc.loc[scores_df.index]

    out = pd.DataFrame(index=Xc.columns, columns=axes, dtype=float)
    method = corr_cfg.get("method", "pearson")

    if method in ('pearson','spearman'):
        f = pearsonr if method=='pearson' else spearmanr
        for abi in Xc.columns:
            x = Xc[abi].to_numpy()
            for j, ax in enumerate(axes):
                r, _ = f(x, S[:, j])
                out.loc[abi, ax] = r
        return out

    if method == 'regression':
        for abi in Xc.columns:
            x = Xc[[abi]].to_numpy()
            for j, ax in enumerate(axes):
                y = S[:, j]
                reg = LinearRegression().fit(x, y)
                out.loc[abi, ax] = float(reg.coef_[0])
        return out

    raise ValueError("Unrecognized association method ('pearson'|'spearman'|'regression').")


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIF calculation identical to the ANCOVA code:
    - variance_inflation_factor
    - if error -> np.inf
    """
    if df.shape[1] == 0:
        return pd.DataFrame({"variable": [], "VIF": []})

    vif_data = pd.DataFrame()
    vif_data["variable"] = df.columns

    vals = df.values
    vifs = []
    for i in range(df.shape[1]):
        try:
            vifs.append(variance_inflation_factor(vals, i))
        except Exception:
            vifs.append(np.inf)

    vif_data["VIF"] = vifs
    return vif_data


def iterative_vif_filter(X: pd.DataFrame, cfg: Dict[str, Any]):
    """
    Iterative filter identical to ANCOVA:
    - works on X raw numeric values
    - removes the column with the highest VIF if > threshold
    - continues until all <= threshold or 1 variable remains
    - does NOT use min_variables and max_iter
    """
    if not cfg.get("enable", True):
        Xnum = X.select_dtypes(include=[np.number]).copy()
        init_vif = calculate_vif(Xnum)
        return X, [], init_vif, init_vif

    Xnum = X.select_dtypes(include=[np.number]).copy()
    Xnum = Xnum.apply(pd.to_numeric, errors="coerce").dropna()

    if Xnum.shape[1] <= 1:
        init_vif = calculate_vif(Xnum)
        return X, [], init_vif, init_vif

    threshold = float(cfg.get("threshold", 10.0))

    current_vars = list(Xnum.columns)
    X_sub = Xnum[current_vars].copy()

    initial_vif_df = calculate_vif(X_sub)

    removed = []
    while len(current_vars) > 1:
        vif_df = calculate_vif(X_sub)
        max_vif = vif_df["VIF"].max()

        if not np.isfinite(max_vif) or max_vif <= threshold:
            break

        drop_var = vif_df.sort_values("VIF", ascending=False)["variable"].iloc[0]

        if drop_var in current_vars:
            current_vars.remove(drop_var)
            removed.append(drop_var)
            X_sub = X_sub[current_vars]

    final_vif_df = calculate_vif(X_sub)

    X_kept = X[current_vars].copy()
    non_numeric_cols = [c for c in X.columns if c not in Xnum.columns]
    if non_numeric_cols:
        X_kept = pd.concat([X_kept, X[non_numeric_cols]], axis=1)

    return X_kept, removed, initial_vif_df, final_vif_df


def rda_manual_param(Y: pd.DataFrame, X: pd.DataFrame, conf: Dict[str, Any], factors_for_plot=None):
    cfg_vegan = conf.get("vegan", {})
    fit_cfg = cfg_vegan.get("fit", {"use_intercept": False, "method": "pinv"})
    inertia_method = conf.get("rda", {}).get("inertia_method", "var")

    Y_t = transform_response_param(Y, conf["transformY"])

    X_trans = transform_covariates(X, conf["Xscaling"])
    X_scaled, _ = scale_covariates(X_trans, conf["Xscaling"])
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X_trans.columns)

    if factors_for_plot is None:
        factors_for_plot = {}
    X_with_factors_df = add_factors_to_X(X_scaled_df, factors_for_plot, cfg_vegan)
    X_scaled2 = X_with_factors_df.to_numpy(dtype=float)

    part_cfg = cfg_vegan.get("partial_rda", {})
    conditioned_cov_cols = []
    if part_cfg.get("enable", False):
        covs = part_cfg.get("covariates_to_condition", []) or []
        conditioned_cov_cols = [c for c in covs if c in X_with_factors_df.columns]
        C = X_with_factors_df[conditioned_cov_cols].to_numpy(dtype=float) if conditioned_cov_cols else np.zeros((len(X), 0))
        Y_t, X_scaled2 = residualize_on_covariates(Y_t, X_scaled2, C, fit_cfg)

    Y_hat = constrained_fitted_values(Y_t, X_scaled2, fit_cfg)

    k = int(conf["rda"]["n_components"]) or 2
    scores_c, loadings_c, eigvals_c, _ = pca_on_matrix(Y_hat, k, inertia_method=inertia_method)

    unconstrained = None
    if cfg_vegan.get("compute_unconstrained", False):
        Y_res = Y_t - Y_hat
        scores_u, loadings_u, eigvals_u, _ = pca_on_matrix(Y_res, k, inertia_method=inertia_method)
        unconstrained = {"scores": scores_u, "loadings": loadings_u, "eigvals": eigvals_u}

    vref = conf["rda"]["variance_reference"]
    if vref == 'Y_total':
        denom = total_inertia_from_method(Y_t, inertia_method)
    elif vref == 'Y_hat':
        denom = total_inertia_from_method(Y_hat, inertia_method)
    else:
        raise ValueError("variance_reference not recognized ('Y_total'|'Y_hat').")

    per_axis = eigvals_c / denom if denom > 0 else np.full_like(eigvals_c, np.nan)
    constrained_total = eigvals_c.sum() / denom if denom > 0 else np.nan

    scores_df  = pd.DataFrame(scores_c,  index=Y.index,    columns=[f'RDA{i+1}' for i in range(scores_c.shape[1])])
    load_df    = pd.DataFrame(loadings_c, index=Y.columns, columns=[f'RDA{i+1}' for i in range(loadings_c.shape[1])])

    R2 = R2adj = total_inertia = constrained_inertia = None
    if cfg_vegan.get("compute_R2_adj", False):
        total_inertia_for_R2 = total_inertia_from_method(Y_t, inertia_method)
        R2, R2adj, total_inertia, constrained_inertia = compute_R2_and_adj(
            eigvals_c, total_inertia_for_R2, n_samples=Y_t.shape[0], n_X=X_scaled2.shape[1]
        )

    extras = {
        "unconstrained": unconstrained,
        "R2": R2,
        "R2adj": R2adj,
        "total_inertia": total_inertia,
        "constrained_inertia": constrained_inertia,
        "X_final_columns": list(X_with_factors_df.columns),
        "X_final_scaled": X_scaled2,
        "conditioned_cov_cols": conditioned_cov_cols
    }

    return scores_df, load_df, per_axis, constrained_total, eigvals_c, Y_t, Y_hat, extras



output_dir = conf_output_path
output_rda = os.path.join(output_dir, "RDA")
os.makedirs(output_rda, exist_ok=True)

df = read_csv_auto(clean_path)
params = pd.read_csv(parameters_csv, sep=';', header=None, encoding='utf-8-sig')

columns = params.iloc[0].astype(str).str.strip().tolist()
labels  = params.iloc[1].astype(str).str.strip().tolist()

agg_row = 2 if params.shape[0] > 2 else None
agg_cols = [col for col, lab in zip(columns, params.iloc[agg_row]) if lab == 'A'] if agg_row is not None else []

biotic_vars  = [col for col, lab in zip(columns, labels) if lab == 'Y']
abiotic_vars = [col for col, lab in zip(columns, labels) if lab == 'X']
factor_vars  = [col for col, lab in zip(columns, labels) if lab == 'f']

aggregation_needed = len(agg_cols) > 0
n_species_col_name = CONFIG["n_species_col_name"]
is_nspecies_biotic = n_species_col_name in biotic_vars

if aggregation_needed:
    print("Aggregation requested! Aggregating data as specified in Parameter.csv...")

    group_keys = []
    for c in agg_cols + factor_vars:
        if c in df.columns and c not in group_keys:
            group_keys.append(c)

    biotic_cols_no_nspecies = [c for c in biotic_vars if c != n_species_col_name]
    agg_dict = {col: 'mean' for col in biotic_cols_no_nspecies + abiotic_vars if col in df.columns}

    grouped = df.groupby(group_keys).agg(agg_dict).reset_index()

    if is_nspecies_biotic and 'acceptedNameUsage' in df.columns:
        n_species = (
            df.groupby(group_keys)['acceptedNameUsage']
              .nunique()
              .reset_index(name=n_species_col_name)
        )
        grouped = pd.merge(grouped, n_species, on=group_keys, how='left')

    grouped.to_csv(os.path.join(output_rda, "Aggregated_output.csv"), index=False, encoding='utf-8-sig')
    print("Aggregated table saved to: output/Aggregated_output.csv")

    use_cols = [c for c in biotic_vars + abiotic_vars if c in grouped.columns]
    data_rda = grouped[use_cols].dropna()
    Y = data_rda[[c for c in biotic_vars if c in data_rda.columns]]
    X = data_rda[[c for c in abiotic_vars if c in data_rda.columns]]

    factors_for_plot = {}
    for fcol in factor_vars:
        if fcol in grouped.columns:
            factors_for_plot[fcol] = grouped.loc[data_rda.index, fcol].astype(str)
else:
    print("No aggregation requested. Using original data table...")
    biotic_vars_actual  = [v for v in biotic_vars  if v in df.columns]
    abiotic_vars_actual = [v for v in abiotic_vars if v in df.columns]

    data_rda = df[biotic_vars_actual + abiotic_vars_actual].dropna()
    Y = data_rda[biotic_vars_actual]
    X = data_rda[abiotic_vars_actual]

    factors_for_plot = {}
    for fcol in factor_vars:
        if fcol in df.columns:
            factors_for_plot[fcol] = df.loc[data_rda.index, fcol].astype(str)



if CONFIG["vif"].get("enable", True):
    X_kept, removed_vars, vif_initial_df, vif_final_df = iterative_vif_filter(X, CONFIG["vif"])

    vif_initial_df.to_csv(os.path.join(output_rda, 'RDA_abiotic_VIF_initial.csv'),
                          index=False, encoding='utf-8-sig')
    vif_final_df.to_csv(os.path.join(output_rda, 'RDA_abiotic_VIF_Final.csv'),
                        index=False, encoding='utf-8-sig')
    if removed_vars:
        print(f"Abiotic variables excluded due to multicollinearity (threshold ={CONFIG['vif']['threshold']}): {removed_vars}")
else:
    X_kept = X.copy()
    removed_vars = []

abiotic_vars = [c for c in X_kept.columns if c in abiotic_vars]
X = X_kept[abiotic_vars]



scores_df, loadings_df, per_axis_vs_total, prop_constrained_total, eigvals, Y_t, Y_hat, extras = \
    rda_manual_param(Y, X, CONFIG, factors_for_plot=factors_for_plot)

scores_df.to_csv(os.path.join(output_rda, 'RDA_scores.csv'), encoding='utf-8-sig')
loadings_df.to_csv(os.path.join(output_rda, 'RDA_biotic_loadings.csv'), encoding='utf-8-sig')

with open(os.path.join(output_rda, 'RDA_variance_explained.txt'), 'w', encoding='utf-8-sig') as f:
    for i, v in enumerate(per_axis_vs_total):
        f.write(f"RDA{i+1}: {v*100:.2f}% (ref={CONFIG['rda']['variance_reference']})\n")
    f.write(f"Constrained total (all RDA axes): {prop_constrained_total*100:.2f}%\n")

print("RDA completed. Results saved in:", output_rda)


cfg_vegan = CONFIG.get("vegan", {})
inertia_method = CONFIG.get("rda", {}).get("inertia_method", "var")

if cfg_vegan.get("compute_unconstrained", False) and extras.get("unconstrained") is not None:
    unc = extras["unconstrained"]
    k = int(CONFIG["rda"]["n_components"]) or 2

    unc_scores_df = pd.DataFrame(unc["scores"], index=Y.index,
                                 columns=[f"PC_res{i+1}" for i in range(unc["scores"].shape[1])])
    unc_load_df = pd.DataFrame(unc["loadings"], index=Y.columns,
                               columns=[f"PC_res{i+1}" for i in range(unc["loadings"].shape[1])])

    unc_scores_df.to_csv(os.path.join(output_rda, "RDA_unconstrained_scores.csv"), encoding='utf-8-sig')
    unc_load_df.to_csv(os.path.join(output_rda, "RDA_unconstrained_loadings.csv"), encoding='utf-8-sig')
    print("Unconstrained (residual) PCA saved in:", output_rda)

if cfg_vegan.get("compute_R2_adj", False):
    with open(os.path.join(output_rda, "RDA_R2adj.txt"), "w", encoding="utf-8-sig") as f:
        f.write(f"R2: {extras['R2']:.6f}\n")
        f.write(f"R2adj: {extras['R2adj']:.6f}\n")
    print("R2 and R2adj saved in:", output_rda)

if cfg_vegan.get("save_inertia_table", False):
    inertia_path = os.path.join(output_rda, "RDA_inertia_table.csv")

    total_inertia = extras.get("total_inertia")
    constrained_inertia = extras.get("constrained_inertia")

    unconstrained_inertia = None
    if extras.get("unconstrained") is not None:
        unconstrained_inertia = float(np.sum(extras["unconstrained"]["eigvals"]))

    inertia_df = pd.DataFrame({
        "component": ["Total", "Constrained", "Unconstrained"],
        "inertia": [total_inertia, constrained_inertia, unconstrained_inertia],
        "proportion": [
            1.0,
            constrained_inertia / total_inertia if total_inertia else np.nan,
            unconstrained_inertia / total_inertia if (total_inertia and unconstrained_inertia is not None) else np.nan
        ],
        "method": [inertia_method]*3
    })
    inertia_df.to_csv(inertia_path, index=False, encoding="utf-8-sig")
    print("Inertia table saved to:", inertia_path)



X_for_corr = X.loc[Y.index]
abiotic_assoc = compute_env_associations(scores_df, X_for_corr, CONFIG["corr"])
abiotic_assoc.to_csv(os.path.join(output_rda, 'RDA_abiotic_associations.csv'), encoding='utf-8-sig')
print("Abiotic associations with RDA axes saved in:", output_rda)



perm_cfg = cfg_vegan.get("permutation_tests", {})
if perm_cfg.get("enable", False):
    term_names = extras.get("X_final_columns", list(X.columns))

    perm_results = permutation_anova_rda(
        Y_t,
        extras["X_final_scaled"],
        CONFIG,
        cfg_vegan.get("fit", {}),
        n_perm=int(perm_cfg.get("n_permutations", 999)),
        seed=int(perm_cfg.get("seed", 0)),
        test_global=bool(perm_cfg.get("test_global", True)),
        test_axes=bool(perm_cfg.get("test_axes", False)),
        test_terms=bool(perm_cfg.get("test_terms", False)),
        term_names=term_names
    )

    outp = os.path.join(output_rda, "RDA_permutation_tests.txt")
    with open(outp, "w", encoding='utf-8-sig') as f:
        for k, v in perm_results.items():
            f.write(f"[{k}]\n{v}\n\n")
    print("Permutation tests saved in:", outp)



scores_plot, loadings_plot = apply_biplot_scaling_param(
    scores_df.to_numpy(), loadings_df.to_numpy(), eigvals, CONFIG["biplot"]
)

scores_plot_df   = pd.DataFrame(scores_plot,  index=scores_df.index, columns=scores_df.columns)
loadings_plot_df = pd.DataFrame(loadings_plot, index=loadings_df.index, columns=loadings_df.columns)

fig, ax = plt.subplots(figsize=CONFIG["plot"]["figsize"])

if CONFIG["biplot"]["scaling"] == 'none':
    biotic_arrow_factor  = CONFIG["biplot"]["biotic_arrow_factor"]
    biotic_label_factor  = CONFIG["biplot"]["biotic_label_factor"]
    abiotic_scale        = CONFIG["biplot"]["abiotic_scale"]
else:
    biotic_arrow_factor  = 1.0
    biotic_label_factor  = 1.05
    abiotic_scale        = 1.0

biotic_arrow_enlarge_factor  = CONFIG["biplot"].get("biotic_arrow_enlarge_factor", 1.8)
abiotic_arrow_enlarge_factor = CONFIG["biplot"].get("abiotic_arrow_enlarge_factor", 1.0)  # <-- usato qui

factor_to_use = None
for fcol in factor_vars:
    if fcol in factors_for_plot:
        factor_to_use = fcol
        break

if factor_to_use is not None:
    factor_series = factors_for_plot[factor_to_use]
    unique_factors = pd.unique(factor_series)
    colors = plt.cm.get_cmap('tab10', len(unique_factors))
    for idx, fct in enumerate(unique_factors):
        mask = (factor_series == fct).to_numpy()
        ax.scatter(scores_plot_df.loc[mask, 'RDA1'],
                   scores_plot_df.loc[mask, 'RDA2'],
                   label=str(fct), alpha=0.8, color=colors(idx))
    ax.legend(title=factor_to_use)
else:
    ax.scatter(scores_plot_df['RDA1'], scores_plot_df['RDA2'], color='grey', alpha=0.7, label='Samples')

for i, var in enumerate(loadings_plot_df.index):
    ax.arrow(0, 0,
             loadings_plot_df.iloc[i, 0] * biotic_arrow_factor * biotic_arrow_enlarge_factor,
             loadings_plot_df.iloc[i, 1] * biotic_arrow_factor * biotic_arrow_enlarge_factor,
             color='blue', head_width=0.08, head_length=0.10, alpha=0.8, length_includes_head=True)
    ax.text(loadings_plot_df.iloc[i, 0] * biotic_label_factor * biotic_arrow_enlarge_factor,
            loadings_plot_df.iloc[i, 1] * biotic_label_factor * biotic_arrow_enlarge_factor,
            var, color='blue', ha='center', va='center', fontsize=9)

for abi in abiotic_assoc.index:
    c1 = abiotic_assoc.columns[0] if len(abiotic_assoc.columns) > 0 else 'RDA1'
    c2 = abiotic_assoc.columns[1] if len(abiotic_assoc.columns) > 1 else c1
    corr1 = float(abiotic_assoc.loc[abi, c1])
    corr2 = float(abiotic_assoc.loc[abi, c2])

    ax.arrow(
        0, 0,
        corr1 * abiotic_scale * abiotic_arrow_enlarge_factor,
        corr2 * abiotic_scale * abiotic_arrow_enlarge_factor,
        color='red', head_width=0.05, head_length=0.07, alpha=0.7, length_includes_head=True
    )
    ax.text(
        corr1 * abiotic_scale * abiotic_arrow_enlarge_factor * 1.17,
        corr2 * abiotic_scale * abiotic_arrow_enlarge_factor * 1.17,
        abi, color='red', fontsize=9
    )

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(f"RDA1 ({per_axis_vs_total[0]*100:.1f}%)")
ax.set_ylabel(f"RDA2 ({per_axis_vs_total[1]*100:.1f}%)")
ax.set_title(CONFIG["plot"]["title"])
plt.tight_layout()
fig.savefig(os.path.join(output_rda, 'RDA_biplot.png'), dpi=CONFIG["plot"]["dpi"])
plt.show()

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
