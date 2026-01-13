import pandas as pd
import warnings
import os
from itertools import combinations
from scipy.stats import probplot
import numpy as np
import re
from patsy import build_design_matrices
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import csv
from scipy import stats

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


CONFIG = {

    "OUTPUT_CSV_SEP": ";",

    "ENABLE_VIF": True,
    "VIF_THRESHOLD": 5.0,

    "SCALING_MODE": "none",  # "none" | "mean" | "zscore" | "log"

    "ALPHA": 0.05,
    "ANOVA_TYPE": "auto",  # "I" | "II" | "III" | "auto"

    "INTERACTIONS": True,
    "INTERACTIONS_P": 0.05,
    "INTERACTIONS_FxC": True,
    "INTERACTIONS_FxF": False,

    "INTERACTION_WHITELIST": None,  # es. [("month","Temperature"), ("month","Salinity")]

    "ROBUST_COV_TYPE": "None",  # None/"none" to disable optional HC1, HC2 e HAC
    "HAC_MAXLAGS": 1,

    "ROBUST_ANOVA_LM": False,

    "PAIRWISE_ADJUST": "holm",
    "TUKEY_ENABLED": True,
    "TUKEY_MODE": "residualized",  # "raw" | "residualized"

    "PLOT_FORMAT": "png",
    "PLOT_DPI": 200,

    "LOG_LEVEL": "silent",  # "silent" | "info" | "debug"

    "CONTRASTS": "sum",  # "sum" (recommended) | "treatment"

    
    "PRE_OUTLIERS_ENABLED": False,
    "PRE_OUTLIERS_METHOD": "mad",  # "mad" | "iqr" | "zscore"
    "PRE_OUTLIERS_THR": 3.5,
    "PRE_OUTLIERS_BY_GROUP": False,
    "PRE_OUTLIERS_GROUP_FACTOR": "auto",
    "PRE_OUTLIERS_SAVE_CSV": True,

    
    "RUN_BACKWARD_SELECTION": True,
    "SELECTION_MODE": "BIC",
    "PVALUE_ALPHA": 0.05,
    "IC": "BIC",

    "PROTECT_INTERACTIONS": False,
}

def _log(level: str, msg: str):
    order = {"silent": 0, "info": 1, "debug": 2}
    if order.get(CONFIG["LOG_LEVEL"], 1) >= order.get(level, 1):
        print(msg)

output_dir = conf_output_path
output_ancova = os.path.join(output_dir, "ANCOVA")
os.makedirs(output_ancova, exist_ok=True)

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
    df_ = pd.read_csv(path, sep=sep, decimal=decimal, encoding='utf-8-sig')
    df_.columns = df_.columns.str.strip().str.replace('\uFEFF', '', regex=True)
    return df_

df = read_csv_auto(clean_path)

params = pd.read_csv(
    parameters_csv,
    sep=';', header=None, encoding='utf-8-sig'
)

columns = params.iloc[0].astype(str).str.strip().tolist()
labels  = params.iloc[1].astype(str).str.strip().tolist()

agg_row = 2 if params.shape[0] > 2 else None
agg_cols = [col for col, lab in zip(columns, params.iloc[agg_row]) if lab == 'A'] if agg_row is not None else []
aggregation_needed = len(agg_cols) > 0

biotic_vars  = [col for col, lab in zip(columns, labels) if lab == 'Y']
abiotic_vars = [col for col, lab in zip(columns, labels) if lab == 'X']
factor_vars  = [col for col, lab in zip(columns, labels) if lab == 'f']

n_species_col_name = 'N Species'
is_nspecies_biotic = n_species_col_name in biotic_vars

if aggregation_needed:
    _log("info", "Aggregation requested. Aggregating as specified in Parameter.csv...")
    biotic_cols_no_nspecies = [c for c in biotic_vars if c != n_species_col_name]
    agg_dict = {col: 'mean' for col in biotic_cols_no_nspecies + abiotic_vars}
    grouped = df.groupby(agg_cols).agg(agg_dict).reset_index()

    if is_nspecies_biotic:
        if 'acceptedNameUsage' in df.columns:
            n_species = (
                df.groupby(agg_cols)['acceptedNameUsage']
                  .nunique()
                  .reset_index(name=n_species_col_name)
            )
            grouped = pd.merge(grouped, n_species, on=agg_cols, how='left')
        else:
            _log("info", "   [WARN] 'acceptedNameUsage' not found: skipping 'N Species' computation.")

    keep_cols = [col for col in biotic_vars + abiotic_vars + factor_vars if col in grouped.columns]
    data_for_ancova = grouped[keep_cols].dropna()
else:
    _log("info", "No aggregation requested. Using original data table...")
    biotic_vars_actual = [v for v in biotic_vars if v in df.columns]
    data_for_ancova = df[biotic_vars_actual + abiotic_vars + factor_vars].dropna()

def q(col):
    return f'Q("{col}")'

def factor_term(f: str) -> str:
    c = (CONFIG.get("CONTRASTS", "sum") or "sum").lower()
    if c == "treatment":
        return f'C({q(f)})'
    return f'C({q(f)}, Sum)'

for f in factor_vars:
    if f in data_for_ancova.columns:
        data_for_ancova[f] = data_for_ancova[f].astype('category')

def sanitize_design(df_in, factors, covs):
    factors_ok = [f for f in factors if f in df_in.columns and df_in[f].nunique(dropna=True) >= 2]
    covs_ok = []
    for a in covs:
        if a in df_in.columns:
            s = pd.to_numeric(df_in[a], errors='coerce').std(ddof=0)
            if pd.notnull(s) and s > 0:
                covs_ok.append(a)
    return factors_ok, covs_ok

factor_vars, abiotic_vars = sanitize_design(data_for_ancova, factor_vars, abiotic_vars)

def transform_covariate(series: pd.Series, mode: str) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if mode == "none":
        return s
    elif mode == "mean":
        return s - s.mean()
    elif mode == "zscore":
        std = s.std(ddof=0)
        return (s - s.mean()) / std if pd.notnull(std) and std > 0 else s*0
    elif mode == "log":
        if (s <= -1).any():
            shift = 1 - s.min()
            return np.log1p(s + shift)
        return np.log1p(s)
    else:
        _log("info", f"[WARN] SCALING_MODE '{mode}' not recognized. Using 'none'.")
        return s

suffix_map = {"none": "", "mean": "_c", "zscore": "_z", "log": "_log"}
scale_suffix = suffix_map.get(CONFIG["SCALING_MODE"], "")

abiotic_vars_actual = [a for a in abiotic_vars if a in data_for_ancova.columns]
for a in abiotic_vars_actual:
    s_tr = transform_covariate(data_for_ancova[a], CONFIG["SCALING_MODE"])
    data_for_ancova[f"{a}{scale_suffix}"] = s_tr

def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

def anova_type(model, typ: int):
    """
    ANOVA with optional and configurable robustness.
    If ROBUST_ANOVA_LM=False -> classic anova_lm (identical to code 2).
    """
    try:
        if not np.isfinite(model.df_resid) or (model.df_resid <= 0):
            return None, 0

        robust_opt = None

        if CONFIG.get("ROBUST_ANOVA_LM", True):
            cov_type = CONFIG.get("ROBUST_COV_TYPE", None)
            if cov_type is not None:
                ct = str(cov_type).lower()
                if ct in ("hc0", "hc1", "hc2", "hc3"):
                    robust_opt = ct  # statsmodels wants lowercase

        if robust_opt is not None:
            tbl = anova_lm(model, typ=typ, robust=robust_opt)
        else:
            tbl = anova_lm(model, typ=typ)

        return tbl, typ
    except Exception:
        return None, 0

def anova_type3_with_safety(model):
    for t in (3, 2, 1):
        tbl, _ = anova_type(model, t)
        if tbl is not None:
            return tbl, t
    return pd.DataFrame({'note': ['ANOVA not computable for this model/data']}), 0

def run_anova_with_choice(model, choice: str):
    ch = (choice or "").upper()
    if ch == "I":
        return anova_type(model, 1)[0], 1
    if ch == "II":
        return anova_type(model, 2)[0], 2
    if ch == "III":
        return anova_type(model, 3)[0], 3
    return anova_type3_with_safety(model)

def calculate_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data["variable"] = X_df.columns
    vals = []
    for i in range(X_df.shape[1]):
        try:
            vals.append(variance_inflation_factor(X_df.values, i))
        except Exception:
            vals.append(np.inf)
    vif_data["VIF"] = vals
    return vif_data

def fit_ols(formula: str, data: pd.DataFrame):
    """
    Final fit for coefficients: remains robust if ROBUST_COV_TYPE is set.
    (as in code 2)
    """
    cov_type = CONFIG.get("ROBUST_COV_TYPE", None)
    if cov_type is None or str(cov_type).lower() in ("", "none", "off", "false"):
        return smf.ols(formula, data=data).fit()

    cov_type_up = str(cov_type).upper()
    cov_kwds = {}
    if cov_type_up == "HAC":
        cov_kwds = {"maxlags": int(CONFIG.get("HAC_MAXLAGS", 1))}
    return smf.ols(formula, data=data).fit(cov_type=cov_type_up, cov_kwds=cov_kwds)

def fit_ols_classic(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit()

def anova_on_classic(formula: str, data: pd.DataFrame):
    classic = fit_ols_classic(formula, data)
    tbl, typ = run_anova_with_choice(classic, CONFIG["ANOVA_TYPE"])
    return tbl, typ


def add_partial_eta_squared(anova_tbl: pd.DataFrame) -> pd.DataFrame:
    if anova_tbl is None or anova_tbl.empty:
        return anova_tbl

    tbl = anova_tbl.copy()

    resid_row = None
    for cand in ["Residual", "Residuals", "resid", "error"]:
        if cand in tbl.index:
            resid_row = cand
            break

    if resid_row is None or "sum_sq" not in tbl.columns:
        tbl["eta2_p"] = np.nan
        return tbl

    ss_error = tbl.loc[resid_row, "sum_sq"]
    if not np.isfinite(ss_error) or ss_error <= 0:
        tbl["eta2_p"] = np.nan
        return tbl

    eta_vals = []
    for idx, row in tbl.iterrows():
        if idx == resid_row:
            eta_vals.append(np.nan)
        else:
            ss_eff = row.get("sum_sq", np.nan)
            if np.isfinite(ss_eff):
                eta_vals.append(ss_eff / (ss_eff + ss_error))
            else:
                eta_vals.append(np.nan)

    tbl["eta2_p"] = eta_vals
    return tbl


def _split_formula_terms(formula: str):
    lhs, rhs = formula.split("~")
    lhs = lhs.strip()
    rhs_terms = [t.strip() for t in rhs.split("+") if t.strip()]
    return lhs, rhs_terms

def _get_ic_value(model, ic="AIC"):
    ic = (ic or "AIC").upper()
    return model.bic if ic == "BIC" else model.aic

def _identify_interaction_terms(terms):
    interactions = []
    mains = set()
    for t in terms:
        if ":" in t or "*" in t:
            interactions.append(t)
            clean = t.replace("*", ":")
            parts = [p.strip() for p in clean.split(":") if p.strip()]
            mains.update(parts)
    return interactions, mains

def backward_selection_pvalue(formula: str, data: pd.DataFrame, alpha=0.05,
                              protected_interactions=None):
    lhs, terms = _split_formula_terms(formula)
    if len(terms) <= 1:
        return formula, fit_ols_classic(formula, data)

    current_terms = terms[:]
    best_formula = formula
    best_model = fit_ols_classic(best_formula, data)

    while True:
        current_formula = f"{lhs} ~ " + " + ".join(current_terms)
        m = fit_ols_classic(current_formula, data)

        tbl, _ = run_anova_with_choice(m, "III")
        if tbl is None or "PR(>F)" not in tbl.columns:
            break

        removable_terms = current_terms[:]

        if protected_interactions:
            for t in protected_interactions:
                if t in removable_terms:
                    removable_terms.remove(t)

            _, protected_mains = _identify_interaction_terms(protected_interactions)
            for t in current_terms:
                for pm in protected_mains:
                    if t.startswith(pm) and t in removable_terms:
                        removable_terms.remove(t)

        pvals = tbl["PR(>F)"].copy()
        pvals = pvals.drop(labels=[x for x in pvals.index if str(x).lower().startswith("resid")],
                           errors="ignore")

        candidate_p = {}
        for idx, p in pvals.items():
            idx_str = str(idx)
            for r in removable_terms:
                if idx_str in r:
                    candidate_p[idx_str] = float(p)
                    break

        if not candidate_p:
            break

        worst_label = max(candidate_p, key=candidate_p.get)
        worst_p = candidate_p[worst_label]

        if worst_p <= alpha:
            best_formula, best_model = current_formula, m
            break

        removable = None
        for t in removable_terms:
            if worst_label in t:
                removable = t
                break

        if removable is None or len(current_terms) <= 1:
            break

        current_terms.remove(removable)
        best_formula, best_model = current_formula, m

    return best_formula, best_model

def backward_selection_ic(formula: str, data: pd.DataFrame, ic="AIC",
                          protected_interactions=None):
    lhs, terms = _split_formula_terms(formula)
    if len(terms) <= 1:
        return formula, fit_ols_classic(formula, data)

    current_terms = terms[:]
    current_formula = f"{lhs} ~ " + " + ".join(current_terms)
    current_model = fit_ols_classic(current_formula, data)
    current_ic = _get_ic_value(current_model, ic)

    while True:
        best_candidate = None
        best_ic = current_ic

        removable_terms = current_terms[:]
        if protected_interactions:
            for t in protected_interactions:
                if t in removable_terms:
                    removable_terms.remove(t)

            _, protected_mains = _identify_interaction_terms(protected_interactions)
            for t in current_terms:
                for pm in protected_mains:
                    if t.startswith(pm) and t in removable_terms:
                        removable_terms.remove(t)

        for t in removable_terms:
            cand_terms = [x for x in current_terms if x != t]
            cand_formula = f"{lhs} ~ " + " + ".join(cand_terms)
            try:
                cand_model = fit_ols_classic(cand_formula, data)
                cand_ic = _get_ic_value(cand_model, ic)
                if cand_ic < best_ic - 1e-9:
                    best_ic = cand_ic
                    best_candidate = (cand_terms, cand_formula, cand_model)
            except Exception:
                continue

        if best_candidate is None:
            break

        current_terms, current_formula, current_model = best_candidate
        current_ic = best_ic

    return current_formula, current_model

def run_backward_selection_if_enabled(formula: str, data: pd.DataFrame,
                                      protected_interactions=None):
    if not CONFIG.get("RUN_BACKWARD_SELECTION", False):
        return formula, None

    mode = (CONFIG.get("SELECTION_MODE", "IC") or "IC").upper()
    alpha = float(CONFIG.get("PVALUE_ALPHA", 0.05))
    ic = (CONFIG.get("IC", "AIC") or "AIC").upper()

    if mode == "PVALUE":
        return backward_selection_pvalue(
            formula, data, alpha=alpha,
            protected_interactions=protected_interactions
        )
    else:
        return backward_selection_ic(
            formula, data, ic=ic,
            protected_interactions=protected_interactions
        )


def prefilter_outliers_y(data: pd.DataFrame, y_col: str,
                         method: str = "mad",
                         thr: float = 3.5,
                         group_col: str | None = None):
    method = (method or "mad").lower()
    thr = float(thr)

    if group_col is None or group_col not in data.columns:
        groups = [("ALL", data)]
    else:
        groups = list(data.groupby(group_col, dropna=False))

    keep_idx, removed_idx = [], []
    for gname, g in groups:
        y = pd.to_numeric(g[y_col], errors="coerce")
        valid = y.dropna()
        if len(valid) < 3:
            keep_idx.extend(g.index.tolist())
            continue

        if method == "mad":
            med = np.median(valid)
            mad = np.median(np.abs(valid - med))
            mad = mad if mad > 0 else np.std(valid, ddof=0)
            z = 0.6745 * (y - med) / mad if mad > 0 else pd.Series(0, index=g.index)
            flag = np.abs(z) > thr

        elif method == "zscore":
            mu = valid.mean()
            sd = valid.std(ddof=0)
            z = (y - mu) / sd if sd > 0 else pd.Series(0, index=g.index)
            flag = np.abs(z) > thr

        elif method == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            low = q1 - thr * iqr
            high = q3 + thr * iqr
            flag = (y < low) | (y > high)

        else:
            raise ValueError(f"PRE_OUTLIERS_METHOD '{method}' not recognized.")

        removed_idx.extend(g.index[flag.fillna(False)].tolist())
        keep_idx.extend(g.index[~flag.fillna(False)].tolist())

    data_kept = data.loc[keep_idx].copy()
    data_removed = data.loc[removed_idx].copy()
    info = {
        "method": method,
        "thr": thr,
        "group_col": group_col,
        "n_in": len(data),
        "n_removed": len(data_removed),
        "n_out": len(data_kept),
    }
    return data_kept, data_removed, info


def _mode_or_first(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iat[0] if not m.empty else s.dropna().iloc[0]

def emms_for_factor(model, data, factor, covariates, other_factors, alpha=0.05):
    ref = {}
    for c in covariates:
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c]):
            ref[c] = float(pd.to_numeric(data[c], errors='coerce').mean())
        else:
            ref[c] = _mode_or_first(data[c]) if c in data.columns else 0.0

    for f in other_factors:
        if f in data.columns:
            ref[f] = _mode_or_first(data[f].astype('category'))

    if factor not in data.columns:
        raise ValueError(f"Factor '{factor}' not present in data")
    levs = pd.Categorical(data[factor]).categories
    if len(levs) < 2:
        raise ValueError(f"Factor '{factor}' has < 2 useful levels")

    rows = []
    for lev in levs:
        row = ref.copy()
        row[factor] = lev
        rows.append(row)
    new_df = pd.DataFrame(rows)

    design_info = model.model.data.design_info
    exog = build_design_matrices([design_info], new_df, return_type='dataframe')[0]

    pred = model.get_prediction(new_df).summary_frame(alpha=alpha)
    out = pd.DataFrame({
        factor: levs,
        "emm": pred["mean"].values,
        "se": pred["mean_se"].values,
        "ci_low": pred["mean_ci_lower"].values,
        "ci_high": pred["mean_ci_upper"].values
    })
    return out, exog, new_df

def pairwise_emm_contrasts(model, exog, levels, alpha=0.05, p_adjust="holm"):
    results = []
    for i, j in combinations(range(len(levels)), 2):
        L = exog.iloc[i].values - exog.iloc[j].values
        tt = model.t_test(L)
        diff = float(tt.effect)
        se   = float(tt.sd)
        tval = float(tt.tvalue)
        pval = float(tt.pvalue)
        df_  = float(model.df_resid)
        tcrit = stats.t.ppf(1 - alpha/2, df_) if df_ > 0 else np.nan
        ci_lo = diff - tcrit * se if np.isfinite(tcrit) else np.nan
        ci_hi = diff + tcrit * se if np.isfinite(tcrit) else np.nan
        results.append([levels[i], levels[j], diff, se, tval, df_, pval, ci_lo, ci_hi])

    pw = pd.DataFrame(results, columns=[
        "level_i", "level_j", "diff", "se", "t", "df_resid", "p_raw", "ci_low", "ci_high"
    ])
    if len(pw) > 0:
        pw["p_adj"] = multipletests(pw["p_raw"].values, method=p_adjust)[1]
        pw["alpha"] = alpha
        pw["reject_H0"] = pw["p_adj"] < alpha
    return pw

def run_tukey_for_factor(y_col: str, factor_name: str, data: pd.DataFrame,
                         covariates: list, all_factors: list, out_dir: str):
    mode = (CONFIG.get("TUKEY_MODE", "residualized") or "residualized").lower()
    needed_cols = [y_col, factor_name] + covariates + [f for f in all_factors if f != factor_name]
    needed_cols = [c for c in needed_cols if c in data.columns]
    d = data[needed_cols].dropna().copy()
    if len(d) == 0:
        return

    if mode == "raw" or (not covariates and len(all_factors) <= 1):
        endog = pd.to_numeric(d[y_col], errors="coerce")
        groups = d[factor_name].astype("category")
    else:
        other_factors = [f for f in all_factors if f != factor_name and f in d.columns]
        main_covs    = [q(c) for c in covariates if c in d.columns]
        main_factors = [factor_term(f) for f in other_factors]
        base_terms   = main_covs + main_factors
        formula = (f"{q(y_col)} ~ " + " + ".join(base_terms)) if base_terms else f"{q(y_col)} ~ 1"
        base_model = fit_ols(formula, d)
        endog = base_model.resid
        groups = d[factor_name].astype("category")

    try:
        tk = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=CONFIG["ALPHA"])
        summ = tk.summary()
        data_rows = summ.data[1:]
        headers = summ.data[0]
        tukey_df = pd.DataFrame(data_rows, columns=headers)

        os.makedirs(out_dir, exist_ok=True)
        tukey_df.to_csv(
            os.path.join(out_dir, f"{y_col}_Tukey_{safe_name(factor_name)}_{mode}.csv"),
            index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
        )
        with open(os.path.join(out_dir, f"{y_col}_Tukey_{safe_name(factor_name)}_{mode}.txt"),
                  "w", encoding="utf-8") as ftxt:
            ftxt.write(str(summ))
    except Exception as e:
        _log("info", f"[WARN] Tukey HSD failed for {y_col} ~ {factor_name}: {e}")


def plot_ancova_lines_with_ci(
    data, model, y_col, factor_vars, abiotic_vars, out_dir,
    covar_name=None, n_points=120, alpha_ci=0.05,
    plot_format="png", plot_dpi=200
):
    if not factor_vars:
        return
    fac = factor_vars[0]
    if fac not in data.columns:
        return

    if not pd.api.types.is_categorical_dtype(data[fac]):
        data = data.copy()
        data[fac] = data[fac].astype("category")

    if covar_name is None:
        if not abiotic_vars:
            return
        covar_name = abiotic_vars[0]

    suffixes = ["_c", "_z", "_log", ""]
    covar_transformed = None
    for sfx in suffixes:
        cand = covar_name + sfx
        if cand in data.columns:
            covar_transformed = cand
            break
    if covar_transformed is None:
        return

    levels = pd.Categorical(data[fac]).categories
    x_obs = pd.to_numeric(data[covar_transformed], errors="coerce").dropna()
    if len(x_obs) == 0:
        return

    xgrid = np.linspace(x_obs.min(), x_obs.max(), n_points)
    new_base = {}

    for a in abiotic_vars:
        for sfx in suffixes:
            cand = a + sfx
            if cand == covar_transformed:
                continue
            if cand in data.columns:
                if pd.api.types.is_numeric_dtype(data[cand]):
                    new_base[cand] = float(pd.to_numeric(data[cand], errors="coerce").mean())
                else:
                    m = data[cand].mode(dropna=True)
                    new_base[cand] = m.iat[0] if not m.empty else data[cand].dropna().iloc[0]
                break

    for f in factor_vars:
        if f == fac:
            continue
        m = data[f].mode(dropna=True)
        new_base[f] = m.iat[0] if not m.empty else data[f].dropna().iloc[0]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=pd.to_numeric(data[covar_transformed], errors="coerce"),
        y=data[y_col],
        hue=data[fac],
        alpha=0.3,
        legend=False
    )

    line_handles = []
    for lev in levels:
        new_df = pd.DataFrame({**new_base, fac: [lev] * len(xgrid)})
        new_df[covar_transformed] = xgrid

        pred = model.get_prediction(new_df)
        sf = pred.summary_frame(alpha=alpha_ci)
        h_line, = plt.plot(xgrid, sf["mean"], label=str(lev))
        plt.fill_between(xgrid, sf["mean_ci_lower"], sf["mean_ci_upper"], alpha=0.15)
        line_handles.append(h_line)

    plt.xlabel(covar_transformed)
    plt.ylabel(y_col)
    plt.title(f"Regressions by level of {fac} with {int((1-alpha_ci)*100)}% CI – {y_col}")
    plt.legend(handles=line_handles, title=f"{fac} (ANCOVA factor)", loc="best")
    plt.tight_layout()

    ext = plot_format.lower()
    fname = f"{y_col}_Lines_by_{fac}_vs_{safe_name(covar_transformed)}.{ext}".replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname), dpi=plot_dpi, format=ext)
    plt.close()

def plot_emm_points_for_factor(emms_tbl: pd.DataFrame, factor: str, y_col: str,
                               out_dir: str, plot_format: str = "png", plot_dpi: int = 200):
    if emms_tbl is None or emms_tbl.empty:
        return

    levs = list(emms_tbl[factor].astype(str))
    x = np.arange(len(levs))
    y = emms_tbl["emm"].values
    ylow = emms_tbl["ci_low"].values
    yhigh = emms_tbl["ci_high"].values
    yerr = np.vstack([y - ylow, yhigh - y])

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=5)
    plt.xticks(x, levs, rotation=0)
    plt.xlabel(factor)
    plt.ylabel(f"EMM of {y_col}")
    plt.title(f"EMM (LS-means) with CI across {factor} levels – {y_col}")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    ext = plot_format.lower()
    plt.savefig(os.path.join(out_dir, f"{y_col}_EMMplot_{safe_name(factor)}.{ext}"),
                dpi=plot_dpi, format=ext)
    plt.close()


summary_list = []
pairwise_all_rows = []
vif_exclusion_global_log = []

for y_col in [c for c in biotic_vars if c in data_for_ancova.columns and c != n_species_col_name]:
    _log("info", f"ANCOVA for: {y_col}")

    y_root = os.path.join(output_ancova, safe_name(y_col))
    os.makedirs(y_root, exist_ok=True)

    dirs = {
        "ANOVA": os.path.join(y_root, "01_ANCOVA"),
        "VIF": os.path.join(y_root, "02_VIF"),
        "DIAGNOSTICS": os.path.join(y_root, "03_DIAGNOSTICS"),
        "PLOTS": os.path.join(y_root, "04_PLOTS"),
        "PLOTS_LINES": os.path.join(y_root, "04_PLOTS", "ANCOVA_Lines"),
        "PLOTS_EMM": os.path.join(y_root, "04_PLOTS", "Factor_EMM_Plots"),
        "POSTHOC": os.path.join(y_root, "05_POSTHOC"),
        "POSTHOC_EMM": os.path.join(y_root, "05_POSTHOC", "EMMs"),
        "POSTHOC_PAIRWISE": os.path.join(y_root, "05_POSTHOC", "Pairwise"),
        "POSTHOC_TUKEY": os.path.join(y_root, "05_POSTHOC", "TukeyHSD"),
        "DATA": os.path.join(y_root, "00_DATA"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    factor_vars_actual = [f for f in factor_vars
                          if f in data_for_ancova.columns
                          and data_for_ancova[f].nunique() >= 2]
    abio_vars_used = [a for a in abiotic_vars if a in data_for_ancova.columns]

    covariate_transformed = [
        f"{a}{scale_suffix}" for a in abio_vars_used
        if f"{a}{scale_suffix}" in data_for_ancova.columns
    ]

    base_cols = [y_col] + factor_vars_actual + covariate_transformed
    base_cols = [c for c in base_cols if c in data_for_ancova.columns]
    data_y = data_for_ancova[base_cols].dropna().copy()
    N = len(data_y)

    if CONFIG.get("PRE_OUTLIERS_ENABLED", False):
        method = CONFIG.get("PRE_OUTLIERS_METHOD", "mad")
        thr = CONFIG.get("PRE_OUTLIERS_THR", 3.5)

        group_col = None
        if CONFIG.get("PRE_OUTLIERS_BY_GROUP", True) and factor_vars_actual:
            gfac = CONFIG.get("PRE_OUTLIERS_GROUP_FACTOR", "auto")
            if str(gfac).lower() == "auto":
                group_col = factor_vars_actual[0]
            elif gfac in data_y.columns:
                group_col = gfac

        data_y_filt, data_y_removed, pre_info = prefilter_outliers_y(
            data=data_y, y_col=y_col, method=method, thr=thr, group_col=group_col
        )

        _log("info",
             f"   Pre-outliers ({pre_info['method']}, thr={pre_info['thr']}, group={pre_info['group_col']}): "
             f"removed {pre_info['n_removed']} / {pre_info['n_in']}")

        data_y = data_y_filt
        N = len(data_y)

        if CONFIG.get("PRE_OUTLIERS_SAVE_CSV", True):
            data_y.to_csv(os.path.join(dirs["DATA"], f"{safe_name(y_col)}_data_AFTER_preoutliers.csv"),
                          index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"])
            if len(data_y_removed) > 0:
                data_y_removed.to_csv(os.path.join(dirs["DATA"], f"{safe_name(y_col)}_preoutliers_REMOVED.csv"),
                                      index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"])
            with open(os.path.join(dirs["DATA"], f"{safe_name(y_col)}_preoutliers_info.txt"), "w", encoding="utf-8") as f:
                f.write(str(pre_info))

    vif_history = []
    exclusion_steps = []
    if CONFIG["ENABLE_VIF"] and len(abio_vars_used) > 1 and N > 0:
        X_abio = data_for_ancova.loc[data_y.index, abio_vars_used].copy()
        X_abio = X_abio.apply(pd.to_numeric, errors="coerce").dropna()

        if len(X_abio) > 0:
            while len(X_abio.columns) > 1:
                vif_df = calculate_vif(X_abio)
                vif_history.append(vif_df.copy())
                max_vif = vif_df["VIF"].max()
                if max_vif > CONFIG["VIF_THRESHOLD"]:
                    drop_var = vif_df.sort_values("VIF", ascending=False)["variable"].iloc[0]
                    exclusion_steps.append(
                        f"Excluded: {drop_var} (VIF={max_vif:.2f} > {CONFIG['VIF_THRESHOLD']})"
                    )
                    _log("debug", f"   Dropping '{drop_var}' (VIF={max_vif:.2f}) for {y_col}")
                    abio_vars_used.remove(drop_var)
                    X_abio = X_abio[abio_vars_used]
                else:
                    break
    elif not CONFIG["ENABLE_VIF"]:
        _log("info", "   VIF check DISABLED.")

    if vif_history:
        vif_history[0].to_csv(os.path.join(dirs["VIF"], f"{safe_name(y_col)}_VIF_initial.csv"),
                              index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"])
        vif_history[-1].to_csv(os.path.join(dirs["VIF"], f"{safe_name(y_col)}_VIF_final.csv")),
                              
    with open(os.path.join(dirs["VIF"], f"{safe_name(y_col)}_vif_exclusion_steps.txt"), "w", encoding="utf-8") as flog:
        if exclusion_steps:
            flog.write("\n".join(exclusion_steps))
        else:
            if CONFIG["ENABLE_VIF"] and len(abio_vars_used) > 1:
                flog.write(f"No exclusions. All VIF <= {CONFIG['VIF_THRESHOLD']} or <= 1 covariate available.")
            elif not CONFIG["ENABLE_VIF"]:
                flog.write("VIF check DISABLED by user configuration.")
            else:
                flog.write("No exclusions. All VIF <= threshold or <= 1 covariate available.")

    vif_exclusion_global_log.append(f"{y_col}:")
    if exclusion_steps:
        vif_exclusion_global_log.extend(exclusion_steps)
    else:
        if CONFIG["ENABLE_VIF"] and len(abio_vars_used) > 1:
            vif_exclusion_global_log.append(
                f"  No exclusions. All VIF <= {CONFIG['VIF_THRESHOLD']} or <= 1 covariate available."
            )
        elif not CONFIG["ENABLE_VIF"]:
            vif_exclusion_global_log.append("  VIF check DISABLED by user configuration.")
        else:
            vif_exclusion_global_log.append("  No exclusions. All VIF <= threshold or <= 1 covariate available.")
    vif_exclusion_global_log.append("")

    covariate_transformed = [
        f"{a}{scale_suffix}" for a in abio_vars_used
        if f"{a}{scale_suffix}" in data_for_ancova.columns
    ]

    model_cols = [y_col] + factor_vars_actual + covariate_transformed
    model_cols = [c for c in model_cols if c in data_y.columns]
    data_model = data_y[model_cols].dropna().copy()
    N = len(data_model)

    decided_formula = None
    decided_model = None
    decided_type = None
    anova_table = None

    min_needed_params = 1 + len(factor_vars_actual) + len(covariate_transformed)
    main_factors = [factor_term(f) for f in factor_vars_actual]
    main_covs = [q(c) for c in covariate_transformed]

    if not factor_vars_actual and not covariate_transformed:
        decided_formula = f"{q(y_col)} ~ 1"
        anova_table, decided_type = anova_on_classic(decided_formula, data_model)
        has_sig_inter = False
        sig_interactions = []
        inter_terms = []

    else:
        base_terms = main_factors + main_covs
        formula_reduced = f"{q(y_col)} ~ " + " + ".join(base_terms) if base_terms else f"{q(y_col)} ~ 1"

        has_sig_inter = False
        sig_interactions = []
        inter_terms = []

        if (
            CONFIG.get("INTERACTIONS", True)
            and factor_vars_actual
            and covariate_transformed
            and N > (min_needed_params + 2)
        ):
            if CONFIG.get("INTERACTIONS_FxC", True):
                wl = CONFIG.get("INTERACTION_WHITELIST", None)

                for f_raw, f_term in zip(factor_vars_actual, main_factors):
                    for c_raw, c_term in zip(covariate_transformed, main_covs):
                        if wl is not None:
                            cov_raw_name = c_raw.replace(scale_suffix, "")
                            if (f_raw, cov_raw_name) not in wl:
                                continue
                        inter_terms.append(f"{f_term}:{c_term}")

            if CONFIG.get("INTERACTIONS_FxF", False) and len(main_factors) > 1:
                for i, fi in enumerate(main_factors):
                    for j, fj in enumerate(main_factors):
                        if j > i:
                            inter_terms.append(f"{fi}:{fj}")

            if inter_terms:
                formula_full = f"{q(y_col)} ~ " + " + ".join(base_terms + inter_terms)
                anova_full_tbl, type_full = anova_on_classic(formula_full, data_model)

                if (
                    isinstance(anova_full_tbl, pd.DataFrame)
                    and "PR(>F)" in anova_full_tbl.columns
                    and type_full in (1, 2, 3)
                ):
                    for idx, p in anova_full_tbl["PR(>F)"].items():
                        idx_str = str(idx)
                        if ":" in idx_str and pd.notnull(p) and (p < CONFIG["INTERACTIONS_P"]):
                            sig_interactions.append(idx_str)

                    has_sig_inter = len(sig_interactions) > 0

                if has_sig_inter:
                    decided_formula = formula_full
                    anova_table = anova_full_tbl
                    decided_type = type_full
                    _log("info", f"   Significant interactions kept: {sig_interactions}")
                else:
                    decided_formula = formula_reduced
                    anova_table, decided_type = anova_on_classic(decided_formula, data_model)
                    _log("info", "   No significant interactions: using REDUCED model.")
            else:
                decided_formula = formula_reduced
                anova_table, decided_type = anova_on_classic(decided_formula, data_model)
                _log("info", "   No interaction terms built: using REDUCED model.")

        else:
            decided_formula = formula_reduced
            anova_table, decided_type = anova_on_classic(decided_formula, data_model)
            _log("info", "   Thin data or interactions OFF: direct REDUCED model.")

    if CONFIG.get("PROTECT_INTERACTIONS", True) and has_sig_inter:
        protected_interactions = sig_interactions[:]
    else:
        protected_interactions = []

    sel_formula, _ = run_backward_selection_if_enabled(
        decided_formula, data_model, protected_interactions=protected_interactions
    )
    if sel_formula != decided_formula:
        _log("info", f"   Backward selection: {decided_formula}  ->  {sel_formula}")
        decided_formula = sel_formula
        anova_table, decided_type = anova_on_classic(decided_formula, data_model)

    decided_model = fit_ols(decided_formula, data_model)

    classic_model_for_diag = fit_ols_classic(decided_formula, data_model)

    type_label = {3: "typeIII", 2: "typeII", 1: "typeI", 0: "typeNA"}.get(decided_type, "typeNA")
    anova_fname = f"{safe_name(y_col)}_ANCOVA_{type_label}_FINAL.csv"

    anova_table = add_partial_eta_squared(anova_table)

    anova_table.to_csv(os.path.join(dirs["ANOVA"], anova_fname),
                       encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"])

    y_pred = decided_model.predict(data_model)
    resid = decided_model.resid
    ext = CONFIG["PLOT_FORMAT"].lower()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=data_model[y_col], y=y_pred)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Observed vs Predicted - {y_col}")
    plt.plot(
        [data_model[y_col].min(), data_model[y_col].max()],
        [data_model[y_col].min(), data_model[y_col].max()],
        "k--", alpha=0.6
    )
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["PLOTS"], f"{safe_name(y_col)}_Observed_vs_Predicted.{ext}"),
                dpi=CONFIG["PLOT_DPI"], format=ext)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=resid)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted - {y_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["PLOTS"], f"{safe_name(y_col)}_Residuals_vs_Predicted.{ext}"),
                dpi=CONFIG["PLOT_DPI"], format=ext)
    plt.close()

    plt.figure(figsize=(6, 4))
    probplot(resid, dist="norm", plot=plt)
    plt.title(f"QQ plot of residuals - {y_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["PLOTS"], f"{safe_name(y_col)}_QQplot_Residuals.{ext}"),
                dpi=CONFIG["PLOT_DPI"], format=ext)
    plt.close()

    n_resid = len(resid)
    if 3 <= n_resid <= 5000:
        shapiro_stat, shapiro_p = shapiro(resid)
        shapiro_note = ""
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        shapiro_note = "Shapiro not performed: sample size out of bounds."

    with open(os.path.join(dirs["DIAGNOSTICS"], f"{safe_name(y_col)}_Shapiro_test.txt"),
              "w", encoding="utf-8") as ftest:
        ftest.write("Shapiro–Wilk test for normality of residuals:\n")
        if np.isnan(shapiro_stat):
            ftest.write(shapiro_note + "\n")
        else:
            ftest.write(f"Statistic: {shapiro_stat:.4f}\n")
            ftest.write(f"P-value: {shapiro_p:.4g}\n")

    try:
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
            classic_model_for_diag.resid,
            classic_model_for_diag.model.exog
        )
    except Exception as e:
        bp_lm, bp_lm_p, bp_f, bp_f_p = np.nan, np.nan, np.nan, np.nan
        _log("info", f"[WARN] Breusch–Pagan test failed for {y_col}: {e}")

    with open(os.path.join(dirs["DIAGNOSTICS"], f"{safe_name(y_col)}_BreuschPagan_test.txt"),
              "w", encoding="utf-8") as fbp:
        fbp.write("Breusch–Pagan test for heteroscedasticity (Pagan test):\n")
        fbp.write(f"LM statistic: {bp_lm:.4f}\n")
        fbp.write(f"LM p-value:  {bp_lm_p:.4g}\n")
        fbp.write(f"F statistic: {bp_f:.4f}\n")
        fbp.write(f"F p-value:  {bp_f_p:.4g}\n")

    with open(os.path.join(dirs["DIAGNOSTICS"], f"{safe_name(y_col)}_ancova_summary.txt"),
              "w", encoding="utf-8") as fsum:
        fsum.write(decided_model.summary().as_text())

    with open(os.path.join(dirs["DIAGNOSTICS"], "diagnostics.txt"), "w", encoding="utf-8") as fd:
        fd.write(f"N = {N}\n")
        fd.write(f"df_resid = {decided_model.df_resid:.1f}\n")
        fd.write(f"Final formula: {decided_formula}\n")
        fd.write(f"Factors used: {factor_vars_actual}\n")
        fd.write(f"Transformed covariates used: {covariate_transformed}\n")
        fd.write(f"ANOVA type: {type_label}\n")
        fd.write(f"Backward selection enabled: {CONFIG.get('RUN_BACKWARD_SELECTION', False)}\n")
        if CONFIG.get("RUN_BACKWARD_SELECTION", False):
            fd.write(f"   selection_mode={CONFIG.get('SELECTION_MODE')}, "
                     f"pvalue_alpha={CONFIG.get('PVALUE_ALPHA')}, ic={CONFIG.get('IC')}\n")
            fd.write(f"   PROTECT_INTERACTIONS={CONFIG.get('PROTECT_INTERACTIONS', True)}\n")
            fd.write(f"   protected_interactions={protected_interactions}\n")
        fd.write(f"Pre-outliers enabled: {CONFIG.get('PRE_OUTLIERS_ENABLED', False)}\n")
        if CONFIG.get("PRE_OUTLIERS_ENABLED", False):
            fd.write(f"   method={CONFIG.get('PRE_OUTLIERS_METHOD')}, thr={CONFIG.get('PRE_OUTLIERS_THR')}, "
                     f"by_group={CONFIG.get('PRE_OUTLIERS_BY_GROUP')}\n")
        fd.write("\nBreusch–Pagan:\n")
        fd.write(f"   LM={bp_lm:.4f}, p={bp_lm_p:.4g}, F={bp_f:.4f}, p={bp_f_p:.4g}\n")

    eta2_p_max = np.nan
    try:
        if isinstance(anova_table, pd.DataFrame) and "eta2_p" in anova_table.columns:
            eta2_p_max = anova_table["eta2_p"].dropna().max()
    except Exception:
        eta2_p_max = np.nan

    coef_dict = {f"coef_{k}": v for k, v in decided_model.params.items()}
    coef_dict.update({
        "biotic_var": y_col,
        "r2": decided_model.rsquared,
        "adj_r2": decided_model.rsquared_adj,
        "model_pvalue": decided_model.f_pvalue,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "bp_lm_stat": bp_lm,
        "bp_lm_p": bp_lm_p,
        "bp_f_stat": bp_f,
        "bp_f_p": bp_f_p,
        "heteroscedasticity_reject": (bp_lm_p < CONFIG["ALPHA"]) if np.isfinite(bp_lm_p) else np.nan,
        "formula": decided_model.model.formula if hasattr(decided_model, "model") else "NA",
        "anova_type": type_label,
        "predictors_after_vif": ",".join(factor_vars_actual + covariate_transformed),
        "eta2_p_max": eta2_p_max,
    })
    summary_list.append(coef_dict)

    abiotic_vars_actual_loop = [a for a in abio_vars_used if a in data_for_ancova.columns]
    factor_vars_actual_loop = [f for f in factor_vars
                               if f in data_for_ancova.columns and data_for_ancova[f].nunique() >= 2]

    covs_for_plots = []
    for a in abiotic_vars_actual_loop:
        cand_cols = [f"{a}{scale_suffix}", f"{a}_c", f"{a}_z", f"{a}_log", a]
        for cc in cand_cols:
            if cc in data_for_ancova.columns:
                series = pd.to_numeric(data_for_ancova[cc], errors="coerce")
                if pd.api.types.is_numeric_dtype(series):
                    covs_for_plots.append(a)
                break

    for fac in factor_vars_actual_loop:
        factor_order = [fac] + [f for f in factor_vars_actual_loop if f != fac]
        for cov in covs_for_plots:
            plot_ancova_lines_with_ci(
                data=data_model,
                model=decided_model,
                y_col=y_col,
                factor_vars=factor_order,
                abiotic_vars=abiotic_vars_actual_loop,
                out_dir=dirs["PLOTS_LINES"],
                covar_name=cov,
                n_points=120,
                alpha_ci=CONFIG["ALPHA"],
                plot_format=CONFIG["PLOT_FORMAT"],
                plot_dpi=CONFIG["PLOT_DPI"]
            )

    for fac_posthoc in factor_vars_actual:
        try:
            covs_in_model = [c for c in data_model.columns if c not in [y_col] + factor_vars_actual]

            emms_tbl, exog_emm, _ = emms_for_factor(
                model=decided_model,
                data=data_model,
                factor=fac_posthoc,
                covariates=covs_in_model,
                other_factors=[f for f in factor_vars_actual if f != fac_posthoc],
                alpha=CONFIG["ALPHA"]
            )

            emms_tbl.to_csv(
                os.path.join(dirs["POSTHOC_EMM"], f"{safe_name(y_col)}_EMMs_{safe_name(fac_posthoc)}.csv"),
                index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
            )

            plot_emm_points_for_factor(
                emms_tbl=emms_tbl,
                factor=fac_posthoc,
                y_col=y_col,
                out_dir=dirs["PLOTS_EMM"],
                plot_format=CONFIG["PLOT_FORMAT"],
                plot_dpi=CONFIG["PLOT_DPI"]
            )

            levels = list(emms_tbl[fac_posthoc].astype(str))
            pw = pairwise_emm_contrasts(
                model=decided_model,
                exog=exog_emm,
                levels=levels,
                alpha=CONFIG["ALPHA"],
                p_adjust=CONFIG["PAIRWISE_ADJUST"]
            )

            pw.to_csv(
                os.path.join(dirs["POSTHOC_PAIRWISE"],
                             f"{safe_name(y_col)}_PairwiseEMM_{safe_name(fac_posthoc)}_{CONFIG['PAIRWISE_ADJUST']}.csv"),
                index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
            )

            if len(pw) > 0:
                pw_glob = pw.copy()
                pw_glob.insert(0, "factor", fac_posthoc)
                pw_glob.insert(0, "biotic_var", y_col)
                pw_glob.insert(0, "method", f"EMM_{CONFIG['PAIRWISE_ADJUST'].upper()}")
                pairwise_all_rows.append(pw_glob)

        except Exception as e:
            _log("info", f"[WARN] EMMs/pairwise failed for factor {fac_posthoc} on {y_col}: {e}")

    if CONFIG.get("TUKEY_ENABLED", True) and factor_vars_actual_loop:
        for fac in factor_vars_actual_loop:
            run_tukey_for_factor(
                y_col=y_col,
                factor_name=fac,
                data=data_model,
                covariates=covariate_transformed,
                all_factors=factor_vars_actual_loop,
                out_dir=dirs["POSTHOC_TUKEY"],
            )


global_dirs = {
    "GLOBAL": os.path.join(output_ancova, "_GLOBAL"),
}
for d in global_dirs.values():
    os.makedirs(d, exist_ok=True)

with open(os.path.join(global_dirs["GLOBAL"], "vif_exclusion_log.txt"), "w", encoding="utf-8") as flog:
    flog.write("\n".join(vif_exclusion_global_log))

results_df = pd.DataFrame(summary_list)
results_df.to_csv(
    os.path.join(global_dirs["GLOBAL"], "ancova_summary_all.csv"),
    index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
)

if len(pairwise_all_rows) > 0:
    pairwise_all_df = pd.concat(pairwise_all_rows, ignore_index=True)
    pairwise_all_df.to_csv(
        os.path.join(global_dirs["GLOBAL"], "pairwise_all_factors_all_Y.csv"),
        index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
    )

print("All outputs saved to:", output_ancova)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
