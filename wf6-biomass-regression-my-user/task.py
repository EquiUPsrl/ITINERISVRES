import pandas as pd
import warnings
import os
from scipy.stats import probplot
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_transformation_path', action='store', type=str, required=True, dest='data_transformation_path')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

data_transformation_path = args.data_transformation_path.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'

warnings.filterwarnings("ignore")


CONFIG = {
    "OUTPUT_CSV_SEP": ";",

    "ENABLE_VIF": True,
    "VIF_THRESHOLD": 5.0,

    "SCALING_MODE": "none",  # "none" | "mean" | "zscore" | "log"

    "ALPHA": 0.05,

    "INTERACTIONS": True,
    "INTERACTIONS_FxC": True,
    "INTERACTIONS_FxF": False,
    "INTERACTION_WHITELIST": None,  # es. [("month","Temperature")]

    "ROBUST_COV_TYPE": "HC3",  # None/"none" to disable; HC0/HC1/HC2/HC3/HAC
    "HAC_MAXLAGS": 1,

    "RUN_BACKWARD_SELECTION": True,
    "SELECTION_MODE": "BIC",   # "PVALUE" | "AIC"/"BIC"
    "PVALUE_ALPHA": 0.05,
    "IC": "BIC",

    "PLOT_FORMAT": "png",
    "PLOT_DPI": 200,

    "LOG_LEVEL": "silent",  # "silent" | "info" | "debug"

    "CONTRASTS": "sum",  # "sum" (recommended) | "treatment"

    "PARAM_COLUMNS_ROW": 0,  # row with column names
    "PARAM_LABELS_ROW": 4,   # 5th row (index 4) with X/Y/f
}

def _log(level: str, msg: str):
    order = {"silent": 0, "info": 1, "debug": 2}
    if order.get(CONFIG["LOG_LEVEL"], 1) >= order.get(level, 1):
        print(msg)

output_dir = conf_output_path
output_regression = os.path.join(output_dir, "REGRESSION")
os.makedirs(output_regression, exist_ok=True)

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

df = read_csv_auto(data_transformation_path)

params = pd.read_csv(
    parameters_csv,
    sep=';', header=None, encoding='utf-8-sig'
)

col_row = int(CONFIG.get("PARAM_COLUMNS_ROW", 0))
lab_row = int(CONFIG.get("PARAM_LABELS_ROW", 4))

if params.shape[0] <= col_row:
    raise ValueError("Parameter.csv: missing columns row.")
columns = params.iloc[col_row].astype(str).str.strip().tolist()

if params.shape[0] > lab_row:
    labels = params.iloc[lab_row].astype(str).str.strip().tolist()
else:
    _log("info", f"[WARN] Parameter.csv has no row {lab_row+1}; fallback to row 2.")
    labels = params.iloc[1].astype(str).str.strip().tolist()

agg_row = None
for r in range(params.shape[0]):
    if r in (col_row, lab_row):
        continue
    row_vals = params.iloc[r].astype(str).str.strip().tolist()
    if any(v.upper() == "A" for v in row_vals):
        agg_row = r
        break

agg_cols = []
if agg_row is not None:
    agg_cols = [col for col, lab in zip(columns, params.iloc[agg_row]) if str(lab).strip().upper() == 'A']

aggregation_needed = len(agg_cols) > 0

biotic_vars  = [col for col, lab in zip(columns, labels) if str(lab).strip() == 'Y']
abiotic_vars = [col for col, lab in zip(columns, labels) if str(lab).strip() == 'X']
factor_vars  = [col for col, lab in zip(columns, labels) if str(lab).strip() == 'f']

n_species_col_name = 'N Species'
is_nspecies_biotic = n_species_col_name in biotic_vars

if aggregation_needed:
    _log("info", f"Aggregation requested using row {agg_row+1}. Grouping by: {agg_cols}")
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
    data_for_model = grouped[keep_cols].dropna()
else:
    _log("info", "No aggregation requested. Using original data table...")
    biotic_vars_actual = [v for v in biotic_vars if v in df.columns]
    data_for_model = df[biotic_vars_actual + abiotic_vars + factor_vars].dropna()

def q(col):
    return f'Q("{col}")'

def factor_term(f: str) -> str:
    c = (CONFIG.get("CONTRASTS", "sum") or "sum").lower()
    if c == "treatment":
        return f'C({q(f)})'
    return f'C({q(f)}, Sum)'

for f in factor_vars:
    if f in data_for_model.columns:
        data_for_model[f] = data_for_model[f].astype('category')

def sanitize_design(df_in, factors, covs):
    factors_ok = [f for f in factors if f in df_in.columns and df_in[f].nunique(dropna=True) >= 2]
    covs_ok = []
    for a in covs:
        if a in df_in.columns:
            s = pd.to_numeric(df_in[a], errors='coerce').std(ddof=0)
            if pd.notnull(s) and s > 0:
                covs_ok.append(a)
    return factors_ok, covs_ok

factor_vars, abiotic_vars = sanitize_design(data_for_model, factor_vars, abiotic_vars)

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

abiotic_vars_actual = [a for a in abiotic_vars if a in data_for_model.columns]
for a in abiotic_vars_actual:
    s_tr = transform_covariate(data_for_model[a], CONFIG["SCALING_MODE"])
    data_for_model[f"{a}{scale_suffix}"] = s_tr

def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

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
    """Final fit for inference (robust if requested)."""
    cov_type = CONFIG.get("ROBUST_COV_TYPE", None)
    if cov_type is None or str(cov_type).lower() in ("", "none", "off", "false"):
        return smf.ols(formula, data=data).fit()

    cov_type_up = str(cov_type).upper()
    cov_kwds = {}
    if cov_type_up == "HAC":
        cov_kwds = {"maxlags": int(CONFIG.get("HAC_MAXLAGS", 1))}
    return smf.ols(formula, data=data).fit(cov_type=cov_type_up, cov_kwds=cov_kwds)

def fit_ols_classic(formula: str, data: pd.DataFrame):
    """Classic fit for diagnostics (non-robust)."""
    return smf.ols(formula, data=data).fit()


def _split_formula_terms(formula: str):
    lhs, rhs = formula.split("~")
    lhs = lhs.strip()
    rhs_terms = [t.strip() for t in rhs.split("+") if t.strip()]
    return lhs, rhs_terms

def _get_ic_value(model, ic="AIC"):
    ic = (ic or "AIC").upper()
    return model.bic if ic == "BIC" else model.aic

def backward_selection_ic(formula: str, data: pd.DataFrame, ic="AIC"):
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

        for t in current_terms:
            cand_terms = [x for x in current_terms if x != t]
            cand_formula = f"{lhs} ~ " + " + ".join(cand_terms) if cand_terms else f"{lhs} ~ 1"
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

def run_backward_selection_if_enabled(formula: str, data: pd.DataFrame):
    if not CONFIG.get("RUN_BACKWARD_SELECTION", False):
        return formula, None
    ic = (CONFIG.get("IC", "AIC") or "AIC").upper()
    return backward_selection_ic(formula, data, ic=ic)


summary_list = []
vif_exclusion_global_log = []

for y_col in [c for c in biotic_vars if c in data_for_model.columns and c != n_species_col_name]:
    _log("info", f"Regression for: {y_col}")

    y_root = os.path.join(output_regression, safe_name(y_col))
    os.makedirs(y_root, exist_ok=True)

    dirs = {
        "MODEL": os.path.join(y_root, "01_MODEL"),
        "VIF": os.path.join(y_root, "02_VIF"),
        "DIAGNOSTICS": os.path.join(y_root, "03_DIAGNOSTICS"),
        "PLOTS": os.path.join(y_root, "04_PLOTS"),
        "DATA": os.path.join(y_root, "00_DATA"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    factor_vars_actual = [f for f in factor_vars
                          if f in data_for_model.columns
                          and data_for_model[f].nunique() >= 2]
    abio_vars_used = [a for a in abiotic_vars if a in data_for_model.columns]

    covariate_transformed = [
        f"{a}{scale_suffix}" for a in abio_vars_used
        if f"{a}{scale_suffix}" in data_for_model.columns
    ]

    base_cols = [y_col] + factor_vars_actual + covariate_transformed
    base_cols = [c for c in base_cols if c in data_for_model.columns]
    data_y = data_for_model[base_cols].dropna().copy()
    N = len(data_y)

    vif_history = []
    exclusion_steps = []
    if CONFIG["ENABLE_VIF"] and len(abio_vars_used) > 1 and N > 0:
        X_abio = data_for_model.loc[data_y.index, abio_vars_used].copy()
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

    if vif_history:
        vif_history[0].to_csv(os.path.join(dirs["VIF"], f"{safe_name(y_col)}_VIF_initial.csv"),
                              index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"])
        vif_history[-1].to_csv(os.path.join(dirs["VIF"], f"{safe_name(y_col)}_VIF_final.csv"),
                               index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"])

    with open(os.path.join(dirs["VIF"], f"{safe_name(y_col)}_vif_exclusion_steps.txt"), "w", encoding="utf-8") as flog:
        flog.write("\n".join(exclusion_steps) if exclusion_steps else "No VIF exclusions.")

    vif_exclusion_global_log.append(f"{y_col}:")
    vif_exclusion_global_log.extend(exclusion_steps or ["  No VIF exclusions."])
    vif_exclusion_global_log.append("")

    covariate_transformed = [
        f"{a}{scale_suffix}" for a in abio_vars_used
        if f"{a}{scale_suffix}" in data_for_model.columns
    ]

    model_cols = [y_col] + factor_vars_actual + covariate_transformed
    model_cols = [c for c in model_cols if c in data_y.columns]
    data_model = data_y[model_cols].dropna().copy()
    N = len(data_model)

    main_factors = [factor_term(f) for f in factor_vars_actual]
    main_covs = [q(c) for c in covariate_transformed]
    base_terms = main_factors + main_covs

    inter_terms = []
    if CONFIG.get("INTERACTIONS", True):
        if CONFIG.get("INTERACTIONS_FxC", True) and main_factors and main_covs:
            wl = CONFIG.get("INTERACTION_WHITELIST", None)
            for f_raw, f_term in zip(factor_vars_actual, main_factors):
                for c_raw, c_term in zip(covariate_transformed, main_covs):
                    if wl is not None:
                        cov_raw_name = c_raw.replace(scale_suffix, "")
                        if (f_raw, cov_raw_name) not in wl:
                            continue
                    inter_terms.append(f"{f_term}:{c_term}")

        if CONFIG.get("INTERACTIONS_FxF", False) and len(main_factors) > 1:
            for i in range(len(main_factors)):
                for j in range(i + 1, len(main_factors)):
                    inter_terms.append(f"{main_factors[i]}:{main_factors[j]}")

    all_terms = base_terms + inter_terms
    decided_formula = f"{q(y_col)} ~ " + " + ".join(all_terms) if all_terms else f"{q(y_col)} ~ 1"

    sel_formula, _ = run_backward_selection_if_enabled(decided_formula, data_model)
    if sel_formula != decided_formula:
        _log("info", f"Backward selection: {decided_formula} -> {sel_formula}")
        decided_formula = sel_formula

    decided_model = fit_ols(decided_formula, data_model)
    classic_model_for_diag = fit_ols_classic(decided_formula, data_model)

    with open(os.path.join(dirs["MODEL"], f"{safe_name(y_col)}_regression_summary.txt"),
              "w", encoding="utf-8") as fsum:
        fsum.write(decided_model.summary().as_text())

    coef_df = pd.DataFrame({
        "term": decided_model.params.index,
        "coef": decided_model.params.values,
        "se": decided_model.bse.values,
        "t": decided_model.tvalues.values,
        "p": decided_model.pvalues.values
    })
    coef_df.to_csv(
        os.path.join(dirs["MODEL"], f"{safe_name(y_col)}_coefficients.csv"),
        index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
    )

    y_pred = decided_model.predict(data_model)
    resid = decided_model.resid
    ext = (CONFIG.get("PLOT_FORMAT") or "png").lower()
    dpi = int(CONFIG.get("PLOT_DPI", 200))

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
                dpi=dpi, format=ext)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=resid)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted - {y_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["PLOTS"], f"{safe_name(y_col)}_Residuals_vs_Predicted.{ext}"),
                dpi=dpi, format=ext)
    plt.close()

    plt.figure(figsize=(6, 4))
    probplot(resid, dist="norm", plot=plt)
    plt.title(f"QQ plot of residuals - {y_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["PLOTS"], f"{safe_name(y_col)}_QQplot_Residuals.{ext}"),
                dpi=dpi, format=ext)
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
        fbp.write("Breusch–Pagan test for heteroscedasticity:\n")
        fbp.write(f"LM statistic: {bp_lm:.4f}\n")
        fbp.write(f"LM p-value:  {bp_lm_p:.4g}\n")
        fbp.write(f"F statistic: {bp_f:.4f}\n")
        fbp.write(f"F p-value:  {bp_f_p:.4g}\n")

    with open(os.path.join(dirs["DIAGNOSTICS"], "diagnostics.txt"), "w", encoding="utf-8") as fd:
        fd.write(f"N = {N}\n")
        fd.write(f"df_resid = {decided_model.df_resid:.1f}\n")
        fd.write(f"Final formula: {decided_formula}\n")
        fd.write(f"Factors used: {factor_vars_actual}\n")
        fd.write(f"Transformed covariates used: {covariate_transformed}\n")
        fd.write(f"Backward selection enabled: {CONFIG.get('RUN_BACKWARD_SELECTION', False)}\n")
        if CONFIG.get("RUN_BACKWARD_SELECTION", False):
            fd.write(f"   selection_mode={CONFIG.get('SELECTION_MODE')}, "
                     f"ic={CONFIG.get('IC')}\n")
        fd.write("\nBreusch–Pagan:\n")
        fd.write(f"   LM={bp_lm:.4f}, p={bp_lm_p:.4g}, F={bp_f:.4f}, p={bp_f_p:.4g}\n")

    coef_dict = {f"coef_{k}": v for k, v in decided_model.params.items()}
    coef_dict.update({
        "biotic_var": y_col,
        "n": N,
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
        "predictors_after_vif": ",".join(factor_vars_actual + covariate_transformed),
    })
    summary_list.append(coef_dict)



global_dir = os.path.join(output_regression, "_GLOBAL")
os.makedirs(global_dir, exist_ok=True)

with open(os.path.join(global_dir, "vif_exclusion_log.txt"), "w", encoding="utf-8") as flog:
    flog.write("\n".join(vif_exclusion_global_log))

results_df = pd.DataFrame(summary_list)
results_df.to_csv(
    os.path.join(global_dir, "regression_summary_all.csv"),
    index=False, encoding="utf-8-sig", sep=CONFIG["OUTPUT_CSV_SEP"]
)

print("All outputs saved to:", output_regression)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
