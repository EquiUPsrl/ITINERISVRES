import os
from statsmodels.graphics.gofplots import qqplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import statsmodels.formula.api as smf
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from datetime import datetime

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filtered_input', action='store', type=str, required=True, dest='filtered_input')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

filtered_input = args.filtered_input.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

CENTER_CONTINUOUS = True       # center numeric predictors (recommended)
CENTER_SUFFIX = "_c"

COV_TYPE = "HC3"

HAC_MAXLAGS = None             # e.g., 1 or 4; if None -> automatic rule-of-thumb
HAC_KERNEL = "bartlett"        # "bartlett", "uniform", "parzen", "qs"



RUN_BACKWARD_SELECTION = False  # True -> run selection; False -> full model only

SELECTION_MODE = "IC"      # "PVALUE" or "IC"

PVALUE_ALPHA   = 0.05          # threshold for backward p-value
IC_TOL = 1e-6
IC = "AIC"                     # "BIC" or "AIC" for backward selection (if SELECTION_MODE="IC")

ENFORCE_VIF = True
VIF_THRESHOLD = 5.0
VIF_RELEASE_INTERACTIONS = True  # can remove interactions to reduce VIF (breaks hierarchy)

ALPHA = 0.05                   # p < ALPHA → asterisk (*) in tables/plots
COMPUTE_VIF = False

KEEP_ALL_FACTORS = False

FACTOR_CONTRAST = "sum"   # oppure "treatment"


data_path = filtered_input
param_path = parameters_csv
output_dir = conf_output_path
results_dir = os.path.join(output_dir, "regression_analysis")
os.makedirs(results_dir, exist_ok=True)

full_results_dir = os.path.join(results_dir, "full_model")
os.makedirs(full_results_dir, exist_ok=True)

selected_results_dir = os.path.join(results_dir, "selected_model")

log_path = os.path.join(results_dir, "vif_enforcement_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] VIF enforcement log (EN)\n")
    f.write(f"Factor contrasts: {FACTOR_CONTRAST}\n")


def read_csv_auto_sep(filepath, guess_rows=5):
    """Read a CSV by guessing the separator among: ',', ';', '\t', '|'."""
    possible_seps = [',', ';', '\t', '|']
    best_cols, best_sep = 0, ','
    for sep in possible_seps:
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=guess_rows)
            if df.shape[1] > best_cols:
                best_cols, best_sep = df.shape[1], sep
        except Exception:
            continue
    df_full = pd.read_csv(filepath, sep=best_sep)
    print(f"Read '{filepath}' with sep '{best_sep}' ({df_full.shape[1]} columns)")
    return df_full

def strip_center_suffix_if_any(s):
    return s[:-len(CENTER_SUFFIX)] if CENTER_CONTINUOUS and s.endswith(CENTER_SUFFIX) else s

def cat_for_formula(cat_name: str) -> str:
    """
    Returns the expression C(...) to use in the formula for the cat_name factor,
    based on the FACTOR_CONTRAST configuration.
    """
    fc = str(FACTOR_CONTRAST).lower()
    if fc == "sum":
        return f"C({cat_name}, Sum)"
    elif fc == "treatment":
        return f"C({cat_name}, Treatment)"
    else:
        raise ValueError(f"FACTOR_CONTRAST must be 'sum' or 'treatment', got: {FACTOR_CONTRAST}")

def clean_term_label(term: str) -> str:
    """
    Create human-readable labels from coefficient names (handles C(f, Contr)[cod.level] and interactions).
    Ignores the contrast type (Sum/Treatment) in the label, displaying only the factor name.
    """

    m = re.match(r"(.+?):C\(([^,]+)(?:,[^)]+)?\)\[[^.]+\.(.+)\]$", term)
    if m:
        left, var, level = m.groups()
        left_clean = " × ".join(strip_center_suffix_if_any(p) for p in left.split(":"))
        return f"{left_clean} × {var} = {level}"

    m2 = re.match(r"C\(([^,]+)(?:,[^)]+)?\)\[[^.]+\.(.+)\]$", term)
    if m2:
        var, level = m2.groups()
        return f"{var} = {level}"

    m3 = re.match(r"C\(([^,]+)(?:,[^)]+)?\)\[[^\.]+\.(.+)\]$", term)
    if m3:
        var, level = m3.groups()
        return f"{var} = {level}"

    if ':' in term:
        return " × ".join(strip_center_suffix_if_any(p) for p in term.split(':'))

    return strip_center_suffix_if_any(term)

def _auto_hac_maxlags(n):
    """Rule-of-thumb default for HAC maxlags (Newey–West)."""
    return max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))

def maybe_fit(formula, data, cov_type):
    """Fit OLS with classic, heteroskedasticity-robust, or HAC-robust SE."""
    model = smf.ols(formula, data=data)
    if cov_type:
        ct = str(cov_type).upper()
        if ct == "HAC":
            n = data.shape[0]
            maxlags = HAC_MAXLAGS if HAC_MAXLAGS is not None else _auto_hac_maxlags(n)
            return model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": HAC_KERNEL})
        else:
            return model.fit(cov_type=ct)
    return model.fit()

def write_log(log_path, msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def compute_design_vif(formula, data):
    """Return df with design columns (including Intercept) and VIF."""
    y_mat, X_mat = patsy.dmatrices(formula, data=data, return_type="dataframe")
    vif_vals = [variance_inflation_factor(X_mat.values, i) for i in range(X_mat.shape[1])]
    return pd.DataFrame({"DesignCol": X_mat.columns, "VIF": vif_vals})

def map_design_col_to_term(design_col):
    """
    Map a design column to a higher-level term.
    Use cat_for_formula(...) to return strings that match
    the terms in the formula (including the contrast type).
    """
    m = re.match(r"^([^:]+):C\(([^,]+)(?:,[^)]+)?\)\[[^\]]+\]$", design_col)
    if m:
        x, cat = m.groups()
        return f"{x}:{cat_for_formula(cat)}"

    m = re.match(r"^C\(([^,]+)(?:,[^)]+)?\)\[[^\]]+\]:([^:]+)$", design_col)
    if m:
        cat, x = m.groups()
        return f"{x}:{cat_for_formula(cat)}"

    m = re.match(r"^C\(([^,]+)(?:,[^)]+)?\)\[[^\]]+\]$", design_col)
    if m:
        cat = m.group(1)
        return cat_for_formula(cat)

    return design_col

def split_term_tokens(term: str):
    return [t.strip() for t in term.split(":")]

def interaction_contains_main(interaction: str, main: str) -> bool:
    return main in split_term_tokens(interaction)

def drop_term_respecting_hierarchy(terms, term_to_drop):
    if term_to_drop not in terms:
        return False
    if ":" in term_to_drop:
        terms.remove(term_to_drop)
        return True
    if any(interaction_contains_main(inter, term_to_drop) for inter in terms if ':' in inter):
        return False
    terms.remove(term_to_drop)
    return True

def pick_interaction_to_drop_by_vif(active_terms, data, yvar, cov_type, restrict_to_main=None):
    if not active_terms:
        return None
    fml = f"{yvar} ~ " + " + ".join(active_terms)
    dv = compute_design_vif(fml, data)
    dv = dv[dv["DesignCol"] != "Intercept"].copy()
    if dv.empty:
        return None
    dv["Term"] = dv["DesignCol"].apply(map_design_col_to_term)
    term_vif = dv.groupby("Term")["VIF"].max().reset_index()
    inters = term_vif[term_vif["Term"].str.contains(":")].copy()
    if restrict_to_main is not None:
        inters = inters[inters["Term"].apply(lambda t: interaction_contains_main(t, restrict_to_main))]
    if inters.empty:
        return None
    inters = inters.sort_values("VIF", ascending=False)
    for t in inters["Term"]:
        if t in active_terms:
            return t
    return None

def factors_with_single_level(df_like, factors):
    kept, removed = [], []
    for fvar in factors:
        if fvar not in df_like.columns:
            removed.append((fvar, "missing"))
            continue
        lvls = pd.Series(df_like[fvar]).dropna().unique()
        if len(lvls) <= 1:
            removed.append((fvar, f"n_levels={len(lvls)}"))
        else:
            kept.append(fvar)
    return kept, removed

def rebuild_terms_after_factor_filter(X_for_formula, f_vars_kept):
    main = X_for_formula + [cat_for_formula(cat) for cat in f_vars_kept]
    inter = [f"{x}:{cat_for_formula(cat)}" for x in X_for_formula for cat in f_vars_kept]
    return main, inter

def backward_elimination_ic_with_vif(data, yvar, main_effects, interactions, cov_type, ic,
                                     enforce_vif, vif_threshold, release_interactions,
                                     log_path):
    ic = ic.upper()
    def get_ic(m): return m.bic if ic == "BIC" else m.aic
    def fit(ts):
        fml = f"{yvar} ~ " + " + ".join(ts) if ts else f"{yvar} ~ 1"
        return maybe_fit(fml, data, cov_type), fml

    terms = main_effects + interactions
    model, formula = fit(terms)

    improved = True
    while improved and len(terms) > 0:
        improved = False
        best_ic, best_term, best_model = get_ic(model), None, None
        current_inters = [t for t in terms if ':' in t]
        candidates = [t for t in interactions if t in terms] + [
            t for t in main_effects if t in terms and all(not interaction_contains_main(inter, t) for inter in current_inters)
        ]

        if enforce_vif:
            design_vif = compute_design_vif(formula, data)
            design_vif = design_vif[design_vif["DesignCol"] != "Intercept"]
            if not design_vif.empty:
                design_vif["Term"] = design_vif["DesignCol"].apply(map_design_col_to_term)
                term_vif = design_vif.groupby("Term")["VIF"].max().reset_index()
                offenders = term_vif[term_vif["VIF"] > vif_threshold]
                if not offenders.empty:
                    offenders["is_inter"] = offenders["Term"].str.contains(":")
                    offenders = offenders.sort_values(["is_inter", "VIF"], ascending=[False, False])
                    if not release_interactions:
                        offenders = offenders[
                            offenders["is_inter"] |
                            ~offenders["Term"].apply(lambda t: any(interaction_contains_main(inter, t) for inter in current_inters))
                        ]
                    if not offenders.empty:
                        candidates = offenders["Term"].tolist()

        for t in candidates:
            if t not in terms:
                continue
            ts = [x for x in terms if x != t]
            m, _ = fit(ts)
            this_ic = get_ic(m)
            if this_ic < best_ic - IC_TOL:
                best_ic, best_term, best_model = this_ic, t, m

        if best_term is not None:
            write_log(log_path, f"[IC] Removed '{best_term}' (improved {ic}: {best_ic:.3f})")
            terms.remove(best_term)
            model, formula = best_model, f"{yvar} ~ " + " + ".join(terms) if terms else f"{yvar} ~ 1"
            improved = True
            continue

        if enforce_vif:
            design_vif = compute_design_vif(formula, data)
            design_vif = design_vif[design_vif["DesignCol"] != "Intercept"]
            if not design_vif.empty:
                design_vif["Term"] = design_vif["DesignCol"].apply(map_design_col_to_term)
                term_vif = design_vif.groupby("Term")["VIF"].max().reset_index()
                offenders = term_vif[term_vif["VIF"] > vif_threshold]
                if not offenders.empty:
                    offenders["is_inter"] = offenders["Term"].str.contains(":")
                    offenders = offenders.sort_values(["is_inter","VIF"], ascending=[False, False])

                    removed = False
                    for _, row in offenders.iterrows():
                        cand_term, cand_vif, is_inter = row["Term"], float(row["VIF"]), bool(row["is_inter"])
                        if drop_term_respecting_hierarchy(terms, cand_term):
                            write_log(log_path, f"[VIF] Removed '{cand_term}' (max VIF ≈ {cand_vif:.2f})")
                            model, formula = fit(terms)
                            improved, removed = True, True
                            break
                        if (not is_inter) and release_interactions:
                            inter_to_drop = pick_interaction_to_drop_by_vif(
                                terms, data, yvar, cov_type, restrict_to_main=cand_term
                            )
                            if inter_to_drop:
                                terms.remove(inter_to_drop)
                                write_log(log_path, f"[VIF] Removed interaction '{inter_to_drop}' to release '{cand_term}'")
                                model, formula = fit(terms)
                                improved, removed = True, True
                                break
                    if removed:
                        continue

    final_formula = f"{yvar} ~ " + " + ".join(terms) if terms else f"{yvar} ~ 1"
    return model, final_formula

def backward_elimination_pval_like_code1(data, yvar, main_effects, interactions, alpha=0.05):
    """
    - start from main effects + interactions
    - estimate OLS with classic SE
    - compute 'per-term' p-value as the mean p-value of design columns containing the term name
    - remove the term with the highest mean p-value if > alpha
    - stop when all terms have p-value <= alpha
    """
    terms = main_effects + interactions
    while True and terms:
        formula_test = f"{yvar} ~ " + " + ".join(terms)
        model = smf.ols(formula_test, data=data).fit()
        pvals = model.pvalues.drop('Intercept', errors='ignore')

        term_pvals = {}
        for term in terms:
            rel_cols = [idx for idx in pvals.index if term in idx]
            term_pvals[term] = pvals[rel_cols].mean() if rel_cols else 0.0

        worst_term = max(term_pvals, key=term_pvals.get)
        if term_pvals[worst_term] > alpha:
            terms.remove(worst_term)
        else:
            break

    final_formula = f"{yvar} ~ " + " + ".join(terms) if terms else f"{yvar} ~ 1"
    final_model = smf.ols(final_formula, data=data).fit()
    return final_model, final_formula

def run_residual_diagnostics(best_model, final_formula, df_clean, suffix, base_out_dir="output"):
    """
    Essential:
      - CSV: fitted, residuals, studentized_residuals
      - Figures: Residuals vs Fitted (studentized), QQ-plot (studentized)
      - Tests: Shapiro–Wilk (global) + Breusch–Pagan
      - Compact summary (CSV)
      - NEW: Observed vs Fitted with regression line + save line parameters
    """
    plots_dir        = os.path.join(base_out_dir, "residual_diagnostics")
    diagnostic_dir   = os.path.join(base_out_dir, "diagnostics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(diagnostic_dir, exist_ok=True)

    resid_raw   = best_model.resid
    fitted      = best_model.fittedvalues
    influence   = best_model.get_influence()
    resid_stud  = influence.resid_studentized_internal

    try:
        yvar_local = final_formula.split("~", 1)[0].strip()
        y_obs = df_clean.loc[fitted.index, yvar_local]
    except Exception:
        y_obs = df_clean.iloc[fitted.index if hasattr(fitted, "index") else slice(None), 0]

    residuals_csv_path = os.path.join(diagnostic_dir, f"residuals_{suffix}.csv")
    pd.DataFrame({
        "fitted": fitted,
        "residuals": resid_raw,
        "studentized_residuals": resid_stud,
    }).to_csv(residuals_csv_path, index=False, encoding="utf-8", sep=";")

    fig_rvf_path = os.path.join(plots_dir, f"residuals_vs_fitted_{suffix}.png")
    plt.figure(figsize=(8,6))
    plt.scatter(fitted, resid_stud, alpha=0.7, edgecolor='k', linewidth=0.3)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Fitted values")
    plt.ylabel("Studentized residuals")
    plt.title(f"Residuals vs Fitted — [{suffix}]")
    plt.tight_layout()
    plt.savefig(fig_rvf_path, dpi=300)
    plt.close()

    fig_qq_path = os.path.join(plots_dir, f"qqplot_studentized_{suffix}.png")
    plt.figure(figsize=(6,6))
    qqplot(resid_stud, line='45', fit=True)
    plt.title(f"QQ plot — Studentized residuals [{suffix}]")
    plt.tight_layout()
    plt.savefig(fig_qq_path, dpi=300)
    plt.close()

    fig_ovf_path = os.path.join(plots_dir, f"observed_vs_fitted_with_regline_{suffix}.png")
    try:
        b, a = np.polyfit(fitted, y_obs, 1)
        slope, intercept = b, a
        x_line = np.linspace(np.min(fitted), np.max(fitted), 200)
        y_line = intercept + slope * x_line

        plt.figure(figsize=(8,6))
        plt.scatter(fitted, y_obs, alpha=0.6, edgecolor='k', linewidth=0.3)
        plt.plot(x_line, y_line, linewidth=2)
        diag_x = np.linspace(np.min(fitted), np.max(fitted), 2)
        plt.plot(diag_x, diag_x, linestyle='--')
        plt.xlabel("Fitted values")
        plt.ylabel("Observed values")
        plt.title(f"Observed vs Fitted — regression line [{suffix}]\n y = {intercept:.3f} + {slope:.3f}·fitted")
        plt.tight_layout()
        plt.savefig(fig_ovf_path, dpi=300)
        plt.close()

        regline_txt = os.path.join(diagnostic_dir, f"regression_line_params_{suffix}.txt")
        with open(regline_txt, "w", encoding="utf-8") as f:
            f.write("Observed vs Fitted — regression line\n")
            f.write(f"Intercept: {intercept:.6f}\n")
            f.write(f"Slope: {slope:.6f}\n")
    except Exception as e:
        slope = intercept = np.nan
        regline_txt = os.path.join(diagnostic_dir, f"regression_line_params_{suffix}.txt")
        with open(regline_txt, "w", encoding="utf-8") as f:
            f.write(f"ERROR computing regression line: {e}\n")

    shap_txt_path = os.path.join(diagnostic_dir, f"shapiro_residuals_global_{suffix}.txt")
    try:
        shap_stat, shap_p = shapiro(resid_raw)
        with open(shap_txt_path, "w", encoding="utf-8") as f:
            f.write(
                f"Shapiro–Wilk (global residuals)\n"
                f"statistic: {shap_stat:.4f}\n"
                f"p-value:  {shap_p:.6g}\n"
            )
    except Exception as e:
        shap_stat, shap_p = np.nan, np.nan
        with open(shap_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Shapiro–Wilk ERROR: {e}\n")

    bp_txt_path = os.path.join(diagnostic_dir, f"heteroskedasticity_breusch_pagan_{suffix}.txt")
    try:
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(best_model.resid, best_model.model.exog)
        with open(bp_txt_path, "w", encoding="utf-8") as f:
            f.write("Breusch–Pagan test (heteroskedasticity)\n")
            f.write(f"LM statistic: {bp_lm:.4f}, LM p-value: {bp_lm_p:.6g}\n")
            f.write(f"F  statistic: {bp_f:.4f},  F p-value:  {bp_f_p:.6g}\n")
    except Exception as e:
        bp_lm, bp_lm_p, bp_f, bp_f_p = (np.nan,)*4
        with open(bp_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Breusch–Pagan ERROR: {e}\n")

    rows = [
        {"Group": "—", "Test": "Shapiro–Wilk", "Assumption": "Normality",
         "Statistic": shap_stat, "p_value": shap_p,
         "Assumption_met": ("Yes" if (not np.isnan(shap_p) and shap_p >= 0.05) else ("No" if not np.isnan(shap_p) else "n/a")),
         "Output_file": shap_txt_path},
        {"Group": "—", "Test": "Breusch–Pagan (LM)", "Assumption": "Homoscedasticity",
         "Statistic": bp_lm, "p_value": bp_lm_p,
         "Assumption_met": ("Yes" if (not np.isnan(bp_lm_p) and bp_lm_p >= 0.05) else ("No" if not np.isnan(bp_lm_p) else "n/a")),
         "Output_file": bp_txt_path},
        {"Group": "—", "Test": "Breusch–Pagan (F)", "Assumption": "Homoscedasticity",
         "Statistic": bp_f, "p_value": bp_f_p,
         "Assumption_met": ("Yes" if (not np.isnan(bp_f_p) and bp_f_p >= 0.05) else ("No" if not np.isnan(bp_f_p) else "n/a")),
         "Output_file": bp_txt_path},
        {"Group": "—", "Test": "Figure", "Assumption": "Residuals vs Fitted",
         "Statistic": None, "p_value": None, "Assumption_met": None, "Output_file": fig_rvf_path},
        {"Group": "—", "Test": "Figure", "Assumption": "QQ plot (studentized)",
         "Statistic": None, "p_value": None, "Assumption_met": None, "Output_file": fig_qq_path},
        {"Group": "—", "Test": "Figure", "Assumption": "Observed vs Fitted (regression line)",
         "Statistic": None, "p_value": None, "Assumption_met": None, "Output_file": fig_ovf_path},
        {"Group": "—", "Test": "TXT", "Assumption": "Regression line parameters OvsF",
         "Statistic": intercept if 'intercept' in locals() else None, "p_value": slope if 'slope' in locals() else None,
         "Assumption_met": None, "Output_file": regline_txt},
        {"Group": "—", "Test": "CSV", "Assumption": "Residuals export",
         "Statistic": None, "p_value": None, "Assumption_met": None, "Output_file": residuals_csv_path},
    ]
    res_table_path = os.path.join(diagnostic_dir, f"residuals_analysis_{suffix}.csv")
    pd.DataFrame(rows)[["Group","Test","Assumption","Statistic","p_value","Assumption_met","Output_file"]].to_csv(
        res_table_path, index=False, sep=";"
    )

    write_log(log_path, "Residual diagnostics (essential) completed.")
    write_log(log_path, f"- Plots: {plots_dir}")
    write_log(log_path, f"- Reports and tables: {diagnostic_dir}")
    write_log(log_path, f"- Summary: {res_table_path}")


df = read_csv_auto_sep(data_path)
param = read_csv_auto_sep(param_path)
param_row = param.iloc[0]

Y_vars = [c for c in param.columns if str(param_row[c]).strip().upper() == "Y"]
X_vars = [c for c in param.columns if str(param_row[c]).strip().upper() == "X"]
f_vars = [c for c in param.columns if str(param_row[c]).strip().lower() == "f"]

if not Y_vars:
    raise ValueError("No Y variable in Parameter.csv")
yvar = Y_vars[0]

missing = [x for x in ([yvar] + X_vars + f_vars) if x not in df.columns]
if missing:
    raise KeyError(f"Missing columns in dataset: {missing}")

df = df.copy()
if CENTER_CONTINUOUS:
    num_X = [x for x in X_vars if pd.api.types.is_numeric_dtype(df[x])]
    for var in num_X:
        df[f"{var}{CENTER_SUFFIX}"] = df[var] - df[var].mean()
    X_for_formula = [f"{var}{CENTER_SUFFIX}" if var in num_X else var for var in X_vars]
else:
    X_for_formula = X_vars

main_effects_init = X_for_formula + [cat_for_formula(cat) for cat in f_vars]
interactions_init = [f"{x}:{cat_for_formula(cat)}" for x in X_for_formula for cat in f_vars]
formula_full_init = (
    f"{yvar} ~ " + " + ".join(main_effects_init + interactions_init)
    if (main_effects_init or interactions_init) else f"{yvar} ~ 1"
)

needed_cols = [yvar] + X_for_formula + f_vars
df_clean = df[needed_cols].dropna().copy()

if KEEP_ALL_FACTORS:
    f_vars_kept = list(f_vars)   # CODE 1 style: no filter
    f_vars_removed = []
    write_log(log_path, "[Factor filter] DISABLED by config: keeping all factors as in CODE 1.")
else:
    f_vars_kept, f_vars_removed = factors_with_single_level(df_clean, f_vars)
    if f_vars_removed:
        for (fvar, reason) in f_vars_removed:
            write_log(log_path, f"[Factor filter] Removed factor '{fvar}' (reason: {reason})")
    else:
        write_log(log_path, "[Factor filter] No factors removed (all have ≥ 2 levels).")

main_effects, interactions = rebuild_terms_after_factor_filter(X_for_formula, f_vars_kept)
terms_full = main_effects + interactions
formula_full = f"{yvar} ~ " + " + ".join(terms_full) if terms_full else f"{yvar} ~ 1"

write_log(log_path, f"Full formula (POST-FACTOR-FILTER): {formula_full}")
write_log(log_path, f"df_clean shape: {df_clean.shape}")

model_full = maybe_fit(formula_full, df_clean, COV_TYPE)

with open(os.path.join(full_results_dir, "selected_formula_FULL.txt"), "w", encoding="utf-8") as f:
    f.write(formula_full + "\n")

with open(os.path.join(full_results_dir, "regression_full_summary.txt"), "w", encoding="utf-8") as f:
    f.write(model_full.summary().as_text())

try:
    run_residual_diagnostics(model_full, formula_full, df_clean, suffix="FULL", base_out_dir=full_results_dir)
except Exception as e:
    write_log(log_path, f"Residual diagnostics ERROR (FULL): {e}")

best_model = None
final_formula = None

if RUN_BACKWARD_SELECTION:
    os.makedirs(selected_results_dir, exist_ok=True)
    write_log(log_path, "[SELECTION] Backward selection ENABLED.")

    if SELECTION_MODE.upper() == "PVALUE":
        if COV_TYPE:
            write_log(log_path, f"[WARN] SELECTION_MODE=PVALUE uses classic OLS for selection; COV_TYPE={COV_TYPE} ignored in this phase.")
        if ENFORCE_VIF:
            write_log(log_path, "[WARN] SELECTION_MODE=PVALUE does not enforce VIF during selection.")

        best_model, final_formula = backward_elimination_pval_like_code1(
            df_clean, yvar, main_effects, interactions, alpha=PVALUE_ALPHA
        )
        write_log(log_path, f"[PVALUE] Selected formula: {final_formula}")

        if COV_TYPE:
            best_model = maybe_fit(final_formula, df_clean, COV_TYPE)

    else:
        best_model, final_formula = backward_elimination_ic_with_vif(
            df_clean, yvar, main_effects, interactions, COV_TYPE, IC,
            enforce_vif=ENFORCE_VIF, vif_threshold=VIF_THRESHOLD,
            release_interactions=VIF_RELEASE_INTERACTIONS, log_path=log_path
        )
        write_log(log_path, f"[IC] Selected (pre-enforcement) formula: {final_formula}")

        if ENFORCE_VIF:
            changed = True
            while changed:
                changed = False
                design_vif = compute_design_vif(final_formula, df_clean)
                design_vif = design_vif[design_vif["DesignCol"] != "Intercept"]
                if design_vif.empty:
                    break
                design_vif["Term"] = design_vif["DesignCol"].apply(map_design_col_to_term)
                term_vif = design_vif.groupby("Term")["VIF"].max().reset_index()
                max_row = term_vif.loc[term_vif["VIF"].idxmax()]
                max_term, max_vif = max_row["Term"], float(max_row["VIF"])

                if max_vif > VIF_THRESHOLD:
                    terms_current = final_formula.split("~", 1)[1].strip()
                    terms_current = [t.strip() for t in terms_current.split("+")]
                    if terms_current == ['1']:
                        write_log(log_path, f"[POST-VIF] Only intercept left; stopping.")
                        break

                    if drop_term_respecting_hierarchy(terms_current, max_term):
                        write_log(log_path, f"[POST-VIF] Removed '{max_term}' (max VIF ≈ {max_vif:.2f})")
                        final_formula = f"{yvar} ~ " + " + ".join(terms_current) if terms_current else f"{yvar} ~ 1"
                        best_model = maybe_fit(final_formula, df_clean, COV_TYPE)
                        changed = True
                        continue

                    if VIF_RELEASE_INTERACTIONS:
                        inter_to_drop = pick_interaction_to_drop_by_vif(
                            terms_current, df_clean, yvar, COV_TYPE, restrict_to_main=max_term
                        )
                        if inter_to_drop:
                            terms_current.remove(inter_to_drop)
                            write_log(log_path, f"[POST-VIF] Removed interaction '{inter_to_drop}' to release '{max_term}'")
                            final_formula = f"{yvar} ~ " + " + ".join(terms_current) if terms_current else f"{yvar} ~ 1"
                            best_model = maybe_fit(final_formula, df_clean, COV_TYPE)
                            changed = True
                            continue

                    write_log(log_path, f"[POST-VIF] Cannot reduce '{max_term}' due to hierarchy; VIF remains {max_vif:.2f}")
                    break
else:
    write_log(log_path, "[SELECTION] Backward selection DISABLED: only full model will be produced.")

if RUN_BACKWARD_SELECTION and best_model is not None and final_formula is not None:
    if SELECTION_MODE.upper() == "PVALUE":
        sel_name = "regression_selected_summary.txt"
        fml_name = "selected_formula_PVALUE.txt"
        sel_tag  = "PVALUE"
    else:
        sel_name = f"regression_selected_summary_{IC}.txt"
        fml_name = f"selected_formula_{IC}.txt"
        sel_tag  = IC.upper()

    with open(os.path.join(selected_results_dir, sel_name), "w", encoding="utf-8") as f:
        f.write(best_model.summary().as_text())
    with open(os.path.join(selected_results_dir, fml_name), "w", encoding="utf-8") as f:
        f.write(final_formula + "\n")
    write_log(log_path, f"[SELECTION] Final formula ({sel_tag}): {final_formula}")

    ci_sel = best_model.conf_int()
    summary_table = pd.DataFrame({
        "Term": best_model.params.index,
        "Coefficient": best_model.params.values,
        "Std. Error": best_model.bse.values,
        "t-Statistic": best_model.tvalues.values,
        "P-value": best_model.pvalues.values,
        "Conf. 95% low": ci_sel[0].values,
        "Conf. 95% high": ci_sel[1].values
    })
    summary_table["Label"] = summary_table["Term"].apply(clean_term_label)

    mask = summary_table["P-value"] < ALPHA
    summary_table.loc[mask, "Label"] = summary_table.loc[mask, "Label"].astype(str) + " *"
    
    summary_table.to_csv(
        os.path.join(selected_results_dir, f"selected_model_table_{sel_tag}.csv"),
        index=False, sep=";"
    )
    write_log(log_path, "[SELECTION] Saved coefficients table (selected).")

    df_plot = pd.DataFrame({
        "Term": best_model.params.index,
        "Coefficient": best_model.params.values,
        "SE": best_model.bse.values,
        "P_value": best_model.pvalues.values,
        "Conf_low": ci_sel[0].values,
        "Conf_high": ci_sel[1].values
    })
    df_plot = df_plot[df_plot["Term"] != "Intercept"].copy()
    df_plot["t_abs"] = (df_plot["Coefficient"] / df_plot["SE"]).abs()
    df_plot["abs_coef"] = df_plot["Coefficient"].abs()
    df_plot = df_plot.sort_values(["t_abs", "abs_coef"], ascending=[False, False]).drop(columns="abs_coef").reset_index(drop=True)
    df_plot["Label"] = df_plot["Term"].apply(clean_term_label)

    mask = df_plot["P_value"] < ALPHA
    df_plot.loc[mask, "Label"] = df_plot.loc[mask, "Label"].astype(str) + " *"

    y = np.arange(len(df_plot))
    plt.figure(figsize=(10, max(6, 0.4 * len(df_plot) + 2)))
    for yi, lo, hi in zip(y, df_plot["Conf_low"], df_plot["Conf_high"]):
        plt.hlines(yi, lo, hi, linewidth=2, color='dimgray')
    plt.scatter(df_plot["Coefficient"], y, s=70, color="black", zorder=3)
    plt.axvline(0, linestyle='--', linewidth=1.5, color="red")
    plt.yticks(y, df_plot["Label"])
    plt.gca().invert_yaxis()
    cov_str = f"(robust SE: {COV_TYPE})" if COV_TYPE else "(classic SE)"
    center_str = "centered" if CENTER_CONTINUOUS else "not centered"
    plt.title(f"Coefficients and 95% CI — Selected model [{sel_tag}]\n"
              f"{cov_str}, predictors {center_str}\n(* marks p < {ALPHA:.2f})")
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(selected_results_dir, f"coef_plot_selected_points_importance_sig_{sel_tag}.png"), dpi=300)
    plt.close()
    write_log(log_path, "[SELECTION] Saved coefficient plot (selected).")

    if COMPUTE_VIF:
        vif_final = compute_design_vif(final_formula, df_clean)
        vif_final = vif_final[vif_final["DesignCol"] != "Intercept"]
        vif_final.to_csv(
            os.path.join(selected_results_dir, f"VIF_selected_model_{sel_tag}.csv"),
            index=False, sep=";"
        )
        vmax = float(vif_final["VIF"].max()) if not vif_final.empty else 0.0
        write_log(log_path, f"[SELECTION] Final VIF saved (selected). Max VIF = {vmax:.2f} (threshold = {VIF_THRESHOLD:.2f})")

    try:
        run_residual_diagnostics(best_model, final_formula, df_clean, suffix=sel_tag, base_out_dir=selected_results_dir)
    except Exception as e:
        write_log(log_path, f"Residual diagnostics ERROR (selected): {e}")

    write_log(log_path, "All selected-model results saved in: " + selected_results_dir)

ci_full = model_full.conf_int()
summary_table_full = pd.DataFrame({
    "Term": model_full.params.index,
    "Coefficient": model_full.params.values,
    "Std. Error": model_full.bse.values,
    "t-Statistic": model_full.tvalues.values,
    "P-value": model_full.pvalues.values,
    "Conf. 95% low": ci_full[0].values,
    "Conf. 95% high": ci_full[1].values
})
summary_table_full["Label"] = summary_table_full["Term"].apply(clean_term_label)

mask = summary_table_full["P-value"] < ALPHA
summary_table_full.loc[mask, "Label"] = summary_table_full.loc[mask, "Label"].astype(str) + " *"


summary_table_full.to_csv(
    os.path.join(full_results_dir, "full_model_table.csv"),
    index=False, sep=";"
)
write_log(log_path, "Saved coefficients table (full).")

df_plot_full = pd.DataFrame({
    "Term": model_full.params.index,
    "Coefficient": model_full.params.values,
    "SE": model_full.bse.values,
    "P_value": model_full.pvalues.values,
    "Conf_low": ci_full[0].values,
    "Conf_high": ci_full[1].values
})
df_plot_full = df_plot_full[df_plot_full["Term"] != "Intercept"].copy()
df_plot_full["t_abs"] = (df_plot_full["Coefficient"] / df_plot_full["SE"]).abs()
df_plot_full["abs_coef"] = df_plot_full["Coefficient"].abs()
df_plot_full = df_plot_full.sort_values(["t_abs", "abs_coef"], ascending=[False, False]).drop(columns="abs_coef").reset_index(drop=True)
df_plot_full["Label"] = df_plot_full["Term"].apply(clean_term_label)

mask = df_plot_full["P_value"] < ALPHA
df_plot_full.loc[mask, "Label"] = df_plot_full.loc[mask, "Label"].astype(str) + " *"

y = np.arange(len(df_plot_full))
plt.figure(figsize=(10, max(6, 0.4 * len(df_plot_full) + 2)))
for yi, lo, hi in zip(y, df_plot_full["Conf_low"], df_plot_full["Conf_high"]):
    plt.hlines(yi, lo, hi, linewidth=2, color='dimgray')
plt.scatter(df_plot_full["Coefficient"], y, s=70, color="black", zorder=3)
plt.axvline(0, linestyle='--', linewidth=1.5, color="red")
plt.yticks(y, df_plot_full["Label"])
plt.gca().invert_yaxis()
cov_str_full = f"(robust SE: {COV_TYPE})" if COV_TYPE else "(classic SE)"
center_str_full = "centered" if CENTER_CONTINUOUS else "not centered"
plt.title(f"Coefficients and 95% CI — Full model\n"
          f"{cov_str_full}, predictors {center_str_full}\n(* marks p < {ALPHA:.2f})")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(full_results_dir, "coef_plot_full_points_importance_sig.png"), dpi=300)
plt.close()
write_log(log_path, "Saved coefficient plot (full).")

if COMPUTE_VIF:
    vif_full = compute_design_vif(formula_full, df_clean)
    vif_full = vif_full[vif_full["DesignCol"] != "Intercept"]
    vif_full.to_csv(
        os.path.join(full_results_dir, "VIF_full_model.csv"),
        index=False, sep=";"
    )
    vmax_full = float(vif_full["VIF"].max()) if not vif_full.empty else 0.0
    write_log(log_path, f"Final VIF (full) saved. Max VIF = {vmax_full:.2f} (threshold = {VIF_THRESHOLD:.2f})")

write_log(log_path, "All results saved in (EN): " + results_dir)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
