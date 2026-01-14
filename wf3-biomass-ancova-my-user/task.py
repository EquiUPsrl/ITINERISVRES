import os
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import statsmodels.formula.api as smf
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import build_design_matrices
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from scipy import stats
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

CENTER_CONTINUOUS = True
CENTER_SUFFIX = "_c"

COV_TYPE = "HC3"

HAC_MAXLAGS = None
HAC_KERNEL = "bartlett"

RUN_BACKWARD_SELECTION = True

SELECTION_MODE = "IC"   # "PVALUE" | "IC"
PVALUE_ALPHA   = 0.05
IC             = "AIC"  # "AIC" | "BIC"

ENFORCE_VIF = True
VIF_THRESHOLD = 3.0
VIF_RELEASE_INTERACTIONS = True
IC_TOL = 1e-6

ALPHA = 0.05
COMPUTE_VIF = True
KEEP_ALL_FACTORS = False

CONTRASTS = "sum"
ANOVA_TYPE = "III"
PAIRWISE_ADJUST = "holm"
TUKEY_ENABLED = True
TUKEY_MODE = "residualized"

PARAM_SELECT_MODE   = "index"    # "index" | "filter"
PARAM_ROW_INDEX     = 2
PARAM_FILTER_COL    = "Model"
PARAM_FILTER_VALUE  = "Modello1"

data_path = filtered_input
param_path = parameters_csv
output_dir = conf_output_path
results_dir = os.path.join(output_dir, "ANCOVA")

run_subdir = results_dir
os.makedirs(run_subdir, exist_ok=True)

full_model_dir = os.path.join(run_subdir, "full_model")
os.makedirs(full_model_dir, exist_ok=True)

if RUN_BACKWARD_SELECTION:
    selected_model_dir = os.path.join(run_subdir, "selected_model")
    os.makedirs(selected_model_dir, exist_ok=True)
else:
    selected_model_dir = full_model_dir

def read_csv_auto_sep(filepath, guess_rows=5):
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

def write_log(log_path, msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def factor_term(f: str) -> str:
    c = (CONTRASTS or "sum").lower()
    if c == "sum":
        return f"C({f}, Sum)"
    return f"C({f})"

def strip_center_suffix_if_any(s):
    return s[:-len(CENTER_SUFFIX)] if CENTER_CONTINUOUS and s.endswith(CENTER_SUFFIX) else s

def clean_term_label(term: str) -> str:
    m = re.match(r"(.+?):C\(([^,\)]+)(?:,\s*Sum)?\)\[T\.([^\]]+)\]$", term)
    if m:
        left, var, level = m.groups()
        left = ":".join(strip_center_suffix_if_any(p) for p in left.split(":"))
        return f"{left} × {var} = {level}"
    m2 = re.match(r"C\(([^,\)]+)(?:,\s*Sum)?\)\[T\.([^\]]+)\]$", term)
    if m2:
        return f"{m2.group(1)} = {m2.group(2)}"
    if ':' in term:
        return " × ".join(strip_center_suffix_if_any(p) for p in term.split(':'))
    return strip_center_suffix_if_any(term)

def _auto_hac_maxlags(n):
    return max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))

def maybe_fit(formula, data, cov_type):
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

def compute_design_vif(formula, data):
    y_mat, X_mat = patsy.dmatrices(formula, data=data, return_type="dataframe")
    vif_vals = [variance_inflation_factor(X_mat.values, i) for i in range(X_mat.shape[1])]
    return pd.DataFrame({"DesignCol": X_mat.columns, "VIF": vif_vals})

def split_term_tokens(term: str):
    return [t.strip() for t in term.split(":")]

def interaction_contains_main(interaction: str, main: str) -> bool:
    return main in split_term_tokens(interaction)

def map_design_col_to_term(design_col):
    contrast_suffix = ", Sum" if (CONTRASTS or "sum").lower() == "sum" else ""

    m = re.match(r"^([^:]+):C\(([^,\)]+)(?:,\s*Sum)?\)\[[^\]]+\]$", design_col)
    if m:
        x, cat = m.groups()
        return f"{x}:C({cat}{contrast_suffix})"

    m = re.match(r"^C\(([^,\)]+)(?:,\s*Sum)?\)\[[^\]]+\]:([^:]+)$", design_col)
    if m:
        cat, x = m.groups()
        return f"{x}:C({cat}{contrast_suffix})"

    m = re.match(r"^C\(([^,\)]+)(?:,\s*Sum)?\)\[[^\]]+\]$", design_col)
    if m:
        return f"C({m.group(1)}{contrast_suffix})"

    return design_col

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
    main = X_for_formula + [factor_term(cat) for cat in f_vars_kept]
    inter = [f"{x}:{factor_term(cat)}" for x in X_for_formula for cat in f_vars_kept]
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
            t for t in main_effects
            if t in terms and all(not interaction_contains_main(inter, t) for inter in current_inters)
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
                            ~offenders["Term"].apply(
                                lambda t: any(interaction_contains_main(inter, t) for inter in current_inters)
                            )
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
            model, formula = best_model, (f"{yvar} ~ " + " + ".join(terms) if terms else f"{yvar} ~ 1")
            improved = True
            continue

        if enforce_vif:
            design_vif = compute_design_vif(formula, data)
            design_vif = design_vif[design_vif["DesignCol"] != "Intercept"]
            if not design_vif.empty:
                design_vif["Term"] = design_vif["DesignCol"].apply(map_design_col_to_term)
                term_vif = design_vif.groupby("Term")["VIF"].max().reset_index()
                max_row = term_vif.loc[term_vif["VIF"].idxmax()]
                max_term, max_vif = max_row["Term"], float(max_row["VIF"])

                if max_vif > vif_threshold:
                    final_formula = f"{yvar} ~ " + " + ".join(terms) if terms else f"{yvar} ~ 1"
                    terms_current = final_formula.split("~", 1)[1].strip()
                    terms_current = [t.strip() for t in terms_current.split("+")]
                    if terms_current == ['1']:
                        write_log(log_path, f"[POST-VIF/IC] Only intercept left; stopping.")
                        break

                    if drop_term_respecting_hierarchy(terms_current, max_term):
                        write_log(log_path,
                                  f"[POST-VIF/IC] Removed '{max_term}' (max VIF ≈ {max_vif:.2f})")
                        final_formula = f"{yvar} ~ " + " + ".join(terms_current) if terms_current else f"{yvar} ~ 1"
                        model, formula = maybe_fit(final_formula, data, cov_type), final_formula
                        terms = terms_current
                        improved = True
                        continue

                    if release_interactions:
                        inter_to_drop = pick_interaction_to_drop_by_vif(
                            terms_current, data, yvar, cov_type, restrict_to_main=max_term
                        )
                        if inter_to_drop:
                            terms_current.remove(inter_to_drop)
                            write_log(
                                log_path,
                                f"[POST-VIF/IC] Removed interaction '{inter_to_drop}' "
                                f"to release '{max_term}'"
                            )
                            final_formula = f"{yvar} ~ " + " + ".join(terms_current) if terms_current else f"{yvar} ~ 1"
                            model, formula = maybe_fit(final_formula, data, cov_type), final_formula
                            terms = terms_current
                            improved = True
                            continue

                    write_log(
                        log_path,
                        f"[POST-VIF/IC] Cannot reduce '{max_term}' due to hierarchy; "
                        f"VIF remains {max_vif:.2f}"
                    )
                    break

    final_formula = f"{yvar} ~ " + " + ".join(terms) if terms else f"{yvar} ~ 1"
    return model, final_formula

def backward_elimination_pval_like_code1(
    data,
    yvar,
    main_effects,
    interactions,
    alpha=0.05,
    cov_type=None
):
    terms = main_effects + interactions

    while True and terms:
        formula_test = f"{yvar} ~ " + " + ".join(terms)
        model = maybe_fit(formula_test, data, cov_type)
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
    final_model = maybe_fit(final_formula, data, cov_type)
    return final_model, final_formula

def run_residual_diagnostics(best_model, final_formula, df_clean, suffix, base_out_dir, log_path=None):
    plots_dir        = os.path.join(base_out_dir, "residual_diagnostics")
    diagnostic_dir   = os.path.join(base_out_dir, "diagnostics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(diagnostic_dir, exist_ok=True)

    try:
        resid_raw   = np.asarray(best_model.resid).astype(float)
    except Exception as e:
        resid_raw = np.array([])
        if log_path: write_log(log_path, f"[DIAG] resid extract error: {e}")
    try:
        fitted      = np.asarray(best_model.fittedvalues).astype(float)
    except Exception as e:
        fitted = np.array([])
        if log_path: write_log(log_path, f"[DIAG] fitted extract error: {e}")

    try:
        influence   = best_model.get_influence()
        resid_stud  = np.asarray(influence.resid_studentized_internal).astype(float)
    except Exception as e:
        if log_path: write_log(log_path, f"[DIAG] influence error: {e}; fallback to raw residuals")
        resid_stud = resid_raw.copy()

    try:
        residuals_csv_path = os.path.join(diagnostic_dir, f"residuals_{suffix}.csv")
        pd.DataFrame({
            "fitted": fitted,
            "residuals": resid_raw,
            "studentized_residuals": resid_stud,
        }).to_csv(residuals_csv_path, index=False, encoding="utf-8", sep=";")
    except Exception as e:
        if log_path: write_log(log_path, f"[DIAG] residual CSV error: {e}")

    try:
        m_plot = np.isfinite(fitted) & np.isfinite(resid_stud)
    except Exception:
        m_plot = np.zeros_like(resid_stud, dtype=bool)

    try:
        fig_rvf_path = os.path.join(plots_dir, f"residuals_vs_fitted_{suffix}.png")
        plt.figure(figsize=(8,6))
        if m_plot.any():
            plt.scatter(fitted[m_plot], resid_stud[m_plot], alpha=0.7, edgecolor='k', linewidth=0.3)
        plt.axhline(0, linestyle='--')
        plt.xlabel("Fitted values")
        plt.ylabel("Studentized residuals")
        plt.title(f"Residuals vs Fitted — [{suffix}]")
        plt.tight_layout()
        plt.savefig(fig_rvf_path, dpi=300)
        plt.close()
    except Exception as e:
        if log_path: write_log(log_path, f"[DIAG] RVF plot error: {e}")
        try: plt.close()
        except Exception: pass

    try:
        if suffix == "FULL":
            fig_qq_path = os.path.join(plots_dir, "qqplot_studentize.png")
        else:
            fig_qq_path = os.path.join(plots_dir, f"qqplot_studentized_{suffix}.png")

        plt.figure(figsize=(6,6))
        mqq = np.isfinite(resid_stud)
        if mqq.any():
            qqplot(resid_stud[mqq], line='45', fit=True)
        else:
            plt.text(0.5, 0.5, "No finite residuals", ha='center', va='center')
        plt.title(f"QQ plot — Studentized residuals [{suffix}]")
        plt.tight_layout()
        plt.savefig(fig_qq_path, dpi=300)
        plt.close()
    except Exception as e:
        if log_path: write_log(log_path, f"[DIAG] QQ plot error: {e}")
        try: plt.close()
        except Exception: pass

    try:
        if suffix == "FULL":
            shap_txt_path = os.path.join(diagnostic_dir, "shapiro_residuals_global.txt")
        else:
            shap_txt_path = os.path.join(diagnostic_dir, f"shapiro_residuals_global_{suffix}.txt")

        mfinite = np.isfinite(resid_raw)
        ruse = resid_raw[mfinite]
        if ruse.size >= 3:
            if ruse.size > 5000:
                rng = np.random.default_rng(123)
                ruse = rng.choice(ruse, size=5000, replace=False)
            shap_stat, shap_p = shapiro(ruse)
            with open(shap_txt_path, "w", encoding="utf-8") as f:
                f.write("Shapiro-Wilk (global residuals)\n")
                f.write(f"n_used: {ruse.size}\n")
                f.write(f"statistic: {shap_stat:.4f}\n")
                f.write(f"p-value:  {shap_p:.6g}\n")
        else:
            with open(shap_txt_path, "w", encoding="utf-8") as f:
                f.write("Shapiro-Wilk not run: too few usable residuals (n<3).\n")
    except Exception as e:
        try:
            with open(shap_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Shapiro-Wilk ERROR: {e}\n")
        except Exception:
            pass

    try:
        if suffix == "FULL":
            bp_txt_path = os.path.join(diagnostic_dir, "heteroskedasticity_breusch_pagan.txt")
        else:
            bp_txt_path = os.path.join(diagnostic_dir, f"heteroskedasticity_breusch_pagan_{suffix}.txt")

        exog = getattr(best_model.model, 'exog', None)
        if exog is not None and resid_raw.size == exog.shape[0]:
            m_bp = np.all(np.isfinite(exog), axis=1) & np.isfinite(resid_raw)
            bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid_raw[m_bp], exog[m_bp, :])
            with open(bp_txt_path, "w", encoding="utf-8") as f:
                f.write("Breusch-Pagan test (heteroskedasticity)\n")
                f.write(f"LM statistic: {bp_lm:.4f}, LM p-value: {bp_lm_p:.6g}\n")
                f.write(f"F  statistic: {bp_f:.4f},  F p-value:  {bp_f_p:.6g}\n")
        else:
            with open(bp_txt_path, "w", encoding="utf-8") as f:
                f.write("Breusch-Pagan not run: incompatible shapes or missing exog.\n")
    except Exception as e:
        try:
            with open(bp_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Breusch-Pagan ERROR: {e}\n")
        except Exception:
            pass

    if log_path:
        write_log(log_path, "Residual diagnostics (robust) completed.")
        write_log(log_path, f"- Plots: {plots_dir}")
        write_log(log_path, f"- Reports and tables: {diagnostic_dir}")

def plot_regression_line(model, data, yvar, covariate, ref_cols, out_dir, suffix):
    os.makedirs(out_dir, exist_ok=True)
    x_obs = pd.to_numeric(data[covariate], errors='coerce').dropna()
    if x_obs.empty:
        return
    xgrid = np.linspace(x_obs.min(), x_obs.max(), 120)
    ref = {}
    for c in ref_cols:
        if c == covariate:
            continue
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c]):
            ref[c] = float(pd.to_numeric(data[c], errors='coerce').mean())
        elif c in data.columns:
            m = data[c].mode(dropna=True)
            ref[c] = (m.iat[0] if not m.empty else data[c].dropna().iloc[0]) if data[c].dropna().size else np.nan
    new_df = pd.DataFrame({**ref, covariate: xgrid})
    sf = model.get_prediction(new_df).summary_frame(alpha=ALPHA)

    plt.figure(figsize=(8,6))
    plt.scatter(data[covariate], data[yvar], alpha=0.5, s=25)
    plt.plot(xgrid, sf['mean'])
    plt.fill_between(xgrid, sf['mean_ci_lower'], sf['mean_ci_upper'], alpha=0.2)
    plt.xlabel(covariate)
    plt.ylabel(yvar)
    plt.title(f"Regression {yvar} ~ {covariate} — [{suffix}]")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"regression_line_{suffix}_{covariate}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def _anova_try(model, typ):
    robust_opt = None
    if COV_TYPE:
        ct = str(COV_TYPE).lower()
        if ct in ("hc0", "hc1", "hc2", "hc3"):
            robust_opt = ct

    try:
        if robust_opt is not None:
            return anova_lm(model, typ=typ, robust=robust_opt)
        else:
            return anova_lm(model, typ=typ)
    except Exception:
        return None

def anova_on_classic(model, choice="auto"):
    ch = (choice or "").upper()
    if ch == "I":
        return _anova_try(model, 1)
    if ch == "II":
        return _anova_try(model, 2)
    if ch == "III":
        return _anova_try(model, 3)
    for t in (3, 2, 1):
        tbl = _anova_try(model, t)
        if tbl is not None:
            return tbl
    return None

def _mode_or_first(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iat[0] if not m.empty else (s.dropna().iloc[0] if s.dropna().size else np.nan)

def emms_for_factor(model, data, factor, covariates, other_factors, alpha=0.05):
    ref = {}
    for c in covariates:
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c]):
            ref[c] = float(pd.to_numeric(data[c], errors='coerce').mean())
        elif c in data.columns:
            ref[c] = _mode_or_first(data[c])
    for f in other_factors:
        if f in data.columns:
            ref[f] = _mode_or_first(data[f].astype('category'))

    if factor not in data.columns:
        raise ValueError(f"Factor '{factor}' not present in data")
    levs = pd.Categorical(data[factor]).categories
    if len(levs) < 2:
        raise ValueError(f"Factor '{factor}' has fewer than 2 levels")

    new_df = pd.DataFrame([{**ref, factor: lev} for lev in levs])
    pred = model.get_prediction(new_df).summary_frame(alpha=alpha)
    emms = pd.DataFrame({
        factor: levs.astype(str),
        "emm": pred["mean"].values,
        "se": pred["mean_se"].values,
        "ci_low": pred["mean_ci_lower"].values,
        "ci_high": pred["mean_ci_upper"].values
    })

    design_info = model.model.data.design_info
    exog = build_design_matrices([design_info], new_df, return_type='dataframe')[0]
    return emms, exog, list(levs.astype(str))

def pairwise_emm_contrasts(model, exog, levels, alpha=0.05, adjust="holm"):
    rows = []
    for i, j in combinations(range(len(levels)), 2):
        L = exog.iloc[i].values - exog.iloc[j].values
        tt = model.t_test(L)
        diff, se, tval, pval = float(tt.effect), float(tt.sd), float(tt.tvalue), float(tt.pvalue)
        df = float(model.df_resid)
        tcrit = stats.t.ppf(1 - alpha/2, df) if df > 0 else np.nan
        ci_lo = diff - tcrit * se if np.isfinite(tcrit) else np.nan
        ci_hi = diff + tcrit * se if np.isfinite(tcrit) else np.nan
        rows.append([levels[i], levels[j], diff, se, tval, df, pval, ci_lo, ci_hi])

    out = pd.DataFrame(rows, columns=["level_i","level_j","diff","se","t","df","p_raw","ci_low","ci_high"])
    if len(out):
        out["p_adj"] = multipletests(out["p_raw"].values, method=adjust)[1]
        out["reject"] = out["p_adj"] < alpha
    return out

def tukey_for_factor(y_col: str, factor_name: str, data: pd.DataFrame,
                     covariates: list, all_factors: list, alpha=0.05,
                     mode="residualized", out_dir="output"):
    needed = [y_col, factor_name] + list(covariates) + [f for f in all_factors if f != factor_name]
    d = data[[c for c in needed if c in data.columns]].dropna().copy()
    if len(d) == 0:
        return
    if mode == "raw" or (not covariates and len(all_factors) <= 1):
        endog = pd.to_numeric(d[y_col], errors='coerce')
        groups = d[factor_name].astype('category')
    else:
        other_factors = [f for f in all_factors if f != factor_name and f in d.columns]
        base_terms = []
        if covariates:    base_terms = base_terms + [c for c in covariates if c in d.columns]
        if other_factors: base_terms = base_terms + [factor_term(f) for f in other_factors]
        formula = f"{y_col} ~ " + " + ".join(base_terms) if base_terms else f"{y_col} ~ 1"
        base_model = smf.ols(formula, data=d).fit()
        endog = base_model.resid
        groups = d[factor_name].astype('category')

    try:
        tk = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)
        summ = tk.summary()
        rows = summ.data[1:]
        headers = summ.data[0]
        df_tk = pd.DataFrame(rows, columns=headers)
        os.makedirs(out_dir, exist_ok=True)
        df_tk.to_csv(os.path.join(out_dir, f"{y_col}_Tukey_{factor_name}_{mode}.csv"),
                     index=False, encoding="utf-8-sig", sep=";")
    except Exception as e:
        print(f"[WARN] Tukey HSD failed for {y_col} ~ {factor_name}: {e}")


log_path = os.path.join(run_subdir, "vif_enforcement_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] VIF enforcement log + ANCOVA additions\n")

df = read_csv_auto_sep(data_path)
param = read_csv_auto_sep(param_path)

if PARAM_SELECT_MODE.lower() == "filter":
    if PARAM_FILTER_COL not in param.columns:
        raise KeyError(f"Filter column '{PARAM_FILTER_COL}' not found in Parameter.csv")
    sel = param[param[PARAM_FILTER_COL] == PARAM_FILTER_VALUE]
    if sel.empty:
        raise ValueError(f"No row in Parameter.csv with {PARAM_FILTER_COL} == {PARAM_FILTER_VALUE!r}")
    param_row = sel.iloc[0]
    write_log(log_path, f"[Parameter] Using filtered row {PARAM_FILTER_COL}={PARAM_FILTER_VALUE!r}")
else:
    if not (0 <= int(PARAM_ROW_INDEX) < len(param)):
        raise IndexError(f"PARAM_ROW_INDEX out of range: {PARAM_ROW_INDEX} (available rows: 0..{len(param)-1})")
    param_row = param.iloc[int(PARAM_ROW_INDEX)]
    write_log(log_path, f"[Parameter] Using row by index: {PARAM_ROW_INDEX}")

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

main_effects_init = X_for_formula + [factor_term(cat) for cat in f_vars]
interactions_init = [f"{x}:{factor_term(cat)}" for x in X_for_formula for cat in f_vars]
formula_full_init = (
    f"{yvar} ~ " + " + ".join(main_effects_init + interactions_init)
    if (main_effects_init or interactions_init)
    else f"{yvar} ~ 1"
)

needed_cols = [yvar] + X_for_formula + f_vars
df_clean = df[needed_cols].dropna().copy()

if KEEP_ALL_FACTORS:
    f_vars_kept = list(f_vars)
    f_vars_removed = []
    write_log(log_path, "[Factor filter] DISABLED: keeping all factors.")
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
with open(os.path.join(full_model_dir, "regression_full_summary.txt"), "w", encoding="utf-8") as f:
    f.write(model_full.summary().as_text())

try:
    run_residual_diagnostics(model_full, formula_full, df_clean,
                             suffix="FULL", base_out_dir=full_model_dir, log_path=log_path)
except Exception as e:
    write_log(log_path, f"Residual diagnostics ERROR (FULL): {e}")

_full_plots_dir = os.path.join(full_model_dir, "residual_diagnostics")
_full_diag_dir  = os.path.join(full_model_dir, "diagnostics")
expected_full = [
    os.path.join(_full_plots_dir,  "residuals_vs_fitted_FULL.png"),
    os.path.join(_full_plots_dir,  "qqplot_studentize.png"),
    os.path.join(_full_diag_dir,   "heteroskedasticity_breusch_pagan.txt"),
    os.path.join(_full_diag_dir,   "shapiro_residuals_global.txt"),
]
missing_full = [p for p in expected_full if not os.path.exists(p)]
if missing_full:
    write_log(log_path, f"[WARN] Missing FULL artifacts: {missing_full}. Regenerating...")
    try:
        run_residual_diagnostics(model_full, formula_full, df_clean,
                                 suffix="FULL", base_out_dir=full_model_dir, log_path=log_path)
    except Exception as e:
        write_log(log_path, f"[ERROR] Regenerating FULL artifacts failed: {e}")

if RUN_BACKWARD_SELECTION:
    if SELECTION_MODE.upper() == "PVALUE":
        if ENFORCE_VIF:
            write_log(log_path,
                      "[WARN] SELECTION_MODE=PVALUE does not apply VIF enforcement during selection.")
        best_model, final_formula = backward_elimination_pval_like_code1(
            df_clean,
            yvar,
            main_effects,
            interactions,
            alpha=PVALUE_ALPHA,
            cov_type=COV_TYPE
        )
        write_log(log_path, f"[PVALUE] Selected formula: {final_formula}")
    else:
        best_model, final_formula = backward_elimination_ic_with_vif(
            df_clean, yvar, main_effects, interactions, COV_TYPE, IC,
            enforce_vif=ENFORCE_VIF, vif_threshold=VIF_THRESHOLD,
            release_interactions=VIF_RELEASE_INTERACTIONS, log_path=log_path
        )
        write_log(log_path, f"[IC] Selected (with VIF) formula: {final_formula}")
else:
    write_log(log_path,
              "[SELECTION] Skipped: RUN_BACKWARD_SELECTION=False — using FULL model as final.")
    best_model = model_full
    final_formula = formula_full

MODEL_TAG = (SELECTION_MODE.upper() if RUN_BACKWARD_SELECTION else "FULL")

if RUN_BACKWARD_SELECTION:
    sel_name = ("regression_selected_summary.txt"
                if SELECTION_MODE.upper() == "PVALUE"
                else f"regression_selected_summary_{IC}.txt")
    fml_name = ("selected_formula_PVALUE.txt"
                if SELECTION_MODE.upper() == "PVALUE"
                else f"selected_formula_{IC}.txt")
else:
    sel_name = None
    fml_name = "selected_formula_FULL.txt"

if sel_name is not None:
    with open(os.path.join(selected_model_dir, sel_name), "w", encoding="utf-8") as f:
        f.write(best_model.summary().as_text())

with open(os.path.join(selected_model_dir, fml_name), "w", encoding="utf-8") as f:
    f.write(final_formula + "\n")
write_log(log_path, f"Final formula: {final_formula}")

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

summary_table.to_csv(os.path.join(selected_model_dir,
                                  f"selected_model_table_{MODEL_TAG}.csv"),
                     index=False, sep=";")

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
df_plot = df_plot.sort_values(["t_abs", "abs_coef"],
                              ascending=[False, False]) \
                 .drop(columns="abs_coef").reset_index(drop=True)
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
plt.title(f"Coefficients and 95% CI — {MODEL_TAG}\n"
          f"{cov_str}, predictors {center_str}\n"
          f"(* marks p < {ALPHA:.2f})")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(selected_model_dir,
                         f"coef_plot_selected_points_importance_sig_{MODEL_TAG}.png"),
            dpi=300)
plt.close()

ci_full = model_full.conf_int()
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
df_plot_full = df_plot_full.sort_values(["t_abs", "abs_coef"],
                                        ascending=[False, False]) \
                           .drop(columns="abs_coef").reset_index(drop=True)
df_plot_full["Label"] = df_plot_full["Term"].apply(clean_term_label)

mask = df_plot_full["P_value"] < ALPHA
df_plot_full.loc[mask, "Label"] = df_plot_full.loc[mask, "Label"].astype(str) + " *"

y_full = np.arange(len(df_plot_full))
plt.figure(figsize=(10, max(6, 0.4 * len(df_plot_full) + 2)))
for yi, lo, hi in zip(y_full, df_plot_full["Conf_low"], df_plot_full["Conf_high"]):
    plt.hlines(yi, lo, hi, linewidth=2, color='dimgray')
plt.scatter(df_plot_full["Coefficient"], y_full, s=70, color="black", zorder=3)
plt.axvline(0, linestyle='--', linewidth=1.5, color="red")

plt.yticks(y_full, df_plot_full["Label"])
plt.gca().invert_yaxis()

plt.title(f"Coefficients and 95% CI — FULL\n"
          f"{cov_str}, predictors {center_str}\n"
          f"(* marks p < {ALPHA:.2f})")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(full_model_dir,
                         "coef_plot_selected_points_importance_sig_FULL.png"),
            dpi=300)
plt.close()

if COMPUTE_VIF:
    vif_full = compute_design_vif(formula_full, df_clean)
    vif_full = vif_full[vif_full["DesignCol"] != "Intercept"]
    vif_full.to_csv(
        os.path.join(full_model_dir, "VIF_selected_model_FULL.csv"),
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )
    vmax_full = float(vif_full["VIF"].max()) if not vif_full.empty else 0.0
    write_log(
        log_path,
        f"FULL model VIF saved. Max VIF = {vmax_full:.2f} "
        f"(threshold = {VIF_THRESHOLD:.2f})"
    )

if COMPUTE_VIF:
    vif_final = compute_design_vif(final_formula, df_clean)
    vif_final = vif_final[vif_final["DesignCol"] != "Intercept"]
    vif_final.to_csv(
        os.path.join(selected_model_dir, f"VIF_selected_model_{MODEL_TAG}.csv"),
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )
    vmax = float(vif_final["VIF"].max()) if not vif_final.empty else 0.0
    write_log(log_path,
              f"Final VIF saved. Max VIF = {vmax:.2f} "
              f"(threshold = {VIF_THRESHOLD:.2f})")

param_names = list(best_model.params.index)
p = len(param_names)

def find_indices_for_regex(pattern):
    rgx = re.compile(pattern)
    return [i for i, name in enumerate(param_names) if rgx.match(name)]

groups = []
for cat in f_vars_kept:
    pat = rf"^C\({re.escape(cat)}(?:,\s*Sum)?\)\[.+\]$"
    idxs = find_indices_for_regex(pat)
    if idxs:
        groups.append({"group": f"Factor: {cat}", "indices": idxs})
for x in X_for_formula:
    for cat in f_vars_kept:
        pat1 = rf"^{re.escape(x)}:C\({re.escape(cat)}(?:,\s*Sum)?\)\[.+\]$"
        pat2 = rf"^C\({re.escape(cat)}(?:,\s*Sum)?\)\[.+\]:{re.escape(x)}$"
        idxs = sorted(set(find_indices_for_regex(pat1) + find_indices_for_regex(pat2)))
        if idxs:
            groups.append({"group": f"Interaction: {x} × {cat}", "indices": idxs})

wald_rows, mapping_rows = [], []
for g in groups:
    idxs = g["indices"]
    if not idxs:
        continue
    R = np.zeros((len(idxs), p))
    for r, j in enumerate(idxs):
        R[r, j] = 1.0
        mapping_rows.append({"Group": g["group"], "ParamName": param_names[j]})
    res = best_model.wald_test(R)
    try:
        stat_val = float(np.asarray(res.statistic).squeeze())
    except Exception:
        stat_val = np.nan
    try:
        pval = float(res.pvalue)
    except Exception:
        pval = np.nan
    wald_rows.append({
        "Group": g["group"],
        "Num params (df_num)": len(idxs),
        "Statistic": stat_val,
        "p-value": pval
    })
wald_df = pd.DataFrame(wald_rows)
wald_map_df = pd.DataFrame(mapping_rows)

try:
    run_residual_diagnostics(best_model, final_formula, df_clean,
                             suffix=MODEL_TAG, base_out_dir=selected_model_dir,
                             log_path=log_path)
except Exception as e:
    write_log(log_path, f"Residual diagnostics ERROR (essential): {e}")

anova_full_tbl = anova_on_classic(model_full, choice=ANOVA_TYPE)
if anova_full_tbl is not None:
    type_lab = (ANOVA_TYPE or "auto").upper()

    if "sum_sq" in anova_full_tbl.columns:
        if "Residual" in anova_full_tbl.index:
            ss_error_full = float(anova_full_tbl.loc["Residual", "sum_sq"])
        else:
            ss_error_full = float(anova_full_tbl["sum_sq"].iloc[-1])

        anova_full_tbl["eta2_p"] = np.nan
        mask_effects_full = anova_full_tbl.index != "Residual"
        ss_effects_full = anova_full_tbl.loc[mask_effects_full, "sum_sq"]
        anova_full_tbl.loc[mask_effects_full, "eta2_p"] = (
            ss_effects_full / (ss_effects_full + ss_error_full)
        )
    else:
        write_log(
            log_path,
            "[ANCOVA FULL] 'sum_sq' not present in ANOVA table: cannot compute η²_p."
        )

    anova_full_tbl.to_csv(
        os.path.join(full_model_dir, f"ANCOVA_table_type{type_lab}_FULL.csv"),
        encoding="utf-8-sig",
        sep=";"
    )
    write_log(log_path, f"Saved ANCOVA FULL table (type {type_lab}) with eta2_p.")
else:
    write_log(log_path,
              "ANCOVA FULL table could not be computed (all ANOVA types failed).")

anova_tbl = anova_on_classic(best_model, choice=ANOVA_TYPE)
if anova_tbl is not None:
    type_lab = (ANOVA_TYPE or "auto").upper()

    if "sum_sq" in anova_tbl.columns:
        if "Residual" in anova_tbl.index:
            ss_error = float(anova_tbl.loc["Residual", "sum_sq"])
        else:
            ss_error = float(anova_tbl["sum_sq"].iloc[-1])

        anova_tbl["eta2_p"] = np.nan
        mask_effects = anova_tbl.index != "Residual"
        ss_effects = anova_tbl.loc[mask_effects, "sum_sq"]
        anova_tbl.loc[mask_effects, "eta2_p"] = ss_effects / (ss_effects + ss_error)
    else:
        write_log(log_path,
                  "[ANCOVA] 'sum_sq' not present in ANOVA table: cannot compute η²_p.")

    anova_tbl.to_csv(
        os.path.join(selected_model_dir, f"ANCOVA_table_type{type_lab}.csv"),
        encoding="utf-8-sig",
        sep=";"
    )
    write_log(log_path, f"Saved ANCOVA table (type {type_lab}) with eta2_p.")
else:
    write_log(log_path,
              "ANCOVA table could not be computed (all ANOVA types failed).")

if RUN_BACKWARD_SELECTION:
    emm_dir_full = os.path.join(full_model_dir, "EMM")
    os.makedirs(emm_dir_full, exist_ok=True)

    for fac in f_vars_kept:
        if fac not in df_clean.columns or df_clean[fac].nunique() < 2:
            continue
        covs_in_model_full = [c for c in df_clean.columns if c not in [yvar] + f_vars_kept]
        try:
            emms_tbl_full, exog_emm_full, levels_full = emms_for_factor(
                model=model_full,
                data=df_clean,
                factor=fac,
                covariates=covs_in_model_full,
                other_factors=[f for f in f_vars_kept if f != fac],
                alpha=ALPHA
            )
            emms_path_full = os.path.join(emm_dir_full, f"EMMs_{fac}_FULL.csv")
            emms_tbl_full.to_csv(emms_path_full, index=False, encoding="utf-8-sig", sep=";")

            xs = np.arange(len(emms_tbl_full))
            yv  = emms_tbl_full["emm"].values
            lo = emms_tbl_full["ci_low"].values
            hi = emms_tbl_full["ci_high"].values
            yerr = np.vstack([yv - lo, hi - yv])
            plt.figure(figsize=(7,5))
            plt.errorbar(xs, yv, yerr=yerr, fmt='o', capsize=5)
            plt.xticks(xs, emms_tbl_full[fac].astype(str))
            plt.xlabel(fac); plt.ylabel(f"EMM of {yvar}")
            plt.title(f"EMM (LS-means) with CI — {fac} [FULL]")
            plt.tight_layout()
            plt.savefig(os.path.join(emm_dir_full, f"{yvar}_EMMplot_{fac}_FULL.png"), dpi=300)
            plt.close()

            pw_full = pairwise_emm_contrasts(model_full, exog_emm_full, levels_full,
                                             alpha=ALPHA, adjust=PAIRWISE_ADJUST)
            pw_path_full = os.path.join(emm_dir_full,
                                        f"PairwiseEMM_{fac}_{PAIRWISE_ADJUST}_FULL.csv")
            pw_full.to_csv(pw_path_full, index=False, encoding='utf-8-sig', sep=";")
        except Exception as e:
            write_log(log_path, f"[WARN] EMM/pairwise FULL failed for factor {fac}: {e}")

emm_dir = os.path.join(selected_model_dir, "EMM")
os.makedirs(emm_dir, exist_ok=True)

for fac in f_vars_kept:
    if fac not in df_clean.columns or df_clean[fac].nunique() < 2:
        continue
    covs_in_model = [c for c in df_clean.columns if c not in [yvar] + f_vars_kept]
    try:
        emms_tbl, exog_emm, levels = emms_for_factor(
            model=best_model,
            data=df_clean,
            factor=fac,
            covariates=covs_in_model,
            other_factors=[f for f in f_vars_kept if f != fac],
            alpha=ALPHA
        )
        emms_path = os.path.join(emm_dir, f"EMMs_{fac}.csv")
        emms_tbl.to_csv(emms_path, index=False, encoding="utf-8-sig", sep=";")

        xs = np.arange(len(emms_tbl))
        yv  = emms_tbl["emm"].values
        lo = emms_tbl["ci_low"].values
        hi = emms_tbl["ci_high"].values
        yerr = np.vstack([yv - lo, hi - yv])
        plt.figure(figsize=(7,5))
        plt.errorbar(xs, yv, yerr=yerr, fmt='o', capsize=5)
        plt.xticks(xs, emms_tbl[fac].astype(str))
        plt.xlabel(fac); plt.ylabel(f"EMM of {yvar}")
        plt.title(f"EMM (LS-means) with CI — {fac}")
        plt.tight_layout()
        plt.savefig(os.path.join(emm_dir, f"{yvar}_EMMplot_{fac}.png"), dpi=300)
        plt.close()

        pw = pairwise_emm_contrasts(best_model, exog_emm, levels,
                                    alpha=ALPHA, adjust=PAIRWISE_ADJUST)
        pw_path = os.path.join(emm_dir,
                               f"PairwiseEMM_{fac}_{PAIRWISE_ADJUST}.csv")
        pw.to_csv(pw_path, index=False, encoding='utf-8-sig', sep=";")
    except Exception as e:
        write_log(log_path, f"[WARN] EMM/pairwise failed for factor {fac}: {e}")

def _term_contains_var(term: str, var: str) -> bool:
    return (
        (var == term)
        or term.startswith(var + ":")
        or term.endswith(":" + var)
        or ((":" + var + ":") in (":" + term + ":"))
    )

line_dir = os.path.join(selected_model_dir, "ANCOVA_Lines")
os.makedirs(line_dir, exist_ok=True)

_rhs = final_formula.split("~", 1)[1].strip() if "~" in final_formula else ""
rhs_terms = [t.strip() for t in _rhs.split("+")] if _rhs else []

num_covs_in_model = [
    x for x in X_for_formula
    if (x in df_clean.columns)
    and pd.api.types.is_numeric_dtype(df_clean[x])
    and any(_term_contains_var(t, x) for t in rhs_terms)
]

_rhs_full = formula_full.split("~", 1)[1].strip() if "~" in formula_full else ""
rhs_terms_full = [t.strip() for t in _rhs_full.split("+")] if _rhs_full else []

num_covs_in_model_full = [
    x for x in X_for_formula
    if (x in df_clean.columns)
    and pd.api.types.is_numeric_dtype(df_clean[x])
    and any(_term_contains_var(t, x) for t in rhs_terms_full)
]

if RUN_BACKWARD_SELECTION and len(num_covs_in_model_full) > 0:
    cov0_full = num_covs_in_model_full[0]
    x_obs_full = pd.to_numeric(df_clean[cov0_full], errors="coerce").dropna()
    if len(x_obs_full) > 0:
        xgrid_full = np.linspace(x_obs_full.min(), x_obs_full.max(), 120)
        line_dir_full = os.path.join(full_model_dir, "ANCOVA_Lines")
        os.makedirs(line_dir_full, exist_ok=True)

        for fac in f_vars_kept:
            fac_in_model_full = any(("C(" + fac) in t for t in rhs_terms_full)
            if not fac_in_model_full:
                continue
            if (fac not in df_clean.columns) or (df_clean[fac].nunique() < 2):
                continue

            new_base_full = {}
            for c in X_for_formula:
                if c == cov0_full:
                    continue
                if (c in df_clean.columns) and pd.api.types.is_numeric_dtype(df_clean[c]):
                    new_base_full[c] = float(pd.to_numeric(df_clean[c],
                                                           errors="coerce").mean())
            for fct in f_vars_kept:
                if fct == fac:
                    continue
                if fct in df_clean.columns:
                    new_base_full[fct] = _mode_or_first(df_clean[fct].astype("category"))

            fig, ax = plt.subplots(figsize=(8, 6))
            for lev in pd.Categorical(df_clean[fac]).categories:
                new_df_full = pd.DataFrame({**new_base_full,
                                            fac: [lev] * len(xgrid_full),
                                            cov0_full: xgrid_full})
                sf_full = model_full.get_prediction(new_df_full).summary_frame(alpha=ALPHA)
                ln = ax.plot(xgrid_full, sf_full["mean"], label=str(lev))
                color = ln[0].get_color()
                ax.fill_between(xgrid_full,
                                sf_full["mean_ci_lower"],
                                sf_full["mean_ci_upper"],
                                alpha=0.15,
                                color=color)
                mask = (df_clean[fac] == lev)
                ax.scatter(
                    df_clean.loc[mask, cov0_full],
                    df_clean.loc[mask, yvar],
                    s=18,
                    alpha=0.35,
                    color=color
                )
            ax.set_xlabel(cov0_full)
            ax.set_ylabel(yvar)
            ax.set_title(
                f"Regression lines by {fac} with {cov0_full} — MODEL FULL "
                f"(CI {int((1-ALPHA)*100)}%)"
            )
            ax.legend(title=fac)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    line_dir_full,
                    f"{yvar}_Lines_by_{fac}_vs_{cov0_full}_FULL.png"
                ),
                dpi=300
            )
            plt.close(fig)

if len(num_covs_in_model) > 0:
    cov0 = num_covs_in_model[0]
    x_obs = pd.to_numeric(df_clean[cov0], errors="coerce").dropna()
    if len(x_obs) > 0:
        xgrid = np.linspace(x_obs.min(), x_obs.max(), 120)

        for fac in f_vars_kept:
            fac_in_model = any(("C(" + fac) in t for t in rhs_terms)
            if not fac_in_model:
                continue
            if (fac not in df_clean.columns) or (df_clean[fac].nunique() < 2):
                continue

            new_base = {}
            for c in X_for_formula:
                if c == cov0:
                    continue
                if (c in df_clean.columns) and pd.api.types.is_numeric_dtype(df_clean[c]):
                    new_base[c] = float(pd.to_numeric(df_clean[c],
                                                      errors="coerce").mean())
            for fct in f_vars_kept:
                if fct == fac:
                    continue
                if fct in df_clean.columns:
                    new_base[fct] = _mode_or_first(df_clean[fct].astype("category"))

            fig, ax = plt.subplots(figsize=(8, 6))
            for lev in pd.Categorical(df_clean[fac]).categories:
                new_df = pd.DataFrame({**new_base,
                                       fac: [lev] * len(xgrid),
                                       cov0: xgrid})
                sf = best_model.get_prediction(new_df).summary_frame(alpha=ALPHA)
                ln = ax.plot(xgrid, sf["mean"], label=str(lev))
                color = ln[0].get_color()
                ax.fill_between(xgrid,
                                sf["mean_ci_lower"],
                                sf["mean_ci_upper"],
                                alpha=0.15,
                                color=color)
                mask = (df_clean[fac] == lev)
                ax.scatter(
                    df_clean.loc[mask, cov0],
                    df_clean.loc[mask, yvar],
                    s=18,
                    alpha=0.35,
                    color=color
                )
            ax.set_xlabel(cov0)
            ax.set_ylabel(yvar)
            ax.set_title(
                f"Regression lines by {fac} with {cov0} — MODEL {MODEL_TAG} "
                f"(CI {int((1-ALPHA)*100)}%)"
            )
            ax.legend(title=fac)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    line_dir,
                    f"{yvar}_Lines_by_{fac}_vs_{cov0}_{MODEL_TAG}.png"
                ),
                dpi=300
            )
            plt.close(fig)

regline_dir_full  = os.path.join(full_model_dir, "Regression_Lines")
regline_dir_final = os.path.join(selected_model_dir, "Regression_Lines")
os.makedirs(regline_dir_full, exist_ok=True)
os.makedirs(regline_dir_final, exist_ok=True)

if num_covs_in_model:
    cov_main = num_covs_in_model[0]
    try:
        plot_regression_line(model_full, df_clean, yvar, cov_main,
                             X_for_formula + f_vars_kept, regline_dir_full,
                             suffix="FULL")
    except Exception as e:
        write_log(log_path, f"[WARN] Regression line (FULL) failed: {e}")
    try:
        plot_regression_line(best_model, df_clean, yvar, cov_main,
                             X_for_formula + f_vars_kept, regline_dir_final,
                             suffix=MODEL_TAG)
    except Exception as e:
        write_log(log_path, f"[WARN] Regression line (FINAL) failed: {e}")

if TUKEY_ENABLED and f_vars_kept and RUN_BACKWARD_SELECTION:
    tk_dir_full = os.path.join(full_model_dir, "TukeyHSD")
    os.makedirs(tk_dir_full, exist_ok=True)

    rhs_full_tk = formula_full.split("~", 1)[1].strip()
    rhs_terms_full_tk = [t.strip() for t in rhs_full_tk.split("+")] if rhs_full_tk else []
    covariates_in_model_full = [
        x for x in X_for_formula
        if any(x == t or x+":" in t or ":"+x in t for t in rhs_terms_full_tk)
    ]

    for fac in f_vars_kept:
        try:
            tukey_for_factor(
                y_col=yvar,
                factor_name=fac,
                data=df_clean,
                covariates=covariates_in_model_full,
                all_factors=f_vars_kept,
                alpha=ALPHA,
                mode=TUKEY_MODE,
                out_dir=tk_dir_full
            )
        except Exception as e:
            write_log(log_path, f"[WARN] Tukey FULL failed for {fac}: {e}")

if TUKEY_ENABLED and f_vars_kept:
    tk_dir = os.path.join(selected_model_dir, "TukeyHSD")
    os.makedirs(tk_dir, exist_ok=True)
    rhs = final_formula.split("~",1)[1].strip()
    rhs_terms = [t.strip() for t in rhs.split("+")] if rhs else []
    covariates_in_model = [
        x for x in X_for_formula
        if any(x == t or x+":" in t or ":"+x in t for t in rhs_terms)
    ]
    for fac in f_vars_kept:
        try:
            tukey_for_factor(
                y_col=yvar,
                factor_name=fac,
                data=df_clean,
                covariates=covariates_in_model,
                all_factors=f_vars_kept,
                alpha=ALPHA,
                mode=TUKEY_MODE,
                out_dir=tk_dir
            )
        except Exception as e:
            write_log(log_path, f"[WARN] Tukey failed for {fac}: {e}")

write_log(log_path, "All results saved in: " + run_subdir)
print("Done. ANCOVA outputs available at:", run_subdir)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
