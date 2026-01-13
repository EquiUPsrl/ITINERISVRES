import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--aggregated_data_file', action='store', type=str, required=True, dest='aggregated_data_file')

arg_parser.add_argument('--parameters_file_csv', action='store', type=str, required=True, dest='parameters_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

aggregated_data_file = args.aggregated_data_file.replace('"','')
parameters_file_csv = args.parameters_file_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

STANDARDIZE_PREDICTORS = True # Set to False to disable

ROBUST_HC3_DEFAULT = False # Robust HC3 errors
ROBUST_HAC_DEFAULT = True # Newey–West/HAC
HAC_MAXLAGS_DEFAULT = 'auto' # 'auto' (Andrews' rule) or an integer >=1

enable_vif = True # If True, reduces multicollinearity by eliminating predictors with VIF high
VIF_THRESHOLD = 10 # VIF threshold above which a variable is removed


output_dir = conf_output_path

regression_all_dir = os.path.join(output_dir, 'Regression')
os.makedirs(regression_all_dir, exist_ok=True)

bio_path = aggregated_data_file #os.path.join(output_dir, 'aggregated_data.csv')
param_path = parameters_file_csv

custom_na = ['NA', 'n.c', 'n.a.', '', 'NaN', 'nan', 'NULL', 'null']

df = pd.read_csv(bio_path, sep=None, engine='python', encoding='ISO-8859-1', na_values=custom_na)
param_df = pd.read_csv(param_path, sep=None, engine='python', encoding='ISO-8859-1', na_values=custom_na)

roles = param_df.iloc[2]

target_col = roles[roles == 'Y'].index[0]
predictor_cols_all = roles[roles == 'X'].index.tolist()

predictor_cols = [col for col in predictor_cols_all if col in df.columns]
missing_predictors = [col for col in predictor_cols_all if col not in df.columns]
if missing_predictors:
    print("Warning: The following predictors are absent from the data and will be ignored:", missing_predictors)

level_cols = False  # roles[roles == 'L'].index.tolist()

def _to_bool(x, default=None):
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s in ('true', 't', '1', 'yes', 'y', 'si', 'sì'):
        return True
    if s in ('false', 'f', '0', 'no', 'n'):
        return False
    return default

def _to_maxlags(x, default='auto'):
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s == 'auto':
        return 'auto'
    try:
        v = int(s)
        return v if v >= 1 else default
    except:
        return default

ROBUST_HC3 = ROBUST_HC3_DEFAULT
ROBUST_HAC = ROBUST_HAC_DEFAULT
HAC_MAXLAGS = HAC_MAXLAGS_DEFAULT

if 'ROBUST_HC3' in param_df.columns:
    vals = param_df['ROBUST_HC3'].dropna()
    if not vals.empty:
        ROBUST_HC3 = _to_bool(vals.iloc[0], ROBUST_HC3)

if 'ROBUST_HAC' in param_df.columns:
    vals = param_df['ROBUST_HAC'].dropna()
    if not vals.empty:
        ROBUST_HAC = _to_bool(vals.iloc[0], ROBUST_HAC)

if 'HAC_MAXLAGS' in param_df.columns:
    vals = param_df['HAC_MAXLAGS'].dropna()
    if not vals.empty:
        HAC_MAXLAGS = _to_maxlags(vals.iloc[0], HAC_MAXLAGS)

ROBUST_OPTIONS_DEFAULT = dict(
    enable_hc3=bool(ROBUST_HC3),
    enable_hac=bool(ROBUST_HAC),
    hac_maxlags=HAC_MAXLAGS           # 'auto' or int>=1
)

def save_text(text, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def format_coefficients_table(df_coef):
    header = f"{'Variable':20s} {'Coefficient':>12s} {'Std Error':>12s} {'p-value':>10s} {'Conf Low':>12s} {'Conf High':>12s}\n"
    lines = [header, '-'*80 + '\n']
    for idx, row in df_coef.iterrows():
        name = str(idx)
        lines.append(f"{name:20s} {row['Coefficient']:12.4f} {row['Std Error']:12.4f} {row['p-value']:10.4g} {row['Conf. Low']:12.4f} {row['Conf. High']:12.4f}\n")
    return ''.join(lines)

def format_regularized_table(series, name):
    header = f"{'Predictor':20s} {name:>12s}\n"
    lines = [header, '-'*35 + '\n']
    for idx, val in series.items():
        lines.append(f"{str(idx):20s} {val:12.4f}\n")
    return ''.join(lines)

def plot_residuals_vs_fitted(fitted, residuals, folder, prefix):
    plt.figure(figsize=(8,6))
    plt.scatter(fitted, residuals, edgecolor='k', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Fitted - {prefix}')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{prefix}_residuals_vs_fitted.png'))
    plt.close()

def plot_predicted_vs_observed(y, fitted, folder, prefix):
    plt.figure(figsize=(8,6))
    plt.scatter(y, fitted, edgecolor='k', alpha=0.7)
    min_val = np.nanmin([y.min(), fitted.min()])
    max_val = np.nanmax([y.max(), fitted.max()])
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')  # y = x
    plt.xlabel('Observed')
    plt.ylabel('Predicted (Fitted)')
    plt.title(f'Predicted vs Observed - {prefix}')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{prefix}_predicted_vs_observed.png'))
    plt.close()

def plot_qq(residuals, folder, prefix):
    plt.figure(figsize=(8,6))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title(f'QQ Plot of Residuals - {prefix}')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{prefix}_qqplot_residuals.png'))
    plt.close()

def newey_west_maxlags(n):
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))

def robust_tables(model, cov_type, **cov_kwds):
    """
    Wrapper to obtain results with robust covariances.
    - HC* (HC0/HC1/HC2/HC3) DO NOT accept extra kwargs
    - HAC (Newey–West) accepts kwargs as maxlags (we use it here); we NO LONGER handle use_correction
    """
    cov_type = cov_type.upper()
    if cov_type in ('HC0', 'HC1', 'HC2', 'HC3'):
        r = model.get_robustcov_results(cov_type=cov_type)
    else:
        r = model.get_robustcov_results(cov_type=cov_type, **cov_kwds)

    ci = r.conf_int()  # ndarray (k, 2)
    coef_df = pd.DataFrame({
        'Coefficient': r.params,
        'Std Error': r.bse,
        'p-value': r.pvalues,
        'Conf. Low': ci[:, 0],
        'Conf. High': ci[:, 1],
    }, index=(r.params.index if hasattr(r.params, 'index') else None))
    return r, coef_df


def reduce_multicollinearity(X, threshold=10.0, verbose=True):
    """
    teratively removes predictors with VIF > threshold.
    Returns the new DataFrame with the remaining variables
    and a list of tuples (removed_variable, VIF_value).
    """
    X = X.copy()
    dropped_vars = []
    dropped = True
    while dropped:
        dropped = False
        X_const = sm.add_constant(X)
        vifs = pd.Series(
            [variance_inflation_factor(X_const.values, i)
             for i in range(1, X_const.shape[1])],
            index=X.columns
        )
        max_vif = vifs.max()
        if max_vif > threshold:
            drop_col = vifs.idxmax()
            dropped_vars.append((drop_col, max_vif))
            X.drop(columns=[drop_col], inplace=True)
            dropped = True
            if verbose:
                print(f"Rimosso '{drop_col}' (VIF={max_vif:.2f}) per ridurre la multicollinearità.")
    return X, dropped_vars


def perform_regression(
    data,
    predictors,
    target,
    prefix='output',
    folder='.',
    skip_files=False,
    robust_options=None,
    ridge_alpha=1.0,
    lasso_alpha=0.1,
    standardize=STANDARDIZE_PREDICTORS
):
    """
    robust_options: dict with keys:
        - enable_hc3: bool (default from global/Parameter.csv)
        - enable_hac: bool (default from global/Parameter.csv)
        - hac_maxlags: 'auto' | int | None (default 'auto')
    standardize: bool
        If True, Ridge/Lasso uses standardized predictors (z-scores).
        If False, they use predictors in the original units.
        OLS always remains unstandardized.
    """
    if robust_options is None:
        robust_options = ROBUST_OPTIONS_DEFAULT.copy()
    else:
        tmp = ROBUST_OPTIONS_DEFAULT.copy()
        tmp.update(robust_options)
        robust_options = tmp

    variable_predictors = [col for col in predictors if data[col].nunique(dropna=True) > 1]
    if len(variable_predictors) == 0:
        print(f"No variable with variation in {prefix}, regression skipped.")
        return

    for col in variable_predictors + [target]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=variable_predictors + [target])
    if data.empty:
        print(f"No useful data after NA drop in {prefix}.")
        return

    X = data[variable_predictors]
    y = data[target]

    vif_log_path = os.path.join(folder, f'{prefix}_vif_exclusion_log.txt')
    if enable_vif:
        print(f"Multicollinearity reduction enabled (VIF threshold = {VIF_THRESHOLD})...")
        X, dropped_vars = reduce_multicollinearity(X, threshold=VIF_THRESHOLD, verbose=True)
        variable_predictors = X.columns.tolist()

        if dropped_vars:
            os.makedirs(folder, exist_ok=True)
            with open(vif_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Variables excluded for VIF > {VIF_THRESHOLD}\n")
                f.write("-" * 50 + "\n")
                for name, vif_val in dropped_vars:
                    f.write(f"{name:25s} VIF={vif_val:.4f}\n")
            print(f"VIF Exclusion Log saved in: {vif_log_path}")
        else:
            with open(vif_log_path, 'w', encoding='utf-8') as f:
                f.write(f"No excluded variables: all VIFs ≤ {VIF_THRESHOLD}\n")
            print(f"No VIF exclusions. Log saved in: {vif_log_path}")

        print(f"Remaining predictors after VIF reduction: {variable_predictors}")

    else:
        with open(vif_log_path, 'w', encoding='utf-8') as f:
            f.write("VIF reduction disabled (enable_vif=False)\n")

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    os.makedirs(folder, exist_ok=True)

    save_text(model.summary().as_text(), os.path.join(folder, f'{prefix}_regression_summary.txt'))

    coef_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std Error': model.bse,
        'p-value': model.pvalues,
        'Conf. Low': model.conf_int()[0],
        'Conf. High': model.conf_int()[1]
    })
    if not skip_files:
        coef_txt = format_coefficients_table(coef_df)
        save_text(coef_txt, os.path.join(folder, f'{prefix}_coefficients.txt'))

    resid = model.resid
    fitted = model.fittedvalues

    if len(resid) >= 3:
        if len(resid) > 5000:
            rng = np.random.default_rng(0)
            shapiro_sample = rng.choice(resid, size=5000, replace=False)
            W, p_shapiro = shapiro(shapiro_sample)
            sh_note = "Performed on a random sample of 5000 residues."
        else:
            W, p_shapiro = shapiro(resid)
            sh_note = "Performed on all residues."
    else:
        W, p_shapiro = np.nan, np.nan
        sh_note = "Sample < 3: Shapiro not executable."

    if len(resid) >= 20:
        jb_stat, jb_pvalue, _, _ = sms.jarque_bera(resid)
        jb_res = f"{jb_stat:.4f}"
        jb_p = f"{jb_pvalue:.4f}"
    else:
        jb_res, jb_p = "NaN", "NaN"
        print(f"Warning: sample size {len(resid)} < 20, Jarque-Bera test skipped.")

    bp_test = sms.het_breuschpagan(resid, model.model.exog)
    bp_stat, bp_pvalue = bp_test[0], bp_test[1]
    dw_stat = sms.durbin_watson(resid)

    diagnostics_txt = (
        f"Diagnostic tests on residuals:\n"
        f"Shapiro-Wilk (normality): W={W:.4f}, p-value={p_shapiro:.4f}  ({sh_note})\n"
        f"Jarque-Bera (normality):  stat={jb_res}, p-value={jb_p}\n"
        f"Breusch-Pagan (heteroscedasticity): stat={bp_stat:.4f}, p-value={bp_pvalue:.4f}\n"
        f"Durbin-Watson (autocorrelation): stat={dw_stat:.4f}\n"
    )
    save_text(diagnostics_txt, os.path.join(folder, f'{prefix}_residuals_diagnostics.txt'))

    if not skip_files:
        plot_residuals_vs_fitted(fitted, resid, folder, prefix)
    plot_qq(resid, folder, prefix)
    plot_predicted_vs_observed(y, fitted, folder, prefix)

    if robust_options.get('enable_hc3', True):
        r_hc3, coef_hc3 = robust_tables(model, 'HC3')
        save_text(r_hc3.summary().as_text(), os.path.join(folder, f'{prefix}_regression_summary_HC3.txt'))
        if not skip_files:
            save_text(format_coefficients_table(coef_hc3), os.path.join(folder, f'{prefix}_coefficients_HC3.txt'))

    if robust_options.get('enable_hac', True):
        hac_maxlags = robust_options.get('hac_maxlags', 'auto')
        if isinstance(hac_maxlags, int):
            if hac_maxlags < 1:
                raise ValueError("hac_maxlags must be >= 1 if given as an integer.")
            maxlags = hac_maxlags
            maxlags_origin = "manuale"
        elif hac_maxlags in ('auto', None):
            maxlags = newey_west_maxlags(len(resid))
            maxlags_origin = "auto (Andrews)"
        else:
            raise ValueError("hac_maxlags must be 'auto', None, or an integer >= 1.")

        r_hac, coef_hac = robust_tables(model, 'HAC', maxlags=maxlags)
        save_text(r_hac.summary().as_text(), os.path.join(folder, f'{prefix}_regression_summary_HAC.txt'))
        if not skip_files:
            save_text(format_coefficients_table(coef_hac), os.path.join(folder, f'{prefix}_coefficients_HAC.txt'))
        save_text(f"HAC (Newey–West) maxlags = {maxlags} ({maxlags_origin})\n",
                  os.path.join(folder, f'{prefix}_HAC_settings.txt'))

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X.to_numpy()

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_proc, y)
    lasso = Lasso(alpha=lasso_alpha)
    lasso.fit(X_proc, y)

    ridge_coefs = pd.Series(ridge.coef_, index=variable_predictors)
    lasso_coefs = pd.Series(lasso.coef_, index=variable_predictors)

    ridge_txt = format_regularized_table(ridge_coefs, f"Ridge Coef (alpha={ridge_alpha})")
    lasso_txt = format_regularized_table(lasso_coefs, f"Lasso Coef (alpha={lasso_alpha})")

    reg_summary = (
        f"Ridge Regression coefficients:\n{ridge_txt}\n"
        f"Lasso Regression coefficients:\n{lasso_txt}\n"
        f"Note: Predictor standardization (Ridge/Lasso) = {standardize}\n"
    )
    save_text(reg_summary, os.path.join(folder, f'{prefix}_regularized_coefficients.txt'))

    print(f"Regression results and diagnostics saved with prefix '{prefix}' in {folder}")

print("### Multiple regression on entire dataset ###")

perform_regression(
    df.copy(), predictor_cols, target_col,
    prefix='all_data',
    folder=regression_all_dir,
    skip_files=True,
    robust_options=None,
    ridge_alpha=1.0,
    lasso_alpha=0.1,
    standardize=STANDARDIZE_PREDICTORS 
)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
