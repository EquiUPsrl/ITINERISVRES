import pandas as pd
import sys
import argparse
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
import warnings
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')

arg_parser.add_argument('--parameters_file_csv', action='store', type=str, required=True, dest='parameters_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

final_input = args.final_input.replace('"','')
parameters_file_csv = args.parameters_file_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONFIG = {
    "OUTPUT_BASE_DIR": conf_output_path,
    "OUTPUT_RUN_NAME": "Regression_RF_N",

    "ENABLE_VIF": False,
    "VIF_THRESHOLD": 10.0,

    "SCALING_MODE": "none",

    "INTERACTIONS": False,

    "PLOT_FORMAT": "tiff",   # "png" / "tiff" / "pdf"
    "PLOT_DPI": 200,

    "LOG_LEVEL": "info",     # "silent" | "info" | "debug"

    "RF_N_ESTIMATORS": 350,
    "RF_MAX_DEPTH": 13,        # None = unlimited depth
    "RF_MAX_FEATURES": "sqrt", # "sqrt"/"log2"/float(0-1]/int
    "RF_PERM_N_REPEATS": 10,   # for permutation importance

    "RF_VALIDATE": "split",   # "off" | "split"
    "RF_TEST_SIZE": 0.40,     # test share
    "RF_SHUFFLE": True,       # shuffle before split

    "RANDOM_STATE": 42,     # global seed
    "RF_N_JOBS": -1,        # 1 = deterministic; -1 = all cores
}

parser = argparse.ArgumentParser(
    description="Random Forest with VIF diagnostics, transformations and interactions, with hold-out validation (without aggregation)."
)
parser.add_argument("--output-run-name", type=str)
parser.add_argument("--output-base-dir", type=str)
parser.add_argument("--input-dir", type=str)
parser.add_argument("--input-data-file", type=str)
parser.add_argument("--parameter-file", type=str)

parser.add_argument("--vif", choices=["on", "off"])
parser.add_argument("--vif-threshold", type=float)
parser.add_argument("--scaling-mode", choices=["none", "mean", "zscore", "log"])
parser.add_argument("--interactions", choices=["on", "off"])

parser.add_argument("--plot-format", choices=["png", "pdf", "tiff"])
parser.add_argument("--plot-dpi", type=int)

parser.add_argument("--log-level", choices=["silent", "info", "debug"])

parser.add_argument("--rf-n-estimators", type=int)
parser.add_argument("--rf-max-depth")  # int or "none"
parser.add_argument("--rf-max-features")  # "sqrt"/"log2"/float/int/"none"
parser.add_argument("--rf-perm-n-repeats", type=int, help="Repeats for permutation importance")

parser.add_argument("--rf-validate", choices=["off", "split"])
parser.add_argument("--rf-test-size", type=float)
parser.add_argument("--rf-shuffle", choices=["on", "off"])

parser.add_argument("--random-state", type=int)
parser.add_argument("--rf-n-jobs", type=int)  # 1 for max reproducibility

args, _ = parser.parse_known_args()

def cfg_set(key, val):
    if val is not None:
        CONFIG[key] = val

cfg_set("OUTPUT_RUN_NAME", args.output_run_name)
cfg_set("OUTPUT_BASE_DIR", args.output_base_dir)

cfg_set("VIF_THRESHOLD", args.vif_threshold)
cfg_set("SCALING_MODE", args.scaling_mode)
cfg_set("PLOT_FORMAT", args.plot_format)
cfg_set("PLOT_DPI", args.plot_dpi)
cfg_set("LOG_LEVEL", args.log_level)

if args.vif is not None:
    CONFIG["ENABLE_VIF"] = (args.vif == "on")
if args.interactions is not None:
    CONFIG["INTERACTIONS"] = (args.interactions == "on")

cfg_set("RF_N_ESTIMATORS", args.rf_n_estimators)
cfg_set("RF_PERM_N_REPEATS", args.rf_perm_n_repeats)
cfg_set("RANDOM_STATE", args.random_state)
cfg_set("RF_N_JOBS", args.rf_n_jobs)

cfg_set("RF_TEST_SIZE", args.rf_test_size)
if args.rf_validate is not None:
    CONFIG["RF_VALIDATE"] = args.rf_validate
if args.rf_shuffle is not None:
    CONFIG["RF_SHUFFLE"] = (args.rf_shuffle == "on")

if args.rf_max_depth is not None:
    CONFIG["RF_MAX_DEPTH"] = None if str(args.rf_max_depth).lower() in ("none", "null") else int(args.rf_max_depth)

if args.rf_max_features is not None:
    rf_mf = str(args.rf_max_features).lower()
    if rf_mf == "auto":
        rf_mf = "sqrt"
    if rf_mf in ("none", "null"):
        CONFIG["RF_MAX_FEATURES"] = None
    elif rf_mf in ("sqrt", "log2"):
        CONFIG["RF_MAX_FEATURES"] = rf_mf
    else:
        try:
            if "." in rf_mf:
                val = float(rf_mf)
                if not (0.0 < val <= 1.0):
                    raise ValueError
                CONFIG["RF_MAX_FEATURES"] = val
            else:
                val = int(rf_mf)
                if val < 1:
                    raise ValueError
                CONFIG["RF_MAX_FEATURES"] = val
        except Exception:
            raise ValueError("--rf-max-features must be sqrt/log2/none or a number (int>=1 o float in (0,1])")

input_data_path = final_input
parameter_file_path = parameters_file_csv
output_dir = os.path.join(CONFIG["OUTPUT_BASE_DIR"], CONFIG["OUTPUT_RUN_NAME"])
os.makedirs(output_dir, exist_ok=True)

np.random.seed(CONFIG["RANDOM_STATE"])

def log_info(msg):
    if CONFIG["LOG_LEVEL"] in ("info", "debug"):
        print(msg)

def log_debug(msg):
    if CONFIG["LOG_LEVEL"] == "debug":
        print(msg)

def compute_regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    n    = int(len(y_true))
    return rmse, mae, n

def read_csv_auto(path: str) -> pd.DataFrame:
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

df = read_csv_auto(input_data_path)
params = pd.read_csv(parameter_file_path, sep=';', header=None, encoding='utf-8')

columns = params.iloc[0].astype(str).str.strip().tolist()
labels_row = params.iloc[3].astype(str).str.strip().tolist()

target_vars    = [col for col, lab in zip(columns, labels_row) if lab == 'Y']
predictor_vars = [col for col, lab in zip(columns, labels_row) if lab == 'X']

target_vars_actual = [v for v in target_vars if v in df.columns]
predictor_vars_actual = [v for v in predictor_vars if v in df.columns]
data_for_reg = df[target_vars_actual + predictor_vars_actual].dropna()

def apply_scaling(X: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return X
    if mode == 'log':
        X_tr = X.copy()
        for c in X_tr.columns:
            X_tr[c] = np.log1p(np.clip(X_tr[c], a_min=-0.999999999, a_max=None))
        return X_tr
    if mode == "mean":
        return X - X.mean(axis=0)
    if mode == "zscore":
        std = X.std(axis=0, ddof=0).replace(0, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Z = (X - X.mean(axis=0)) / std
        return Z.fillna(0.0)
    raise ValueError("Unrecognized SCALING_MODE.")

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    vif_data = []
    X_np = X.values
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X_np, i)
        except Exception:
            vif = np.nan
        vif_data.append({'variable': col, 'VIF': float(vif)})
    return pd.DataFrame(vif_data)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, format=CONFIG["PLOT_FORMAT"], dpi=CONFIG["PLOT_DPI"])
    plt.close()

def build_rf():
    kwargs = {
        "n_jobs": CONFIG["RF_N_JOBS"],
        "random_state": CONFIG["RANDOM_STATE"],
    }
    if CONFIG["RF_N_ESTIMATORS"] is not None:
        kwargs["n_estimators"] = CONFIG["RF_N_ESTIMATORS"]
    if CONFIG["RF_MAX_DEPTH"] is not None:
        kwargs["max_depth"] = CONFIG["RF_MAX_DEPTH"]
    if CONFIG["RF_MAX_FEATURES"] is not None:
        kwargs["max_features"] = CONFIG["RF_MAX_FEATURES"]
    return RandomForestRegressor(**kwargs)

summary_list = []
vif_exclusion_global_log = []
last_rf_validation_info = None

for y_col in [c for c in target_vars if c in data_for_reg.columns]:
    if CONFIG["LOG_LEVEL"] != "silent":
        print(f"Processing {y_col} ...")

    y_dir = os.path.join(output_dir, y_col.replace(" ", "_"))
    os.makedirs(y_dir, exist_ok=True)

    base_preds = [v for v in predictor_vars if v in data_for_reg.columns]

    full_cols = [y_col] + base_preds
    full_data = data_for_reg[full_cols].dropna().copy()

    Y_full = full_data[y_col]
    X_full = full_data[base_preds].copy()

    X_full = apply_scaling(X_full, CONFIG["SCALING_MODE"])

    interaction_terms = []
    if CONFIG["INTERACTIONS"] and len(base_preds) >= 2:
        log_info(f"Adding pairwise interactions for {y_col} ...")
        for a, b in combinations(base_preds, 2):
            t = f"{a}:{b}"
            X_full[t] = X_full[a] * X_full[b]
            interaction_terms.append(t)

    predictors_used = base_preds.copy()
    vif_history = []
    exclusion_steps = []
    if CONFIG["ENABLE_VIF"] and len(predictors_used) > 1:
        while True and len(predictors_used) > 1:
            vif_df = calculate_vif(X_full[predictors_used])
            vif_history.append(vif_df.copy())
            max_vif = vif_df['VIF'].max()
            if max_vif > CONFIG["VIF_THRESHOLD"]:
                drop_var = vif_df.sort_values('VIF', ascending=False)['variable'].iloc[0]
                exclusion_steps.append(f"Excluded: {drop_var} (VIF={max_vif:.2f})")
                if CONFIG["LOG_LEVEL"] != "silent":
                    print(f"   Dropping '{drop_var}' (VIF={max_vif:.2f}) for {y_col}")
                predictors_used.remove(drop_var)
            else:
                break
        if len(vif_history) > 0:
            vif_history[0].to_csv(os.path.join(y_dir, f"{y_col}_VIF_initial.csv"), index=False, encoding="utf-8-sig")
            vif_history[-1].to_csv(os.path.join(y_dir, f"{y_col}_VIF_final.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(y_dir, f"{y_col}_vif_exclusion_steps.txt"), "w", encoding="utf-8") as flog:
            flog.write("\n".join(exclusion_steps) if exclusion_steps else f"No exclusions. All VIF <= {CONFIG['VIF_THRESHOLD']}.")
        vif_exclusion_global_log.append(f"{y_col}:")
        vif_exclusion_global_log.extend(exclusion_steps if exclusion_steps else [f"  No exclusions. All VIF <= {CONFIG['VIF_THRESHOLD']}."])
        vif_exclusion_global_log.append("")

    kept_set = set(predictors_used)
    used_interactions = [t for t in interaction_terms if all(p in kept_set for p in t.split(':'))]

    X_design = X_full[predictors_used + used_interactions].copy()
    X = X_design.values
    Y = Y_full.values

    do_split = (CONFIG["RF_VALIDATE"] == "split") and (len(Y) > 3)

    if do_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y,
            test_size=CONFIG["RF_TEST_SIZE"],
            random_state=CONFIG["RANDOM_STATE"],
            shuffle=CONFIG["RF_SHUFFLE"]
        )
    else:
        X_train, y_train = X, Y
        X_test, y_test = None, None

    rf = build_rf()
    rf.fit(X_train, y_train)

    y_pred_tr = rf.predict(X_train)
    r2_tr = rf.score(X_train, y_train)
    rmse_tr, mae_tr, n_tr = compute_regression_metrics(y_train, y_pred_tr)

    if do_split:
        y_pred_ts = rf.predict(X_test)
        r2_ts = float(r2_score(y_test, y_pred_ts))
        rmse_ts, mae_ts, n_ts = compute_regression_metrics(y_test, y_pred_ts)
    else:
        y_pred_ts, r2_ts, rmse_ts, mae_ts, n_ts = None, '', '', '', ''

    try:
        key = re.sub(r'\W+', '', y_col).lower()
        if key in ("speciesnumber", "species_number", "nspecies", "numberofspecies"):
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=y_train, y=y_pred_tr)
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"Observed vs Predicted (TRAIN) - {y_col} (Random Forest)")
            lim_min, lim_max = float(np.nanmin(y_train)), float(np.nanmax(y_train))
            plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.6)

            legacy_name = "speciesNumber_Observed_vs_Predicted_RF_Training.tiff"
            out_legacy = os.path.join(y_dir, legacy_name)
            plt.tight_layout()
            plt.savefig(out_legacy, format="tiff", dpi=CONFIG["PLOT_DPI"])
            plt.close()
    except Exception as e:
        log_debug(f"[LEGACY TRAIN PLOT] skipped for {y_col}: {e}")

    plt.figure(figsize=(6, 6))
    if do_split:
        sns.scatterplot(x=y_test, y=y_pred_ts)
        plt.title(f"Observed vs Predicted (TEST) - {y_col} (Random Forest)")
        lim_min, lim_max = float(np.nanmin(y_test)), float(np.nanmax(y_test))
        out_name = f"{y_col}_Observed_vs_Predicted_RF_TEST.{CONFIG['PLOT_FORMAT']}"
    else:
        sns.scatterplot(x=y_train, y=y_pred_tr)
        plt.title(f"Observed vs Predicted (TRAIN) - {y_col} (Random Forest)")
        lim_min, lim_max = float(np.nanmin(y_train)), float(np.nanmax(y_train))
        out_name = f"{y_col}_Observed_vs_Predicted_RF_TRAIN.{CONFIG['PLOT_FORMAT']}"
    plt.xlabel("Observed"); plt.ylabel("Predicted")
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.6)
    savefig(os.path.join(y_dir, out_name))

    feat_names = list(X_design.columns)
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]

    top_k = min(20, len(imp_df))
    plt.figure(figsize=(8, max(4, 0.3*top_k)))
    sns.barplot(x=imp_df["importance"].iloc[:top_k], y=imp_df["feature"].iloc[:top_k])
    plt.xlabel("Importance (MDI)")
    plt.ylabel("")
    plt.title(f"Random Forest Feature Importance (top {top_k}) - {y_col}")

    try:
        X_perm = X_test if do_split else X_train
        y_perm = y_test if do_split else y_train
        perm = permutation_importance(
            rf, X_perm, y_perm,
            n_repeats=CONFIG["RF_PERM_N_REPEATS"],
            random_state=CONFIG["RANDOM_STATE"],
            scoring="r2",
            n_jobs=CONFIG["RF_N_JOBS"],
        )
        perm_df = pd.DataFrame({
            'feature': feat_names,
            'import_mean': perm.importances_mean,
            'import_std': perm.importances_std,
        }).sort_values('import_mean', ascending=False)
        suffix = "TEST" if do_split else "TRAIN"
        perm_df.to_csv(os.path.join(y_dir, f"{y_col}_rf_permutation_importance_{suffix}.csv"),
                       index=False, encoding="utf-8-sig")

        top_k = min(20, len(perm_df))
        plt.figure(figsize=(8, max(4, 0.3*top_k)))
        plt.barh(perm_df['feature'].iloc[:top_k][::-1],
                 perm_df['import_mean'].iloc[:top_k][::-1],
                 xerr=perm_df['import_std'].iloc[:top_k][::-1])
        plt.xlabel("Permutation importance (Δ R²)")
        plt.ylabel("")
        plt.title(f"Permutation Importance {suffix} (top {top_k}) - {y_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(
            y_dir,
            f"{y_col}_rf_permutation_importance_{suffix}.{CONFIG['PLOT_FORMAT']}"
        ),
            format=CONFIG["PLOT_FORMAT"],
            dpi=CONFIG["PLOT_DPI"]
        )
        plt.close()
    except Exception as e:
        log_debug(f"Permutation importance failed for {y_col}: {e}")
        perm_df = None

    rf_sum_path = os.path.join(y_dir, f"{y_col}_rf_summary.txt")
    with open(rf_sum_path, "w", encoding="utf-8") as fs:
        fs.write(f"Random Forest Summary – {y_col}\n")
        fs.write("=" * (22 + len(y_col)) + "\n\n")
        fs.write("Performance:\n")
        fs.write(f"  R² (train): {r2_tr:.6f}\n")
        fs.write(f"  RMSE train : {rmse_tr:.6f}\n")
        fs.write(f"  MAE  train : {mae_tr:.6f}\n")
        fs.write(f"  n   train  : {n_tr}\n")
        if do_split:
            fs.write(f"\n  R² (test) : {r2_ts:.6f}\n")
            fs.write(f"  RMSE test : {rmse_ts:.6f}\n")
            fs.write(f"  MAE  test : {mae_ts:.6f}\n")
            fs.write(f"  n   test  : {n_ts}\n")
        fs.write("\nModel setup:\n")
        fs.write(f"  Features used     : {len(predictors_used + used_interactions)}\n")
        fs.write(f"  Scaling mode      : {CONFIG['SCALING_MODE']}\n")
        fs.write(f"  Interactions used : {'yes' if used_interactions else 'no'}\n\n")
        fs.write("Hyperparameters:\n")
        fs.write(f"  n_estimators : {CONFIG['RF_N_ESTIMATORS']}\n")
        fs.write(f"  max_depth    : {CONFIG['RF_MAX_DEPTH']}\n")
        fs.write(f"  max_features : {CONFIG['RF_MAX_FEATURES']}\n")
        fs.write(f"  random_state : {CONFIG['RANDOM_STATE']}\n")
        fs.write(f"  n_jobs       : {CONFIG['RF_N_JOBS']}\n")
        if perm_df is not None:
            fs.write("\nTop features (Permutation Importance, mean Δ R²):\n")
            for i, rowp in enumerate(perm_df[['feature','import_mean']].values[:10], start=1):
                f, m = rowp
                fs.write(f"  {i:>2}. {f}: {m:.6f}\n")

    row = {
        'model': 'rf',
        'target_var': y_col,
        'r2_train': float(r2_tr),
        'rmse_train': rmse_tr,
        'mae_train': mae_tr,
        'n_train': n_tr,
        'r2_test': '' if not do_split else float(r2_ts),
        'rmse_test': '' if not do_split else rmse_ts,
        'mae_test': '' if not do_split else mae_ts,
        'n_test': '' if not do_split else n_ts,
        'predictors_used': ','.join(predictors_used + used_interactions),
        'scaling_mode': CONFIG["SCALING_MODE"],
        'interactions': CONFIG["INTERACTIONS"],
        'vif_enabled': CONFIG["ENABLE_VIF"],
        'vif_threshold': CONFIG["VIF_THRESHOLD"],
        'rf_n_estimators': CONFIG["RF_N_ESTIMATORS"],
        'rf_max_depth': '' if CONFIG["RF_MAX_DEPTH"] is None else CONFIG["RF_MAX_DEPTH"],
        'rf_max_features': '' if CONFIG["RF_MAX_FEATURES"] is None else str(CONFIG["RF_MAX_FEATURES"]),
        'rf_perm_n_repeats': CONFIG["RF_PERM_N_REPEATS"],
        'random_state': CONFIG["RANDOM_STATE"],
        'rf_n_jobs': CONFIG["RF_N_JOBS"],
    }
    summary_list.append(row)

    last_rf_validation_info = {
        "did_split": do_split,
        "test_size": CONFIG["RF_TEST_SIZE"],
        "r2_tr": r2_tr,
        "r2_ts": None if not do_split else r2_ts,
        "rmse_ts": None if not do_split else rmse_ts,
        "mae_ts": None if not do_split else mae_ts,
    }

if CONFIG["ENABLE_VIF"]:
    with open(os.path.join(output_dir, "vif_exclusion_log.txt"), "w", encoding="utf-8") as flog:
        flog.write("\n".join(vif_exclusion_global_log) if vif_exclusion_global_log else "No VIF exclusions across variables.")

results_df = pd.DataFrame(summary_list)
results_df.to_csv(os.path.join(output_dir, 'regression_summary_all.csv'), index=False, encoding="utf-8-sig")

if CONFIG["LOG_LEVEL"] != "silent":
    print("Random Forest params:", {
        'n_estimators': CONFIG["RF_N_ESTIMATORS"],
        'max_depth': CONFIG["RF_MAX_DEPTH"],
        'max_features': CONFIG["RF_MAX_FEATURES"],
        'perm_repeats': CONFIG["RF_PERM_N_REPEATS"],
        'random_state': CONFIG["RANDOM_STATE"],
        'n_jobs': CONFIG["RF_N_JOBS"],
    })
    print("VIF:", "ENABLED" if CONFIG["ENABLE_VIF"] else "DISABLED", "| threshold:", CONFIG["VIF_THRESHOLD"])
    if last_rf_validation_info is not None and last_rf_validation_info["did_split"]:
        print(f"RF validation (hold-out {int(CONFIG['RF_TEST_SIZE']*100)}% test): "
              f"R² train={last_rf_validation_info['r2_tr']:.3f}, "
              f"R² test={last_rf_validation_info['r2_ts']:.3f}, "
              f"RMSE test={last_rf_validation_info['rmse_ts']:.3f}, "
              f"MAE test={last_rf_validation_info['mae_ts']:.3f}")
    else:
        print("RF validation: OFF (using full data as train)")
    print("Scaling:", CONFIG["SCALING_MODE"])
    print("Interactions:", CONFIG["INTERACTIONS"])
    print("Plot:", CONFIG["PLOT_FORMAT"], f"{CONFIG['PLOT_DPI']} dpi")
    print("Output dir:", output_dir)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
