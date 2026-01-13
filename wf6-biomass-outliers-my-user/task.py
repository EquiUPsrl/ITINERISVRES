import pandas as pd
import os
import re
import csv
import numpy as np

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

CONFIG = {

    "METHOD": "mad",       # "mad" | "zscore" | "iqr"
    "THR": 4.0,            # threshold for MAD/zscore, or multiplier for IQR
    "RULE": "max",         # "max" | "count"
    "COUNT_K": 1,          # used only if RULE="count"

    "GROUP_MODE": "single_factor",  # "none" | "single_factor" | "all_factors"
    "GROUP_FACTOR": "auto",         # used only if GROUP_MODE="single_factor"

    "N_SPECIES_COL_NAME": "N Species",
}


def read_csv_auto(path: str) -> pd.DataFrame:
    """Auto-detect separator and decimal style (comma vs dot)."""
    with open(path, "rb") as f:
        sample = f.read(8192).decode("utf-8-sig", errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        sep = dialect.delimiter
    except Exception:
        sep = "," if sample.count(",") >= sample.count(";") else ";"

    dot_nums = len(re.findall(r"\d+\.\d+", sample))
    comma_nums = len(re.findall(r"\d+,\d+", sample))
    decimal = "," if (sep == ";" and comma_nums > dot_nums) else "."

    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\uFEFF", "", regex=True)
    return df


def prefilter_outliers_multiy(
    data: pd.DataFrame,
    y_cols: list,
    method: str = "mad",
    thr: float = 3.5,
    group_cols: list | None = None,
    rule: str = "max",
    count_k: int = 1,
):
    """
    Sample-wise global outlier prefilter using all Y columns.
    Always preserves the original row order.
    """
    method = (method or "mad").lower()
    thr = float(thr)

    if not group_cols:
        groups = [("ALL", data)]
        used_group_cols = None
    else:
        group_cols = [c for c in group_cols if c in data.columns]
        if len(group_cols) == 0:
            groups = [("ALL", data)]
            used_group_cols = None
        else:
            groups = list(data.groupby(group_cols, dropna=False, sort=False))
            used_group_cols = group_cols

    outlier_flag = pd.Series(False, index=data.index)

    for gname, g in groups:
        Y = g[y_cols].apply(pd.to_numeric, errors="coerce")
        Z = pd.DataFrame(index=Y.index, columns=Y.columns, dtype=float)

        for c in y_cols:
            v = Y[c].dropna()
            if len(v) < 3:
                Z[c] = 0.0
                continue

            if method == "mad":
                med = np.median(v)
                mad = np.median(np.abs(v - med))
                mad = mad if mad > 0 else np.std(v, ddof=0)
                zc = 0.6745 * (Y[c] - med) / mad if mad > 0 else 0.0

            elif method == "zscore":
                mu = v.mean()
                sd = v.std(ddof=0)
                zc = (Y[c] - mu) / sd if sd > 0 else 0.0

            elif method == "iqr":
                q1 = v.quantile(0.25)
                q3 = v.quantile(0.75)
                iqr = q3 - q1
                low = q1 - thr * iqr
                high = q3 + thr * iqr
                zc = ((Y[c] < low) | (Y[c] > high)).astype(float)

            else:
                raise ValueError(f"Unknown method '{method}'")

            Z[c] = zc

        if rule == "max":
            score = Z.abs().max(axis=1)
            flag = score > thr
        elif rule == "count":
            count_extreme = (Z.abs() > thr).sum(axis=1)
            flag = count_extreme >= count_k
        else:
            raise ValueError("rule must be 'max' or 'count'")

        outlier_flag.loc[g.index] = flag.fillna(False)

    data_removed = data.loc[outlier_flag].copy()
    data_kept = data.loc[~outlier_flag].copy()

    info = {
        "method": method,
        "thr": thr,
        "rule": rule,
        "count_k": count_k,
        "group_cols": used_group_cols,
        "n_in": len(data),
        "n_removed": len(data_removed),
        "n_out": len(data_kept),
    }
    return data_kept, data_removed, info



output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

df = read_csv_auto(data_transformation_path)

params = pd.read_csv(
    parameters_csv,
    sep=";",
    header=None,
    encoding="utf-8-sig",
)

columns = params.iloc[0].astype(str).str.strip().tolist()
labels = params.iloc[1].astype(str).str.strip().tolist()

biotic_vars = [col for col, lab in zip(columns, labels) if lab == "Y"]
abiotic_vars = [col for col, lab in zip(columns, labels) if lab == "X"]
factor_vars = [col for col, lab in zip(columns, labels) if lab == "f"]

n_species_col_name = CONFIG["N_SPECIES_COL_NAME"]

print("Aggregation disabled. Using original table.")
use_cols = [
    c for c in (biotic_vars + abiotic_vars + factor_vars)
    if c in df.columns
]
data_full = df[use_cols].dropna().copy()

y_cols_common = [
    c for c in biotic_vars
    if c in data_full.columns and c != n_species_col_name
]
if len(y_cols_common) == 0:
    raise ValueError("No valid Y columns found for outlier prefilter.")

group_mode = str(CONFIG.get("GROUP_MODE", "single_factor")).lower()
group_cols = None

if group_mode == "none":
    group_cols = None

elif group_mode == "single_factor":
    if len(factor_vars) == 0:
        group_cols = None
    else:
        gf = CONFIG.get("GROUP_FACTOR", "auto")
        if str(gf).lower() == "auto":
            group_cols = [factor_vars[0]]
        elif gf in data_full.columns:
            group_cols = [gf]
        else:
            group_cols = None

elif group_mode == "all_factors":
    group_cols = [f for f in factor_vars if f in data_full.columns]
    if len(group_cols) == 0:
        group_cols = None

else:
    raise ValueError("GROUP_MODE must be: 'none', 'single_factor', or 'all_factors'")

data_clean, data_removed, pre_info = prefilter_outliers_multiy(
    data=data_full,
    y_cols=y_cols_common,
    method=CONFIG["METHOD"],
    thr=CONFIG["THR"],
    group_cols=group_cols,
    rule=CONFIG["RULE"],
    count_k=CONFIG["COUNT_K"],
)

clean_path = os.path.join(output_dir, "data_transformation_clean.csv")
rem_path = os.path.join(output_dir, "data_transformation_outliers_removed.csv")

mask_keep = df.index.isin(data_clean.index)
df_clean_full = df.loc[mask_keep].copy()
df_removed_full = df.loc[~mask_keep].copy()

df_clean_full.to_csv(clean_path, index=False, encoding="utf-8-sig")
df_removed_full.to_csv(rem_path, index=False, encoding="utf-8-sig")

reduced_clean_path = clean_path.replace(".csv", "_reduced.csv")
reduced_removed_path = rem_path.replace(".csv", "_reduced.csv")
data_clean.to_csv(reduced_clean_path, index=False, encoding="utf-8-sig")
data_removed.to_csv(reduced_removed_path, index=False, encoding="utf-8-sig")

with open(os.path.join(output_dir, "GLOBAL_preoutliers_info.txt"), "w", encoding="utf-8") as f:
    f.write(str(pre_info))

print("\n[GLOBAL PRE-OUTLIERS DONE]")
print(pre_info)
print("Clean table saved to:", clean_path)
print("Removed outliers saved to:", rem_path)

file_clean_path = open("/tmp/clean_path_" + id + ".json", "w")
file_clean_path.write(json.dumps(clean_path))
file_clean_path.close()
