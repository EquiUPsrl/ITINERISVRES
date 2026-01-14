import warnings
import os
import pandas as pd
import numpy as np
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--dataCOR_path', action='store', type=str, required=True, dest='dataCOR_path')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')

arg_parser.add_argument('--to_transform', action='store', type=str, required=True, dest='to_transform')


args = arg_parser.parse_args()
print(args)

id = args.id

dataCOR_path = args.dataCOR_path.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')
to_transform = json.loads(args.to_transform)


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

warnings.filterwarnings("ignore")

def detect_sep(path, fallback=';'):
    """Detect the separator among , ; \t | using csv.Sniffer, with heuristic fallback."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        for line in sample.splitlines():
            if line.strip():
                counts = {d: line.count(d) for d in [',', ';', '\t', '|']}
                delim = max(counts, key=counts.get)
                return delim if counts[delim] > 0 else fallback
        return fallback

def read_csv_any_separator(path, nrows=None, header='infer', dtype=None):
    """Robust CSV reader that automatically detects column separators."""
    sep = detect_sep(path)
    return pd.read_csv(path, sep=sep, nrows=nrows, header=header, dtype=dtype)

param_path = parameters_csv

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

REMOVE_OUTLIERS = False   # True to enable outlier removal, False to disable
IQR_K = 1.5               # classic IQR threshold 1.5 (higher = more permissive)
MIN_ROWS_AFTER = 10       # safety net: do not apply if fewer rows would remain

CLEAN_TREAT_ZERO_AS_MISSING = True   # True: remove rows with 0 in key columns
CLEAN_ZERO_TOLERANCE = 0.0            # >0 to treat |x| <= tolerance as zero

def clean_data_post(df, cols, treat_zero_as_missing=False, zero_tolerance=0.0):
    """
    Keep rows where ALL the given columns are valid.
    Valid = non-NA and, optionally, non-zero (with tolerance).
    """
    if not cols:
        return df.copy()

    mask_valid = df[cols].notna()

    if treat_zero_as_missing:
        if zero_tolerance and zero_tolerance > 0:
            mask_nonzero = ~df[cols].apply(
                lambda s: s.astype(float).abs() <= zero_tolerance if pd.api.types.is_numeric_dtype(s) else False
            )
        else:
            mask_nonzero = df[cols] != 0
        mask_valid = mask_valid & mask_nonzero

    return df[mask_valid.all(axis=1)]

def compute_iqr_bounds(series, k=1.5):
    """Return (Q1, Q3, IQR, lower, upper). If IQR=0 return None."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return None
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return float(q1), float(q3), float(iqr), float(lower), float(upper)

def remove_outliers_iqr(df, cols, k=1.5):
    """
    Remove rows with IQR outliers in ANY of the columns in 'cols'.
    Returns: filtered_df, report_by_column, outlier_rows (with offending columns).
    """
    if not cols:
        report = pd.DataFrame(columns=["column","Q1","Q3","IQR","Lower","Upper","n_outliers","pct_outliers"])
        out_rows = pd.DataFrame(columns=["_row_index","_outlier_columns"])
        return df.copy(), report, out_rows

    thresholds = {}
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        b = compute_iqr_bounds(df[col].astype(float), k=k)
        if b is not None:
            thresholds[col] = b

    if not thresholds:
        report = pd.DataFrame(columns=["column","Q1","Q3","IQR","Lower","Upper","n_outliers","pct_outliers"])
        out_rows = pd.DataFrame(columns=["_row_index","_outlier_columns"])
        return df.copy(), report, out_rows

    n0 = len(df)
    outlier_mask_any = pd.Series(False, index=df.index)
    outlier_cols_map = {idx: [] for idx in df.index}

    for col, (q1, q3, iqr, lower, upper) in thresholds.items():
        m = ~(df[col].between(lower, upper, inclusive="both"))
        outlier_mask_any |= m
        for idx in df.index[m]:
            outlier_cols_map[idx].append(col)

    outlier_rows = df[outlier_mask_any].copy()
    if not outlier_rows.empty:
        outlier_rows["_row_index"] = outlier_rows.index
        outlier_rows["_outlier_columns"] = outlier_rows["_row_index"].map(
            lambda i: ",".join(outlier_cols_map.get(i, []))
        )
        out_rows = outlier_rows[["_row_index", "_outlier_columns"] + cols]
    else:
        out_rows = pd.DataFrame(columns=["_row_index","_outlier_columns"] + cols)

    rep_rows = []
    for col, (q1, q3, iqr, lower, upper) in thresholds.items():
        if col in df.columns:
            mcol = ~(df[col].between(lower, upper, inclusive="both"))
            n_out = int(mcol.sum())
            pct = (n_out / n0 * 100.0) if n0 > 0 else 0.0
            rep_rows.append({
                "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                "Lower": lower, "Upper": upper,
                "n_outliers": n_out, "pct_outliers": pct
            })
    report = pd.DataFrame(rep_rows).sort_values("n_outliers", ascending=False)

    df_filtered = df[~outlier_mask_any].copy()
    return df_filtered, report, out_rows



data_tr = read_csv_any_separator(dataCOR_path)

param_df = read_csv_any_separator(param_path, header=None, dtype=str)
if param_df.shape[0] < 2:
    raise ValueError("Parameter.csv must contain at least 2 rows: names (row 1) and flags (row 2).")

col_names = [str(v).strip() for v in param_df.iloc[0].tolist()]
flags_xyf = [str(v).strip().upper() for v in param_df.iloc[1].tolist()]
if param_df.shape[0] >= 3:
    flags_A = [str(v).strip().upper() for v in param_df.iloc[2].tolist()]
else:
    flags_A = [""] * len(col_names)

keywords = {'L', 'X', 'Y', 'F'}  # uppercase
cols_of_interest_idx = [
    i for i in range(len(col_names))
    if any(str(param_df.iloc[r, i]).strip().upper() in keywords for r in range(min(3, param_df.shape[0])))
]
cols_of_interest = [col_names[i] for i in cols_of_interest_idx if col_names[i] in data_tr.columns]

non_empty_cols = [c for c in cols_of_interest if not data_tr[c].isna().all()]
if not non_empty_cols:
    print("⚠️ No non-empty columns found for cleaning. No rows will be removed.")
    dataCOR_clean = data_tr.copy()
else:
    dataCOR_clean = clean_data_post(
        data_tr,
        non_empty_cols,
        treat_zero_as_missing=CLEAN_TREAT_ZERO_AS_MISSING,
        zero_tolerance=CLEAN_ZERO_TOLERANCE,
    )

print(f"[CLEAN] Columns actually used for cleaning: {non_empty_cols}")
print(f"[CLEAN] Option 'zero as missing' active: {CLEAN_TREAT_ZERO_AS_MISSING} (tolerance={CLEAN_ZERO_TOLERANCE})")
print(f"[CLEAN] Rows after cleaning: {dataCOR_clean.shape[0]}")

outliers_report_path = None
outliers_rows_path = None

if REMOVE_OUTLIERS:
    cols_for_outliers = [c for c in to_transform if c in dataCOR_clean.columns]
    df_no_out, report_df, out_rows_df = remove_outliers_iqr(dataCOR_clean, cols_for_outliers, k=IQR_K)

    if len(df_no_out) >= MIN_ROWS_AFTER:
        outliers_report_path = os.path.join(output_dir, f"outliers_report_IQR{IQR_K}.csv")
        report_df.to_csv(outliers_report_path, index=False)
        outliers_rows_path = os.path.join(output_dir, f"outliers_rows_IQR{IQR_K}.csv")
        out_rows_df.to_csv(outliers_rows_path, index=False)

        print(f"[OUTLIER] IQR k={IQR_K} on columns: {cols_for_outliers}")
        if not report_df.empty:
            print(f"[OUTLIER] Report saved: {outliers_report_path}")
        else:
            print("[OUTLIER] No valid threshold for IQR (columns nearly constant or non-numeric).")
        if not out_rows_df.empty:
            print(f"[OUTLIER] Outlier rows saved: {outliers_rows_path}")
            print(f"[OUTLIER] Before: {len(dataCOR_clean)} | After: {len(df_no_out)} | Removed: {len(out_rows_df)}")
        else:
            print("[OUTLIER] No outlier rows detected.")

        dataCOR_clean = df_no_out.copy()
    else:
        print(f"[OUTLIER] Removal skipped: remaining rows ({len(df_no_out)}) < MIN_ROWS_AFTER ({MIN_ROWS_AFTER}).")

cleaned_file_path = os.path.join(output_dir, 'dataCOR_clean_selectedCols.csv')
dataCOR_clean.to_csv(cleaned_file_path, index=False, sep=',', decimal='.')
print(f"Final cleaned file saved to: {cleaned_file_path}")

file_cleaned_file_path = open("/tmp/cleaned_file_path_" + id + ".json", "w")
file_cleaned_file_path.write(json.dumps(cleaned_file_path))
file_cleaned_file_path.close()
