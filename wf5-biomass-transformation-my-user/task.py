import pandas as pd
import os
import numpy as np
import warnings
import re
from scipy import stats
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

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

CONFIG = {
    "OUTPUT_DATA_FILE": "data_transformation.csv",
    "OUTPUT_REPORT_FILE": "transformation_report.csv",
}


def read_csv_auto(path: str) -> pd.DataFrame:
    """
    Reads CSV trying to understand separator (',' or ';') and decimal ('.' or ',').
    """
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
    df.columns = df.columns.str.strip().str.replace('\uFEFF', '', regex=True)
    return df


def _nanmin_safe(arr: np.ndarray, default_val=0.0):
    arr_no_nan = arr[~np.isnan(arr)]
    if arr_no_nan.size == 0:
        return default_val
    return np.min(arr_no_nan)


def safe_boxcox(series: pd.Series):
    x = series.to_numpy(dtype=float)
    min_val = _nanmin_safe(x, default_val=0.0)
    shift = 0.0
    if min_val <= 0:
        shift = abs(min_val) + 1e-6
        x = x + shift

    mask = ~np.isnan(x)
    if not np.any(mask):
        return series.to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformed, _ = stats.boxcox(x[mask])

    full = np.full_like(x, np.nan, dtype=float)
    full[mask] = transformed
    return full


def safe_zscore(series: pd.Series):
    x = series.astype(float).to_numpy()
    good = ~np.isnan(x)
    if not np.any(good):
        return x
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std == 0 or np.isnan(std):
        out = np.zeros_like(x, dtype=float)
        out[~good] = np.nan
        return out
    return (x - mean) / std


def plain_log(series: pd.Series, base10: bool = False):
    """
    Log WITHOUT shift:
    - values ​​<= 0 -> -inf / NaN (standard NumPy behavior)
    """
    x = series.astype(float).to_numpy()
    return np.log10(x) if base10 else np.log(x)


def shifted_log(series: pd.Series, base10: bool = False, eps: float = 1e-6):
    """
    Log WITH adaptive shift: shift = |min(x)| + eps if min(x) <= 0
    """
    x = series.astype(float).to_numpy()
    min_val = _nanmin_safe(x, default_val=0.0)
    shift = 0.0
    if min_val <= 0:
        shift = abs(min_val) + eps
        x = x + shift
    return np.log10(x) if base10 else np.log(x)


def log1p(series: pd.Series, base10: bool = False):
    """
    log(1+x): requires x > -1; for x <= -1 -> -inf/NaN
    """
    x = series.astype(float).to_numpy()
    return np.log10(1 + x) if base10 else np.log1p(x)


def sqrt_with_shift(series: pd.Series):
    """
    sqrt with shift to avoid negatives.
    """
    x = series.astype(float).to_numpy()
    shift = 0.0
    min_val = _nanmin_safe(x, default_val=0.0)
    if min_val < 0:
        shift = -min_val
    return np.sqrt(x + shift)


def sqrt_plain(series: pd.Series):
    """
    Classic square root WITHOUT shift:
    for values ​​< 0 -> NaN (NumPy behavior).
    """
    x = series.astype(float).to_numpy()
    with np.errstate(invalid="ignore"):
        return np.sqrt(x)


def _candidate_transforms(values: pd.Series):
    """
    Generates candidate transformations for 'auto'.
    Includes both classic sqrt and shift sqrt.
    """
    ser = values.astype(float)

    cand = {
        "raw":         ser.to_numpy(),
        "logShift":    shifted_log(ser, base10=False),
        "log1p":       log1p(ser, base10=False),
        "log10Shift":  shifted_log(ser, base10=True),
        "log10p":      log1p(ser, base10=True),
        "sqrt":        sqrt_plain(ser),        # classica
        "sqrtShift":   sqrt_with_shift(ser),   # con shift
        "cuberoot":    np.cbrt(ser.astype(float).to_numpy()),
        "boxcox":      safe_boxcox(ser),
        "zscore":      safe_zscore(ser),
    }

    return cand


def _normality_score(x_np: np.ndarray):
    xx = x_np[~np.isnan(x_np)]
    if len(xx) < 5:
        return (0.0, np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = stats.shapiro(xx)
        except Exception:
            return (0.0, np.inf)
    try:
        sk = np.abs(stats.skew(xx, bias=False))
    except Exception:
        sk = np.inf
    return (p, sk)


def choose_best_transform(values: pd.Series):
    cands = _candidate_transforms(values)
    best_label, best_p, best_sk = None, -1.0, np.inf
    for label, arr in cands.items():
        p_val, skew_abs = _normality_score(arr.astype(float))
        if (p_val > best_p) or (p_val == best_p and skew_abs < best_sk):
            best_label, best_p, best_sk = label, p_val, skew_abs
    if best_label is None:
        return "raw", values.to_numpy(dtype=float)
    return best_label, cands[best_label]


def apply_manual_transform(series: pd.Series, transform_kind: str):
    """
    Manual options:
    - 'log' / 'log10' => log WITHOUT shift
    - 'logShift' / 'log10Shift' => log WITH adaptive shift
    - 'log1p' / 'log10p' => log(1+x)
    - 'sqrt' => classic root (no shift, negatives -> NaN)
    - 'sqrtShift' => root with adaptive shift
    - 'cuberoot', 'boxcox', 'zscore'
    - 'none' / 'raw' => no transformation
    """
    tk = str(transform_kind).strip().lower()

    if tk in ["log", "ln", "loge", "lognat"]:
        return {"data_single": plain_log(series, base10=False), "note": "manual -> log (no shift)"}
    if tk in ["log10", "log_10", "log10e"]:
        return {"data_single": plain_log(series, base10=True), "note": "manual -> log10 (no shift)"}

    if tk in ["logshift", "log_shift", "log+shift"]:
        return {"data_single": shifted_log(series, base10=False), "note": "manual -> logShift"}
    if tk in ["log10shift", "log10_shift", "log10+shift"]:
        return {"data_single": shifted_log(series, base10=True), "note": "manual -> log10Shift"}

    if tk in ["log1p", "log+1", "log_plus1"]:
        return {"data_single": log1p(series, base10=False), "note": "manual -> log1p"}
    if tk in ["log10p", "log10+1", "log10_plus1", "log10_1p"]:
        return {"data_single": log1p(series, base10=True), "note": "manual -> log10p"}

    if tk in ["sqrt", "square_root"]:
        return {"data_single": sqrt_plain(series), "note": "manual -> sqrt (no shift)"}

    if tk in ["sqrtshift", "sqrt_shift", "sqrt+shift"]:
        return {"data_single": sqrt_with_shift(series), "note": "manual -> sqrtShift (with shift)"}

    if tk in ["cuberoot", "cube_root", "cbrt"]:
        return {"data_single": np.cbrt(series.astype(float).to_numpy()), "note": "manual -> cuberoot"}
    if tk in ["box-cox", "boxcox", "box_cox"]:
        return {"data_single": safe_boxcox(series), "note": "manual -> boxcox"}
    if tk in ["z", "zscore", "standardize", "standardise", "std"]:
        return {"data_single": safe_zscore(series), "note": "manual -> zscore"}

    if tk in ["none", "raw", "identity", "na", ""]:
        return {"data_single": series.to_numpy(dtype=float), "note": "manual -> none"}

    warnings.warn(f"Unknown manual transform '{transform_kind}'. Using raw values.")
    return {"data_single": series.to_numpy(dtype=float),
            "note": f"manual -> none (unrecognized '{transform_kind}')"}


def parse_instruction_cell(cell_value: str):
    """
    Interprets values like: "", "A", "auto", "sqrt", "sqrtShift", "log1p", "log10Shift", ...
    Note: 'A' is ignored (no aggregation in this version).
    """
    if not isinstance(cell_value, str):
        raw = "" if pd.isna(cell_value) else str(cell_value)
    else:
        raw = cell_value
    raw = raw.strip()
    if raw == "":
        return {"mode": "none", "manual_kind": None}

    parts = re.split(r"[|,;/\s]+", raw)
    parts = [p.strip() for p in parts if p.strip() != ""]

    mode = "none"
    manual_kind = None
    for p in parts:
        up = p.upper()
        if up == "AUTO":
            mode = "auto"
        elif up == "A":
            continue  # ignored
        else:
            mode = "manual"
            manual_kind = p
    return {"mode": mode, "manual_kind": manual_kind}


def main():
    input_data_path = final_input
    parameter_file_path = parameters_file_csv
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_auto(input_data_path)
    params = pd.read_csv(parameter_file_path, sep=';', header=None, encoding='utf-8-sig')

    columns = params.iloc[0].astype(str).str.strip().tolist()
    row_cmds = params.iloc[4].tolist()

    instructions = {col: parse_instruction_cell(cell) for col, cell in zip(columns, row_cmds)}

    final_df = df.copy()
    report_rows = []

    for col in columns:
        if col not in final_df.columns:
            report_rows.append({"column": col, "treatment": "column not in data -> skipped"})
            continue

        info = instructions.get(col, {"mode": "none", "manual_kind": None})
        mode = info["mode"]
        manual_kind = info["manual_kind"]

        if mode == "auto":
            if pd.api.types.is_numeric_dtype(final_df[col]):
                try:
                    best_label, best_values = choose_best_transform(final_df[col])
                except Exception as e:
                    warnings.warn(f"Auto transform failed for {col}: {e}")
                    best_label, best_values = "raw", final_df[col].to_numpy(dtype=float)
                final_df[col] = best_values
                report_rows.append({"column": col, "treatment": f"auto -> {best_label}"})
            else:
                report_rows.append({"column": col, "treatment": "auto requested but non-numeric -> none"})
            continue

        if mode == "manual":
            if pd.api.types.is_numeric_dtype(final_df[col]):
                try:
                    res = apply_manual_transform(final_df[col], manual_kind)
                    final_df[col] = res["data_single"]
                    report_rows.append({"column": col, "treatment": res["note"]})
                except Exception as e:
                    warnings.warn(f"Manual transform '{manual_kind}' failed for {col}: {e}")
                    report_rows.append({"column": col, "treatment": f"manual '{manual_kind}' failed -> none"})
            else:
                report_rows.append({"column": col, "treatment": f"manual '{manual_kind}' requested but non-numeric -> none"})
            continue

        report_rows.append({"column": col, "treatment": "none"})

    data_out_path = os.path.join(output_dir, CONFIG["OUTPUT_DATA_FILE"])
    report_out_path = os.path.join(output_dir, CONFIG["OUTPUT_REPORT_FILE"])

    final_df.to_csv(data_out_path, index=False, encoding="utf-8-sig", sep=';')
    print(f"[OK] Transformed dataset saved in: {data_out_path}")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_out_path, index=False, encoding="utf-8-sig", sep=';')
    print(f"[OK] Transformation report saved in: {report_out_path}")


if __name__ == "__main__":
    main()

data_transformation_file = os.path.join(output_dir, CONFIG["OUTPUT_DATA_FILE"])

file_data_transformation_file = open("/tmp/data_transformation_file_" + id + ".json", "w")
file_data_transformation_file.write(json.dumps(data_transformation_file))
file_data_transformation_file.close()
