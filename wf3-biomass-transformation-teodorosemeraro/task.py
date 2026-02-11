import warnings
import os
import numpy as np
import pandas as pd
import json
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_csv', action='store', type=str, required=True, dest='input_csv')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

input_csv = args.input_csv.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

warnings.filterwarnings("ignore")

input_path = input_csv
param_path = parameters_csv

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

APPLY_TRANSFORM = True            # True to apply a transformation, False to skip
TRANSFORM_KIND = 'log10'            # 'none' | 'log' | 'log1p' | 'log10' | 'log2' | 'normalize' | 'standardize' | 'robust'

LOG_KIND = 'Log10'              # 'natural' (=ln), 'log10', 'log2' (used if TRANSFORM_KIND == 'log')
LOG_HANDLE_NONPOS = 'clip'        # 'clip' | 'shift' | 'nan'
LOG_EPS = np.finfo(float).eps     # used when clipping
LOG_SHIFT_SMALL = 1e-9            # used by 'shift' when min<=0 (adds |min| + small constant)

NORM_FEATURE_RANGE = (0.0, 1.0)   # output range for normalization

ROBUST_CENTER = False              # True: center on the median
ROBUST_SCALE_IQR = False           # True: scale by IQR (Q3−Q1)

SAVE_TRANSFORM_PARAMS = True


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




def _log_series(s: pd.Series, base: str = 'natural', nonpos='clip', eps=np.finfo(float).eps, shift_small=1e-9):
    s = s.astype(float)
    if nonpos == 'clip':
        s = s.clip(lower=eps)
    elif nonpos == 'shift':
        m = s.min(skipna=True)
        if pd.notna(m) and m <= 0:
            s = s + (abs(m) + shift_small)
    elif nonpos == 'nan':
        s = s.where(s > 0, np.nan)
    else:
        raise ValueError("LOG_HANDLE_NONPOS must be 'clip' | 'shift' | 'nan'")

    if base == 'natural':
        return np.log(s)
    elif base == 'log10':
        return np.log10(s)
    elif base == 'log2':
        return np.log2(s)
    else:
        raise ValueError("LOG_KIND must be 'natural' | 'log10' | 'log2'")

def _normalize_series(s: pd.Series, feature_range=(0.0, 1.0)):
    s = s.astype(float)
    a, b = feature_range
    smin = s.min(skipna=True); smax = s.max(skipna=True)
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.nan, index=s.index)
    return (s - smin) / (smax - smin) * (b - a) + a

def _standardize_series(s: pd.Series):
    s = s.astype(float)
    mu = s.mean(skipna=True); sd = s.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def _robust_series(s: pd.Series, center=True, scale_iqr=True):
    s = s.astype(float)
    med = s.median(skipna=True)
    q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
    out = s.copy()
    if center:
        out = out - med
    if scale_iqr:
        if pd.isna(iqr) or iqr == 0:
            return pd.Series(np.nan, index=s.index)
        out = out / iqr
    return out

def transform_columns(df: pd.DataFrame, cols: list, kind: str, params: dict):
    """
    Apply transformation 'kind' to numeric columns 'cols'.
    Returns (transformed_df, parameters_used).
    """
    df2 = df.copy()
    used_params = {}

    for c in cols:
        if c not in df2.columns or not pd.api.types.is_numeric_dtype(df2[c]):
            continue

        s = df2[c]
        if kind == 'none':
            used_params[c] = {'kind': 'none'}
            continue
        elif kind == 'log':
            base = params.get('LOG_KIND', 'natural')
            nonpos = params.get('LOG_HANDLE_NONPOS', 'clip')
            eps = params.get('LOG_EPS', np.finfo(float).eps)
            shift_small = params.get('LOG_SHIFT_SMALL', 1e-9)
            df2[c] = _log_series(s, base=base, nonpos=nonpos, eps=eps, shift_small=shift_small)
            used_params[c] = {'kind': f'log-{base}', 'nonpos': nonpos, 'eps': float(eps), 'shift_small': shift_small}
        elif kind == 'log1p':
            df2[c] = np.log1p(s.astype(float))
            used_params[c] = {'kind': 'log1p'}
        elif kind == 'log10':
            df2[c] = _log_series(s, base='log10', nonpos=params.get('LOG_HANDLE_NONPOS','clip'),
                                 eps=params.get('LOG_EPS', np.finfo(float).eps),
                                 shift_small=params.get('LOG_SHIFT_SMALL', 1e-9))
            used_params[c] = {'kind': 'log-log10'}
        elif kind == 'log2':
            df2[c] = _log_series(s, base='log2', nonpos=params.get('LOG_HANDLE_NONPOS','clip'),
                                 eps=params.get('LOG_EPS', np.finfo(float).eps),
                                 shift_small=params.get('LOG_SHIFT_SMALL', 1e-9))
            used_params[c] = {'kind': 'log-log2'}
        elif kind == 'normalize':
            fr = params.get('NORM_FEATURE_RANGE', (0.0, 1.0))
            df2[c] = _normalize_series(s, feature_range=fr)
            used_params[c] = {'kind': 'normalize', 'feature_range': list(fr)}
        elif kind == 'standardize':
            mu = s.astype(float).mean(skipna=True); sd = s.astype(float).std(skipna=True, ddof=0)
            df2[c] = _standardize_series(s)
            used_params[c] = {'kind': 'standardize', 'mean': float(mu) if pd.notna(mu) else None,
                              'std': float(sd) if pd.notna(sd) else None}
        elif kind == 'robust':
            df2[c] = _robust_series(s, center=params.get('ROBUST_CENTER', True),
                                    scale_iqr=params.get('ROBUST_SCALE_IQR', True))
            q1 = s.quantile(0.25); q3 = s.quantile(0.75); med = s.median()
            used_params[c] = {'kind': 'robust', 'median': float(med) if pd.notna(med) else None,
                              'q1': float(q1) if pd.notna(q1) else None,
                              'q3': float(q3) if pd.notna(q3) else None}
        else:
            raise ValueError("Invalid TRANSFORM_KIND")

    return df2, used_params

param_df = read_csv_any_separator(param_path, header=None, dtype=str)
if param_df.shape[0] < 2:
    raise ValueError("Parameter.csv must contain at least 2 rows: names (row 1) and flags (row 2).")

col_names = [str(v).strip() for v in param_df.iloc[0].tolist()]
flags_xyf = [str(v).strip().upper() for v in param_df.iloc[1].tolist()]
if param_df.shape[0] >= 3:
    flags_A = [str(v).strip().upper() for v in param_df.iloc[2].tolist()]
else:
    flags_A = [""] * len(col_names)

data = read_csv_any_separator(input_path)
data_tr = data.copy()

xy_cols = [col_names[i] for i, flag in enumerate(flags_xyf) if flag in {'X', 'Y'} and i < len(col_names)]
to_transform = [c for c in xy_cols if c in data_tr.columns and pd.api.types.is_numeric_dtype(data_tr[c])]

print(f"[TRANSFORM] X/Y columns detected: {xy_cols}")
print(f"[TRANSFORM] Numeric columns to transform: {to_transform}")

params = dict(
    LOG_KIND=LOG_KIND,
    LOG_HANDLE_NONPOS=LOG_HANDLE_NONPOS,
    LOG_EPS=LOG_EPS,
    LOG_SHIFT_SMALL=LOG_SHIFT_SMALL,
    NORM_FEATURE_RANGE=NORM_FEATURE_RANGE,
    ROBUST_CENTER=ROBUST_CENTER,
    ROBUST_SCALE_IQR=ROBUST_SCALE_IQR,
)

if APPLY_TRANSFORM and TRANSFORM_KIND != 'none' and to_transform:
    data_tr, tr_params = transform_columns(data_tr, to_transform, TRANSFORM_KIND, params)
    print(f"[TRANSFORM] Applied transformation: {TRANSFORM_KIND}")
else:
    tr_params = {c: {'kind': 'none'} for c in to_transform}
    print("[TRANSFORM] No transformation applied: using original data.")

dataCOR_path = os.path.join(output_dir, 'dataCOR.csv')
data_tr.to_csv(dataCOR_path, index=False, sep=',', decimal='.')
print(f"Transformed/original file saved to: {dataCOR_path}")

if SAVE_TRANSFORM_PARAMS:
    with open(os.path.join(output_dir, 'transform_params.json'), 'w', encoding='utf-8') as f:
        json.dump(tr_params, f, ensure_ascii=False, indent=2)

file_dataCOR_path = open("/tmp/dataCOR_path_" + id + ".json", "w")
file_dataCOR_path.write(json.dumps(dataCOR_path))
file_dataCOR_path.close()
file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
