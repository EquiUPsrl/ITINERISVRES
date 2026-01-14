import warnings
import os
import pandas as pd
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--cleaned_file_path', action='store', type=str, required=True, dest='cleaned_file_path')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

cleaned_file_path = args.cleaned_file_path.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')


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

param_df = read_csv_any_separator(param_path, header=None, dtype=str)
if param_df.shape[0] < 2:
    raise ValueError("Parameter.csv must contain at least 2 rows: names (row 1) and flags (row 2).")

col_names = [str(v).strip() for v in param_df.iloc[0].tolist()]
flags_xyf = [str(v).strip().upper() for v in param_df.iloc[1].tolist()]
if param_df.shape[0] >= 3:
    flags_A = [str(v).strip().upper() for v in param_df.iloc[2].tolist()]
else:
    flags_A = [""] * len(col_names)

data_df = pd.read_csv(cleaned_file_path, sep=',', decimal='.')

agg_cols = [name for name, flag in zip(col_names, flags_A) if flag == 'A' and name in data_df.columns]
factor_cols = [name for name, flag in zip(col_names, flags_xyf) if flag == 'F' and name in data_df.columns]
groupby_cols = agg_cols + factor_cols

print(f"[AGG] Aggregation columns (A): {agg_cols}")
print(f"[AGG] Factor columns (F): {factor_cols}")
print(f"[AGG] Columns for groupby: {groupby_cols}")

output_groupby_path = os.path.join(output_dir, 'dataCOR_groupby.csv')

if not agg_cols:
    print("No column with flag 'A'. Aggregation will NOT be performed!")
    data_df.to_csv(output_groupby_path, index=False, sep=',', decimal='.')
    print(f"Non-aggregated file saved to {output_groupby_path}")
else:
    agg_dict = {}
    for col in data_df.columns:
        if col in groupby_cols:
            continue
        if pd.api.types.is_numeric_dtype(data_df[col]):
            agg_dict[col] = 'mean'
    if not agg_dict:
        print("No numeric column to aggregate! Saving only groupby columns without duplicates.")
        grouped_df = data_df[groupby_cols].drop_duplicates().copy()
    else:
        grouped_df = data_df.groupby(groupby_cols, dropna=True).agg(agg_dict).reset_index()

    grouped_df.to_csv(output_groupby_path, index=False, sep=',', decimal='.')
    print(f"Aggregated table saved to {output_groupby_path}")
    print(grouped_df.head())

file_output_groupby_path = open("/tmp/output_groupby_path_" + id + ".json", "w")
file_output_groupby_path.write(json.dumps(output_groupby_path))
file_output_groupby_path.close()
