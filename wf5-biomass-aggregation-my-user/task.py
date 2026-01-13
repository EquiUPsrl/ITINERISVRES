import pandas as pd
import os
import re
import warnings
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_transformation_file', action='store', type=str, required=True, dest='data_transformation_file')

arg_parser.add_argument('--parameters_file_csv', action='store', type=str, required=True, dest='parameters_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

data_transformation_file = args.data_transformation_file.replace('"','')
parameters_file_csv = args.parameters_file_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

input_data_file = data_transformation_file

CONFIG = {
    "AGG_OUTPUT_FILE": "aggregated_data.csv",

    "AGG_FUNC_NUMERIC": "mean",

    "RICHNESS_COL": "speciesNumber",
}

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
    df.columns = df.columns.str.strip().str.replace('\uFEFF', '', regex=True)
    return df

def read_params_auto(path: str) -> pd.DataFrame:
    with open(path, 'rb') as f:
        sample = f.read(8192).decode('utf-8-sig', errors='ignore')
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',;')
        sep = dialect.delimiter
    except Exception:
        sep = ',' if sample.count(',') >= sample.count(';') else ';'
    params = pd.read_csv(path, sep=sep, header=None, encoding='utf-8-sig')
    params.iloc[0] = params.iloc[0].astype(str).str.strip().str.replace('\uFEFF', '', regex=True)
    return params

def parse_instruction_cell(cell_value):
    """
    Accepts typical combinations:
      "", "A", "mean", "sum", "A mean", "A;sum", "A, mean"...
    Return:
      {"is_agg": bool, "agg_func": "mean"|"sum"|None, "raw": str}
    """
    if not isinstance(cell_value, str):
        raw = "" if pd.isna(cell_value) else str(cell_value)
    else:
        raw = cell_value
    raw_norm = raw.strip()
    if raw_norm == "":
        return {"is_agg": False, "agg_func": None, "raw": raw}

    parts = re.split(r"[|,;/\s]+", raw_norm)
    parts = [p.strip() for p in parts if p.strip() != ""]

    is_agg = any(p.upper() == "A" for p in parts)
    agg_func = None
    for p in parts:
        pl = p.lower()
        if pl in ("mean", "sum"):
            agg_func = pl
            break

    return {"is_agg": is_agg, "agg_func": agg_func, "raw": raw_norm}


data_path = input_data_file
param_path = parameters_file_csv
out_dir = output_dir
os.makedirs(out_dir, exist_ok=True)

aggregated_data_file = input_data_file

df = read_csv_auto(data_path)
params = read_params_auto(param_path)

try:
    param_columns = params.iloc[0].astype(str).str.strip().tolist()
except Exception:
    raise RuntimeError(f"{param_path} it does not have row 0 with column names.")

if len(params.index) <= 5:
    raise RuntimeError(f"{param_path} does not have line 6 (index 5) with instructions.")

row_cmds = params.iloc[5].tolist()  # riga 6

df.columns = df.columns.str.strip().str.replace('\uFEFF', '', regex=True)

instructions = {col: parse_instruction_cell(cell) for col, cell in zip(param_columns, row_cmds)}

df_cols_set = set(df.columns)
missing_cols = [c for c in param_columns if c not in df_cols_set]
if missing_cols:
    warnings.warn(
        f"In the {param_path} file there are columns NOT present in the dataset: "
        + ", ".join(missing_cols)
    )

agg_keys = [c for c, info in instructions.items() if info.get("is_agg", False) and c in df_cols_set]

agg_dict = {}
default_func = CONFIG["AGG_FUNC_NUMERIC"]

for col in df.columns:
    if col in agg_keys:
        inf = instructions.get(col, {})
        if inf.get("agg_func") in ("mean", "sum"):
            warnings.warn(
                f"Marcatura '{inf['agg_func']}' sulla colonna chiave '{col}' "
                "ignorata (le chiavi non vengono aggregate)."
            )
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        col_func = instructions.get(col, {}).get("agg_func") or default_func
        agg_dict[col] = col_func
    else:
        inf = instructions.get(col, {})
        if inf.get("agg_func") in ("mean", "sum"):
            warnings.warn(
                f"The column '{col}' is non-numeric but has been marked '{inf['agg_func']}'. "
                "The aggregate function will be ignored for this column."
            )

if len(agg_keys) == 0:
    warnings.warn("No columns marked 'A' in row 6: unless copy of original data.")
    grouped = df.copy()
else:
    if len(agg_dict) == 0:
        grouped = df[agg_keys].drop_duplicates().reset_index(drop=True)
    else:
        grouped = df.groupby(agg_keys).agg(agg_dict).reset_index()

    richness_col = CONFIG["RICHNESS_COL"]
    if "acceptedNameUsage" in df.columns:
        richness_df = (
            df.groupby(agg_keys)['acceptedNameUsage']
              .nunique()
              .reset_index(name=richness_col)
        )
        if richness_col in grouped.columns:
            grouped = grouped.drop(columns=[richness_col])
        grouped = pd.merge(grouped, richness_df, on=agg_keys, how='left')

aggregated_data_file = os.path.join(out_dir, CONFIG["AGG_OUTPUT_FILE"])
grouped.to_csv(aggregated_data_file, index=False, encoding="utf-8-sig", sep=';')
print(f"[OK] Aggregated data saved in: {aggregated_data_file}")

print("\n=== Aggregation plan applied ===")
if agg_keys:
    print("Keys (A):", ", ".join(agg_keys))
else:
    print("Keys (A): <nessuna>")
if agg_dict:
    for c, f in agg_dict.items():
        print(f"- {c}: {f}")
else:
    print("<no aggregate numeric column>")

file_aggregated_data_file = open("/tmp/aggregated_data_file_" + id + ".json", "w")
file_aggregated_data_file.write(json.dumps(aggregated_data_file))
file_aggregated_data_file.close()
