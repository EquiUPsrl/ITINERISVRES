import os
import pandas as pd

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

def read_csv_any_separator(path, nrows=None, header='infer'):
    """Reads a CSV with ',' or ';' delimiter automatically."""
    try:
        df = pd.read_csv(path, sep=',', nrows=nrows, header=header)
        if df.shape[1] == 1 and ';' in df.columns[0]:
            df = pd.read_csv(path, sep=';', nrows=nrows, header=header)
    except Exception:
        df = pd.read_csv(path, sep=';', nrows=nrows, header=header)
    return df

output_dir = conf_output_path

corr_dir = os.path.join(output_dir, 'correlations')
os.makedirs(corr_dir, exist_ok=True)

param_path = parameters_csv
output_path = os.path.join(corr_dir, 'correlation_results.csv')

param_df = read_csv_any_separator(param_path, header=None)
col_names = [str(col).strip() for col in param_df.iloc[0].tolist()]
selectors = [str(sel).strip().upper() for sel in param_df.iloc[1].tolist()]

data_path = filtered_input
print(f"📦 Using aggregated file: {data_path}")

data_df = read_csv_any_separator(data_path)

x_cols = [name for name, sel in zip(col_names, selectors) if sel == "X"]
y_cols = [name for name, sel in zip(col_names, selectors) if sel == "Y"]
y_col = y_cols[0] if y_cols else None

if y_col is None:
    print("Flag Y not found in the second row of Parameter.csv.")
    print("Values found in the second row:", selectors)
    raise ValueError("No target Y column found in the second row of Parameter.csv.")

results = []
for x in x_cols:
    if x in data_df.columns and y_col in data_df.columns:
        try:
            corr = data_df[[x, y_col]].corr().iloc[0, 1]
        except Exception:
            corr = None
        results.append({'X': x, 'Y': y_col, 'Pearson_correlation': corr})
    else:
        results.append({'X': x, 'Y': y_col, 'Pearson_correlation': None})

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"Results saved in {output_path}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
