import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--statistics_path', action='store', type=str, required=True, dest='statistics_path')


args = arg_parser.parse_args()
print(args)

id = args.id

statistics_path = args.statistics_path.replace('"','')



output_base_folder = statistics_path

target_columns = ["mean", "min", "max"]

output_subfolder = "corrected"
threshold = 1.5   # theshold: 1.5 = +50%

stats_path = os.path.join(output_base_folder, output_subfolder)
os.makedirs(stats_path, exist_ok=True)

def trova_prossimo_valido(values, start_index, prev_val):
    """Trova il prossimo valore valido dopo start_index."""
    for j in range(start_index + 1, len(values)):
        next_val = values[j]
        if not pd.isna(next_val) and next_val <= threshold * prev_val:
            return next_val
    return None  # no valid values found

for filename in os.listdir(output_base_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(output_base_folder, filename)
        print(f"Elaborazione di {filename}...")

        df = pd.read_csv(file_path)

        for target_col in target_columns:

            if target_col not in df.columns:
                print(f"⚠️ Column '{target_col}' not found in {filename}, skipped.")
                continue
    
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            values = df[target_col].tolist()
    
            for i in range(1, len(values)):
                prev_val = values[i - 1]
                curr_val = values[i]
    
                invalid = pd.isna(curr_val) #or (curr_val > threshold * prev_val)
    
                if invalid:
                    next_valid = trova_prossimo_valido(values, i, prev_val)
    
                    if next_valid is not None:
                        corrected = (prev_val + next_valid) / 2
                        values[i] = corrected
                    else:
                        values[i] = prev_val
    
            df[target_col] = values

        new_path = os.path.join(stats_path, filename)

        df.to_csv(new_path, index=False)
        print(f"✅ Correct file saved in: {new_path}\n")

file_stats_path = open("/tmp/stats_path_" + id + ".json", "w")
file_stats_path.write(json.dumps(stats_path))
file_stats_path.close()
