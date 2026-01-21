from glob import glob
import pandas as pd
import os
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--stats_path', action='store', type=str, required=True, dest='stats_path')


args = arg_parser.parse_args()
print(args)

id = args.id

stats_path = args.stats_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/work/' + 'output'

input_folder = stats_path #os.path.join("work", "output", "Time Series Statistics")
csv_files = sorted(glob(os.path.join(input_folder, "*.csv")))

col_to_extract = "mean"

dfs = []

for i, file in enumerate(csv_files, start=1):
    df = pd.read_csv(file)
    
    df_sub = df[[col_to_extract]].copy()
    
    filename = os.path.basename(file)
    match = re.search(r"SubChl_(.+)\.csv$", filename)
    species_name = match.group(1).strip() if match else filename.strip()
    df_sub.rename(columns={col_to_extract: species_name}, inplace=True)
    
    dfs.append(df_sub)

result = pd.concat(dfs, axis=1)

result_T = result.T
result_T.index.name = "species"

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "link_WF3_transposed.csv")
result_T.to_csv(output_file)

print("File for WF3 connection saved in:", output_file)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
