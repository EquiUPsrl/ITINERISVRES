import os
from glob import glob
import pandas as pd
import shutil

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--interval', action='store', type=str, required=True, dest='interval')

arg_parser.add_argument('--pearson_tmp_path', action='store', type=str, required=True, dest='pearson_tmp_path')


args = arg_parser.parse_args()
print(args)

id = args.id

interval = args.interval.replace('"','')
pearson_tmp_path = args.pearson_tmp_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1_1/work/' + 'output'

output_dir = conf_output_path
input_base_folder = pearson_tmp_path #os.path.join(work_path, "tmp", "Pearson correlation coefficient")
output_folder = os.path.join(output_dir, "Pearson correlation coefficient")
os.makedirs(output_folder, exist_ok=True)

for subfolder in os.listdir(input_base_folder):
    subfolder_path = os.path.join(input_base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        if not os.listdir(subfolder_path):  # Se la cartella è vuota
            shutil.rmtree(subfolder_path)
            print(f"Empty folder deleted: {subfolder_path}")


moving_average_column = "pearson_correlation"

pd_freq = interval #"8D"  # "MS" = menthly, "D" = daily, "W" = weekly, "8D", "15D", ecc.
freq_label_map = {
    "MS": "Monthly",
    "D": "Daily",
    "W": "Weekly",
    "8D": "Every 8 days",
    "15D": "Every 15 days"
}
freq_label = freq_label_map.get(pd_freq, pd_freq)

m_map = {
    "MS": 12,      # 12 month in a year
    "D": 365,     # 365 days
    "W": 52,      # 52 weeks
    "8D": 46,     # 8 days ~ 45 period/year
    "15D": 24     # 15 days ~ 24 period/year
}
frequency = m_map.get(pd_freq, 12)  # Default 12 if key not found

order_column = None

csv_files = sorted(glob(os.path.join(input_base_folder, "*.csv")))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    if order_column and order_column in df.columns:
        df = df.sort_values(order_column)

    df[f"{moving_average_column}_rolling_{frequency}"] = df[moving_average_column].rolling(window=frequency, min_periods=frequency).mean()

    out_csv_file = os.path.join(output_folder, os.path.basename(csv_file))
    df.to_csv(out_csv_file, index=False)
    print(f"Moving average file saved in: {out_csv_file}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
