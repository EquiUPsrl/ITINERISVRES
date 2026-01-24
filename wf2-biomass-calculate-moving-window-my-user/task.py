import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_stats_path', action='store', type=str, required=True, dest='final_stats_path')

arg_parser.add_argument('--param_interval', action='store', type=str, required=True, dest='param_interval')

args = arg_parser.parse_args()
print(args)

id = args.id

final_stats_path = args.final_stats_path.replace('"','')

param_interval = args.param_interval.replace('"','')

conf_output_path = conf_output_path = '/tmp/data/WF2/work/' + 'output'

output_dir = conf_output_path
output_base_folder = final_stats_path #os.path.join(work_path, "output", "Time Series Statistics", "corrected")
stats_output_path = output_base_folder
moving_average_column = "mean" 

pd_freq = param_interval  # "M" = monthly, "D" = daily, "W" = weekly, "8D", "15D", ecc.
freq_label_map = {
    "MS": "Monthly",
    "D": "Daily",
    "W": "Weekly",
    "8D": "Every 8 days",
    "15D": "Every 15 days"
}
freq_label = freq_label_map.get(pd_freq, pd_freq)

m_map = {
    "MS": 12,      # 12 months in one year
    "D": 365,     # 365 days
    "W": 52,      # 52 weeks
    "8D": 46,     # 8 days ~ 46 periods/year
    "15D": 24     # 15 days ~ 24 periods/year
}
frequency = m_map.get(pd_freq, 12)  # Default 12 if key not found


for filename in os.listdir(output_base_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(output_base_folder, filename)
        print(f"Process {filename}...")

        df = pd.read_csv(file_path)

        if moving_average_column in df.columns:
            col_name = f"Moving_average_{moving_average_column}_{pd_freq}"
            df[col_name] = df[moving_average_column].rolling(window=frequency, min_periods=frequency).mean()
            df.attrs["media_mobile_frequenza"] = freq_label
    
        output_csv = os.path.join(stats_output_path, filename)
        df.to_csv(output_csv, index=False)
        print(f"  Statistics (with moving average {freq_label}) saved in: {output_csv}")

print("\nOperation completed!")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
file_stats_output_path = open("/tmp/stats_output_path_" + id + ".json", "w")
file_stats_output_path.write(json.dumps(stats_output_path))
file_stats_output_path.close()
