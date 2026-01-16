import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--interval', action='store', type=str, required=True, dest='interval')

arg_parser.add_argument('--stats_path', action='store', type=str, required=True, dest='stats_path')


args = arg_parser.parse_args()
print(args)

id = args.id

interval = args.interval.replace('"','')
stats_path = args.stats_path.replace('"','')


conf_base_path = conf_base_path = '/tmp/data/WF1_1/work/'

work_path = conf_base_path
output_base_folder = stats_path #os.path.join(work_path, "output", "Time Series Statistics", "corrected")
final_stats_path = output_base_folder
moving_average_column = "mean"

pd_freq = interval #"8D"  # "M" = monthly, "D" = daily, "W" = weekly, "8D", "15D", ecc.
freq_label_map = {
    "MS": "Monthly",
    "D": "Daily",
    "W": "Weekly",
    "8D": "Every 8 days",
    "15D": "Every 15 days"
}
freq_label = freq_label_map.get(pd_freq, pd_freq)

m_map = {
    "MS": 12,      # 12 months in a year
    "D": 365,     # 365 days
    "W": 52,      # 52 weeks
    "8D": 46,     # 8 days ~ 46 periods/year
    "15D": 24     # 15 days ~ 24 periods/year
}
frequency = m_map.get(pd_freq, 12)  # Default 12 if key not found

for filename in os.listdir(output_base_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(output_base_folder, filename)
        print(f"Processing of {filename}...")

        df = pd.read_csv(file_path)

        if moving_average_column in df.columns:
            col_name = f"Moving_average_{moving_average_column}_{pd_freq}"
            df[col_name] = df[moving_average_column].rolling(window=frequency, min_periods=frequency).mean()
            df.attrs["moving_average_frequency"] = freq_label
    
        output_csv = os.path.join(final_stats_path, filename)
        df.to_csv(output_csv, index=False)
        print(f"  Statistics (with moving average {freq_label}) saved in: {output_csv}")

print("\nOperation COMPLETED!")

file_final_stats_path = open("/tmp/final_stats_path_" + id + ".json", "w")
file_final_stats_path.write(json.dumps(final_stats_path))
file_final_stats_path.close()
