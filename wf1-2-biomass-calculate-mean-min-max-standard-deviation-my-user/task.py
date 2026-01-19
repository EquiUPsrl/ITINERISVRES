import os
import rasterio
import numpy as np
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--image_subset_base', action='store', type=str, required=True, dest='image_subset_base')

arg_parser.add_argument('--interval', action='store', type=str, required=True, dest='interval')


args = arg_parser.parse_args()
print(args)

id = args.id

image_subset_base = args.image_subset_base.replace('"','')
interval = args.interval.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1_2/work/' + 'output'

input_base_folder = image_subset_base #os.path.join(work_path, "tmp", "ImageSubset_4")
output_dir = conf_output_path
stats_path = os.path.join(output_dir, "Time Series Statistics")

moving_average_column = "mean"

pd_freq = interval  # "M" = monthly, "D" = daily, "W" = weekly, "8D", "15D", ecc.
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
    "8D": 46,     # 8 days ~ 45 period/year
    "15D": 24     # 15 days ~ 24 period/year
}
frequency = m_map.get(pd_freq, 12)  # Default 12 if key not found


def calculate_statistics(input_file):
    with rasterio.open(input_file) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        statistics = {
            'mean': np.nanmean(data),
            'sd': np.nanstd(data),
            'min': np.nanmin(data),
            'max': np.nanmax(data)
        }
    return statistics

os.makedirs(stats_path, exist_ok=True)

for gruppo in sorted(os.listdir(input_base_folder)):
    gruppo_folder = os.path.join(input_base_folder, gruppo)
    if not os.path.isdir(gruppo_folder):
        continue

    print(f"\nCalculate statistics for: {gruppo}")
    input_files = sorted([
        os.path.join(gruppo_folder, f) for f in os.listdir(gruppo_folder) if f.endswith(".tif")
    ])
    if not input_files:
        print(f"  No .tif files found in {gruppo_folder}")
        continue

    all_statistics = {
        os.path.basename(f): calculate_statistics(f)
        for f in input_files
    }
    df = pd.DataFrame.from_dict(all_statistics, orient='index').reset_index().rename(columns={'index': 'filename'})

    if moving_average_column in df.columns:
        col_name = f"Moving_average_{moving_average_column}_{pd_freq}"
        df[col_name] = df[moving_average_column].rolling(window=frequency, min_periods=frequency).mean()
        df.attrs["media_mobile_frequenza"] = freq_label

    output_csv = os.path.join(stats_path, f"all_statistics_{gruppo}.csv")
    df.to_csv(output_csv, index=False)
    print(f"  Statistics (with moving average {freq_label}) saved in: {output_csv}")

print("\nOperation completed!")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
file_stats_path = open("/tmp/stats_path_" + id + ".json", "w")
file_stats_path.write(json.dumps(stats_path))
file_stats_path.close()
