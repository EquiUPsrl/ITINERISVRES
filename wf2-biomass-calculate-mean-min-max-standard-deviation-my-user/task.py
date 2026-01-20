import os
import rasterio
import numpy as np
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_dataset_path', action='store', type=str, required=True, dest='final_dataset_path')


args = arg_parser.parse_args()
print(args)

id = args.id

final_dataset_path = args.final_dataset_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/work/' + 'output'

output_dir = conf_output_path
input_base_folder = final_dataset_path
statistics_path = os.path.join(output_dir, "Time Series Statistics")

def calculate_statistics(input_file):
    with rasterio.open(input_file) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        
        statistics = {
            'mean': np.nanmean(data) if np.any(~np.isnan(data)) else np.nan,
            'sd': np.nanstd(data, ddof=0) if np.any(~np.isnan(data)) else np.nan,
            'min': np.nanmin(data) if np.any(~np.isnan(data)) else np.nan,
            'max': np.nanmax(data) if np.any(~np.isnan(data)) else np.nan
        }
    return statistics

os.makedirs(statistics_path, exist_ok=True)

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

    output_csv = os.path.join(statistics_path, f"all_statistics_{gruppo}.csv")
    df.to_csv(output_csv, index=False)
    print(f"  Statistiche salvate in: {output_csv}")

print("\nOperation completed!")

file_statistics_path = open("/tmp/statistics_path_" + id + ".json", "w")
file_statistics_path.write(json.dumps(statistics_path))
file_statistics_path.close()
