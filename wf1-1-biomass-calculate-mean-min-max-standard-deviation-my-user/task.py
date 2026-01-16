import os
import rasterio
import numpy as np
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--image_subset_path', action='store', type=str, required=True, dest='image_subset_path')


args = arg_parser.parse_args()
print(args)

id = args.id

image_subset_path = args.image_subset_path.replace('"','')


conf_base_path = conf_base_path = '/tmp/data/WF1_1/work/'
conf_output_path = conf_output_path = '/tmp/data/WF1_1/work/' + 'output'

work_path = conf_base_path
input_base_folder = image_subset_path #os.path.join(work_path, "tmp", "ImageSubset")
statistics_path = os.path.join(conf_output_path, "Time Series Statistics")

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
    print(f"  Statistics saved in: {output_csv}")

print("\nOperation COMPLETED!")

file_statistics_path = open("/tmp/statistics_path_" + id + ".json", "w")
file_statistics_path.write(json.dumps(statistics_path))
file_statistics_path.close()
