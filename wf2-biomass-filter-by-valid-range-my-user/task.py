import pandas as pd
import os
import numpy as np
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--appeears_raster_csv', action='store', type=str, required=True, dest='appeears_raster_csv')

arg_parser.add_argument('--image_subset_base', action='store', type=str, required=True, dest='image_subset_base')


args = arg_parser.parse_args()
print(args)

id = args.id

appeears_raster_csv = args.appeears_raster_csv.replace('"','')
image_subset_base = args.image_subset_base.replace('"','')



valid_range_path = image_subset_base

df = pd.read_csv(appeears_raster_csv, sep=';')

df_sel = df[df["TaskID"].notna() & (df["TaskID"] != "")]

var_data = {
    row["Variable"]: row.to_dict()
    for _, row in df_sel.iterrows()
}

raster_extensions = [".tif", ".tiff", ".asc", ".grb", ".grib", ".img"]

for subdir in os.listdir(valid_range_path):
    if subdir.startswith('.'):
        continue

    subdir_path = os.path.join(valid_range_path, subdir)

    raster_files = []
    for root, dirs, files in os.walk(subdir_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if file.startswith('.'):
                continue
            if os.path.splitext(file)[1].lower() in raster_extensions:
                raster_files.append(os.path.join(root, file))


    print(f"\nIn the folder {subdir_path} (recursively) I found {len(raster_files)} raster.")
    if not raster_files:
        print(f"No rasters found in {subdir_path} (or subfolders).")
        continue

    raster_files = sorted(raster_files)

    print(f"Apply valid range for {subdir}: {var_data[subdir]}")

    for raster_file in raster_files:
    
        min_val = float(var_data[subdir]["MIN"])
        max_val = float(var_data[subdir]["MAX"])
        nodata_val = np.nan

        
        with rasterio.open(raster_file) as src:
            data = src.read(1).astype('float32')  # Provides support for NaN
            profile = src.profile.copy()
    
            profile.update(dtype='float32', nodata=nodata_val)
    
            invalid_mask = (data < min_val) | (data > max_val)
            data_filtered = np.where(invalid_mask, np.nan, data)
    
            with rasterio.open(raster_file, 'w', **profile) as dst:
                dst.write(data_filtered, 1)
        
    print(f"Rasters filtered and saved in: {valid_range_path}")

file_valid_range_path = open("/tmp/valid_range_path_" + id + ".json", "w")
file_valid_range_path.write(json.dumps(valid_range_path))
file_valid_range_path.close()
