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

arg_parser.add_argument('--valid_range_path', action='store', type=str, required=True, dest='valid_range_path')


args = arg_parser.parse_args()
print(args)

id = args.id

appeears_raster_csv = args.appeears_raster_csv.replace('"','')
valid_range_path = args.valid_range_path.replace('"','')



scale_factor_path = valid_range_path

df = pd.read_csv(appeears_raster_csv, sep=';')

df_sel = df[df["TaskID"].notna() & (df["TaskID"] != "")]

var_data = {
    row["Variable"]: row.to_dict()
    for _, row in df_sel.iterrows()
}

raster_extensions = [".tif", ".tiff", ".asc", ".grb", ".grib", ".img"]

for subdir in os.listdir(scale_factor_path):
    if subdir.startswith('.'):
        continue

    subdir_path = os.path.join(scale_factor_path, subdir)

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

    print(f"Apply scale factor for {subdir}: {var_data[subdir]}")

    for raster_file in raster_files:
        
        scale_factor = float(var_data[subdir]["Scale Factor"].replace(',', '.'))

        
        with rasterio.open(raster_file) as src:
            data = src.read(1).astype(np.float32)  # Convert to float for scale
            profile = src.profile
            
            scaled_data = data * scale_factor
        
            scaled_data[np.isnan(scaled_data)] = np.nan
            final_nodata_val = np.nan  # Or choose a float value like -9999.0
        
            profile.update(
                dtype='float32',
                nodata=final_nodata_val,
            )
        
            with rasterio.open(raster_file, 'w', **profile) as dst:
                dst.write(scaled_data.astype(np.float32), 1)
    
    print("Scale factor applied to rasters: " + subdir)

file_scale_factor_path = open("/tmp/scale_factor_path_" + id + ".json", "w")
file_scale_factor_path.write(json.dumps(scale_factor_path))
file_scale_factor_path.close()
