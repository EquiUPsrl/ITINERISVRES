from rasterio.mask import mask
import sklearn
import os
import joblib
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--image_subset_base', action='store', type=str, required=True, dest='image_subset_base')

arg_parser.add_argument('--model_path', action='store', type=str, required=True, dest='model_path')


args = arg_parser.parse_args()
print(args)

id = args.id

image_subset_base = args.image_subset_base.replace('"','')
model_path = args.model_path.replace('"','')


conf_base_path = conf_base_path = '/tmp/data/WF1_1/work/'
conf_tmp_path = conf_tmp_path = '/tmp/data/WF1_1/work/' + 'tmp'

print(sklearn.__version__)


work_path = conf_base_path
tmp_dir = conf_tmp_path
image_subset_path = image_subset_base

output_path = image_subset_path
os.makedirs(output_path, exist_ok=True)

model_bundle = joblib.load(model_path)
model = model_bundle["model"]
transformations = model_bundle["transformations"]

print("\n--- INFO model dictionary (transformations) ---")
for k, v in transformations.items():
    print(f"{k}: {v}")

predictors = transformations["predictors"]
log_features = set(transformations.get("log_features", []))
target_log = transformations.get("target_log", False)
target_var = transformations.get("target_var", "target")

out_raster_path = os.path.join(output_path, target_var)
os.makedirs(out_raster_path, exist_ok=True)

subfolders = ["Sub" + var for var in predictors]
files = []
for subfolder in subfolders:
    raster_dir = os.path.join(image_subset_path, subfolder)
    tif_files = sorted([f for f in os.listdir(raster_dir) if f.endswith(".tif")])

    print(f"{subfolder}: {len(tif_files)} .tif files found in {raster_dir}")
    
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {raster_dir}")
    files.append(tif_files)

if not all(len(files[0]) == len(flist) for flist in files):
    raise ValueError("Subfolders do not have the same number of images!")

n_imgs = len(files[0])
print(f"Found {n_imgs} image combinations to process.")

csv_rows = []
header = ['output_file'] + predictors

for idx in range(n_imgs):
    rasters = {}
    input_files = []

    for var, subfolder, tif_list in zip(predictors, subfolders, files):
        raster_path = os.path.join(image_subset_path, subfolder, tif_list[idx])
        input_files.append(tif_list[idx])
        
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(np.float32)
            rasters[var] = arr
            if idx == 0:
                shape = arr.shape

    if not all(r.shape == shape for r in rasters.values()):
        raise ValueError(f"Le immagini nella combinazione {idx+1} non hanno la stessa shape!")

    mask = np.ones(shape, dtype=bool)
    for arr in rasters.values():
        mask &= ~np.isnan(arr)

    X = []
    for var in predictors:
        arr = rasters[var][mask]
        if var in log_features:
            arr_log = np.log(arr)
            arr = arr_log
        X.append(arr)
    X = np.vstack(X).T

    X_df = pd.DataFrame(X, columns=predictors)

    if not np.isfinite(X_df).all().all() :
        mask_inf = ~np.isfinite(X_df)

    y_pred = np.full(shape, np.nan, dtype=np.float32)
    if X.shape[0] > 0:
        y_pred_flat = model.predict(X_df)
        y_pred[mask] = y_pred_flat
    else:
        print(f"Warning: No valid pixels in the combination {idx+1}")

    if target_log:
        y_pred = np.exp(y_pred)

    if idx == 0:
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        plt.title(f"Input: {predictors[0]}")
        plt.imshow(rasters[predictors[0]], cmap='gray')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.title(f"Output prediction ({target_var})")
        plt.imshow(y_pred, cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    out_name = f"{target_var}_{idx+1:03d}.tif"
    out_raster_path = os.path.join(output_path, target_var, out_name)
    template_raster_path = os.path.join(image_subset_path, subfolders[0], files[0][idx])
    with rasterio.open(template_raster_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_raster_path, "w", **profile) as dst:
            dst.write(y_pred.astype(np.float32), 1)
    print(f"Saved: {out_raster_path}")

    csv_rows.append([out_name] + input_files)

csv_path = os.path.join(output_path, f"pairings_{target_var}.csv")
with open(csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(csv_rows)
print(f"Saved CSV file of pairings: {csv_path}")

file_image_subset_path = open("/tmp/image_subset_path_" + id + ".json", "w")
file_image_subset_path.write(json.dumps(image_subset_path))
file_image_subset_path.close()
