from itertools import combinations
import os
from glob import glob
from rasterio.mask import mask
import pandas as pd
import rasterio
import numpy as np
import shutil

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_dataset_path', action='store', type=str, required=True, dest='final_dataset_path')

arg_parser.add_argument('--upscaled_dir', action='store', type=str, required=True, dest='upscaled_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

final_dataset_path = args.final_dataset_path.replace('"','')
upscaled_dir = args.upscaled_dir.replace('"','')


conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)

def correggi_raster(arr, threshold=3.0):
    def trova_prossimo_valido(values, start_idx, prev_val):
        for j in range(start_idx + 1, len(values)):
            if not (np.isnan(values[j]) or values[j] > threshold * prev_val):
                return values[j]
        return None

    flat = arr.astype("float64").flatten().tolist()

    for i in range(1, len(flat)):
        prev_val = flat[i - 1]
        curr_val = flat[i]

        invalid = np.isnan(curr_val) or (curr_val > threshold * prev_val)

        if invalid:
            next_valid = trova_prossimo_valido(flat, i, prev_val)

            if next_valid is not None:
                flat[i] = (prev_val + next_valid) / 2
            else:
                flat[i] = prev_val

    return np.array(flat).reshape(arr.shape)


input_base_folder = final_dataset_path #os.path.join(work_path, "tmp", "ImageSubset")
upscaled_base_folder = upscaled_dir #os.path.join(work_path, "tmp", "ImageSubset_upscaled")
pearson_tmp_dir = os.path.join(conf_tmp_path, "Pearson correlation coefficient")
os.makedirs(pearson_tmp_dir, exist_ok=True)

for subfolder in os.listdir(input_base_folder):
    subfolder_path = os.path.join(input_base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        if not os.listdir(subfolder_path):
            shutil.rmtree(subfolder_path)
            print(f"Empty folder deleted: {subfolder_path}")

subfolders = [
    f for f in os.listdir(input_base_folder)
    if os.path.isdir(os.path.join(input_base_folder, f))
]

print("subfolders", subfolders)

has_1km = any("_1KM" in f for f in subfolders)

subfolders = sorted(subfolders)

for sub1, sub2 in combinations(subfolders, 2):

    print(f"{sub1}_vs_{sub2}")
    if has_1km:
        if "_1KM" in sub1:
            folder1 = os.path.join(input_base_folder, sub1)
        else :
            folder1 = os.path.join(upscaled_base_folder, sub1)
            if not os.path.isdir(folder1):
                folder1 = os.path.join(input_base_folder, sub1)

        if "_1KM" in sub2:
            folder2 = os.path.join(input_base_folder, sub2)
        else :
            folder2 = os.path.join(upscaled_base_folder, sub2)
            if not os.path.isdir(folder2):
                folder2 = os.path.join(input_base_folder, sub2)
    else:
        folder1 = os.path.join(input_base_folder, sub1)
        folder2 = os.path.join(input_base_folder, sub2)

    print(folder1, " vs ", folder2)
    
    files1 = sorted(glob(os.path.join(folder1, "*.tif")))
    files2 = sorted(glob(os.path.join(folder2, "*.tif")))

    N = min(len(files1), len(files2))

    records = []

    for i in range(N):
        file1 = files1[i]
        file2 = files2[i]
        name1 = os.path.basename(file1)
        name2 = os.path.basename(file2)

        arr1 = read_raster(file1)
        arr2 = read_raster(file2)


        if arr1.shape != arr2.shape:
            corr = np.nan
        else:
            v1 = arr1.flatten()
            v2 = arr2.flatten()
            mask = (~np.isnan(v1)) & (~np.isnan(v2))
            v1 = v1[mask]
            v2 = v2[mask]
            corr = np.corrcoef(v1, v2)[0, 1] if len(v1) > 0 else np.nan

        records.append({
            f"{sub1}_file": name1,
            f"{sub2}_file": name2,
            "pearson_correlation": corr
        })

    df = pd.DataFrame(records)
    out_name = f"{sub1}_vs_{sub2}_Pearson.csv"
    output_file = os.path.join(pearson_tmp_dir, out_name)
    df.to_csv(output_file, index=False)
    print(f"Table saved: {output_file}")

file_pearson_tmp_dir = open("/tmp/pearson_tmp_dir_" + id + ".json", "w")
file_pearson_tmp_dir.write(json.dumps(pearson_tmp_dir))
file_pearson_tmp_dir.close()
