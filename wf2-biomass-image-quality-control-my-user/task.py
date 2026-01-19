import os
import pandas as pd
from pathlib import Path
import re
import numpy as np
import rasterio
import shutil

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--appeears_path', action='store', type=str, required=True, dest='appeears_path')

arg_parser.add_argument('--appeears_qc_path', action='store', type=str, required=True, dest='appeears_qc_path')

arg_parser.add_argument('--appeears_raster_csv', action='store', type=str, required=True, dest='appeears_raster_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

appeears_path = args.appeears_path.replace('"','')
appeears_qc_path = args.appeears_qc_path.replace('"','')
appeears_raster_csv = args.appeears_raster_csv.replace('"','')


conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

def extract_date_from_filename(filename):
    """
    Extracts a substring of the type "_doy<sequence>_aid<sequence>.tif" from a filename.

    Parameters:
    - filename: str, MODIS file name
    
    Returns:
    - str: extracted substring or None if no match is found
    """
    pattern = r"(_doy\d+_aid\d+\.tif)$"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        return None





QC_SCHEMAS = {
    "ET_500m": {"layers": {"QC": {"bits": (0,1), "good_values": [0]}}},
    "LE_500m": {"layers": {"QC": {"bits": (0,1), "good_values": [0]}}},
    "PET_500m": {"layers": {"QC": {"bits": (0,1), "good_values": [0]}}},
    "PLE_500m": {"layers": {"QC": {"bits": (0,1), "good_values": [0]}}},
    "Gpp_500m": {"layers": {"QC": {"bits": (0,1), "good_values": [0]}}},
    "PsnNet_500m": {"layers": {"QC": {"bits": (0,1), "good_values": [0]}}},

    "Fpar_500m": {"layers": {"FparLai_QC": {"bits": (0,1), "good_values": [0,1]}}},
    "Lai_500m": {"layers": {"FparLai_QC": {"bits": (0,1), "good_values": [0,1]}}},

    "LST_Day_1KM": {"layers": {"QC_Day": {"bits": (0,1), "good_values": [0]}}},
    "LST_Night_1KM": {"layers": {"QC_Night": {"bits": (0,1), "good_values": [0]}}},

    "sur_refl_b01": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
    "sur_refl_b02": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
    "sur_refl_b03": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
    "sur_refl_b04": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
    "sur_refl_b05": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
    "sur_refl_b06": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
    "sur_refl_b07": {"layers": {"sur_refl_qc_500m": {"bits": (0,1), "good_values": [0b00]}}},
}


def identify_raster_name(filename):
    """
    Returns the base name of the raster (e.g., sur_refl_b01, ET_500m).
    Must match a QC_SCHEMAS key.
    """
    for key in QC_SCHEMAS.keys():
        if key in filename:
            return key
    return None


def filter_raster_by_qc(
        data_file,
        qc_file,
        output_file=None
    ):
    """
    Filters a raster based on QC.

    Automatically retrieves the NoData value from the raster.
    """
    raster_name = identify_raster_name(data_file)
    if raster_name is None:
        raise ValueError(f"Raster not recognized for QC: {data_file}")

    schema = QC_SCHEMAS[raster_name]

    with rasterio.open(data_file) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata_value = src.nodatavals[0]

    with rasterio.open(qc_file) as src:
        qc = src.read(1)

    layer_name = list(schema["layers"].keys())[0]
    bits_range = schema["layers"][layer_name]["bits"]
    good_values = schema["layers"][layer_name]["good_values"]

    start, end = bits_range
    length = end - start + 1
    mask = (qc >> start) & ((1 << length) - 1)

    good_mask = np.isin(mask, good_values)

    data_filtered = np.where(good_mask, data, nodata_value)

    if output_file:
        profile.update(dtype=rasterio.float32, nodata=nodata_value)
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data_filtered.astype(rasterio.float32), 1)
        print(f"Raster filtrato salvato in: {output_file}")

    return data_filtered





image_base = appeears_path 
qc_base = appeears_qc_path

dataset_path = os.path.join(conf_tmp_path, "dataset")
os.makedirs(dataset_path, exist_ok=True)

df = pd.read_csv(appeears_raster_csv, sep=';')

raster_extensions = [".tif", ".tiff", ".asc", ".grb", ".grib", ".img"]

for subdir in os.listdir(image_base):
    if subdir.startswith('.'):
        continue

    subdir_path = os.path.join(image_base, subdir)

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

    count_raster = 0

    qc_value = df.loc[df['Variable'] == subdir, 'QC'].values

    qc_folder_name = qc_value[0]
    qc_folder = Path(os.path.join(qc_base, qc_folder_name))
    tmp_path = os.path.join(dataset_path, subdir)
    os.makedirs(tmp_path, exist_ok=True)
    

    if qc_value:

        for raster_file in raster_files:
            
            search_string = extract_date_from_filename(os.path.basename(raster_file))
            
            matching_files = [f for f in qc_folder.iterdir() if f.is_file() and search_string in f.name and qc_folder_name in f.name]

            output_file = os.path.join(tmp_path, os.path.basename(raster_file))
            
            if matching_files:
                qc_file = matching_files[0]
                psnnet_filtered = filter_raster_by_qc(raster_file, qc_file, output_file)
                count_raster = count_raster + 1
            else:
                first_file = None
                shutil.copy2(raster_file, output_file)
                print("QC not found. Copying the file to " + output_file)
    

file_dataset_path = open("/tmp/dataset_path_" + id + ".json", "w")
file_dataset_path.write(json.dumps(dataset_path))
file_dataset_path.close()
