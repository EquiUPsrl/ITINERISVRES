import os
import pandas as pd
from natsort import natsorted
import numpy as np
import shutil
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--raster_calculator_config_csv', action='store', type=str, required=True, dest='raster_calculator_config_csv')

arg_parser.add_argument('--raster_calculator_formulas_csv', action='store', type=str, required=True, dest='raster_calculator_formulas_csv')

arg_parser.add_argument('--scale_factor_path', action='store', type=str, required=True, dest='scale_factor_path')

arg_parser.add_argument('--upscaled_dir', action='store', type=str, required=True, dest='upscaled_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

raster_calculator_config_csv = args.raster_calculator_config_csv.replace('"','')
raster_calculator_formulas_csv = args.raster_calculator_formulas_csv.replace('"','')
scale_factor_path = args.scale_factor_path.replace('"','')
upscaled_dir = args.upscaled_dir.replace('"','')



def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile
        if nodata is not None:
            array = np.where(array == nodata, np.nan, array)
    return array, profile

def write_raster(path, array, profile):
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)

def raster_calculator(raster_dict, expression, output_path):
    local_vars = {}
    profile = None

    for var, path in raster_dict.items():
        array, profile = read_raster(path)
        local_vars[var] = array

    local_vars.update({"np": np})

    with np.errstate(divide='ignore', invalid='ignore'):
        result = eval(expression, {"__builtins__": {}}, local_vars)
        result = np.where(np.isfinite(result), result, np.nan)

    write_raster(output_path, result, profile)




base_dir = scale_factor_path
upscaled_dir = upscaled_dir

os.makedirs(base_dir, exist_ok=True)

df = pd.read_csv(raster_calculator_config_csv, sep=';')
df_formulas = pd.read_csv(raster_calculator_formulas_csv, sep=';')

df_filtrato = df[df['Name'].notna() & (df['Name'].astype(str).str.strip() != '')]

cartelle_dict = {
    row['Name']: f"{row['Variable']}"
    for _, row in df_filtrato.iterrows()
}

def files_ordinati(cartella):
    return natsorted([
        f for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f))
    ])

cartelle = list(cartelle_dict.values())

has_1KM = any("_1KM" in c for c in cartelle)

liste_file = []
for c in cartelle:
    if has_1KM and "_1KM" not in c:
        dir_to_use = upscaled_dir
    else:
        dir_to_use = base_dir
    liste_file.append(files_ordinati(dir_to_use + c))

lunghezze = [len(lista) for lista in liste_file]
if len(set(lunghezze)) != 1:
    raise ValueError("Le cartelle non contengono lo stesso numero di file.")

coppie = [
    tuple(os.path.join(cartelle[i], file[i]) for i in range(len(cartelle)))
    for file in zip(*liste_file)
]

for index, row in df_formulas.iterrows():
    new_product = row['Product']
    formula = row['Formula']
    print(f"Product: {new_product}, Formula: {formula}")

    output_path = os.path.join(base_dir, new_product)
    shutil.rmtree(output_path, ignore_errors = True)
    os.makedirs(output_path, exist_ok=True)

    idx = 0
    for gruppo in coppie:
    
        idx = idx + 1
        
        raster_dict = {
            chiave: f"{base_dir}{valore}" for chiave, valore in zip(cartelle_dict.keys(), gruppo)
        }
        
        output_file = os.path.join(output_path, new_product + "_result_" + str(idx) + ".tif")
    
        raster_calculator(raster_dict, formula, output_file)
        
    print(f"Generated {idx} rasters in: {output_path}")

final_dataset_path = base_dir

file_final_dataset_path = open("/tmp/final_dataset_path_" + id + ".json", "w")
file_final_dataset_path.write(json.dumps(final_dataset_path))
file_final_dataset_path.close()
