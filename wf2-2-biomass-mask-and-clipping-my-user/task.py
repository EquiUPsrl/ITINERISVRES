import shutil
import os
import fiona
from rasterio.mask import mask
import numpy as np
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--shape_files_dir', action='store', type=str, required=True, dest='shape_files_dir')

arg_parser.add_argument('--verified_dataset_path', action='store', type=str, required=True, dest='verified_dataset_path')


args = arg_parser.parse_args()
print(args)

id = args.id

shape_files_dir = args.shape_files_dir.replace('"','')
verified_dataset_path = args.verified_dataset_path.replace('"','')


conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

print("EXTRACT RASTER PORTION OF INTEREST USING THE SHP FILE")

def mask_and_clipping(input_shape_file, dataset_path, output_dir):
    
    with fiona.open(input_shape_file, 'r') as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]  # Lista di tutti i poligoni
    
    raster_extensions = [".tif", ".tiff", ".asc", ".grb", ".grib", ".img"]
    
    for subdir in os.listdir(dataset_path):
        if subdir.startswith('.'):
            continue
    
        subdir_path = os.path.join(dataset_path, subdir)
    
        raster_files = []
        for root, dirs, files in os.walk(subdir_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
    
            for file in files:
                if file.startswith('.'):
                    continue
                if os.path.splitext(file)[1].lower() in raster_extensions:
                    raster_files.append(os.path.join(root, file))
    
    
        new_subdir = "Chl" if subdir == "CHL" else subdir
        shape_filename_base = os.path.splitext(os.path.basename(input_shape_file))[0]
        output_path = os.path.join(output_dir, f"Sub{new_subdir}_{shape_filename_base}")
        os.makedirs(output_path, exist_ok=True)
    
        print(f"\nIn the {subdir_path} folder (recursively) I found {len(raster_files)} raster.")
        if not raster_files:
            print(f"No rasters found in {subdir_path} (or subfolders).")
            continue
    
        raster_files = sorted(raster_files)
        
        for raster_file in raster_files:
            raster_path = raster_file
        
            with rasterio.open(raster_path) as src:
                out_image, out_transform = mask(src, geoms, crop=True, filled=True)
                
                out_image = out_image.astype('float32')
                
                out_image[out_image == 0] = np.nan  # or: out_image[out_image.mask] = np.nan
        
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "dtype": 'float32',
                    "nodata": np.nan
                })
        
            filename_base = os.path.splitext(os.path.basename(raster_path))[0]
            output_filename = os.path.join(output_path, f"{filename_base}_Sub{subdir}.tif")
            
            with rasterio.open(output_filename, "w", **out_meta) as dest:
                dest.write(out_image)
    
        print("Clipping of rasters " + subdir + " completed. Output saved to " + output_path)
        break;



input_dir = shape_files_dir
tmp_dir = conf_tmp_path
image_subset_base = os.path.join(tmp_dir, "ImageSubset")

print("Delete folder " + image_subset_base)
shutil.rmtree(image_subset_base, ignore_errors = True)
print("Create folder " + image_subset_base)
os.makedirs(image_subset_base, exist_ok=True)

for nome_file in os.listdir(shape_files_dir):
        if nome_file.lower().endswith(".shp"):
            print("Raster clipping via shape file: " + nome_file)
            mask_and_clipping(os.path.join(shape_files_dir, nome_file), verified_dataset_path, image_subset_base)
            

file_image_subset_base = open("/tmp/image_subset_base_" + id + ".json", "w")
file_image_subset_base.write(json.dumps(image_subset_base))
file_image_subset_base.close()
