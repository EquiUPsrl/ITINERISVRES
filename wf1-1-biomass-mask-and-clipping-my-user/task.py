import shutil
import os
import fiona
from rasterio.mask import mask
import numpy as np
import zipfile
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--shape_zip_file', action='store', type=str, required=True, dest='shape_zip_file')

arg_parser.add_argument('--verified_dataset_path', action='store', type=str, required=True, dest='verified_dataset_path')


args = arg_parser.parse_args()
print(args)

id = args.id

shape_zip_file = args.shape_zip_file.replace('"','')
verified_dataset_path = args.verified_dataset_path.replace('"','')


conf_base_path = conf_base_path = '/tmp/data/WF1_1/work/'
conf_tmp_path = conf_tmp_path = '/tmp/data/WF1_1/work/' + 'tmp'

work_path = conf_base_path
tmp_dir = conf_tmp_path


print("EXTRACT RASTER PORTION OF INTEREST USING THE SHP FILE")


input_dir = verified_dataset_path
image_subset_base = os.path.join(tmp_dir, "ImageSubset")

print("I delete the folder " + image_subset_base)
shutil.rmtree(image_subset_base, ignore_errors = True)


def extractZipFile(zip_file, cartella_destinazione):
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(cartella_destinazione)
        print(f"File {zip_file} extracted to: {cartella_destinazione}")
    else:
        print("The ZIP file does not exist: " + zip_file)


print("Extract shape file zip")
basenameShapeFile = os.path.basename(shape_zip_file)
shapeFileDir = os.path.splitext(basenameShapeFile)[0]
shapefile_path = os.path.join(tmp_dir, shapeFileDir)
print("Extract file " + shape_zip_file + " in " + shapefile_path)
extractZipFile(shape_zip_file, shapefile_path)
shapefile_folder = shapefile_path

print("shapefile_path", shapefile_path)

with fiona.open(shapefile_path, 'r') as shapefile:
    geoms = [feature["geometry"] for feature in shapefile]  # List of all polygons

raster_extensions = [".tif", ".tiff", ".asc", ".grb", ".grib", ".img"]

for subdir in os.listdir(input_dir):
    if subdir.startswith('.'):
        continue

    subdir_path = os.path.join(input_dir, subdir)

    raster_files = []
    for root, dirs, files in os.walk(subdir_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if file.startswith('.'):
                continue
            if os.path.splitext(file)[1].lower() in raster_extensions:
                raster_files.append(os.path.join(root, file))


    new_subdir = "Chl" if subdir == "CHL" else subdir
    output_path = os.path.join(image_subset_base, f"Sub{new_subdir}")
    os.makedirs(output_path, exist_ok=True)

    print(f"\nIn the {subdir_path} folder (recursively) I found {len(raster_files)} raster.")
    if not raster_files:
        print(f"No rasters found in {subdir_path} (or subfolders).")
        continue

    raster_files = sorted(raster_files)

    
    for raster_file in raster_files:
        
        raster_path = raster_file #os.path.join(input_dir, raster_file)
    
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, geoms, crop=True, filled=True)
            
            out_image = out_image.astype('float32')
            
            out_image[out_image == 0] = np.nan  # oppure: out_image[out_image.mask] = np.nan
    
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

    print("Clipping of rasters " + subdir + " completed. Output saved to" + output_path)

file_image_subset_base = open("/tmp/image_subset_base_" + id + ".json", "w")
file_image_subset_base.write(json.dumps(image_subset_base))
file_image_subset_base.close()
