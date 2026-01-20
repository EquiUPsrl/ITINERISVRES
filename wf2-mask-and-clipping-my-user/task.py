import shutil
import os
import fiona
import zipfile
from rasterio.mask import mask
import numpy as np
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--dataset_path', action='store', type=str, required=True, dest='dataset_path')

arg_parser.add_argument('--shape_zip_file', action='store', type=str, required=True, dest='shape_zip_file')


args = arg_parser.parse_args()
print(args)

id = args.id

dataset_path = args.dataset_path.replace('"','')
shape_zip_file = args.shape_zip_file.replace('"','')


conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

tmp_dir = conf_tmp_path
shape_file_zip = shape_zip_file


print("EXTRACT RASTER PORTION OF INTEREST USING THE SHP FILE")


input_dir = dataset_path
image_subset_base = os.path.join(tmp_dir, "ImageSubset")

print("Delete folder: " + image_subset_base)
shutil.rmtree(image_subset_base, ignore_errors = True)


def extractZipFile(zip_file, cartella_destinazione):
    if not os.path.exists(zip_file):
        print("The ZIP file does not exist: " + zip_file)
        return

    os.makedirs(cartella_destinazione, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue

            nome_file = os.path.basename(member.filename)
            if not nome_file:  # in caso il nome sia vuoto
                continue

            percorso_estratto = os.path.join(cartella_destinazione, nome_file)

            with zip_ref.open(member) as source, open(percorso_estratto, "wb") as target:
                target.write(source.read())

    print(f"✅ Extracted all files from {zip_file} to: {cartella_destinazione}")




print("Extract shape file from zip")
basenameShapeFile = os.path.basename(shape_file_zip)
shapeFileDir = os.path.splitext(basenameShapeFile)[0]
shapefile_path = os.path.join(tmp_dir, shapeFileDir)
print("Extract file " + shape_file_zip + " in " + shapefile_path)
extractZipFile(shape_file_zip, shapefile_path)
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
    output_path = os.path.join(image_subset_base, f"{new_subdir}")
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

file_image_subset_base = open("/tmp/image_subset_base_" + id + ".json", "w")
file_image_subset_base.write(json.dumps(image_subset_base))
file_image_subset_base.close()
