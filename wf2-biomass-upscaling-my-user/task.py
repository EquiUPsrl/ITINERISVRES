import os
import rasterio
import numpy as np
from rasterio.warp import reproject
from rasterio.enums import Resampling

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--scale_factor_path', action='store', type=str, required=True, dest='scale_factor_path')


args = arg_parser.parse_args()
print(args)

id = args.id

scale_factor_path = args.scale_factor_path.replace('"','')


conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

def upscaling(raster_500m_path, raster_1km_path, output_path):

    raster_1km_files = sorted([f for f in os.listdir(raster_1km_path) if f.endswith(".tif")])
    
    with rasterio.open(os.path.join(raster_1km_path, raster_1km_files[0])) as raster_1km:
        target_crs = raster_1km.crs
        target_transform = raster_1km.transform
        target_width = raster_1km.width
        target_height = raster_1km.height

    os.makedirs(output_path, exist_ok=True)

    raster_500m_files = sorted([f for f in os.listdir(raster_500m_path) if f.endswith(".tif")])

    for raster_file in raster_500m_files:
    
        with rasterio.open(os.path.join(raster_500m_path, raster_file)) as raster_500m:
            upscaled_data = raster_500m.read(
                out_shape=(raster_500m.count, raster_500m.height // 2, raster_500m.width // 2),
                resampling=Resampling.average
            )
        
            upscaled_transform = raster_500m.transform * raster_500m.transform.scale(2, 2)
        
            reprojected_data = np.empty((raster_500m.count, target_height, target_width), dtype=upscaled_data.dtype)
        
            for band in range(raster_500m.count):
                reproject(
                    source=upscaled_data[band],
                    destination=reprojected_data[band],
                    src_transform=upscaled_transform,
                    src_crs=raster_500m.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
        
        with rasterio.open(
            os.path.join(output_path, raster_file),
            'w',
            driver='GTiff',
            height=target_height,
            width=target_width,
            count=raster_500m.count,
            dtype=reprojected_data.dtype,
            crs=target_crs,
            transform=target_transform
        ) as dst:
            dst.write(reprojected_data)
        
    print(f"Upscaling completed. Raster {raster_500m_path} aligned at 1KM")



tmp_dir = conf_tmp_path
input_dir = scale_factor_path #os.path.join(tmp_dir, "ImageSubset")
upscaled_dir = os.path.join(tmp_dir, "ImageSubset_upscaled")
os.makedirs(upscaled_dir, exist_ok=True)

dirs_1KM = []
dirs_500m = []

for subdir in os.listdir(input_dir):
    if subdir.startswith('.'):
        continue

    if "_1KM" in subdir:
        dirs_1KM.append(subdir)
    else:
        dirs_500m.append(subdir)

print("dir_1KM", dirs_1KM)
print("dirs_500m", dirs_500m)

if os.path.exists(upscaled_dir) and os.path.isdir(upscaled_dir):
    present_dirs = [
        name for name in os.listdir(upscaled_dir)
        if os.path.isdir(os.path.join(upscaled_dir, name))
    ]

    missing_dirs = [d for d in dirs_500m if d not in present_dirs]

    print("Folders present:", present_dirs)
    print("Missing folders:", missing_dirs)
else:
    print(f"The folder {upscaled_dir} does not exist.")
    missing_dirs = dirs_500m

if len(dirs_1KM) > 0 and len(missing_dirs) > 0:
    path_1KM = os.path.join(input_dir, dirs_1KM[0])
    for d in missing_dirs:
        path_500m = os.path.join(input_dir, d)
        output_dir = os.path.join(upscaled_dir, d)
        print(f"Upscaling {path_500m} from {path_1KM}")
        upscaling(path_500m, path_1KM, output_dir)

file_upscaled_dir = open("/tmp/upscaled_dir_" + id + ".json", "w")
file_upscaled_dir.write(json.dumps(upscaled_dir))
file_upscaled_dir.close()
