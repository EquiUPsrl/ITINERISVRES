import os
import re
from datetime import datetime
from datetime import timedelta
import rasterio
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--dataset_tmp_path', action='store', type=str, required=True, dest='dataset_tmp_path')


args = arg_parser.parse_args()
print(args)

id = args.id

dataset_tmp_path = args.dataset_tmp_path.replace('"','')



def list_non_empty_dirs(parent_folder):
    """
    Returns a list of non-empty subfolders in parent_folder.
    """
    non_empty_dirs = []

    for name in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, name)

        if os.path.isdir(full_path):
            if os.listdir(full_path):  
                non_empty_dirs.append(full_path)

    return non_empty_dirs

def estrai_date(filename):
    match = re.search(r'doy(\d{4})(\d{3})', filename)
    if match:
        year = int(match.group(1))
        day_of_year = int(match.group(2))
        start = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        end = start + timedelta(days=7)  # Interval MOD21A2: 8 days
        return start, end
    return None, None

def extract_prefix(filename):
    """
    Returns the substring from the first character up to and including '_doy'.
    """
    idx = filename.find("_doy")
    if idx != -1:
        return filename[:idx + 4]  # +4 for including '_doy'
    else:
        return None

def trova_file_mancanti(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
    date_ranges = []

    for f in files:
        start, end = estrai_date(f)
        prefix = extract_prefix(f)
        if start and end:
            date_ranges.append((start, end))

    date_ranges.sort(key=lambda x: x[0])

    missing_files = []
    for i in range(len(date_ranges) - 1):
        current_end = date_ranges[i][1]
        next_start = date_ranges[i + 1][0]

        day = current_end + timedelta(days=1)
        while day < next_start:
            year = day.year
            doy = day.timetuple().tm_yday
            missing_filename = f"{prefix}{year}{doy:03d}000000_aid0001.tif"
            missing_files.append(missing_filename)
            day += timedelta(days=8)

    return sorted(missing_files, key=lambda f: estrai_date(f)[0])

def generate_raster_medium_multiband(folder_raster, cartella_output, missing_files):
    presenti = [f for f in os.listdir(folder_raster) if f.lower().endswith(('.tif', '.tiff'))]
    presenti.sort(key=lambda f: estrai_date(f)[0])
    os.makedirs(cartella_output, exist_ok=True)

    for nome_mancante in missing_files:
        if not nome_mancante.lower().endswith('.tif'):
            nome_mancante += '.tif'

        data_mancante = estrai_date(nome_mancante)[0]

        posizione = None
        for i, nome in enumerate(presenti):
            if estrai_date(nome)[0] > data_mancante:
                posizione = i
                break

        if posizione is None or posizione == 0:
            print(f"⚠️ Unable to calculate mean for {nome_mancante}: previous or next name missing.")
            continue

        file_prev = os.path.join(folder_raster, presenti[posizione - 1])
        file_next = os.path.join(folder_raster, presenti[posizione])

        with rasterio.open(file_prev) as src1, rasterio.open(file_next) as src2:
            if src1.shape != src2.shape or src1.transform != src2.transform or src1.count != src2.count:
                raise ValueError(f"Raster {presenti[posizione-1]} and {presenti[posizione]} are not compatible.")

            dtype_orig = src1.dtypes[0]
            nodata = src1.nodata if src1.nodata is not None else np.nan

            bands, height, width = src1.count, src1.height, src1.width
            media_all = np.empty((bands, height, width), dtype=np.float32)

            for b in range(1, bands+1):
                arr1 = src1.read(b).astype(np.float32)
                arr2 = src2.read(b).astype(np.float32)
                mask = np.isnan(arr1) | np.isnan(arr2) if np.isnan(nodata) else (arr1 == nodata) | (arr2 == nodata)
                media = (arr1 + arr2) / 2.0
                media[mask] = np.nan if np.isnan(nodata) else nodata

                if np.issubdtype(np.dtype(dtype_orig), np.integer):
                    if np.isnan(nodata):
                        media_all[b-1] = media
                    else:
                        info = np.iinfo(dtype_orig)
                        media = np.clip(media, info.min, info.max)
                        media_all[b-1] = np.rint(media).astype(dtype_orig)
                else:
                    media_all[b-1] = media

            final_dtype = dtype_orig if np.issubdtype(np.dtype(dtype_orig), np.integer) and not np.isnan(nodata) else 'float32'

            out_path = os.path.join(cartella_output, nome_mancante)
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=bands,
                dtype=final_dtype,
                crs=src1.crs,
                transform=src1.transform,
                nodata=np.nan if np.isnan(nodata) else nodata,
                compress='LZW'
            ) as dst:
                dst.write(media_all.astype(final_dtype))

        print(f"✅ Creato {nome_mancante} come media di {presenti[posizione-1]} e {presenti[posizione]}")

        presenti.insert(posizione, nome_mancante)



base_path = dataset_tmp_path
dataset_path = base_path

dirs = list_non_empty_dirs(base_path)

print(dirs)

for d in dirs:
    cartella_output = d
    missing_files = trova_file_mancanti(d)
    print(f"Found {len(missing_files)} missing files in {d}.")
    
    if missing_files:
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("✅ No missing files found.")

    generate_raster_medium_multiband(d, cartella_output, missing_files)

file_dataset_path = open("/tmp/dataset_path_" + id + ".json", "w")
file_dataset_path.write(json.dumps(dataset_path))
file_dataset_path.close()
