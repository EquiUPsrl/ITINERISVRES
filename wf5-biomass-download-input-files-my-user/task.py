import os
import requests
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_bounding_box_filter', action='store', type=str, required=True, dest='param_bounding_box_filter')
arg_parser.add_argument('--param_csv_filter', action='store', type=str, required=True, dest='param_csv_filter')
arg_parser.add_argument('--param_input_file_1', action='store', type=str, required=True, dest='param_input_file_1')
arg_parser.add_argument('--param_input_file_2', action='store', type=str, required=True, dest='param_input_file_2')
arg_parser.add_argument('--param_parameters_file', action='store', type=str, required=True, dest='param_parameters_file')
arg_parser.add_argument('--param_raster_dataset', action='store', type=str, required=True, dest='param_raster_dataset')
arg_parser.add_argument('--param_raster_depth', action='store', type=str, required=True, dest='param_raster_depth')
arg_parser.add_argument('--param_shape_coast', action='store', type=str, required=True, dest='param_shape_coast')

args = arg_parser.parse_args()
print(args)

id = args.id


param_bounding_box_filter = args.param_bounding_box_filter.replace('"','')
param_csv_filter = args.param_csv_filter.replace('"','')
param_input_file_1 = args.param_input_file_1.replace('"','')
param_input_file_2 = args.param_input_file_2.replace('"','')
param_parameters_file = args.param_parameters_file.replace('"','')
param_raster_dataset = args.param_raster_dataset.replace('"','')
param_raster_depth = args.param_raster_depth.replace('"','')
param_shape_coast = args.param_shape_coast.replace('"','')


file_urls = [
]

download_dir = "downloaded_files"
os.makedirs(download_dir, exist_ok=True)


def download_file(url, dest_folder):
    """Download a file from a URL into the specified folder."""
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Download completed: {filename}")
        return filepath
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return ""


def is_csv_file(filepath):
    """Check if a file should be treated as a CSV based on its extension."""
    return filepath.lower().endswith(".csv")


def verify_separator(filepath, sep=";"):
    """Verify that the CSV file uses the expected separator."""
    try:
        df = pd.read_csv(filepath, sep=sep, nrows=5)

        if df.shape[1] > 1:
            print(f"OK: {os.path.basename(filepath)} uses separator '{sep}'.")
            return True
        else:
            print(f"WARNING: {os.path.basename(filepath)} may NOT use '{sep}'.")
            return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False




raster_dataset_file = download_file(param_raster_dataset, download_dir)
input_file_1 = download_file(param_input_file_1, download_dir)
input_file_2 = download_file(param_input_file_2, download_dir)
bounding_box_filter = download_file(param_bounding_box_filter, download_dir)
csv_filter = download_file(param_csv_filter, download_dir)
shape_coast = download_file(param_shape_coast, download_dir)
raster_depth = download_file(param_raster_depth, download_dir)
parameters_file = download_file(param_parameters_file, download_dir)



downloaded_files = [
    raster_dataset_file,
    input_file_1,
    input_file_2,
    bounding_box_filter,
    csv_filter,
    shape_coast,
    raster_depth,
    parameters_file
]


print("\nCSV separator check:")
for fp in downloaded_files:
    if is_csv_file(fp):
        verify_separator(fp)
    else:
        print(f"Skipped (not a CSV): {os.path.basename(fp)}")

file_raster_dataset_file = open("/tmp/raster_dataset_file_" + id + ".json", "w")
file_raster_dataset_file.write(json.dumps(raster_dataset_file))
file_raster_dataset_file.close()
