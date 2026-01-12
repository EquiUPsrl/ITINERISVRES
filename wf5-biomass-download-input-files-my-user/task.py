import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_bounding_box_filter', action='store', type=str, required=True, dest='param_bounding_box_filter')
arg_parser.add_argument('--param_filter_file', action='store', type=str, required=True, dest='param_filter_file')
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
param_filter_file = args.param_filter_file.replace('"','')
param_input_file_1 = args.param_input_file_1.replace('"','')
param_input_file_2 = args.param_input_file_2.replace('"','')
param_parameters_file = args.param_parameters_file.replace('"','')
param_raster_dataset = args.param_raster_dataset.replace('"','')
param_raster_depth = args.param_raster_depth.replace('"','')
param_shape_coast = args.param_shape_coast.replace('"','')

conf_base_path = conf_base_path = '/tmp/data/WF5/'

input_dir = os.path.join(conf_base_path, "data")
os.makedirs(input_dir, exist_ok=True)


def download_file(url, dest_folder, file_name=None):
    """Download a file from a URL into the specified folder."""

    filename = file_name if file_name else url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Download completed: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return ""



raster_dataset_file = download_file(param_raster_dataset, input_dir)
input_file_1 = download_file(param_input_file_1, input_dir)
input_file_2 = download_file(param_input_file_2, input_dir)
bounding_box_filter = download_file(param_bounding_box_filter, input_dir)
filter_file = download_file(param_filter_file, input_dir)
shape_coast = download_file(param_shape_coast, input_dir)
raster_depth = download_file(param_raster_depth, input_dir)
parameters_file = download_file(param_parameters_file, input_dir)

file_bounding_box_filter = open("/tmp/bounding_box_filter_" + id + ".json", "w")
file_bounding_box_filter.write(json.dumps(bounding_box_filter))
file_bounding_box_filter.close()
file_filter_file = open("/tmp/filter_file_" + id + ".json", "w")
file_filter_file.write(json.dumps(filter_file))
file_filter_file.close()
file_input_file_1 = open("/tmp/input_file_1_" + id + ".json", "w")
file_input_file_1.write(json.dumps(input_file_1))
file_input_file_1.close()
file_input_file_2 = open("/tmp/input_file_2_" + id + ".json", "w")
file_input_file_2.write(json.dumps(input_file_2))
file_input_file_2.close()
file_parameters_file = open("/tmp/parameters_file_" + id + ".json", "w")
file_parameters_file.write(json.dumps(parameters_file))
file_parameters_file.close()
file_raster_dataset_file = open("/tmp/raster_dataset_file_" + id + ".json", "w")
file_raster_dataset_file.write(json.dumps(raster_dataset_file))
file_raster_dataset_file.close()
file_raster_depth = open("/tmp/raster_depth_" + id + ".json", "w")
file_raster_depth.write(json.dumps(raster_depth))
file_raster_depth.close()
file_shape_coast = open("/tmp/shape_coast_" + id + ".json", "w")
file_shape_coast.write(json.dumps(shape_coast))
file_shape_coast.close()
