import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_config_file', action='store', type=str, required=True, dest='param_config_file')
arg_parser.add_argument('--param_data_file', action='store', type=str, required=True, dest='param_data_file')
arg_parser.add_argument('--param_modis_interval_8d', action='store', type=str, required=True, dest='param_modis_interval_8d')
arg_parser.add_argument('--param_parameters_file', action='store', type=str, required=True, dest='param_parameters_file')
arg_parser.add_argument('--param_raster_dataset', action='store', type=str, required=True, dest='param_raster_dataset')
arg_parser.add_argument('--param_shape_zip_file', action='store', type=str, required=True, dest='param_shape_zip_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_config_file = args.param_config_file.replace('"','')
param_data_file = args.param_data_file.replace('"','')
param_modis_interval_8d = args.param_modis_interval_8d.replace('"','')
param_parameters_file = args.param_parameters_file.replace('"','')
param_raster_dataset = args.param_raster_dataset.replace('"','')
param_shape_zip_file = args.param_shape_zip_file.replace('"','')

conf_input_path = conf_input_path = '/tmp/data/WF1_1/work/' + 'input'

input_dir = conf_input_path
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
data_file = download_file(param_data_file, input_dir)
parameters_file = download_file(param_parameters_file, input_dir)
config_file = download_file(param_config_file, input_dir, 'config.csv')
shape_zip_file = download_file(param_shape_zip_file, input_dir)
modis_interval_8d = download_file(param_modis_interval_8d, input_dir)

file_config_file = open("/tmp/config_file_" + id + ".json", "w")
file_config_file.write(json.dumps(config_file))
file_config_file.close()
file_data_file = open("/tmp/data_file_" + id + ".json", "w")
file_data_file.write(json.dumps(data_file))
file_data_file.close()
file_modis_interval_8d = open("/tmp/modis_interval_8d_" + id + ".json", "w")
file_modis_interval_8d.write(json.dumps(modis_interval_8d))
file_modis_interval_8d.close()
file_parameters_file = open("/tmp/parameters_file_" + id + ".json", "w")
file_parameters_file.write(json.dumps(parameters_file))
file_parameters_file.close()
file_raster_dataset_file = open("/tmp/raster_dataset_file_" + id + ".json", "w")
file_raster_dataset_file.write(json.dumps(raster_dataset_file))
file_raster_dataset_file.close()
file_shape_zip_file = open("/tmp/shape_zip_file_" + id + ".json", "w")
file_shape_zip_file.write(json.dumps(shape_zip_file))
file_shape_zip_file.close()
