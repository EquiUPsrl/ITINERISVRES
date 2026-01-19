import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_aster_calculator_formulas', action='store', type=str, required=True, dest='param_aster_calculator_formulas')
arg_parser.add_argument('--param_config_file', action='store', type=str, required=True, dest='param_config_file')
arg_parser.add_argument('--param_raster_appeears', action='store', type=str, required=True, dest='param_raster_appeears')
arg_parser.add_argument('--param_raster_calculator_config', action='store', type=str, required=True, dest='param_raster_calculator_config')
arg_parser.add_argument('--param_raster_zip_file', action='store', type=str, required=True, dest='param_raster_zip_file')
arg_parser.add_argument('--param_shape_zip_file', action='store', type=str, required=True, dest='param_shape_zip_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_aster_calculator_formulas = args.param_aster_calculator_formulas.replace('"','')
param_config_file = args.param_config_file.replace('"','')
param_raster_appeears = args.param_raster_appeears.replace('"','')
param_raster_calculator_config = args.param_raster_calculator_config.replace('"','')
param_raster_zip_file = args.param_raster_zip_file.replace('"','')
param_shape_zip_file = args.param_shape_zip_file.replace('"','')

conf_input_path = conf_input_path = '/tmp/data/WF2/work/' + 'input'

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



appeears_raster_file = download_file(param_raster_appeears, input_dir)
raster_zip_file = download_file(param_raster_zip_file, input_dir)
config_file = download_file(param_config_file, input_dir, 'config.csv')
shape_zip_file = download_file(param_shape_zip_file, input_dir)
raster_calculator_config = download_file(param_raster_calculator_config, input_dir)
raster_calculator_formulas = download_file(param_aster_calculator_formulas, input_dir)

file_appeears_raster_file = open("/tmp/appeears_raster_file_" + id + ".json", "w")
file_appeears_raster_file.write(json.dumps(appeears_raster_file))
file_appeears_raster_file.close()
file_config_file = open("/tmp/config_file_" + id + ".json", "w")
file_config_file.write(json.dumps(config_file))
file_config_file.close()
file_raster_calculator_config = open("/tmp/raster_calculator_config_" + id + ".json", "w")
file_raster_calculator_config.write(json.dumps(raster_calculator_config))
file_raster_calculator_config.close()
file_raster_calculator_formulas = open("/tmp/raster_calculator_formulas_" + id + ".json", "w")
file_raster_calculator_formulas.write(json.dumps(raster_calculator_formulas))
file_raster_calculator_formulas.close()
file_raster_zip_file = open("/tmp/raster_zip_file_" + id + ".json", "w")
file_raster_zip_file.write(json.dumps(raster_zip_file))
file_raster_zip_file.close()
file_shape_zip_file = open("/tmp/shape_zip_file_" + id + ".json", "w")
file_shape_zip_file.write(json.dumps(shape_zip_file))
file_shape_zip_file.close()
