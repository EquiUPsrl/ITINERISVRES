import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_input_abiotic', action='store', type=str, required=True, dest='param_input_abiotic')
arg_parser.add_argument('--param_input_biotic', action='store', type=str, required=True, dest='param_input_biotic')
arg_parser.add_argument('--param_input_traits', action='store', type=str, required=True, dest='param_input_traits')
arg_parser.add_argument('--param_locations_config', action='store', type=str, required=True, dest='param_locations_config')

args = arg_parser.parse_args()
print(args)

id = args.id


param_input_abiotic = args.param_input_abiotic.replace('"','')
param_input_biotic = args.param_input_biotic.replace('"','')
param_input_traits = args.param_input_traits.replace('"','')
param_locations_config = args.param_locations_config.replace('"','')

conf_base_path = conf_base_path = '/tmp/data/WF2/'

input_dir = os.path.join(conf_base_path, "data")
os.makedirs(input_dir, exist_ok=True)

print("input_dir -> ", input_dir)


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



input_biotic_file = download_file(param_input_biotic, input_dir)
input_abiotic_file = download_file(param_input_abiotic, input_dir)
input_traits_file = download_file(param_input_traits, input_dir)
locations_config_file = download_file(param_locations_config, input_dir, 'locations_config.csv')

file_input_abiotic_file = open("/tmp/input_abiotic_file_" + id + ".json", "w")
file_input_abiotic_file.write(json.dumps(input_abiotic_file))
file_input_abiotic_file.close()
file_input_biotic_file = open("/tmp/input_biotic_file_" + id + ".json", "w")
file_input_biotic_file.write(json.dumps(input_biotic_file))
file_input_biotic_file.close()
file_input_traits_file = open("/tmp/input_traits_file_" + id + ".json", "w")
file_input_traits_file.write(json.dumps(input_traits_file))
file_input_traits_file.close()
file_locations_config_file = open("/tmp/locations_config_file_" + id + ".json", "w")
file_locations_config_file.write(json.dumps(locations_config_file))
file_locations_config_file.close()
