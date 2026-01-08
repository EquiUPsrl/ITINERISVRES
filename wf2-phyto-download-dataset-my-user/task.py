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



input_biotic = download_file(param_input_biotic, input_dir)
input_abiotic = download_file(param_input_abiotic, input_dir)
input_traits = download_file(param_input_traits, input_dir)
locations_config = download_file(param_locations_config, input_dir, 'locations_config.csv')

file_input_abiotic = open("/tmp/input_abiotic_" + id + ".json", "w")
file_input_abiotic.write(json.dumps(input_abiotic))
file_input_abiotic.close()
file_input_biotic = open("/tmp/input_biotic_" + id + ".json", "w")
file_input_biotic.write(json.dumps(input_biotic))
file_input_biotic.close()
file_input_traits = open("/tmp/input_traits_" + id + ".json", "w")
file_input_traits.write(json.dumps(input_traits))
file_input_traits.close()
file_locations_config = open("/tmp/locations_config_" + id + ".json", "w")
file_locations_config.write(json.dumps(locations_config))
file_locations_config.close()
