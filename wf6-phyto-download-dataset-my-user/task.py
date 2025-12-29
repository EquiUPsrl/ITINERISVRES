import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_abiotic_file', action='store', type=str, required=True, dest='param_abiotic_file')
arg_parser.add_argument('--param_biotic_file', action='store', type=str, required=True, dest='param_biotic_file')
arg_parser.add_argument('--param_config_file', action='store', type=str, required=True, dest='param_config_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_abiotic_file = args.param_abiotic_file.replace('"','')
param_biotic_file = args.param_biotic_file.replace('"','')
param_config_file = args.param_config_file.replace('"','')

conf_input_path = conf_input_path = '/tmp/data/WF6/data'

input_dir = conf_input_path
os.makedirs(input_dir, exist_ok=True)

print("input_dir -> ", input_dir)


def download_file(url, dest_folder):
    """Download a file from a URL into the specified folder."""
    filename = url.split("/")[-1]
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



biotic_file = download_file(param_biotic_file, input_dir)
abiotic_file = download_file(param_abiotic_file, input_dir)
config_file = download_file(param_config_file, input_dir)

file_abiotic_file = open("/tmp/abiotic_file_" + id + ".json", "w")
file_abiotic_file.write(json.dumps(abiotic_file))
file_abiotic_file.close()
file_biotic_file = open("/tmp/biotic_file_" + id + ".json", "w")
file_biotic_file.write(json.dumps(biotic_file))
file_biotic_file.close()
file_config_file = open("/tmp/config_file_" + id + ".json", "w")
file_config_file.write(json.dumps(config_file))
file_config_file.close()
