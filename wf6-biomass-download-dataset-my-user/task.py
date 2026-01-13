import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_filter_file', action='store', type=str, required=True, dest='param_filter_file')
arg_parser.add_argument('--param_input_file', action='store', type=str, required=True, dest='param_input_file')
arg_parser.add_argument('--param_parameters_file', action='store', type=str, required=True, dest='param_parameters_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_filter_file = args.param_filter_file.replace('"','')
param_input_file = args.param_input_file.replace('"','')
param_parameters_file = args.param_parameters_file.replace('"','')

conf_base_path = conf_base_path = '/tmp/data/WF6/'

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



input_file = download_file(param_input_file, input_dir)
filter_file = download_file(param_filter_file, input_dir)
parameters_file = download_file(param_parameters_file, input_dir)

file_filter_file = open("/tmp/filter_file_" + id + ".json", "w")
file_filter_file.write(json.dumps(filter_file))
file_filter_file.close()
file_input_file = open("/tmp/input_file_" + id + ".json", "w")
file_input_file.write(json.dumps(input_file))
file_input_file.close()
file_parameters_file = open("/tmp/parameters_file_" + id + ".json", "w")
file_parameters_file.write(json.dumps(parameters_file))
file_parameters_file.close()
