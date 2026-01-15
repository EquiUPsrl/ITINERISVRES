import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_data_file', action='store', type=str, required=True, dest='param_data_file')
arg_parser.add_argument('--param_parameters_file', action='store', type=str, required=True, dest='param_parameters_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_data_file = args.param_data_file.replace('"','')
param_parameters_file = args.param_parameters_file.replace('"','')

conf_input_path = conf_input_path = '/tmp/data/WF1_0/' + 'input'

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



data_file = download_file(param_data_file, input_dir)
parameters_file = download_file(param_parameters_file, input_dir)

file_data_file = open("/tmp/data_file_" + id + ".json", "w")
file_data_file.write(json.dumps(data_file))
file_data_file.close()
file_parameters_file = open("/tmp/parameters_file_" + id + ".json", "w")
file_parameters_file.write(json.dumps(parameters_file))
file_parameters_file.close()
