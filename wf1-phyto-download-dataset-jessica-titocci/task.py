import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_config_file', action='store', type=str, required=True, dest='param_config_file')
arg_parser.add_argument('--param_dataportal_file', action='store', type=str, required=True, dest='param_dataportal_file')
arg_parser.add_argument('--param_formulas_file', action='store', type=str, required=True, dest='param_formulas_file')
arg_parser.add_argument('--param_input_file', action='store', type=str, required=True, dest='param_input_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_config_file = args.param_config_file.replace('"','')
param_dataportal_file = args.param_dataportal_file.replace('"','')
param_formulas_file = args.param_formulas_file.replace('"','')
param_input_file = args.param_input_file.replace('"','')

conf_base_path = conf_base_path = '/tmp/data/WF1/'

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
dataportal_file = download_file(param_dataportal_file, input_dir)
formulas_file = download_file(param_formulas_file, input_dir)
config_file = download_file(param_config_file, input_dir, 'config.csv')

file_config_file = open("/tmp/config_file_" + id + ".json", "w")
file_config_file.write(json.dumps(config_file))
file_config_file.close()
file_dataportal_file = open("/tmp/dataportal_file_" + id + ".json", "w")
file_dataportal_file.write(json.dumps(dataportal_file))
file_dataportal_file.close()
file_formulas_file = open("/tmp/formulas_file_" + id + ".json", "w")
file_formulas_file.write(json.dumps(formulas_file))
file_formulas_file.close()
file_input_file = open("/tmp/input_file_" + id + ".json", "w")
file_input_file.write(json.dumps(input_file))
file_input_file.close()
