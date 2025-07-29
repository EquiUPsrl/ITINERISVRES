import json

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_parameter_file', action='store', type=str, required=True, dest='param_parameter_file')
arg_parser.add_argument('--param_prediction_file', action='store', type=str, required=True, dest='param_prediction_file')
arg_parser.add_argument('--param_training_file', action='store', type=str, required=True, dest='param_training_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_parameter_file = args.param_parameter_file.replace('"','')
param_prediction_file = args.param_prediction_file.replace('"','')
param_training_file = args.param_training_file.replace('"','')

conf_base_path = conf_base_path = '/tmp/data/'
conf_output_path = conf_output_path = '/tmp/data/output/'

params = {
    "param_training_file": param_training_file,
    "param_prediction_file": param_prediction_file,
    "param_parameter_file": param_parameter_file,
    "conf_base_path": conf_base_path,
    "conf_output_path": conf_output_path
}

with open("/tmp/data/wf4-biomass-params.json", "w") as f:
    json.dump(params, f)

