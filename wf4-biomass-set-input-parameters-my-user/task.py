import os
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


params = {
    "param_training_file": param_training_file,
    "param_prediction_file": param_prediction_file,
    "param_parameter_file": param_parameter_file
}

params_path = "/tmp/WF4/data/wf4-biomass-params.json"

os.makedirs(os.path.dirname(params_path), exist_ok=True)

with open(params_path, "w") as f:
    json.dump(params, f)

file_params_path = open("/tmp/params_path_" + id + ".json", "w")
file_params_path.write(json.dumps(params_path))
file_params_path.close()
