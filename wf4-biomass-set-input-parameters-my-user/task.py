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

conf_base_path = conf_base_path = '/tmp/WF4/'

params = {
    "param_training_file": param_training_file,
    "param_prediction_file": param_prediction_file,
    "param_parameter_file": param_parameter_file
}

params_path = os.path.join(conf_base_path, "data", "wf4-biomass-params.json")

os.makedirs(os.path.dirname(params_path), exist_ok=True)

with open(params_path, "w") as f:
    json.dump(params, f)

remote_training_file = param_training_file
remote_prediction_file = param_prediction_file
remote_parameter_file = param_parameter_file

file_remote_parameter_file = open("/tmp/remote_parameter_file_" + id + ".json", "w")
file_remote_parameter_file.write(json.dumps(remote_parameter_file))
file_remote_parameter_file.close()
file_remote_prediction_file = open("/tmp/remote_prediction_file_" + id + ".json", "w")
file_remote_prediction_file.write(json.dumps(remote_prediction_file))
file_remote_prediction_file.close()
file_remote_training_file = open("/tmp/remote_training_file_" + id + ".json", "w")
file_remote_training_file.write(json.dumps(remote_training_file))
file_remote_training_file.close()
