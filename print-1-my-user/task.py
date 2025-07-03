
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_file', action='store', type=str, required=True, dest='param_file')

args = arg_parser.parse_args()
print(args)

id = args.id


param_file = args.param_file.replace('"','')

conf_dati_file = conf_dati_file = 'work/input/TrainingData.csv'

print(conf_dati_file)
print(param_file)
test = "hello"

file_test = open("/tmp/test_" + id + ".json", "w")
file_test.write(json.dumps(test))
file_test.close()
