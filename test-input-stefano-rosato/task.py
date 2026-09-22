
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')



args = arg_parser.parse_args()
print(args)

id = args.id




test_input = "verifica-ambiente"

file_test_input = open("/tmp/test_input_" + id + ".json", "w")
file_test_input.write(json.dumps(test_input))
file_test_input.close()
