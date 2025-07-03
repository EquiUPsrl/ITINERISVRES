
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')



args = arg_parser.parse_args()
print(args)

id = args.id




test = "ciao"

file_test = open("/tmp/test_" + id + ".json", "w")
file_test.write(json.dumps(test))
file_test.close()
