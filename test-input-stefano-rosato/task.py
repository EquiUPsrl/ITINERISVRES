from pathlib import Path

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')



args = arg_parser.parse_args()
print(args)

id = args.id




test_input = "verifica-ambiente"


test_file = Path("/tmp/data/test_gateway_workflow.txt")
test_file.write_text("Output prodotto dal workflow NaaVRE\n", encoding="utf-8")
print("File prodotto:", test_file)

file_test_input = open("/tmp/test_input_" + id + ".json", "w")
file_test_input.write(json.dumps(test_input))
file_test_input.close()
