
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')



args = arg_parser.parse_args()
print(args)

id = args.id




datain: str = 'data/Overalldataset-harmonized2.csv'

file_datain = open("/tmp/datain_" + id + ".json", "w")
file_datain.write(json.dumps(datain))
file_datain.close()
