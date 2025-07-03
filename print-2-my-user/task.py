
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--test', action='store', type=str, required=True, dest='test')


args = arg_parser.parse_args()
print(args)

id = args.id

test = args.test.replace('"','')



print(test)

