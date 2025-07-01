
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')



args = arg_parser.parse_args()
print(args)

id = args.id




input_datain_file = 'data/Phytoplankton__Progetto_Strategico_2009_2012_Australia.csv'
input_operator_file = 'data/2_FILEinformativo_OPERATORE.csv'

file_input_datain_file = open("/tmp/input_datain_file_" + id + ".json", "w")
file_input_datain_file.write(json.dumps(input_datain_file))
file_input_datain_file.close()
file_input_operator_file = open("/tmp/input_operator_file_" + id + ".json", "w")
file_input_operator_file.write(json.dumps(input_operator_file))
file_input_operator_file.close()
