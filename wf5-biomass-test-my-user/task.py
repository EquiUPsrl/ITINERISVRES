
import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')



args = arg_parser.parse_args()
print(args)

id = args.id




conf_base_path = "/tmp/data/WF5/"

conf_input_path = conf_base_path + "input"
conf_temp_path = conf_base_path + "tmp"
conf_output_path = conf_base_path + "output"

oceancolor = ["CHL", "SST"]
start_year = "2003"
end_year = "2003"

file_end_year = open("/tmp/end_year_" + id + ".json", "w")
file_end_year.write(json.dumps(end_year))
file_end_year.close()
file_oceancolor = open("/tmp/oceancolor_" + id + ".json", "w")
file_oceancolor.write(json.dumps(oceancolor))
file_oceancolor.close()
file_start_year = open("/tmp/start_year_" + id + ".json", "w")
file_start_year.write(json.dumps(start_year))
file_start_year.close()
