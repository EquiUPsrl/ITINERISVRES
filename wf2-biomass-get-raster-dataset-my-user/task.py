import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--raster_zip_csv', action='store', type=str, required=True, dest='raster_zip_csv')

arg_parser.add_argument('--param_end_year', action='store', type=str, required=True, dest='param_end_year')
arg_parser.add_argument('--param_start_year', action='store', type=str, required=True, dest='param_start_year')

args = arg_parser.parse_args()
print(args)

id = args.id

raster_zip_csv = args.raster_zip_csv.replace('"','')

param_end_year = args.param_end_year.replace('"','')
param_start_year = args.param_start_year.replace('"','')


df = pd.read_csv(raster_zip_csv, sep=";")

zip_files = []
oceancolor = []
ocean_productivity = []

start_year = param_start_year
end_year = param_end_year

for _, row in df.iterrows():
    product = row['products']
    zip_link = row['zip_file']

    if pd.notna(zip_link) and zip_link.strip() != "":
        zip_files.append(product + "||" + zip_link.strip())

file_end_year = open("/tmp/end_year_" + id + ".json", "w")
file_end_year.write(json.dumps(end_year))
file_end_year.close()
file_start_year = open("/tmp/start_year_" + id + ".json", "w")
file_start_year.write(json.dumps(start_year))
file_start_year.close()
file_zip_files = open("/tmp/zip_files_" + id + ".json", "w")
file_zip_files.write(json.dumps(zip_files))
file_zip_files.close()
