import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--raster_zip_csv', action='store', type=str, required=True, dest='raster_zip_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

raster_zip_csv = args.raster_zip_csv.replace('"','')



df = pd.read_csv(raster_zip_csv, sep=";")

zip_files = []
oceancolor = []
ocean_productivity = []

for _, row in df.iterrows():
    product = row['products']
    zip_link = row['zip_file']

    if pd.notna(zip_link) and zip_link.strip() != "":
        zip_files.append(product + "||" + zip_link.strip())

file_zip_files = open("/tmp/zip_files_" + id + ".json", "w")
file_zip_files.write(json.dumps(zip_files))
file_zip_files.close()
