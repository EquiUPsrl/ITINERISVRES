import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--raster_dataset_csv', action='store', type=str, required=True, dest='raster_dataset_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

raster_dataset_csv = args.raster_dataset_csv.replace('"','')



df = pd.read_csv(raster_dataset_csv, sep=";")

zip_files = []
oceancolor = []
ocean_productivity = []

start_year = int(df.iloc[0]["start_year"])
end_year = int(df.iloc[0]["end_year"])
interval = df.iloc[0]["interval"]
resolution = df.iloc[0]["resolution"]

for _, row in df.iterrows():
    product = row['products']
    download_type = row['download']
    zip_link = row['zip_file']

    if pd.notna(zip_link) and zip_link.strip() != "":
        zip_files.append(product + "||" + zip_link.strip())
    elif download_type == "oceancolor" and (pd.isna(zip_link) or zip_link.strip() == ""):
        oceancolor.append(product)

file_end_year = open("/tmp/end_year_" + id + ".json", "w")
file_end_year.write(json.dumps(end_year))
file_end_year.close()
file_interval = open("/tmp/interval_" + id + ".json", "w")
file_interval.write(json.dumps(interval))
file_interval.close()
file_ocean_productivity = open("/tmp/ocean_productivity_" + id + ".json", "w")
file_ocean_productivity.write(json.dumps(ocean_productivity))
file_ocean_productivity.close()
file_oceancolor = open("/tmp/oceancolor_" + id + ".json", "w")
file_oceancolor.write(json.dumps(oceancolor))
file_oceancolor.close()
file_resolution = open("/tmp/resolution_" + id + ".json", "w")
file_resolution.write(json.dumps(resolution))
file_resolution.close()
file_start_year = open("/tmp/start_year_" + id + ".json", "w")
file_start_year.write(json.dumps(start_year))
file_start_year.close()
file_zip_files = open("/tmp/zip_files_" + id + ".json", "w")
file_zip_files.write(json.dumps(zip_files))
file_zip_files.close()
