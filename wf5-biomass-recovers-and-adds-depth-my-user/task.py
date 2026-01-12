import os
import pandas as pd
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_with_distance', action='store', type=str, required=True, dest='input_with_distance')

arg_parser.add_argument('--raster_depth', action='store', type=str, required=True, dest='raster_depth')


args = arg_parser.parse_args()
print(args)

id = args.id

input_with_distance = args.input_with_distance.replace('"','')
raster_depth = args.raster_depth.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

def add_depth_inplace(csv_file, raster_file, output_csv):
    df = pd.read_csv(csv_file, delimiter=';')

    if not {'LATITUDE', 'LONGITUDE'}.issubset(df.columns):
        raise ValueError("The CSV file must contain the columns 'LATITUDE' and 'LONGITUDE'")

    coords = [(x, y) for x, y in zip(df['LONGITUDE'], df['LATITUDE'])]

    with rasterio.open(raster_file) as src:
        values = list(src.sample(coords))

    depths = [v[0] if v.size > 0 else None for v in values]

    df['depth'] = depths

    df.to_csv(output_csv, index=False, sep=';')

    print(f"Added 'depth' column and create file: {output_csv}")



output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

final_input = os.path.join(output_dir, 'final_input.csv')

add_depth_inplace(
    csv_file=input_with_distance,
    raster_file=raster_depth,
    output_csv=final_input
)

file_final_input = open("/tmp/final_input_" + id + ".json", "w")
file_final_input.write(json.dumps(final_input))
file_final_input.close()
