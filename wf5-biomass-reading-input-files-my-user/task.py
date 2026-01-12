import warnings
import os
import pandas as pd
import csv
import json

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--bounding_box_filter_csv', action='store', type=str, required=True, dest='bounding_box_filter_csv')

arg_parser.add_argument('--input_file_1_csv', action='store', type=str, required=True, dest='input_file_1_csv')

arg_parser.add_argument('--input_file_2_csv', action='store', type=str, required=True, dest='input_file_2_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

bounding_box_filter_csv = args.bounding_box_filter_csv.replace('"','')
input_file_1_csv = args.input_file_1_csv.replace('"','')
input_file_2_csv = args.input_file_2_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

warnings.filterwarnings("ignore")

input_file_1 = input_file_1_csv
input_file_2 = input_file_2_csv
input_bbox = bounding_box_filter_csv


def filter_csv_by_bbox(csv_path, bbox):
    df = pd.read_csv(csv_path, delimiter=";")

    for col in ['LATITUDE', 'LONGITUDE']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

    filtered_df = df[
        (df['LATITUDE'] >= bbox['min_lat']) &
        (df['LATITUDE'] <= bbox['max_lat']) &
        (df['LONGITUDE'] >= bbox['min_lon']) &
        (df['LONGITUDE'] <= bbox['max_lon'])
    ]

    return filtered_df


def csv_to_json(csv_path, json_path=None):
    data = {}

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if len(row) != 2 or not row[1].strip():
                return {}  # Returns an empty JSON object if a value is missing

            key = row[0].strip().strip('"')
            try:
                value = float(row[1])
            except ValueError:
                value = row[1].strip()

            data[key] = value

    if json_path and data:
        with open(json_path, "w", encoding="utf-8") as jsonfile:
            json.dump(data, jsonfile, indent=4)

    return data





output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

if len(input_file_1) == 0 and len(input_file_2) == 0 :
    raise ValueError("At least one file is required")

if len(input_file_1) > 0 and len(input_file_2) > 0 :
    print("2 existing files")
    df1 = pd.read_csv(input_file_1, delimiter=";")
    if len(input_bbox) > 0 :
        print("Existing bounding box: apply filter to file 1")
        bbox = csv_to_json(input_bbox)
        df1 = filter_csv_by_bbox(input_file_1, bbox)
    df2 = pd.read_csv(input_file_2, delimiter=";")
    final_df = pd.concat([df2, df1], ignore_index=True)

if len(input_file_1) > 0 and len(input_file_2) == 0 :
    print("Only 1 existing file")
    final_df = pd.read_csv(input_file_1, delimiter=";")
    if len(input_bbox) > 0 :
        print("Existing bounding box: apply filter to file 1")
        bbox = csv_to_json(input_bbox)
        final_df = filter_csv_by_bbox(input_file_1, bbox)

final_input = os.path.join(output_dir, 'input.csv')
final_df.to_csv(final_input, index=False, sep=';')

print(f"Input file saved in: {final_input}")

file_final_input = open("/tmp/final_input_" + id + ".json", "w")
file_final_input.write(json.dumps(final_input))
file_final_input.close()
