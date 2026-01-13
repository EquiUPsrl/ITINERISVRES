import os
import pandas as pd
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filter_csv', action='store', type=str, required=True, dest='filter_csv')

arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')


args = arg_parser.parse_args()
print(args)

id = args.id

filter_csv = args.filter_csv.replace('"','')
final_input = args.final_input.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

def apply_csv_filter(csv_path, filtro_dict):
    df = pd.read_csv(csv_path, delimiter=';')
    
    for col, regole in filtro_dict.items():
        if 'value' in regole:
            df = df[df[col] == regole['value']]
        else:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            df = df.dropna(subset=[col])

            print(regole)
            
            if 'min' in regole:
                df = df[df[col] >= regole['min']]
            if 'max' in regole:
                df = df[df[col] <= regole['max']]

    return df


def csv_to_json_filter(csv_path):
    """
    Reads a CSV with the header key;min;max;value and returns a JSON.

    - If 'value' is a value, it returns only {'value':value}.
    - Otherwise, it returns only the fields between min and max.
    """
    result = {}

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            key = row['key']
            value_field = row['value'].strip()
            
            if value_field:
                result[key] = {'value': value_field}
            else:
                entry = {}
                if row['min'].strip():
                    entry['min'] = float(row['min'])
                if row['max'].strip():
                    entry['max'] = float(row['max'])
                if entry:  # adds only if there is at least min or max
                    result[key] = entry

    return result


filtro = csv_to_json_filter(filter_csv)

print(f"Filter to apply: {filtro}")

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

filtered_input = final_input

if filtro :
    filtered_input = os.path.join(output_dir, 'filtered_input.csv')
    df = apply_csv_filter(final_input, filtro)
    df.to_csv(filtered_input, index=False, sep=';')
    print(f"Filtered file saved in: {filtered_input}")
else :
    print("No filters applied")
    

file_filtered_input = open("/tmp/filtered_input_" + id + ".json", "w")
file_filtered_input.write(json.dumps(filtered_input))
file_filtered_input.close()
