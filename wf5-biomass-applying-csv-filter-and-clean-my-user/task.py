import pandas as pd
import csv

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filter_csv', action='store', type=str, required=True, dest='filter_csv')

arg_parser.add_argument('--input_file', action='store', type=str, required=True, dest='input_file')


args = arg_parser.parse_args()
print(args)

id = args.id

filter_csv = args.filter_csv.replace('"','')
input_file = args.input_file.replace('"','')



def filtra_csv(csv_path, filtro_dict):
    df = pd.read_csv(csv_path, delimiter=';')
    
    for col, regole in filtro_dict.items():
        if 'valore' in regole:
            df = df[df[col] == regole['valore']]
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
                result[key] = {'valore': value_field}
            else:
                entry = {}
                if row['min'].strip():
                    entry['min'] = float(row['min'])
                if row['max'].strip():
                    entry['max'] = float(row['max'])
                if entry:  # aggiunge solo se c'è almeno min o max
                    result[key] = entry

    return result


filtro = csv_to_json_filter(filter_csv)

filtro['depth'] = {'min': -999999999}

print(f"Filter to apply: {filtro}")

if filtro :
    df = filtra_csv(input_file, filtro)
    df.to_csv(input_file, index=False, sep=';')
    print(f"Filtered file saved in: {input_file}")
else :
    print("No filters applied")

final_input = input_file

file_final_input = open("/tmp/final_input_" + id + ".json", "w")
file_final_input.write(json.dumps(final_input))
file_final_input.close()
