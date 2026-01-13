import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filtered_input', action='store', type=str, required=True, dest='filtered_input')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

filtered_input = args.filtered_input.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')



def columns_with_values(csv_path, exclude_columns=None):
    """
    Reads a CSV file with a ';' separator using pandas and returns
    a list of column names containing at least one non-blank value,
    excluding any columns specified in exclude_columns.
    
    Args:
        csv_path (str): path to the CSV file
        exclude_columns (list, optional): list of column names to exclude from the check
    
    Returns:
        list: names of columns with at least one non-blank value, excluding those in exclude_columns
    """
    if exclude_columns is None:
        exclude_columns = []

    df = pd.read_csv(csv_path, sep=';', dtype=str)  # dtype=str to not convert NaN

    colonne_da_controllare = [col for col in df.columns if col not in exclude_columns]

    colonne_valide = [
        col for col in colonne_da_controllare
        if df[col].notna().any() and (df[col].astype(str).str.strip() != '').any()
    ]

    return colonne_valide


param_path = parameters_csv
input_file = filtered_input


df = pd.read_csv(input_file, sep=';')

df.columns = df.columns.str.strip()

required_cols = columns_with_values(param_path, ['Settings']) #['PNN_MEAN', 'CHL-a_MEAN']

print("Required cols: ", required_cols)

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df_clean = df.dropna(subset=required_cols)

df_clean.to_csv(input_file, sep=';', index=False)

print(f"File updated: {input_file}")

file_input_file = open("/tmp/input_file_" + id + ".json", "w")
file_input_file.write(json.dumps(input_file))
file_input_file.close()
