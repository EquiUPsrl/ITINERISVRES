import sys
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_file_csv', action='store', type=str, required=True, dest='data_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

data_file_csv = args.data_file_csv.replace('"','')



def verifica_separatore_punto_e_virgola(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            if header.count(';') < 1:
                print(f"❌ The file '{file_path}' does NOT use the ';' separator.")
                return False
            if '\t' in header:
                print(f"❌ The file '{file_path}' appears to use TAB instead of ';'")
                return False
        return True
    except Exception as e:
        print(f"❌ Error reading '{file_path}': {e}")
        return False


data_csv = data_file_csv


if not verifica_separatore_punto_e_virgola(data_csv):
    sys.exit("🛑 Script aborted: One or more files do not have a ';' separator..")

print("✅ All files use the ';' separator..\n")



print("\n---")
print(f"📄 File: {data_csv}")
print("---")

try:
    df = pd.read_csv(data_csv, sep=';')
    print(f"Read with {df.shape[0]} rows and {df.shape[1]} columns.")

    original_rows = df.shape[0]

    df = df.dropna()

    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_clean = df[df_numeric.notna().all(axis=1)]

    removed_rows = original_rows - df_clean.shape[0]
    print(f"Deleted rows: {removed_rows}")
    print(f"Remaining rows: {df_clean.shape[0]}")

    df_clean.to_csv(data_csv, sep=';', index=False)
    print("✅ File overwritten with clean data.")

except Exception as e:
    print(f"❌ Error while cleaning '{data_csv}': {e}")

file_data_csv = open("/tmp/data_csv_" + id + ".json", "w")
file_data_csv.write(json.dumps(data_csv))
file_data_csv.close()
