import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--download_dir', action='store', type=str, required=True, dest='download_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

download_dir = args.download_dir.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF4/' + 'output'

input_folder = download_dir       # cartella contenente i CSV
final_input = os.path.join(conf_output_path, 'merged.csv')  # percorso e nome del file di output

os.makedirs(os.path.dirname(final_input), exist_ok=True)

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
print(f"Trovati {len(csv_files)} file CSV:", csv_files)

df_list = []

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=';', encoding='latin1', low_memory=False)  # fallback se UTF-8 fallisce
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True, sort=False)
merged_df.to_csv(final_input, index=False, sep=';', encoding='utf-8')

print(f"File unito salvato in: {final_input}")

file_final_input = open("/tmp/final_input_" + id + ".json", "w")
file_final_input.write(json.dumps(final_input))
file_final_input.close()
