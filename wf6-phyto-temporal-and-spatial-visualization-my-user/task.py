import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_path', action='store', type=str, required=True, dest='input_path')


args = arg_parser.parse_args()
print(args)

id = args.id

input_path = args.input_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)
print(f"Folder '{output_dir}' ready.")

csv_path = input_path
df_raw = pd.read_csv(csv_path, low_memory=False)

print("Original shape:", df_raw.shape)
df_raw.head()

df = df_raw.copy()
df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')

print("Min date:", df['parsed_date'].min())
print("Max date:", df['parsed_date'].max())
print("Lakes (locality):", df['locality'].unique())

num_cols = df.select_dtypes(include=['number']).columns

event_df = (
    df.groupby(['locality', 'parsed_date'], as_index=False)[num_cols]
      .mean()
      .sort_values(['locality', 'parsed_date'])
)

print("event_df shape:", event_df.shape)
event_df.head()

output_file = os.path.join(output_dir, "event_df.csv")
event_df.to_csv(output_file, index=False)
print(f"Saved file: {output_file}")

file_output_file = open("/tmp/output_file_" + id + ".json", "w")
file_output_file.write(json.dumps(output_file))
file_output_file.close()
