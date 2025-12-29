import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--merged_file', action='store', type=str, required=True, dest='merged_file')


args = arg_parser.parse_args()
print(args)

id = args.id

merged_file = args.merged_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)
print(f"Folder '{output_dir}' ready.")

csv_path = merged_file
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

final_input = os.path.join(output_dir, "final_input.csv")
event_df.to_csv(final_input, index=False)
print(f"Saved file: {final_input}")

file_final_input = open("/tmp/final_input_" + id + ".json", "w")
file_final_input.write(json.dumps(final_input))
file_final_input.close()
