import os
import pandas as pd
import pymannkendall as mk

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_stats_path', action='store', type=str, required=True, dest='final_stats_path')


args = arg_parser.parse_args()
print(args)

id = args.id

final_stats_path = args.final_stats_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1_1/work/' + 'output'

input_folder = final_stats_path #os.path.join(conf_output_path, "Time Series Statistics", "corrected")
output_dir = conf_output_path
output_folder = os.path.join(output_dir, "Analysis of the time series trend")

os.makedirs(output_folder, exist_ok=True)

colname = "mean"  # Change if you want to use another column


for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        print(f"\nProcessing: {file_name}")

        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue

        if colname in data.columns:
            if 'date' in data.columns:
                timeSeries = pd.Series(data[colname].values, index=pd.to_datetime(data['date']))
            else:
                timeSeries = pd.Series(data[colname].values)

            result = mk.original_test(timeSeries)
            tau = result.Tau
            p_value = result.p

            if p_value < 0.05:
                if tau > 0:
                    interpretation = "A statistically significant increasing trend is detected."
                elif tau < 0:
                    interpretation = "A statistically significant decreasing trend is detected."
                else:
                    interpretation = "No trend detected (tau ≈ 0)."
            else:
                if tau > 0:
                    interpretation = "A non-significant increasing trend is detected (not statistically significant)."
                elif tau < 0:
                    interpretation = "A non-significant decreasing trend is detected (not statistically significant)."
                else:
                    interpretation = "No trend detected (tau ≈ 0)."

            base_name = os.path.splitext(file_name)[0]
            result_file = os.path.join(output_folder, f"{base_name}_mann_kendall.txt")

            with open(result_file, "w") as f:
                f.write(f"tau: {tau}\n")
                f.write(f"p-value: {p_value}\n")
                f.write(f"Interpretation: {interpretation}\n")

            print(f"  ➜ Result saved to: {result_file}")
            print(f"  Interpretation: {interpretation}")
        else:
            print(f"  Column '{colname}' not found in file.")

print("\nOperation completed!")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
