import warnings
import os
import re
import pandas as pd
from scipy.stats import kendalltau
import numpy as np
from sklearn.linear_model import TheilSenRegressor

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--stats_output_path', action='store', type=str, required=True, dest='stats_output_path')


args = arg_parser.parse_args()
print(args)

id = args.id

stats_output_path = args.stats_output_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/work/' + 'output'

try:
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

warnings.filterwarnings("ignore")

output_dir = conf_output_path
input_folder = stats_output_path #os.path.join(work_path, "output", "Time Series Statistics", "corrected")
output_folder = os.path.join(output_dir, "Analysis of the time series trend")

os.makedirs(output_folder, exist_ok=True)

colname = "mean"  # Change if you want to use another column

def safe_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_label = safe_label(os.path.splitext(file_name)[0])
        ts_name = f"{file_label}_{colname}"
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

                try:
                    tau, pval = kendalltau(np.arange(len(timeSeries)), timeSeries.values)
                    slope_info = ""
                    if _HAS_SKLEARN:
                        X = np.arange(len(timeSeries)).reshape(-1, 1)
                        tsr = TheilSenRegressor().fit(X, timeSeries.values)
                        slope_info = f"Slope (Theil–Sen): {float(tsr.coef_[0]):.6g}\n"
                    interp = ("Significant increasing trend" if (pval < 0.05 and tau > 0)
                              else "Significant decreasing trend" if (pval < 0.05 and tau < 0)
                              else "No significant monotonic trend (p >= 0.05)")
                    with open(os.path.join(output_folder, f"Kendall_{ts_name}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Kendall's tau: {tau:.4f}\n")
                        f.write(f"p-value: {pval:.4g}\n")
                        if slope_info:
                            f.write(slope_info)
                        f.write(f"Interpretation: {interp}\n")
                    print(f"    Kendall: tau={tau:.3f}, p={pval:.3g} -> {interp}")
                except Exception as e:
                    print(f"WARNING: Kendall test failed: {e}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
