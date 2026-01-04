import os
import pandas as pd
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--bio_file', action='store', type=str, required=True, dest='bio_file')


args = arg_parser.parse_args()
print(args)

id = args.id

bio_file = args.bio_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

output_file_name = "bio_filtered.csv"

apply_abundance_filter = False       # Apply total abundance/biomass filter
abundance_percentile = 10.0          # Minimum percentile for cumulative abundance

apply_frequency_filter = True        # Apply presence frequency filter
min_frequency_ratio = 0.10           # Minimum presence (% of samples) to keep species


df = pd.read_csv(bio_file, header=0, index_col=0, encoding="utf-8")


if apply_abundance_filter:
    col_sums = df.sum(axis=0)
    p_thr = np.percentile(col_sums.to_numpy(), abundance_percentile)
    mask_sum = col_sums >= p_thr
else:
    mask_sum = pd.Series(True, index=df.columns)


if apply_frequency_filter:
    n_rows = df.shape[0]
    positive_counts = (df > 0).sum(axis=0)
    ratio = np.round(positive_counts / n_rows, 3)
    mask_ratio = ratio >= min_frequency_ratio
else:
    mask_ratio = pd.Series(True, index=df.columns)


keep_mask = mask_sum & mask_ratio
kept_cols = df.columns[keep_mask]
filtered = df[kept_cols]


n_total = df.shape[1]
n_kept = filtered.shape[1]
n_removed = n_total - n_kept

print(f"Species total: {n_total}")
print(f"Species kept: {n_kept}")
print(f"Species removed: {n_removed}")


bio_file_filtered = os.path.join(output_dir, output_file_name)
filtered.to_csv(bio_file_filtered, encoding="utf-8")

print("Biotic pivot table filtered saved to: " + bio_file_filtered)

file_bio_file_filtered = open("/tmp/bio_file_filtered_" + id + ".json", "w")
file_bio_file_filtered.write(json.dumps(bio_file_filtered))
file_bio_file_filtered.close()
