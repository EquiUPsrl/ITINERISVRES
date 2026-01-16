import pandas as pd
from scipy.spatial.distance import squareform
import os
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--end_year', action='store', type=str, required=True, dest='end_year')

arg_parser.add_argument('--final_stats_path', action='store', type=str, required=True, dest='final_stats_path')

arg_parser.add_argument('--interval', action='store', type=str, required=True, dest='interval')

arg_parser.add_argument('--start_year', action='store', type=int, required=True, dest='start_year')


args = arg_parser.parse_args()
print(args)

id = args.id

end_year = args.end_year.replace('"','')
final_stats_path = args.final_stats_path.replace('"','')
interval = args.interval.replace('"','')
start_year = args.start_year


conf_output_path = conf_output_path = '/tmp/data/WF1_1/work/' + 'output'

output_dir = conf_output_path
input_dir = final_stats_path #os.path.join(output_dir, "Time Series Statistics", "corrected")
output_dir_base = os.path.join(output_dir, "Recurrence_Plot")

col_name = "mean"
start_date = str(start_year) + '-01-01'
end_date = str(end_year) + '-12-31'
pd_freq = interval #"MS", "M", "D", "W", "8D", "15D", ecc.

freq_label_map = {
    "MS": "Monthly",
    "M": "Monthly",
    "D": "Daily",
    "W": "Weekly",
    "8D": "Every 8 days",
    "15D": "Biweekly"
}
freq_label = freq_label_map.get(pd_freq, pd_freq)

m_map = {
    "MS": 12,
    "D": 365,
    "W": 52,
    "8D": 46,
    "15D": 24
}
obs_per_year = m_map.get(pd_freq, 12)

dates = []

date_start = pd.to_datetime(start_date)
date_end = pd.to_datetime(end_date)

start_year = date_start.year
end_year = date_end.year

for y in range(start_year, end_year + 1):
    start = pd.Timestamp(f"{y}-01-01")
    end = pd.Timestamp(f"{y}-12-31")
    dr = pd.date_range(start=start, end=end, periods=obs_per_year)
    dates.extend(dr)

embedding_dim = 5
time_delay = 1
recurrence_rate = 0.1

def recurrencePlotFRR(
    variable, col, start_date, dates, embedding_dim, time_delay,
    recurrence_rate, file_path, output_folder, freq_label
):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{variable}_recurrence_plot.png")
    
    data = pd.read_csv(file_path, sep=",")
    if col not in data.columns:
        print(f"Column '{col}' not found in {file_path}")
        return
    time_series = data[col].values.astype(float)
    

    date_range = pd.to_datetime(dates)
    
    if len(time_series) != len(date_range):
        raise ValueError(f"Series length ({len(time_series)}) does not match number of dates ({len(date_range)}).")
    
    def takens_embedding(ts, dim, tau):
        n_vectors = len(ts) - (dim - 1) * tau
        if n_vectors <= 0:
            raise ValueError("Time series too short for the specified embedding!")
        return np.array([ts[i:i + dim * tau:tau] for i in range(n_vectors)])
    
    embedded = takens_embedding(time_series, embedding_dim, time_delay)
    
    distance_matrix = squareform(pdist(embedded, metric='euclidean'))
    sorted_distances = np.sort(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
    threshold_index = int(recurrence_rate * len(sorted_distances))
    radius = sorted_distances[threshold_index]
    rp_matrix = (distance_matrix <= radius).astype(int)
    
    dates_embedded = date_range[(embedding_dim - 1) * time_delay:]
    if len(dates_embedded) != len(embedded):
        raise ValueError("Mismatch between embedded vectors and aligned dates!")
    
    years_unique = sorted(dates_embedded.year) 
    years_unique = sorted(set(years_unique))
    
    tick_idx = [dates_embedded.get_loc(dates_embedded[dates_embedded.year == y][0])
                for y in years_unique]    
    tick_labels = [str(y) for y in years_unique]

    
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.imshow(rp_matrix, cmap="Greys", interpolation="nearest", origin='lower', rasterized=False)
    
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels, fontsize=9)
    
    ax.set_title(f"Recurrence Plot ({freq_label}, RR {int(recurrence_rate*100)}%)", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Time", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Recurrence plot saved in: {output_folder}")

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        variable = os.path.splitext(filename)[0]
        output_folder = os.path.join(output_dir_base, variable)
        print(f"Processing {filename} ...")
        try:
            recurrencePlotFRR(
                variable, col_name, start_date, dates, embedding_dim, time_delay,
                recurrence_rate, file_path, output_folder, freq_label
            )
        except Exception as e:
            print(f"Error with {filename}: {e}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
