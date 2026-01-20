import os
import pandas as pd
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--end_year', action='store', type=str, required=True, dest='end_year')

arg_parser.add_argument('--start_year', action='store', type=int, required=True, dest='start_year')

arg_parser.add_argument('--stats_output_path', action='store', type=str, required=True, dest='stats_output_path')


args = arg_parser.parse_args()
print(args)

id = args.id

end_year = args.end_year.replace('"','')
start_year = args.start_year
stats_output_path = args.stats_output_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/work/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF2/work/' + 'input'

output_dir = conf_output_path
input_folder = stats_output_path #os.path.join(work_path, "output", "Time Series Statistics", "corrected")
output_folder = os.path.join(output_dir, "Spline TIFF")
os.makedirs(output_folder, exist_ok=True)

start_date = str(start_year) + '-01-01'
end_date = str(end_year) + '-12-31'
colname = "mean"
smoothing_value = None  # Smoothing spline (None = automatic; increase for more smoothing)

config_file = os.path.join(conf_input_path, "config_spline.csv")

if os.path.exists(config_file):
    config_df = pd.read_csv(config_file, sep=";")
    config_dict = config_df.set_index("variable").to_dict(orient="index")
    print(f"{config_file} loaded successfully.")
else:
    print(f"Warning: config file {config_file} does not exist.")
    config_dict = {}  # opzionale, per evitare errori successivi

print("config_dict: ", config_dict)

pd_freq = "8D"  # "M", "D", "W", "8D", "15D", ecc.
freq_label_map = {
    "MS": "Monthly",
    "D": "Daily",
    "W": "Weekly",
    "8D": "8 days",
    "15D": "15 days"
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

def is_number(value):
    try:
        float(value)  # try to convert to float
        return True
    except (ValueError, TypeError):
        return False

def calcolaSpline(variable, file_path, output_tiff, colname, start_date, dates, smooth_factor):
    data = pd.read_csv(file_path)
    if colname in data.columns:
        date_range = pd.to_datetime(dates)
        end_date = date_range[-1]
        start_year = start_date[:4]
        end_year = str(end_date.year)
        print(f"Process spline for: {os.path.basename(file_path)} | column: {colname} | range: {start_year} - {end_year} | freq: {pd_freq} ({freq_label})")
        series = pd.Series(data[colname].values, index=date_range)
        time_numeric = np.arange(len(series))

        smoothing_value = None
        
        if is_number(smooth_factor):
            values = series.values
            diff = np.diff(values)
            sigma2 = np.var(diff) / 2
            smoothing_value = smooth_factor * sigma2 * len(values)
        
        print("calculated smoothing_value: " + variable, smoothing_value)

        spline_fit = UnivariateSpline(time_numeric, series.values, s=smoothing_value)
        smoothed_values = spline_fit(time_numeric)
        
        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series.values, label='Original Data', color='blue')
        plt.plot(series.index, smoothed_values, label='Smoothing Spline', color='red', linewidth=2)
        plt.title(f"Time Series {variable} ({start_year}–{end_year}) - {colname} - {freq_label}")
        plt.xlabel("Date")
        plt.ylabel(colname)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_tiff, format='tiff')
        plt.close()
        print(f"  Chart saved in: {output_tiff}")

        df_out = pd.DataFrame({
            'date': series.index,
            'original': series.values,
            'spline': smoothed_values
        })
        csv_out = output_tiff.replace('.tiff', '_spline.csv')
        df_out.to_csv(csv_out, index=False)
        print(f"  Spline values saved in: {csv_out}")
    else:
        print(f"WARNING: The column '{colname}' does not exist in {file_path}")

for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        variable = base_name.replace("all_statistics_", "")
        output_tiff = os.path.join(output_folder, f"{base_name}_spline.tiff")

        smooth_factor = None
        
        if variable in config_dict:
            smooth_factor = config_dict[variable].get("smooth_factor", smooth_factor)

        print(f"Processing {file_name} with parameters: "
              f"smooth_factor={smooth_factor}")
        
        calcolaSpline(variable, file_path, output_tiff, colname, start_date, dates, smooth_factor)

print("Operation completed!")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
