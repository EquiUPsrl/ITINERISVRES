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

input_folder = final_stats_path #os.path.join(conf_output_path, "Time Series Statistics", "corrected")
output_dir = conf_output_path
output_folder = os.path.join(output_dir, "Spline TIFF")
os.makedirs(output_folder, exist_ok=True)

start_date = str(start_year) + '-01-01'
end_date = str(end_year) + '-12-31'
colname = "mean"
smoothing_value = None

pd_freq = interval #"8D"  # "M", "D", "W", "8D", "15D", ecc.
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

def calcolaSpline(variable, file_path, output_tiff, colname, start_date, dates, smoothing_value):
    data = pd.read_csv(file_path)
    if colname in data.columns:
        
        date_range = pd.to_datetime(dates)
        
        end_date = date_range[-1]
        start_year = start_date[:4]
        end_year = str(end_date.year)
        print(f"Process splines for: {os.path.basename(file_path)} | Column: {colname} | Range: {start_year} - {end_year} | freq: {pd_freq} ({freq_label})")
        series = pd.Series(data[colname].values, index=date_range)
        time_numeric = np.arange(len(series))
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
        print(f"Chart saved in: {output_tiff}")

        df_out = pd.DataFrame({
            'date': series.index,
            'original': series.values,
            'spline': smoothed_values
        })
        csv_out = output_tiff.replace('.tiff', '_spline.csv')
        df_out.to_csv(csv_out, index=False)
        print(f"Spline values saved in: {csv_out}")
    else:
        print(f"WARNING: The column '{colname}' does not exist in {file_path}")

for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        output_tiff = os.path.join(output_folder, f"{base_name}_spline.tiff")
        variable = base_name
        calcolaSpline(variable, file_path, output_tiff, colname, start_date, dates, smoothing_value)

print("Operation COMPLETED!")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
