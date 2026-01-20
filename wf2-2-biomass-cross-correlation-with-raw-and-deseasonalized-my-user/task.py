import numpy as np
from rasterio.mask import mask
from itertools import combinations
import os
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import STL
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--start_year', action='store', type=int, required=True, dest='start_year')

arg_parser.add_argument('--stats_path', action='store', type=str, required=True, dest='stats_path')


args = arg_parser.parse_args()
print(args)

id = args.id

start_year = args.start_year
stats_path = args.stats_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/work/' + 'output'

"""
Pairwise cross-correlation: RAW + DESEASONALIZED (STL), with explicit lead/lag convention.
- Reads all CSVs from input_folder (each must contain the numeric column `col_name`)
- Builds a synthetic time index from start_date and pd_freq (no dates in the files)
- For each pair (A,B):
    * RAW analysis                      → <output_root>/<A>__<B>/raw/
    * STL residuals (seasonality off)   → <output_root>/<A>__<B>/deseasonalized/
Outputs per analysis (English-only):
    - crosscorr_stem[ _deseasonalized ].png
    - crosscorr_summary[ _deseasonalized ].csv  (with leader/follower fields)
    - lag_correlation_window[ _deseasonalized ].csv
    - aligned_inputs[ _deseasonalized ].csv

Lead/lag convention:
    lag > 0  -> series_1 leads series_2 by `lag` units
    lag < 0  -> series_2 leads series_1 by `abs(lag)` units
"""


input_folder = stats_path #os.path.join("work", "output", "Time Series Statistics")  # Input CSV folder
output_dir = conf_output_path
output_root  = os.path.join(output_dir, "Cross Correlation")       # Output root
col_name     = "mean"            # Numeric column name in each CSV

start_date   = str(start_year) + "-01-01"      # Synthetic timeline start
pd_freq      = "8D" #interval # "MS", "M", "D", "W", "8D", "15D", ...

max_shift_map = {
    "MS": 12,   # months
    "M":  12,
    "W":  52,   # weeks
    "D":  365,  # days
    "8D": 46,   # 8-day periods ≈ 360/8
    "15D": 24,  # 15-day periods ≈ 360/15
}
max_shift = max_shift_map.get(pd_freq, 12)

season_periods_map = {
    "MS": 12,   # yearly seasonality for monthly data
    "M":  12,
    "W":  52,
    "D":  7,    # weekly seasonality for daily data (adjust if needed)
    "8D": 46,
    "15D": 24,
}
season_periods = season_periods_map.get(pd_freq, None)

save_plots = True
save_tables = True
save_aligned_inputs = True

os.makedirs(output_root, exist_ok=True)
print("Working dir:", os.path.abspath(os.getcwd()))
print("Input folder:", os.path.abspath(input_folder))
print("Output root:", os.path.abspath(output_root))
print(f"Frequency: {pd_freq} | season_periods: {season_periods} | max_shift: ±{max_shift}")

def safe_read_series(path, col):
    """Read numeric column from CSV -> numpy array (drops NaNs)."""
    s = pd.read_csv(path, usecols=[col])[col]
    s = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    return s

def build_dates(n, start, freq):
    return pd.date_range(start=start, periods=n, freq=freq)

def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return pearsonr(x, y)

def cross_corr_norm(x: np.ndarray, y: np.ndarray):
    """Normalized cross-correlation (~[-1,1]) vs integer lags."""
    x0 = x - x.mean()
    y0 = y - y.mean()
    n = len(x0)
    denom = np.std(x0) * np.std(y0) * n
    if denom == 0 or n < 2:
        return np.array([0]), np.array([np.nan])
    cc = np.correlate(x0, y0, mode="full") / denom
    lags = np.arange(-n + 1, n)
    return lags, cc

def deseasonalize(values: np.ndarray, period: int | None) -> np.ndarray:
    """
    Remove seasonality using STL (robust). Returns residuals.
    Falls back to original values if period is None or series is too short.
    """
    if period is None or len(values) < max(2*period, 24):
        return values.astype(float)
    stl = STL(values, period=period, robust=True)
    res = stl.fit()
    return res.resid.astype(float)

def sanitize_name(path):
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_")

def unit_label(freq: str) -> str:
    return "months" if freq in ("MS", "M") else "periods"

def lead_lag_from_lag(series1: str, series2: str, lag: int, unit: str):
    """Return leader, follower, lead_by (int), and a human interpretation string."""
    if lag > 0:
        leader, follower, lead_by = series1, series2, lag
        interp = f"{series1} leads {series2} by {lag} {unit}"
    elif lag < 0:
        leader, follower, lead_by = series2, series1, abs(lag)
        interp = f"{series2} leads {series1} by {abs(lag)} {unit}"
    else:
        leader, follower, lead_by = "none", "none", 0
        interp = "No lead/lag (best lag is 0)"
    return leader, follower, lead_by, interp

def plot_stem_with_annotations(lags_win, cc_win, best_lag_no0, xlabel_unit, title, save_path):
    """Stem plot with baseline=0, Lag0 (red), best lag ≠ 0 (green)."""
    plt.figure(figsize=(10, 5))
    plt.stem(lags_win, cc_win, basefmt=" ")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, color='red', linestyle="--", label="Lag 0")
    if best_lag_no0 != 0 and np.isfinite(best_lag_no0):
        plt.axvline(best_lag_no0, color='green', linestyle="--",
                    label=f"Best no-0 lag: {best_lag_no0:+d} {xlabel_unit}")
    plt.title(title)
    plt.xlabel(f"Lag ({xlabel_unit})")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_path, dpi=150)
    plt.close()

def analyze_and_save(df_pair: pd.DataFrame,
                     name1: str, name2: str,
                     out_dir: str,
                     label_suffix: str = "",
                     xlabel_unit: str = "months",
                     max_shift_local: int = 12):
    """
    Compute Pearson, cross-correlation, best lags; save plot + tables for a pair/variant.
    df_pair: dataframe with columns ['date', name1, name2]
    label_suffix: "" for raw, "_deseasonalized" for STL residuals
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"  -> Saving to: {os.path.abspath(out_dir)}")

    x = df_pair[name1].to_numpy()
    y = df_pair[name2].to_numpy()
    n = len(df_pair)
    if n < 2:
        print("  !! Insufficient data after alignment, skipped.")
        return

    r, p = safe_pearson(x, y)

    lags, cc = cross_corr_norm(x, y)

    mask = (lags >= -max_shift_local) & (lags <= max_shift_local)
    lags_win = lags[mask]
    cc_win = cc[mask]

    idx_abs = int(np.nanargmax(cc_win))
    best_lag_abs = int(lags_win[idx_abs])
    best_corr_abs = float(cc_win[idx_abs])

    cc_no0 = cc_win.copy()
    if 0 in lags_win:
        cc_no0[lags_win == 0] = -np.inf
    idx_no0 = int(np.nanargmax(cc_no0))
    best_lag_no0 = int(lags_win[idx_no0])
    best_corr_no0 = float(cc_win[idx_no0])

    leader, follower, lead_by, interp = lead_lag_from_lag(name1, name2, best_lag_no0, xlabel_unit)

    title = (f"Cross-correlation {label_suffix.replace('_', ' ').strip()} "
             f"(±{max_shift_local} {xlabel_unit}): {name1} vs {name2}")
    fname = "crosscorr_stem.png" if not label_suffix else f"crosscorr_stem{label_suffix}.png"
    plot_stem_with_annotations(
        lags_win, cc_win, best_lag_no0, xlabel_unit, title,
        save_path=os.path.join(out_dir, fname)
    )

    if save_tables:
        summary = pd.DataFrame([{
            "series_1": name1,
            "series_2": name2,
            ("pearson_r" + label_suffix): r,
            ("p_value" + label_suffix): p,
            ("best_lag_abs" + label_suffix): best_lag_abs,
            ("best_corr_abs" + label_suffix): best_corr_abs,
            ("best_lag_no0" + label_suffix): best_lag_no0,
            ("best_corr_no0" + label_suffix): best_corr_no0,
            "lag_unit": xlabel_unit,
            "leader": leader,
            "follower": follower,
            (f"lead_by_{xlabel_unit}" + label_suffix): lead_by,
            ("interpretation" + label_suffix): interp,
            "start_date": start_date,
            "freq": pd_freq,
            "season_periods": season_periods,
            "max_shift_window": max_shift_local,
            "n_points": n
        }])
        summ_name = "crosscorr_summary.csv" if not label_suffix else f"crosscorr_summary{label_suffix}.csv"
        summary.to_csv(os.path.join(out_dir, summ_name), index=False)

        lag_corr_df = (pd.DataFrame({"lag": lags_win, "correlation": cc_win})
                       .sort_values("correlation", ascending=False)
                       .reset_index(drop=True))
        lag_tbl_name = "lag_correlation_window.csv" if not label_suffix else f"lag_correlation_window{label_suffix}.csv"
        lag_corr_df.to_csv(os.path.join(out_dir, lag_tbl_name), index=False)

    if save_aligned_inputs:
        aligned_name = "aligned_inputs.csv" if not label_suffix else f"aligned_inputs{label_suffix}.csv"
        df_pair.to_csv(os.path.join(out_dir, aligned_name), index=False)

csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
if len(csv_files) < 2:
    raise RuntimeError(f"At least two CSV files are required in: {os.path.abspath(input_folder)}")
print(f"Found {len(csv_files)} CSV files.")

xlabel_unit = unit_label(pd_freq)

for f1, f2 in combinations(csv_files, 2):
    name1 = sanitize_name(f1)   # this is "series_1" in outputs
    name2 = sanitize_name(f2)   # this is "series_2" in outputs

    pair_root = os.path.join(output_root, f"{name1}__{name2}")
    os.makedirs(pair_root, exist_ok=True)
    print(f"\nPair: {name1} vs {name2}")
    print("Pair folder:", os.path.abspath(pair_root))

    s1 = safe_read_series(f1, col_name)
    s2 = safe_read_series(f2, col_name)
    n = min(len(s1), len(s2))
    if n < 2:
        print("  !! Insufficient data (n<2), skipped.")
        continue
    s1 = s1[:n]
    s2 = s2[:n]
    dates = build_dates(n, start_date, pd_freq)

    out_raw = os.path.join(pair_root, "raw")
    df_raw = pd.DataFrame({"date": dates, name1: s1, name2: s2}).dropna().reset_index(drop=True)
    if len(df_raw) >= 2:
        analyze_and_save(df_raw, name1, name2, out_raw,
                         label_suffix="", xlabel_unit=xlabel_unit, max_shift_local=max_shift)
        print("  ✓ RAW analysis done.")
    else:
        print("  !! RAW: insufficient data after dropna, skipped.")

    out_ds = os.path.join(pair_root, "deseasonalized")
    s1_ds = deseasonalize(s1, season_periods)
    s2_ds = deseasonalize(s2, season_periods)
    df_ds = pd.DataFrame({"date": dates,
                          f"{name1}": s1_ds,  # keep same column names for consistency
                          f"{name2}": s2_ds}).dropna().reset_index(drop=True)
    if len(df_ds) >= 2:
        analyze_and_save(df_ds, name1, name2, out_ds,
                         label_suffix="_deseasonalized", xlabel_unit=xlabel_unit, max_shift_local=max_shift)
        print("  ✓ DESEASONALIZED analysis done.")
    else:
        print("  !! DESEASONALIZED: insufficient data after dropna, skipped.")

print("\nAll results saved under:", os.path.abspath(output_root))

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
