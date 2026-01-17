import pandas as pd
import matplotlib
import warnings
import os
from statsmodels.stats.diagnostic import acorr_ljungbox
import re
from pmdarima import auto_arima
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from glob import glob
from sklearn.linear_model import TheilSenRegressor
import json

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--end_year', action='store', type=int, required=True, dest='end_year')

arg_parser.add_argument('--interval', action='store', type=str, required=True, dest='interval')

arg_parser.add_argument('--start_year', action='store', type=int, required=True, dest='start_year')

arg_parser.add_argument('--stats_path', action='store', type=str, required=True, dest='stats_path')


args = arg_parser.parse_args()
print(args)

id = args.id

end_year = args.end_year
interval = args.interval.replace('"','')
start_year = args.start_year
stats_path = args.stats_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1_2/work/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF1_2/work/' + 'input'

"""
SARIMA fast on 8D series (CHL) with automatic fallback and speed-up options.
"""


matplotlib.use("Agg")


try:
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

warnings.filterwarnings("ignore")

input_base_folder = stats_path #os.path.join(work_path, "output", "Time Series Statistics", "corrected")
output_dir = conf_output_path
output_dir_base = os.path.join(output_dir, "SARIMA")
os.makedirs(output_dir_base, exist_ok=True)

print(input_base_folder)

start_year = int(start_year)
end_year = int(end_year)

start_date = str(start_year) + "-01-01"
end_date = str(end_year) + "-12-31"

col_name = "mean"
pd_freq = interval # 'MS', 'D', 'W', '8D', '15D', '16D'...
alpha = 0.05         # -> 95% CI

forecast_time_in_years = 5

USE_RESAMPLE_16D = True          # True -> use 16D (m=23) to speed up
ROLLING_WINDOW_YEARS = None       # es. 12 -> Use only last 12 years; None to disable
CLIP_UPPER_Q = 0.995              # Soft clip outliers; None to disable

freq_label_map = {"MS": "Monthly","D": "Daily","W": "Weekly","8D": "Every 8 days","15D": "Biweekly","16D": "Every 16 days"}
freq_label = freq_label_map.get(pd_freq, pd_freq)

m_map = {"MS": 12,"D": 365,"W": 52,"8D": 46,"15D": 24,"16D": 23}
m_value = m_map.get("16D" if USE_RESAMPLE_16D else pd_freq, 1)

autoarima_params_fast = {
    "start_p": 0, "max_p": 2,
    "start_q": 0, "max_q": 2,
    "start_P": 0, "max_P": 2,      # thrifty seasonal
    "start_Q": 1, "max_Q": 2,
    "d": None, "D": None, "max_d": 2, "max_D": 1,
    "seasonal": True, "m": m_value,  # <-- use m_value (23 se 16D, 46 se 8D)
    "stepwise": True,                # quick search
    "approximation": True,
    "simple_differencing": False,
    "max_order": 8,
    "suppress_warnings": True,
    "information_criterion": "aicc",
    "seasonal_test": "ch",
    "boxcox": True,
    "error_action": "ignore",
    "with_intercept": True,
    "enforce_stationarity": False,
    "enforce_invertibility": False
}

print(f"*** SARIMA fast settings | freq='{pd_freq}' ({freq_label}) | m={m_value} | col='{col_name}' ***")
print(f"*** auto_arima params: {autoarima_params_fast} ***\n")

def safe_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))

def detect_date_column(df: pd.DataFrame):
    for c in ["Date", "date", "DATA", "data", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def read_any_table(file_path: str) -> pd.DataFrame:
    tried = []
    try:
        return pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        tried.append(f"inferenza: {e}")
    try:
        return pd.read_csv(file_path, sep="\t", engine="python")
    except Exception as e:
        tried.append(f"tab: {e}")
    for enc in (None, "latin-1", "cp1252"):
        try:
            return pd.read_csv(file_path, sep=";", engine="python", encoding=enc)
        except Exception as e:
            tried.append(f"; enc={enc}: {e}")
    raise RuntimeError("Unable to read file. Attempts: " + " | ".join(tried))

def normalize_decimal_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    s = df[column].astype(str).str.strip()
    if s.str.contains(r",\d+$").any():
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    try:
        df[column] = pd.to_numeric(s, errors="coerce")
    except Exception:
        pass
    return df

def ljung_box_dynamic(residuals: pd.Series, n_hist: int) -> pd.DataFrame:
    max_lag = int(min(24, max(5, n_hist // 5)))
    return acorr_ljungbox(residuals, lags=[max_lag], return_df=True)

def maybe_resample(series: pd.Series, target_freq: str | None):
    if not target_freq or target_freq == series.index.freqstr:
        return series
    return series.asfreq(target_freq, method=None).interpolate("linear", limit_direction="both")

def rolling_window(series: pd.Series, years: int | None):
    if years is None:
        return series
    cutoff = series.index.max() - pd.DateOffset(years=years)
    return series[series.index > cutoff]

def fit_with_fallback(ts_series: pd.Series, params: dict):
    try:
        mdl = auto_arima(ts_series, **params)
        return mdl, False
    except Exception as e:
        msg = str(e).lower()
        alloc_err = ("unable to allocate" in msg) or ("memoryerror" in msg) or ("array with shape" in msg)
        fb = params.copy()
        fb.update({"seasonal": False, "m": 1})
        if not alloc_err:
            pass
        mdl = auto_arima(ts_series, **fb)
        return mdl, True

def backtest_mae_mape(train: pd.Series, test: pd.Series, params: dict):
    try:
        mdl = auto_arima(train, **params)
        pred, _ = mdl.predict(n_periods=len(test), return_conf_int=True, alpha=alpha)
        mae = float(np.mean(np.abs(pred - test.values)))
        denom = np.maximum(1e-8, np.abs(test.values))
        mape = float(np.mean(np.abs((test.values - pred) / denom)) * 100.0)
        return mae, mape
    except Exception:
        return np.nan, np.nan

def generate_historical_numeric_index(n_hist):
    """
    Generate a numeric index for historical data:
    0, 1, 2, ..., n_hist-1
    """
    return np.arange(n_hist)


def generate_future_numeric_index(last_hist_idx, n_future):
    """
    Generates a future numeric index starting from last_hist_idx + 1.
    """
    start = last_hist_idx + 1
    return np.arange(start, start + n_future)

def process_file(file_path: str):
    filename = os.path.basename(file_path)
    file_label = safe_label(os.path.splitext(filename)[0])
    ts_name = f"{file_label}_{col_name}"
    print(f"\nProcessing: {filename}")

    data = read_any_table(file_path)

    print("data", data)
    
    if col_name not in data.columns:
        raise RuntimeError(f"Colonna '{col_name}' non trovata in {filename}")
    data = normalize_decimal_column(data, col_name)

    if USE_RESAMPLE_16D:
        data = data.iloc[::2].copy()
    
    series = data[col_name].interpolate(method='linear', limit_direction='both')
    n_hist = len(series)
    idx = generate_historical_numeric_index(n_hist)
    f_idx = generate_future_numeric_index(idx[-1], forecast_periods)
    
    s = pd.Series(series.values, index=idx, name=ts_name)
    s = s.astype(float).dropna()

    print("s", s)





    date_index = pd.date_range(start=start_date, end=end_date + " 23:59:59", periods=n_hist)
    
    print("date_index", date_index)

    years_ahead = forecast_periods / m_value
    
    last_hist_date = date_index[-1]
    future_end_date = last_hist_date + pd.DateOffset(years=years_ahead)
    
    future_index = pd.date_range(
        start=last_hist_date + (date_index[1] - date_index[0]),  # primo passo dopo l'ultimo storico
        end=future_end_date,
        periods=forecast_periods
    )



    


    print("s after maybe_resample", s)

    s = rolling_window(s, ROLLING_WINDOW_YEARS)

    print("s after rolling_window", s)

    if len(s) < 24:
        raise RuntimeError("Series too short (<24 points).")

    out_dir = ensure_dir(os.path.join(output_dir_base, file_label, col_name))
    run_log_path = os.path.join(out_dir, "run_log.txt")
    with open(run_log_path, "a", encoding="utf-8") as lf:
        lf.write("\n=== NEW RUN ===\n")
        lf.write(f"file: {filename}\nlen(ts): {len(s)}\n")
        lf.write(f"params: {json.dumps(autoarima_params_fast)}\n")

    if CLIP_UPPER_Q is not None:
        upper = s.quantile(CLIP_UPPER_Q)
        s = s.clip(upper=upper)

    split = int(len(s) * 0.8)
    train, test = s.iloc[:split], s.iloc[split:]
    mae_bt, mape_bt = backtest_mae_mape(train, test, autoarima_params_fast)

    model, used_fallback = fit_with_fallback(s, autoarima_params_fast)

    try:
        with open(os.path.join(out_dir, f"Summary_SARIMA_{ts_name}.txt"), "w", encoding="utf-8") as f:
            f.write(str(model.summary()))
            f.write("\n\n=== Selected model parameters ===\n")
            f.write(f"order (p,d,q): {model.order}\n")
            try:
                f.write(f"seasonal_order (P,D,Q,m): {model.seasonal_order}\n")
            except Exception:
                f.write("seasonal_order: (no seasonal)\n")
            f.write(f"AIC: {model.aic()}\n")
            try:
                f.write(f"BIC: {model.bic()}\n")
            except Exception:
                pass
            f.write(f"\nBacktest (80/20) -> MAE: {mae_bt:.4f}, MAPE: {mape_bt:.2f}%\n")
            if used_fallback:
                f.write("\n[FALLBACK] Fit performed without seasonality for speed/robustness.\n")
    except Exception as e:
        print(f"WARNING: summary write failed: {e}")

    try:
        resid = model.resid()

        resid.index = date_index
        resid.name = None

        burnin = max(24, m_value * 2)
        resid_plot = resid[burnin:]

        print("resid_plot", resid_plot)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(resid_plot, label="Residuals")
        axes[0].set_title(f"Residuals - {ts_name}")
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(resid_plot, bins=30, alpha=0.7)
        axes[1].set_title(f"Residuals distribution - {ts_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Residuals_{ts_name}.png"), dpi=144, bbox_inches="tight")
        plt.close(fig)

        lb = ljung_box_dynamic(resid_plot, len(s))
        lb.to_csv(os.path.join(out_dir, f"LjungBox_{ts_name}.txt"), sep="\t", index=False)
    except Exception as e:
        print(f"WARNING: residual diagnostics failed: {e}")

    future, conf = model.predict(n_periods=forecast_periods, return_conf_int=True, alpha=alpha)
    last_date = date_index[-1]

    print("future", future)
    
    f_series = pd.Series(future, index=f_idx, name=f"Forecast_{ts_name}")

    

    print("f_idx", f_idx)
    print("f_series", f_series)

    pd.DataFrame({"Date": f_idx, "Forecast": future, "Lower_CI": conf[:, 0], "Upper_CI": conf[:, 1]}) \
        .to_csv(os.path.join(out_dir, f"Forecast_{ts_name}.txt"), sep="\t", index=False)

    s.index = date_index

    f_series.index = future_index
    f_series.name = None

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(s, label="Historic")
        plt.plot(f_series, label="Forecast", linestyle="dashed")
        plt.fill_between(future_index, conf[:, 0], conf[:, 1], alpha=0.2, label=f"{int((1-alpha)*100)}% CI")
        plt.title(f"{ts_name} - SARIMA Trend ({freq_label if not USE_RESAMPLE_16D else 'Every 16 days'})")
        plt.xlabel("Time"); plt.ylabel(col_name)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Plot_{ts_name}.png"), dpi=144, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"WARNING: plot failed: {e}")

    try:
        tau, pval = kendalltau(np.arange(len(f_series)), f_series.values)
        slope_info = ""
        if _HAS_SKLEARN:
            X = np.arange(len(f_series)).reshape(-1, 1)
            tsr = TheilSenRegressor().fit(X, f_series.values)
            slope_info = f"Slope (Theil–Sen): {float(tsr.coef_[0]):.6g}\n"
        interp = ("Significant increasing trend" if (pval < 0.05 and tau > 0)
                  else "Significant decreasing trend" if (pval < 0.05 and tau < 0)
                  else "No significant monotonic trend (p >= 0.05)")
        with open(os.path.join(out_dir, f"Kendall_{ts_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Kendall's tau: {tau:.4f}\n")
            f.write(f"p-value: {pval:.4g}\n")
            if slope_info:
                f.write(slope_info)
            f.write(f"Interpretation: {interp}\n")
        print(f"  - {ts_name}: last hist = {last_date.date()}, first fc = {future_index[0].date()}")
        print(f"    Kendall: tau={tau:.3f}, p={pval:.3g} -> {interp}")
    except Exception as e:
        print(f"WARNING: Kendall test failed: {e}")

    try:
        with open(run_log_path, "a", encoding="utf-8") as lf:
            lf.write(f"model_order: {model.order}\n")
            try:
                lf.write(f"seasonal_order: {model.seasonal_order}\n")
            except Exception:
                lf.write("seasonal_order: (no seasonal)\n")
            lf.write(f"AIC: {model.aic()}\n")
            try:
                lf.write(f"BIC: {model.bic()}\n")
            except Exception:
                pass
            lf.write(f"Backtest MAE: {mae_bt:.4f}, MAPE: {mape_bt:.2f}%\n")
            lf.write(f"fallback_used: {used_fallback}\n")
            lf.write("status: OK\n")
    except Exception:
        pass



action_name = "sarima"
config_file = os.path.join(conf_input_path, "config.csv")

run_action = False

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]

    if not act.empty:
        active = act.loc[act["parameter"] == "active", "value"]
        run_action = not active.empty and active.iloc[0].lower() == "true"

if not run_action:
    print(f"Action '{action_name}' is disabled or config file missing. Cell skipped.")
else:

    p = act.set_index("parameter")["value"]

    forecast_time_in_years = int(p.get("forecast_time_in_years", forecast_time_in_years))
    forecast_periods = m_value * forecast_time_in_years   


    print("Parameters:")
    print("forecast_time_in_years", forecast_time_in_years)
    print("forecast_periods", forecast_periods)
    
    allowed_ext = (".csv", ".txt")
    file_list = [p for p in glob(os.path.join(input_base_folder, "**", "*.*"), recursive=True)
                 if p.lower().endswith(allowed_ext)]

    print(file_list)

    ok, errs = 0, []
    for fp in file_list:
        try:
            process_file(fp)
            ok += 1
        except Exception as e:
            errs.append(f"{os.path.basename(fp)}: {e}")

    print("\nAll operations completed.")
    print(f"Processed files: {len(file_list)} | Success: {ok} | Errors: {len(errs)}")
    if errs:
        print("\nErrors/Warnings:")
        for e in errs:
            print(" -", e)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
