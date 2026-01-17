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
conf_input_path = conf_input_path = '/tmp/data/WF1_1/work/' + 'input'

matplotlib.use("Agg")


try:
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

warnings.filterwarnings("ignore")

input_base_folder = final_stats_path #os.path.join(work_path, "output", "Time Series Statistics", "corrected")
output_dir = conf_output_path
output_dir_base = os.path.join(output_dir, "SARIMAX")
os.makedirs(output_dir_base, exist_ok=True)

start_year = int(start_year)
end_year = int(end_year)

start_date = str(start_year) + "-01-01"
end_date = str(end_year) + "-12-31"

TARGET_VAR = "PP"
TARGET_FILE_NAME = f"all_statistics_{TARGET_VAR}.csv"    # <--- TARGET FILE NAME (without path)
TARGET_COL_NAME = "mean"       # numeric column to use
COMMON_PD_FREQ = interval # 'MS', 'D', 'W', '8D', '15D', '16D', ...

alpha = 0.05                   # -> 95% CI

USE_RESAMPLE_16D = True        # True -> use 16D (m=23) to speed up
ROLLING_WINDOW_YEARS = None    # e.g. 12 -> last 12 years; None to disable
CLIP_UPPER_Q = 0.995           # soft clipping of target outliers; None to disable

forecast_time_in_years = 5

freq_label_map = {
    "MS": "Monthly",
    "D": "Daily",
    "W": "Weekly",
    "8D": "Every 8 days",
    "15D": "Every 15 days",
    "16D": "Every 16 days"
}
freq_label = freq_label_map.get(COMMON_PD_FREQ, COMMON_PD_FREQ)

m_map = {
    "MS": 12,
    "D": 365,
    "W": 52,
    "8D": 46,
    "15D": 24,
    "16D": 23
}
m_value = m_map.get("16D" if USE_RESAMPLE_16D else COMMON_PD_FREQ, 1)


EXOG_FILE_SELECTION_MODE = "all_except_target"  # or "include_list"

EXOG_FILE_NAMES = [
]

EXOG_MODELING_MODE = "sarima"   # "sarima" | "last" | "mean" | "none"

autoarima_params_fast = {
    "start_p": 0, "max_p": 2,
    "start_q": 0, "max_q": 2,
    "start_P": 0, "max_P": 2,
    "start_Q": 1, "max_Q": 2,
    "d": None, "D": 1, "max_d": 2, "max_D": 1,
    "seasonal": True, "m": m_value,
    "stepwise": True,
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

print(f"*** SARIMAX fast settings | freq='{COMMON_PD_FREQ}' ({freq_label}) | m={m_value} ***")
print(f"*** auto_arima params: {autoarima_params_fast} ***\n")

def safe_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))

def detect_date_column(df: pd.DataFrame):
    """Try to detect a date column among common names."""
    for c in ["Date", "date", "DATA", "data", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def read_any_table(file_path: str) -> pd.DataFrame:
    """
    Try to read a generic text/CSV table with different separators and encodings.
    Raises RuntimeError if all attempts fail.
    """
    tried = []
    try:
        return pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        tried.append(f"inferred sep: {e}")
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
    """
    Normalize a numeric column that may use comma as decimal separator
    and dots as thousand separators.
    """
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
    """Compute Ljung-Box test up to a dynamic lag based on series length."""
    max_lag = int(min(24, max(5, n_hist // 5)))
    return acorr_ljungbox(residuals, lags=[max_lag], return_df=True)

def rolling_window_df(df: pd.DataFrame, years: int | None):
    """Optionally keep only the last N years of data."""
    if years is None:
        return df
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index > cutoff]

def build_series_from_file(file_path: str,
                           value_col: str,
                           pd_freq: str,
                           start_year: int) -> pd.Series:
    """
    Read a file, detect a date column (if present), and build a Series
    with frequency pd_freq.
    """
    data = read_any_table(file_path)
    if value_col not in data.columns:
        raise RuntimeError(f"Column '{value_col}' not found in {os.path.basename(file_path)}")
    data = normalize_decimal_column(data, value_col)

    date_col = detect_date_column(data)
    if date_col:
        data = data[[date_col, value_col]].copy()
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce", dayfirst=False)
        data = data.dropna(subset=[date_col]).drop_duplicates(subset=[date_col], keep="last")
        data = data.sort_values(date_col)
        s = (data.set_index(date_col)[value_col]
                 .asfreq(pd_freq, method=None)
                 .interpolate("linear", limit_direction="both"))
    else:
        series = data[value_col].interpolate(method="linear", limit_direction="both")
        n_hist = len(series)
        idx = pd.date_range(start=f"{start_year}-01-01", periods=n_hist, freq=pd_freq)
        s = pd.Series(series.values, index=idx)
    s = s.astype(float).dropna()
    return s

def fit_with_fallback(ts_series: pd.Series,
                      exog: pd.DataFrame | None,
                      params: dict):
    """
    Fit SARIMAX (via auto_arima) with fallback to non-seasonal in case of
    memory/robustness problems.
    """
    try:
        mdl = auto_arima(ts_series, X=exog, **params)
        return mdl, False
    except Exception as e:
        msg = str(e).lower()
        alloc_err = ("unable to allocate" in msg) or ("memoryerror" in msg) or ("array with shape" in msg)
        fb = params.copy()
        fb.update({"seasonal": False, "m": 1})
        if not alloc_err:
            pass
        mdl = auto_arima(ts_series, X=exog, **fb)
        return mdl, True

def backtest_mae_mape(train_y: pd.Series,
                      test_y: pd.Series,
                      train_X: pd.DataFrame | None,
                      test_X: pd.DataFrame | None,
                      params: dict):
    """
    Simple 80/20 backtest respecting exogenous variables.
    Returns MAE and MAPE.
    """
    try:
        mdl = auto_arima(train_y, X=train_X, **params)
        pred, _ = mdl.predict(
            n_periods=len(test_y),
            X=test_X,
            return_conf_int=True,
            alpha=alpha
        )
        mae = float(np.mean(np.abs(pred - test_y.values)))
        denom = np.maximum(1e-8, np.abs(test_y.values))
        mape = float(np.mean(np.abs((test_y.values - pred) / denom)) * 100.0)
        return mae, mape
    except Exception:
        return np.nan, np.nan


def forecast_exog(df_all: pd.DataFrame,
                  exog_cols: list[str],
                  f_idx,
                  params: dict,
                  mode: str = "sarima") -> pd.DataFrame | None:
    """
    Compute future values of exogenous variables according to 'mode':
      - 'sarima' : univariate SARIMA (auto_arima) for each exogenous variable
      - 'last'   : persistence -> all future steps = last historical value
      - 'mean'   : constant    -> all future steps = historical mean
      - 'none'   : do NOT provide X_future (returns None, exogenous disabled)

    Returns a DataFrame X_future with index=f_idx and columns exog_cols,
    or None if mode == 'none' or exog_cols is empty.
    
    f_idx can be numeric or any index of length compatible with n_periods.
    """
    if not exog_cols:
        return None

    mode = (mode or "sarima").lower().strip()

    if mode == "none":
        return None

    exog_future = {}

    if mode == "sarima":
        for col in exog_cols:
            s = df_all[col].astype(float).dropna()
            if len(s) < 24:
                raise RuntimeError(f"Exogenous variable '{col}' is too short for SARIMA.")
            mdl_exog, _ = fit_with_fallback(s, None, params)
            f_exog = mdl_exog.predict(n_periods=len(f_idx))
            exog_future[col] = pd.Series(f_exog, index=f_idx)

    elif mode == "last":
        for col in exog_cols:
            s = df_all[col].astype(float).dropna()
            if s.empty:
                raise RuntimeError(f"Exogenous variable '{col}' is empty.")
            last_val = s.iloc[-1]
            exog_future[col] = pd.Series(
                np.full(len(f_idx), last_val, dtype=float),
                index=f_idx
            )

    elif mode == "mean":
        for col in exog_cols:
            s = df_all[col].astype(float).dropna()
            if s.empty:
                raise RuntimeError(f"Exogenous variable '{col}' is empty.")
            mean_val = float(s.mean())
            exog_future[col] = pd.Series(
                np.full(len(f_idx), mean_val, dtype=float),
                index=f_idx
            )

    else:
        raise ValueError(f"Unknown EXOG_MODELING_MODE: {mode}")

    X_future = pd.DataFrame(exog_future, index=f_idx)
    return X_future



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



def process_sarimax(target_file: str, exog_files: list[str], forecast_periods):

    target_basename = os.path.basename(target_file)
    target_label = safe_label(os.path.splitext(target_basename)[0])
    ts_name = f"{target_label}_{TARGET_COL_NAME}"

    print(f"\nTarget: {target_basename}")
    if exog_files:
        print("Exogenous files:")
        for ef in exog_files:
            print("  -", os.path.basename(ef))
    else:
        print("No exogenous variables found -> model reduces to SARIMA (no X).")

    y = build_series_from_file(target_file,
                               TARGET_COL_NAME,
                               COMMON_PD_FREQ,
                               start_year)

    exog_series = {}
    for fp in exog_files:
        base = os.path.basename(fp)
        label = safe_label(os.path.splitext(base)[0])
        s_ex = build_series_from_file(fp,
                                      TARGET_COL_NAME,   # same column "mean"
                                      COMMON_PD_FREQ,
                                      start_year)
        s_ex = s_ex.reindex(y.index).interpolate("linear", limit_direction="both")
        exog_series[label] = s_ex

    if exog_series:
        df_all = pd.DataFrame({"target": y})
        for col, s_ex in exog_series.items():
            df_all[col] = s_ex
    else:
        df_all = pd.DataFrame({"target": y})


    if USE_RESAMPLE_16D:
        df_all = df_all.iloc[::2].copy()
        eff_freq = "16D"
    else:
        df_all = df_all.copy()
        eff_freq = COMMON_PD_FREQ

    print("Shape:", df_all.shape)

    df_all.index = df_all.index.normalize()

    OBS_PER_YEAR = m_value
    obs_tot = len(df_all)



    hist_idx = generate_historical_numeric_index(obs_tot)
    print(hist_idx[:10], "...", hist_idx[-3:])
    print("len =", len(hist_idx))
    
    f_idx = generate_future_numeric_index(hist_idx[-1], forecast_periods)
    print(f_idx[:10], "...", f_idx[-3:])
    print("len =", len(f_idx))


    

    df_all.index = hist_idx
    print("Shape history after generate_regular_index:", df_all.shape)

    print("--- After resampling ---")
    print("Shape:", df_all.shape)
    print(df_all.head())
    print(df_all.tail())

    date_index = pd.date_range(start=start_date, end=end_date + " 23:59:59", periods=obs_tot)
    
    print("date_index", date_index)


    years_ahead = forecast_periods / m_value
    
    last_hist_date = date_index[-1]
    future_end_date = last_hist_date + pd.DateOffset(years=years_ahead)
    
    future_index = pd.date_range(
        start=last_hist_date + (date_index[1] - date_index[0]),  # primo passo dopo l'ultimo storico
        end=future_end_date,
        periods=forecast_periods
    )
    
    print("Future premiere:", future_index[0])
    print("Future latest:", future_index[-1])


    df_all = rolling_window_df(df_all, ROLLING_WINDOW_YEARS)

    if len(df_all) < 24:
        raise RuntimeError("Series too short (< 24 points).")

    out_dir = ensure_dir(os.path.join(output_dir_base, target_label, TARGET_COL_NAME))
    run_log_path = os.path.join(out_dir, "run_log.txt")
    with open(run_log_path, "a", encoding="utf-8") as lf:
        lf.write("\n--- NEW RUN (SARIMAX + exog) ---\n")
        lf.write(f"target_file: {target_basename}\nlen(ts): {len(df_all)}\n")
        lf.write(f"params: {json.dumps(autoarima_params_fast)}\n")
        lf.write(f"exog_files: {[os.path.basename(f) for f in exog_files]}\n")
        lf.write(f"exog_modeling_mode: {EXOG_MODELING_MODE}\n")

    if CLIP_UPPER_Q is not None:
        upper = df_all["target"].quantile(CLIP_UPPER_Q)
        df_all["target"] = df_all["target"].clip(upper=upper)

    y_all = df_all["target"].astype(float).dropna()
    exog_cols = [c for c in df_all.columns if c != "target"]

    if EXOG_MODELING_MODE.lower().strip() == "none":
        X_all = None
        exog_cols = []
    else:
        X_all = df_all[exog_cols] if exog_cols else None
        if X_all is not None:
            X_all = X_all.loc[y_all.index]

    split = int(len(y_all) * 0.8)
    y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]
    if X_all is not None:
        X_train, X_test = X_all.iloc[:split, :], X_all.iloc[split:, :]
    else:
        X_train = X_test = None

    mae_bt, mape_bt = backtest_mae_mape(y_train, y_test, X_train, X_test, autoarima_params_fast)

    print("y_all", y_all)
    print("len(y_all)", len(y_all))
    print("X_all", y_all)
    print("len(X_all)", len(X_all))

    model, used_fallback = fit_with_fallback(y_all, X_all, autoarima_params_fast)

    try:
        with open(os.path.join(out_dir, f"Summary_SARIMAX_{ts_name}.txt"), "w", encoding="utf-8") as f:
            f.write(str(model.summary()))
            f.write("\n\n--- Selected model parameters ---\n")
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
            if exog_cols:
                f.write(f"\nExogenous variables ({len(exog_cols)}): {', '.join(exog_cols)}\n")
                f.write(f"Exog forecast mode: {EXOG_MODELING_MODE}\n")
            if used_fallback:
                f.write("\n[FALLBACK] Fit performed without seasonality for speed/robustness.\n")
    except Exception as e:
        print(f"WARNING: summary write failed: {e}")



    print("y_all", y_all)
    print("len(y_all)", len(y_all))
    print("X_all", y_all)
    print("len(X_all)", len(X_all))
    
    print("model", model)
    print("used_fallback", used_fallback)

    try:
        resid = model.resid()

        print("resid", resid)

        resid.index = date_index
        resid.name = None
        
        burnin = max(24, m_value * 2)
        resid_plot = resid[burnin:]

        print("resid_plot", resid_plot)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(resid_plot, label="Residuals (burn-in removed)")
        axes[0].set_title(f"Residuals - {ts_name}")
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(resid_plot, bins=30, alpha=0.7)
        axes[1].set_title(f"Residuals distribution - {ts_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Residuals_{ts_name}.png"),
                    dpi=144, bbox_inches="tight")
        plt.close(fig)

        lb = ljung_box_dynamic(resid_plot, len(resid_plot))
        lb.to_csv(os.path.join(out_dir, f"LjungBox_{ts_name}.txt"),
                  sep="\t", index=False)
    except Exception as e:
        print(f"WARNING: residual diagnostics failed: {e}")


    
    print("=== df_all ===")
    print("Shape:", df_all.shape)
    print("Colonne:", df_all.columns.tolist())
    
    print("NaN per colonna:")
    print(df_all[exog_cols].isna().sum())
    
    finite_mask = np.isfinite(df_all[exog_cols])
    for col in exog_cols:
        if not finite_mask[col].all():
            print(f"Colonna '{col}' contiene valori non finiti o infiniti")
            print(df_all[col][~finite_mask[col]])
    
    
    print("f_idx length:", len(f_idx))
    print("First 5 dates:", f_idx[:5])
    print("Last 5 dates:", f_idx[-5:])

    print("df_all", df_all)
    print("exog_cols", exog_cols)

    if exog_cols:
        X_future = forecast_exog(
            df_all,
            exog_cols,
            f_idx,
            autoarima_params_fast,
            mode=EXOG_MODELING_MODE
        )
    else:
        X_future = None

    print("NaN", X_future.isna().sum())      # count NaN per column
    print("Finite values: ", np.isfinite(X_future).all())  # verifica valori finiti

    print("X_future", X_future)

    print(">>> CALLING predict()")
    print("model used:", model)
    print("X passed to predict:", type(X_future), X_future.shape if X_future is not None else None)
    print("forecast_periods:", forecast_periods)
    
    future, conf = model.predict(
        n_periods=forecast_periods,
        X=X_future,
        return_conf_int=True,
        alpha=alpha
    )
    f_series = pd.Series(future, index=f_idx, name=f"Forecast_{ts_name}")

    print("future:", future[:10])
    print("conf:", conf[:5])
    print("f_series.isna().sum():", f_series.isna().sum())

    print("f_series", f_series)

    if len(future) != len(f_idx):
        raise ValueError(f"Forecast length mismatch: {len(future)} vs {len(f_idx)}")

    pd.DataFrame({
        "Date": f_idx,
        "Forecast": future,
        "Lower_CI": conf[:, 0],
        "Upper_CI": conf[:, 1]
    }).to_csv(os.path.join(out_dir, f"Forecast_{ts_name}.txt"),
             sep="\t", index=False)

    print("y_all index type:", type(y_all.index))
    print("f_series index type:", type(f_series.index))
    print("y_all index dtype:", y_all.index.dtype)
    print("f_series index dtype:", f_series.index.dtype)

    y_all.index = date_index
    y_all.name = None
    last_date = date_index[-1]

    print("y_all", y_all)

    f_series.index = future_index
    f_series.name = None

    print("f_series", f_series)

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(y_all, label="Historical")
        plt.plot(f_series, label="Forecast", linestyle="dashed")
        plt.fill_between(future_index, conf[:, 0], conf[:, 1],
                         alpha=0.2,
                         label=f"{int((1-alpha)*100)}% CI")
        base_title = f"{ts_name} - SARIMAX Trend"
        plt.title(f"{base_title} ({'Every 16 days' if USE_RESAMPLE_16D else freq_label})")
        plt.xlabel("Time")
        plt.ylabel(TARGET_COL_NAME)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Plot_{ts_name}.png"),
                    dpi=144, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"WARNING: plot failed: {e}")

    try:
        tau, pval = kendalltau(np.arange(len(f_series)), f_series.values)
        slope_info = ""
        if _HAS_SKLEARN:
            Xk = np.arange(len(f_series)).reshape(-1, 1)
            tsr = TheilSenRegressor().fit(Xk, f_series.values)
            slope_info = f"Slope (Theil–Sen): {float(tsr.coef_[0]):.6g}\n"
        interp = ("Significant increasing trend"
                  if (pval < 0.05 and tau > 0)
                  else "Significant decreasing trend"
                  if (pval < 0.05 and tau < 0)
                  else "No significant monotonic trend (p >= 0.05)")
        with open(os.path.join(out_dir, f"Kendall_{ts_name}.txt"),
                  "w", encoding="utf-8") as f:
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
            lf.write(f"exog_cols: {exog_cols}\n")
            lf.write(f"exog_modeling_mode: {EXOG_MODELING_MODE}\n")
            lf.write("status: OK\n")
    except Exception:
        pass








action_name = "sarimax"
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

    EXOG_FILE_SELECTION_MODE = p.get("EXOG_FILE_SELECTION_MODE", EXOG_FILE_SELECTION_MODE)
    TARGET_VAR = p.get("TARGET_VAR", TARGET_VAR)
    TARGET_FILE_NAME = f"all_statistics_{TARGET_VAR}.csv"
    forecast_time_in_years = int(p.get("forecast_time_in_years", forecast_time_in_years))

    value = p.get("EXOG_FILE_NAMES")
    if isinstance(value, str) and value.strip():
        EXOG_FILE_NAMES = [x.strip() for x in value.split(",")]

    forecast_periods = m_value * forecast_time_in_years





    print("Parameters:")
    print("EXOG_FILE_SELECTION_MODE", EXOG_FILE_SELECTION_MODE)
    print("TARGET_VAR", TARGET_VAR)
    print("TARGET_FILE_NAME", TARGET_FILE_NAME)
    print("EXOG_FILE_NAMES", EXOG_FILE_NAMES)
    print("forecast_periods", forecast_periods)

    

    allowed_ext = (".csv", ".txt")
    file_list = [p for p in glob(os.path.join(input_base_folder, "**", "*.*"),
                                 recursive=True)
                 if p.lower().endswith(allowed_ext)]
    
    target_file = None
    for fp in file_list:
        if os.path.basename(fp) == TARGET_FILE_NAME:
            target_file = fp
            break
    
    if target_file is None:
        target_file = file_list[0]
        print(f"WARNING: target file '{TARGET_FILE_NAME}' not found. "
              f"Using as target: {os.path.basename(target_file)}")
    
    if EXOG_FILE_SELECTION_MODE == "all_except_target":
        exog_files = [fp for fp in file_list if fp != target_file]
    
    elif EXOG_FILE_SELECTION_MODE == "include_list":
        name_set = set(EXOG_FILE_NAMES)
        exog_files = []
        missing = []
    
        for fp in file_list:
            base = os.path.basename(fp)
            if base in name_set:
                exog_files.append(fp)
    
        found_basenames = {os.path.basename(fp) for fp in exog_files}
        for name in EXOG_FILE_NAMES:
            if name not in found_basenames:
                missing.append(name)
    
        if missing:
            print("WARNING: the following exogenous files were not found in InputS/:",
                  ", ".join(missing))
    
    else:
        raise ValueError(f"Unknown EXOG_FILE_SELECTION_MODE: {EXOG_FILE_SELECTION_MODE}")
    
    ok, errs = 0, []
    process_sarimax(target_file, exog_files, forecast_periods)
    ok += 1
    
    print("\nAll operations completed.")
    print(f"Processed targets: {ok} | Errors: {len(errs)}")
    if errs:
        print("\nErrors/Warnings:")
        for e in errs:
            print(" -", e)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
