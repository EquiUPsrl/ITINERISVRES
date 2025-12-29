import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')


args = arg_parser.parse_args()
print(args)

id = args.id

final_input = args.final_input.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF6/' + 'data'

output_dir = os.path.join(conf_output_path, "SARIMAX")
os.makedirs(output_dir, exist_ok=True)
print(f"Folder '{output_dir}' ready.")

csv_path = final_input
event_df = pd.read_csv(csv_path, low_memory=False)


exog_col = "dissolvedOxygen"


action_name = "sarimax"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    
    exog_col = p.get("exog_col")



def run_sarimax_for_lake(event_df, lake_name, exog_col, single_lake=None,  output_base="output"):
    print(f"\n=== Processing lake: {lake_name} ===")
    
    lake_dir = f"{output_base}/{lake_name}"
    os.makedirs(lake_dir, exist_ok=True)
    print("Saving SARIMAX outputs to:", lake_dir)

    event_df['parsed_date'] = pd.to_datetime(
        event_df['parsed_date'],
        errors='coerce'
    )

    if single_lake:
        lake_df = event_df[event_df["locality"] == lake_name].copy()
    else:
        lake_df = event_df.copy()
    
    lake_df = lake_df.sort_values("parsed_date")
    lake_df["density_log"] = np.log10(lake_df["density"] + 1)
    lake_df = lake_df.set_index("parsed_date")
    lake_df = lake_df[["density", "density_log", exog_col]]
    lake_ts = lake_df.resample("2MS").mean()
    lake_ts["density_log"] = lake_ts["density_log"].interpolate()
    lake_ts[exog_col] = lake_ts[exog_col].interpolate()
    density_log_ts = lake_ts["density_log"]
    exog_ts = lake_ts[exog_col]

    density_log_ts.plot(title=f"Log-density time series - {lake_name}")
    plt.ylabel("log10(density + 1)")
    plt.tight_layout()
    plt.show()

    sarima_exog = SARIMAX(exog_ts, order=(1,1,1), seasonal_order=(1,1,1,6),
                          enforce_stationarity=False, enforce_invertibility=False)
    exog_res = sarima_exog.fit(disp=False)
    print(exog_res.summary())

    plt.figure()
    plot_acf(exog_res.resid.dropna(), lags=20)
    plt.title(f"Residual ACF - {exog_col} SARIMA ({lake_name})")
    plt.tight_layout()
    plt.savefig(f"{lake_dir}/acf_exogenous_{lake_name}.png", dpi=300)
    plt.savefig(f"{lake_dir}/acf_exogenous_{lake_name}.svg")
    plt.close()

    h = 18
    exog_forecast = exog_res.get_forecast(steps=h).predicted_mean
    exog_forecast.name = f"{exog_col}_forecast"

    sarimax_model = SARIMAX(density_log_ts, exog=exog_ts,
                            order=(1,1,1), seasonal_order=(1,1,1,6),
                            enforce_stationarity=False, enforce_invertibility=False)
    sarimax_res = sarimax_model.fit(disp=False)
    print(sarimax_res.summary())

    plt.figure()
    plot_acf(sarimax_res.resid.dropna(), lags=20)
    plt.title(f"Residual ACF - SARIMAX log-density ({lake_name})")
    plt.tight_layout()
    plt.savefig(f"{lake_dir}/acf_sarimax_{lake_name}.png", dpi=300)
    plt.savefig(f"{lake_dir}/acf_sarimax_{lake_name}.svg")
    plt.close()

    ljung_box = acorr_ljungbox(sarimax_res.resid.dropna(), lags=[20], return_df=True)
    ljung_box.to_csv(f"{lake_dir}/ljung_box_{lake_name}.csv")
    
    obs_density = 10**density_log_ts - 1
    fc_res = sarimax_res.get_forecast(steps=h, exog=exog_forecast)
    fc_mean_log = fc_res.predicted_mean
    fc_ci_log = fc_res.conf_int(alpha=0.05)
    fc_mean = 10**fc_mean_log - 1
    fc_lower = 10**fc_ci_log["lower density_log"] - 1
    fc_upper = 10**fc_ci_log["upper density_log"] - 1

    plt.figure(figsize=(11,6))
    def sci_notation(x, pos):
        if x==0: return "0"
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / (10**exponent)
        return r"${:.0f}\times10^{{{}}}$".format(coeff, exponent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(sci_notation))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    plt.plot(obs_density.index, obs_density.values, label="Observed density", color="#1f77b4", linewidth=2.8)
    plt.plot(fc_mean.index, fc_mean.values, label="Forecast", linestyle="--", color="#d95f02", linewidth=2.8)
    plt.fill_between(fc_mean.index, fc_lower, fc_upper, alpha=0.18, color="#74a9cf", label="95% CI")
    plt.grid(False)
    plt.title(f"SARIMAX forecast (density) - {lake_name}", fontsize=17)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("density", fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(loc="upper left", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{lake_dir}/sarimax_forecast_clean_{lake_name}.png", dpi=350)
    plt.savefig(f"{lake_dir}/sarimax_forecast_clean_{lake_name}.svg")
    plt.show()
    
    print("Saved forecast plot for lake:", lake_name)



lakes = list(event_df['locality'].unique()) + ["All lakes"]
for lake in lakes:
    run_sarimax_for_lake(
        event_df, 
        lake, 
        single_lake=None if lake=="All lakes" else lake,
        exog_col=exog_col, 
        output_base=output_dir
    )

