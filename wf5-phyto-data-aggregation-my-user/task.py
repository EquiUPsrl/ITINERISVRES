import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
import os
import pandas as pd
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_csv', action='store', type=str, required=True, dest='input_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

input_csv = args.input_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_temp_path = conf_temp_path = '/tmp/data/WF5/' + 'tmp'
conf_input_path = conf_input_path = '/tmp/data/WF5/' + 'data'

warnings.filterwarnings("ignore", category=ValueWarning)

OUTPUT_DIR = conf_output_path
TMP_DIR = conf_temp_path
CSV_DIR    = os.path.join(OUTPUT_DIR, "csv")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Output folders ready:")
print(" - CSV:", CSV_DIR)
print(" - Plots:", PLOTS_DIR)


value_col = "density" # or organismQuantity

biovolume_col = "biovolume"
carbon_col    = "carbonContent"
biomass_col   = "biomass"

cluster_cols = [
    "country", "locality", "year", "month",
    "parentEventID", "eventID", "shape", "season"
]

taxlev = "acceptedNameUsage"

taxonomic_hierarchy = ["phylum", "class", "order", "family", "genus"]



action_name = "data_aggregation"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    
    value_col = p.get("value_col", value_col)
    biovolume_col = p.get("biovolume_col", biovolume_col)
    carbon_col = p.get("carbon_col", carbon_col)
    biomass_col = p.get("biomass_col", biomass_col)
    taxlev = p.get("taxlev", taxlev)
    
    cluster_cols = []
    value = p.get("cluster_cols")

    if isinstance(value, str) and value.strip():
        cluster_cols = [x.strip() for x in value.split(",")]

    taxonomic_hierarchy = []
    value = p.get("taxonomic_hierarchy")

    if isinstance(value, str) and value.strip():
        taxonomic_hierarchy = [x.strip() for x in value.split(",")]




df = pd.read_csv(input_csv, sep=";", low_memory=False)
print("Raw dataset loaded:", len(df), "records")

value_col = value_col if value_col in df.columns else "organismQuantity"

df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

if "month" in df.columns:
    month_series = pd.to_numeric(df["month"], errors="coerce")
elif "eventDate" in df.columns:
    month_series = pd.to_datetime(df["eventDate"], errors="coerce").dt.month
else:
    month_series = pd.Series([np.nan] * len(df))

def assign_season(m):
    if pd.isna(m):
        return "Unknown"
    m = int(m)
    if m == 4:
        return "Spring"
    elif m == 10:
        return "Autumn"
    else:
        return "Other"

df["month"] = month_series
df["season"] = month_series.apply(assign_season)

df = df[df[value_col] > 0].copy()
print("After removing zero-density rows:", len(df), "records")

df[biovolume_col] = pd.to_numeric(df.get(biovolume_col, 0), errors="coerce").fillna(0)
df[carbon_col]    = pd.to_numeric(df.get(carbon_col, 0), errors="coerce").fillna(0)
df[biomass_col]   = pd.to_numeric(df.get(biomass_col, 0), errors="coerce").fillna(0)

df["totalBiovolume"]     = df[value_col] * df[biovolume_col]
df["totalBiomass"]       = df[value_col] * df[biomass_col]
df["totalCarbonContent"] = df[value_col] * df[carbon_col]

clean_path = os.path.join(TMP_DIR, "final_with_season_and_totals.csv")
df.to_csv(clean_path, index=False)



cluster_cols = [c for c in cluster_cols if c in df.columns]

group_cols = cluster_cols + [taxlev]

sum_params  = [value_col, "totalBiovolume", "totalBiomass", "totalCarbonContent"]
mean_params = ["biovolume", "biomass", "carbonContent"]

agg_dict = {p: "sum" for p in sum_params if p in df.columns}
agg_dict.update({p: "mean" for p in mean_params if p in df.columns})

for tax in taxonomic_hierarchy:
    if tax in df.columns:
        agg_dict[tax] = "first"

df_species = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

if value_col != "density":
    df_species["density"] = df_species[value_col]

species_path = os.path.join(TMP_DIR, "species_aggregated.csv")
df_species.to_csv(species_path, index=False)

aggregated_file = species_path

file_aggregated_file = open("/tmp/aggregated_file_" + id + ".json", "w")
file_aggregated_file.write(json.dumps(aggregated_file))
file_aggregated_file.close()
