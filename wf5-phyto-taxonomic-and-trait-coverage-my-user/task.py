import os
import pandas as pd
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filtered_file', action='store', type=str, required=True, dest='filtered_file')


args = arg_parser.parse_args()
print(args)

id = args.id

filtered_file = args.filtered_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF5/' + 'data'

input_file = filtered_file
output_dir = conf_output_path
CSV_DIR    = os.path.join(output_dir, "csv")

os.makedirs(CSV_DIR, exist_ok=True)

species_col = "acceptedNameUsage"
shape_col = "shape"
group_by_col = "country"


action_name = "taxonomic_and_trait_coverage"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    
    species_col = p.get("species_col", species_col)
    shape_col = p.get("shape_col", shape_col)
    group_by_col = p.get("group_by_col", group_by_col)


df_species = pd.read_csv(input_file, low_memory=False)


for col in [
    "phylum", "class", "family", "order", "genus", "acceptedNameUsage", "shape",
    "country", "locality", "year", "month", "season"
]:
    if col not in df_species.columns:
        df_species[col] = np.nan

def taxonomic_coverage_simple(df_in, group_cols=None):
    if not group_cols:
        groups = [("ALL", df_in)]
        out_col = "group"
    else:
        groups = df_in.groupby(group_cols, dropna=False)
        out_col = "_".join(group_cols)

    rows = []
    for g, sub in groups:
        gval = g if isinstance(g, str) or g == "ALL" else "_".join(map(str, g))
        rows.append({
            out_col: gval,
            "n_records": len(sub),
            "n_phyla": sub["phylum"].nunique(dropna=True),
            "n_classes": sub["class"].nunique(dropna=True),
            "n_families": sub["family"].nunique(dropna=True),
            "n_orders": sub["order"].nunique(dropna=True),
            "n_genus": sub["genus"].nunique(dropna=True),
            "n_shapes": sub["shape"].nunique(dropna=True),
            "n_species": sub["acceptedNameUsage"].nunique(dropna=True),
        })
    return pd.DataFrame(rows)

coverage_by = taxonomic_coverage_simple(df_species, [group_by_col])
taxonomic_coverage_file = os.path.join(CSV_DIR, f"taxonomic_coverage_by_{group_by_col}.csv")
coverage_by.to_csv(
    taxonomic_coverage_file,
    index=False
)
print(f"Saved file: {taxonomic_coverage_file}")



def richness(df_in, group_cols, level):
    name = f"{level}_richness"
    return (
        df_in.groupby(group_cols, dropna=False)[level]
             .nunique()
             .reset_index(name=name)
    )

species_richness_by = richness(df_species, [group_by_col], species_col)
shape_richness_by   = richness(df_species, [group_by_col], shape_col)

richness_file = os.path.join(CSV_DIR, f"richness_by_{group_by_col}.csv")
(
    species_richness_by
    .merge(shape_richness_by, on=group_by_col, how="left")
    .to_csv(richness_file, index=False)
)

print(f"Saved file: {richness_file}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
