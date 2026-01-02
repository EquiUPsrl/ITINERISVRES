import os
import pandas as pd
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--traits_file', action='store', type=str, required=True, dest='traits_file')


args = arg_parser.parse_args()
print(args)

id = args.id

traits_file = args.traits_file.replace('"','')


conf_output_path = conf_output_path = '' + 'output'
conf_input_path = conf_input_path = '' + 'data'

output_dir = os.path.join(conf_output_path, "data_aggregation")
datain = traits_file     # input file
output_file = os.path.join(output_dir, 'agg_data.csv')

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(datain, sep=';', decimal='.', low_memory=False)

df.columns = [c.strip() for c in df.columns]

tax_choice = 'scientificname'

extra_levels = []

split_output_file_by = "" 


action_name = "data_aggregation"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    tax_choice = p.get("tax_choice", tax_choice)
    split_output_file_by = p.get("split_output_file_by", split_output_file_by)
    
    extra_levels = []
    value = p.get("extra_levels")

    if isinstance(value, str) and value.strip():
        extra_levels = [x.strip() for x in value.split(",")]



group_keys = []
if tax_choice in df.columns:
    group_keys.append(tax_choice)
else:
    raise KeyError(f"The chosen taxonomy level '{tax_choice}' is not present in the file columns.")

group_keys += [c for c in extra_levels if c in df.columns]
if not group_keys:
    raise ValueError("No valid grouping key found.")

sum_cols = [c for c in ['density', 'totalbiovolume', 'totalcarboncontent', 'totalbiomass'] if c in df.columns]

mean_cols = [c for c in ['biovolume', 'cellcarboncontent', 'biomass'] if c in df.columns]

for col in mean_cols + sum_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')



mapping_file = os.path.join(conf_input_path, "data_aggregation_mapping.csv")

if os.path.exists(mapping_file):
    print("Data aggregation mapping file exists: " + mapping_file)
    mapping_df = pd.read_csv(mapping_file, sep=";")

    mapping_df["raw_value"] = mapping_df["raw_value"].astype(str).str.strip().str.casefold()

    mapping_dicts = {
        col: dict(zip(sub["raw_value"], sub["normalized_value"]))
        for col, sub in mapping_df.groupby("column")
    }

    for col, col_map in mapping_dicts.items():
        if col in df.columns:
            col_norm = df[col].astype(str).str.strip().str.casefold()
            df[col] = col_norm.map(col_map).fillna(df[col])
else:
    print("Data aggregation mapping file does not exists: " + mapping_file)


abundance = df.groupby(group_keys, dropna=False).size().rename('abundance').reset_index()

agg_dict = {}
for c in sum_cols:
    agg_dict[c] = 'sum'
for c in mean_cols:
    agg_dict[c] = 'mean'

candidate_meta = [c for c in df.columns if c not in group_keys + mean_cols + sum_cols]

drop_exact = {'organismquantity', 'organismquantitytype'}
drop_patterns = [
    r'^[a-h]$',                    # columns a,b,c,d,e,f,g,h
    r'^(length|width|height|diameter|radius|perimeter|area)\b',  # common geometric measures
]

def to_drop(colname: str) -> bool:
    if colname.lower() in drop_exact:
        return True
    for pat in drop_patterns:
        if re.match(pat, colname.lower()):
            return True
    return False

candidate_meta = [c for c in candidate_meta if not to_drop(c)]

g = df.groupby(group_keys, dropna=False)
constant_meta_cols = []
for c in candidate_meta:
    nun = g[c].nunique(dropna=True)
    if (nun > 1).any():
        continue
    constant_meta_cols.append(c)
    agg_dict[c] = 'first'  # all equal in the group, the first one is enough

df_agg = g.agg(agg_dict).reset_index()

df_agg = df_agg.merge(abundance, on=group_keys, how='left')

rename_map = {}
if 'biovolume' in mean_cols:
    rename_map['biovolume'] = 'mean biovolume'
if 'cellcarboncontent' in mean_cols:
    rename_map['cellcarboncontent'] = 'mean carbon content'
if 'biomass' in mean_cols:
    rename_map['biomass'] = 'mean biomass'
df_agg = df_agg.rename(columns=rename_map)

ordered_cols = []
ordered_cols += group_keys
ordered_cols += ['abundance'] if 'abundance' in df_agg.columns else []

ordered_cols += [c for c in ['density', 'totalbiovolume', 'totalcarboncontent', 'totalbiomass'] if c in df_agg.columns]
ordered_cols += [rename_map.get(c, c) for c in ['biovolume', 'cellcarboncontent', 'biomass'] if rename_map.get(c, c) in df_agg.columns]

for c in df.columns:
    if c in constant_meta_cols and c not in group_keys and c not in mean_cols and c not in sum_cols and c not in drop_exact:
        if c not in ordered_cols:
            ordered_cols.append(c)

ordered_cols = [c for c in ordered_cols if c in df_agg.columns]

df_agg = df_agg.reindex(columns=ordered_cols)

df_agg.to_csv(output_file, sep=';', index=False, decimal='.')

print("✅ Grouping made on:", group_keys)
print(f"✅ Aggregate file saved in: {output_file}")


def slugify(s):
    s = str(s)
    s = s.strip().lower().replace(' ', '_')
    s = re.sub(r'[^a-z0-9._-]+', '', s)   # safe characters for filenames
    return s or 'unknown'

if isinstance(split_output_file_by, str) and split_output_file_by.strip():
    output_dir = os.path.join(output_dir, "by_" + split_output_file_by.strip())
    os.makedirs(output_dir, exist_ok=True)
    
    if split_output_file_by in df_agg.columns:
        for col_val, sub in df_agg.groupby(split_output_file_by.strip(), dropna=False):
            fname = f"agg_{slugify(col_val) if pd.notna(col_val) else 'missing'}.csv"
            sub.to_csv(os.path.join(output_dir, fname), sep=';', index=False, decimal='.')
        print(f"✅ File per-country salvati in: {output_dir}")
    else:
        print(f"⚠️ Column '{split_output_file_by}' not present: by-{split_output_file_by} saving skipped.")

data_aggregation_dir = output_dir

file_data_aggregation_dir = open("/tmp/data_aggregation_dir_" + id + ".json", "w")
file_data_aggregation_dir.write(json.dumps(data_aggregation_dir))
file_data_aggregation_dir.close()
