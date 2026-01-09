import os
import pandas as pd

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


conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF1/' + 'data'

def ensure_total_for_mean(df, metric_biomass, metric_numeric):
    if metric_biomass.lower().startswith('mean'):
        out_col = 'total_from_' + metric_biomass.replace(' ', '_').lower()
        if 'abundance' in df.columns:
            df[out_col] = pd.to_numeric(df[metric_biomass], errors='coerce') * pd.to_numeric(df['abundance'], errors='coerce')
        else:
            df[out_col] = pd.to_numeric(df[metric_biomass], errors='coerce')
        return out_col
    return metric_biomass

def top_share_set(data, taxcol, valuecol, thr=1):
    g = data.groupby(taxcol, dropna=False)[valuecol].sum().sort_values(ascending=False)
    g = g.fillna(0)
    total = g.sum()
    if total == 0:
        return set(g.index)
    cumulative = g.cumsum() / total
    return set(cumulative[cumulative <= thr].index)


output_dir = os.path.join(conf_output_path, "data_filtering")
os.makedirs(output_dir, exist_ok=True)



datain = traits_file

filtered_file = traits_file

taxlev = 'scientificname'
metric_numeric = 'density'
metric_biomass = 'biovolume'
thresholds = [1, 0.99, 0.95, 0.90, 0.75]


action_name = "data_filtering"
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

    taxlev = p.get("taxlev", taxlev)
    metric_numeric = p.get("metric_numeric", metric_numeric)
    metric_biomass = p.get("metric_biomass", metric_biomass)

    if "thresholds" in p:
        thresholds = [float(x) for x in p["thresholds"].split(",")]

    df = pd.read_csv(datain, sep=';', decimal='.', low_memory=False)
    df.columns = df.columns.str.strip()
    
    if taxlev not in df.columns:
        raise ValueError(f"Column '{taxlev}' not found in the global dataset.")
    
    for col in [metric_numeric, metric_biomass]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    metric_biomass_effective = ensure_total_for_mean(df, metric_biomass, metric_numeric)
    
    
    thresholds_sorted = sorted(thresholds)
    
    for thr in thresholds_sorted:
        thr_label = str(int(thr * 100))
    
        top_num = top_share_set(df, taxlev, metric_numeric, thr)
        top_bio = top_share_set(df, taxlev, metric_biomass_effective, thr)
    
        keep_taxa = top_num.union(top_bio)
    
        filtered = df[df[taxlev].isin(keep_taxa)].copy()
        rare = df[~df[taxlev].isin(keep_taxa)].copy()
    
        filtered_name = f"filtered_global_{taxlev}_thr{thr_label}_{metric_numeric}_{metric_biomass}.csv"
        rare_name = f"rare_{thr_label}_global_{taxlev}_{metric_numeric}_{metric_biomass}.csv"
    
        filtered_path = os.path.join(output_dir, filtered_name)
        rare_path = os.path.join(output_dir, rare_name)
    
        filtered.to_csv(filtered_path, index=False, sep=';', decimal='.')
        rare.to_csv(rare_path, index=False, sep=';', decimal='.')
    
        print(f"\n✅ thr {thr_label}% | kept {len(keep_taxa)} taxa")
        print(f"   - top{thr_label}% {metric_numeric}: {len(top_num)}")
        print(f"   - top{thr_label}% {metric_biomass} → colonna '{metric_biomass_effective}': {len(top_bio)}")
        print(f"   💾 Filtrato: {filtered_path}")
        print(f"   💾 Rare:     {rare_path}")

        filtered_file = filtered_path
    
    print("\n🏁 Global filtering completed.")

file_filtered_file = open("/tmp/filtered_file_" + id + ".json", "w")
file_filtered_file.write(json.dumps(filtered_file))
file_filtered_file.close()
