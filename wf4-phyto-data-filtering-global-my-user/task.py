import os
from glob import glob
import pandas as pd
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_aggregation_dir', action='store', type=str, required=True, dest='data_aggregation_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

data_aggregation_dir = args.data_aggregation_dir.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF4/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF4/' + 'data'

data_filtering_dir = os.path.join(conf_output_path, "data_filtering")
os.makedirs(data_filtering_dir, exist_ok=True)

input_dir = data_aggregation_dir

taxlev = 'scientificname'
metric_numeric = 'density'
metric_biomass = 'mean biovolume'
thresholds = [1, 0.99, 0.95, 0.90, 0.75]


action_name = "data_filtering"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    
    p = act.set_index("parameter")["value"]
    
    taxlev = p.get("taxlev", taxlev)
    metric_numeric = p.get("metric_numeric", metric_numeric)
    metric_biomass = p.get("metric_biomass", metric_biomass)

    if "thresholds" in p:
        thresholds = [float(x) for x in p["thresholds"].split(",")]


def slugify(s: str) -> str:
    s = str(s).strip().lower().replace(' ', '_')
    s = re.sub(r'[^a-z0-9._-]+', '', s)
    return s or 'unknown'
    
def ensure_total_for_mean(df: pd.DataFrame, metric_biomass: str, metric_numeric: str) -> str:
    """
    If metric_biomass is a 'mean ...', create a consistent 'total' column
    by multiplying by 'abundance' if present, otherwise by 'density' if present.
    Returns the name of the column to use as the 'total' metric.
    """
    m = metric_biomass.strip().lower()
    if m.startswith('total'):
        return metric_biomass

    if m in ('mean biovolume', 'mean biomass', 'mean carbon content'):
        mult_col = None
        if 'abundance' in df.columns:
            mult_col = 'abundance'
        elif 'density' in df.columns:
            mult_col = 'density'
        else:
            return metric_biomass

        out_col = f"total_from_{metric_biomass.replace(' ', '_')}"
        df[out_col] = pd.to_numeric(df.get(metric_biomass), errors='coerce') * pd.to_numeric(df.get(mult_col), errors='coerce')
        return out_col

    return metric_biomass

def top_share_set(data: pd.DataFrame, taxcol: str, valuecol: str, thr: float = 1.0):
    """Returns the set of taxa that make up the first 'thr' of cumulative sum."""
    g = data.groupby(taxcol, dropna=False)[valuecol].sum().sort_values(ascending=False)
    g = g.fillna(0)
    if g.sum() == 0:
        return set(g.index)  # if everything is zero, we don't cut anything
    cumulative = g.cumsum() / g.sum()
    return set(cumulative[cumulative <= thr].index)



files = sorted(glob(os.path.join(input_dir, 'agg_*.csv')))
if not files:
    print(f"⚠️ No files found in '{input_dir}' with pattern 'agg_*.csv'.")
else:
    print(f"🔎 Found {len(files)} files to process in: {input_dir}")

for fp in files:
    base = os.path.basename(fp)
    _name = re.sub(r'^agg_', '', base, flags=re.IGNORECASE)
    _name = re.sub(r'\.csv$', '', _name, flags=re.IGNORECASE)
    _display = _name.replace('_', ' ')
    _slug = slugify(_name)

    if _slug == "united_kingdom_of_great_britain_and_northern_ireland":
        _short = "uk"
    else:
        _short = _slug[:32]  # shorten if too long

    _dir = os.path.join(data_filtering_dir, _slug)
    os.makedirs(_dir, exist_ok=True)

    print(f"\n🌍 Name: {_display}")
    print(f"📄 Input: {fp}")
    print(f"📁 Output dir: {_dir}")

    try:
        df = pd.read_csv(fp, sep=';', decimal='.', low_memory=False)
    except Exception as e:
        print(f"❌ Error in reading {fp}: {e}")
        continue

    df.columns = df.columns.str.strip()

    if taxlev not in df.columns:
        print(f"⚠️ Taxonomic column '{taxlev}' not found in {base}. Skipping.")
        continue

    for col in [metric_numeric, metric_biomass]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    metric_biomass_effective = ensure_total_for_mean(df, metric_biomass, metric_numeric)

    if metric_numeric not in df.columns:
        print(f"⚠️ Numeric metric '{metric_numeric}' not found in {base}. Skipping.")
        continue
    if metric_biomass_effective not in df.columns:
        print(f"⚠️ Biomass metric '{metric_biomass_effective}' not found in {base}. Skip.")
        continue

    for thr in thresholds:
        thr_label = str(int(round(thr * 100)))  # 1.0 -> '100', 0.99 -> '99', ...

        top_num = top_share_set(df, taxlev, metric_numeric, thr)
        top_bio = top_share_set(df, taxlev, metric_biomass_effective, thr)

        keep_taxa = top_num.union(top_bio)
        filtered = df[df[taxlev].isin(keep_taxa)].copy()
        rare = df[~df[taxlev].isin(keep_taxa)].copy()

        filtered_name = f"filtered_{_short}_{taxlev}_thr{thr_label}_{metric_numeric}_{metric_biomass}.csv"
        rare_name     = f"rare_{thr_label}_{_short}_{taxlev}_{metric_numeric}_{metric_biomass}.csv"

        filtered_path = os.path.join(_dir, filtered_name)
        rare_path     = os.path.join(_dir, rare_name)

        filtered.to_csv(filtered_path, index=False, sep=';', decimal='.')
        rare.to_csv(rare_path, index=False, sep=';', decimal='.')

        print(
            f"✅ thr {thr_label}% | kept {len(keep_taxa)} taxa | "
            f"top{thr_label}% {metric_numeric}: {len(top_num)} | "
            f"top{thr_label}% {metric_biomass} (effective: {metric_biomass_effective}): {len(top_bio)}"
        )
        print(f"   💾 Filtered: {filtered_path}")
        print(f"   💾 Rare:     {rare_path}")


print("\n🏁 Processing completed.")

file_data_filtering_dir = open("/tmp/data_filtering_dir_" + id + ".json", "w")
file_data_filtering_dir.write(json.dumps(data_filtering_dir))
file_data_filtering_dir.close()
