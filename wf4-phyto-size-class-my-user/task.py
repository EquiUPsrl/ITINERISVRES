import pandas as pd
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_filtering_dir', action='store', type=str, required=True, dest='data_filtering_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

data_filtering_dir = args.data_filtering_dir.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF4/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF4/' + 'data'

base_dir  = data_filtering_dir

output_dir = os.path.join(conf_output_path, "size_class_distribution")
os.makedirs(output_dir, exist_ok=True)


tax_col   = "scientificname"
x_col = "mean biovolume"   # x_col
y_col = "density"          # y_col

base_log = 2                      # log base 2
n_classes = 16                

group_sizes = [2, 3, 3, 3, 3, 2]  # sum must equal n_classes


action_name = "size_class"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]

    p = act.set_index("parameter")["value"]
    
    tax_col = p.get("tax_col", tax_col)
    x_col = p.get("x_col", x_col)
    y_col = p.get("y_col", y_col)
    base_log = float(p.get("base_log", base_log))
    n_classes = int(p.get("n_classes", n_classes))
    
    if "group_sizes" in p:
        group_sizes = [int(x) for x in p["group_sizes"].split(",")]

print("tax_col", tax_col)
print("x_col", x_col)
print("y_col", y_col)
print("base_log", base_log)
print("n_classes", n_classes)
print("group_sizes", group_sizes)

SUPER_GROUPS = []

start = 1
for size in group_sizes:
    group = tuple(range(start, start + size))
    SUPER_GROUPS.append(group)
    start += size




ALIASES = {
    "united_kingdom_of_great_britain_and_northern_ireland": "uk",
}

def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def load_filtered(path: str) -> pd.DataFrame:
    """Loads filtered/rare file and produces aggregate table by taxon:
       size = media(x_col), density = somma(y_col)."""
    df = pd.read_csv(path, sep=';', decimal='.', low_memory=False)
    df.columns = df.columns.str.strip()

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

    sub = df[[tax_col, x_col, y_col]].dropna()
    sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]

    den = sub.groupby(tax_col, dropna=False)[y_col].sum()
    siz = sub.groupby(tax_col, dropna=False)[x_col].mean()
    out = pd.concat([den, siz], axis=1).reset_index()
    out.columns = [tax_col, y_col, x_col]
    return out

def size_class_index(sizes: np.ndarray, base=2, n_classes=16):
    """Assign class index (1..n_classes) on log(base) scale with width 1."""
    l = np.log(sizes) / np.log(base)
    left = np.floor(l.min())
    edges = left + np.arange(n_classes + 1)  # left, left+1, ..., left+n
    idx = np.digitize(l, edges, right=False)
    idx = np.clip(idx, 1, n_classes)
    return idx.astype(int), edges

def aggregate_classes(df_tax: pd.DataFrame):
    """Distribution across 16 classes and 6 superclasses. Return (table 16, table 6)."""
    idx, _ = size_class_index(df_tax[x_col].to_numpy(), base=base_log, n_classes=n_classes)
    df = df_tax.copy()
    df["class16"] = idx

    dist16 = df.groupby("class16")[y_col].sum().reindex(range(1, n_classes + 1), fill_value=0)

    super_map = {}
    for k, group in enumerate(SUPER_GROUPS, start=1):
        for c in group:
            super_map[c] = k
    df["class6"] = df["class16"].map(super_map)
    dist6 = df.groupby("class6")[y_col].sum().reindex(range(1, 6 + 1), fill_value=0)

    tab16 = dist16.reset_index().rename(columns={"class16": "size_class_16", y_col: "density"})
    tab6  = dist6.reset_index().rename(columns={"class6": "size_class_6",  y_col: "density"})
    return tab16, tab6

def plot_dual_axis(tab6_total: pd.DataFrame, tab6_rare: pd.DataFrame, title: str, out_png: str):
    """Bar plot with double axes: left=totals, right=rare, formatter 2×10^n."""

    x = np.arange(1, 6 + 1)
    w = 0.4

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    def sci_notation(xv, pos):
        if xv == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(xv))))
        coeff = xv / (10 ** exp)
        return rf"{coeff:.1f}×10$^{exp}$"

    ax1.yaxis.set_major_formatter(FuncFormatter(sci_notation))  # Overall
    ax2.yaxis.set_major_formatter(FuncFormatter(sci_notation))  # Rare

    ax1.bar(x - w/2, tab6_total["density"].to_numpy(), width=w, label="Overall", color="#1F78B4")
    ax2.bar(x + w/2, tab6_rare["density"].to_numpy(),  width=w, label="Rare",    color="#AED6F1")

    ax1.set_xlabel("Size classes")
    ax1.set_ylabel("Overall density (cells·L$^{-1}$)", color="#000000")
    ax2.set_ylabel("Rare density (cells·L$^{-1}$)",    color="#000000")
    ax1.set_xticks(x)
    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_csvs(out_dir: str, prefix: str, tab16: pd.DataFrame, tab6: pd.DataFrame):
    """Save CSV for 16 classes and for 6 grouped classes."""
    ensure_dir(out_dir)
    tab16.to_csv(os.path.join(out_dir, f"{prefix}_sizeclasses16.csv"), sep=';', index=False, decimal='.')
    tab6.to_csv(os.path.join(out_dir,  f"{prefix}_sizeclasses6_clustered.csv"), sep=';', index=False, decimal='.')

def pick_best_threshold(paths, regex_pattern):
    """
    From a list of paths, it chooses the one with the highest numerical threshold
    extracted with 'regex_pattern' (e.g., r'thr(\\d+)' or r'rare_(\\d+)').
    If no match, it returns the first one if found; otherwise, it returns None.
    """
    best_p, best_n = None, -1
    for p in paths:
        m = re.search(regex_pattern, os.path.basename(p))
        if m:
            try:
                n = int(m.group(1))
                if n > best_n:
                    best_p, best_n = p, n
            except ValueError:
                pass
    if best_p:
        return best_p
    return paths[0] if paths else None



_dirs = sorted([d for d in glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
if not _dirs:
    print(f"   ⚠️ No folders found in {base_dir}")

for cdir in _dirs:
    folder_name = os.path.basename(cdir)
    out_dir = ensure_dir(os.path.join(output_dir, folder_name))

    f_total = glob(os.path.join(cdir, "filtered_*thr100*mean biovolume.csv"))
    f_rare  = glob(os.path.join(cdir, "rare_99_*mean biovolume.csv"))

    if not f_total:
        f_total = glob(os.path.join(cdir, "filtered_*thr*mean biovolume.csv")) or \
                  glob(os.path.join(cdir, "filtered_*mean biovolume.csv"))
        if f_total:
            f_total = [pick_best_threshold(f_total, r"thr(\d+)")]
    if not f_rare:
        f_rare = glob(os.path.join(cdir, "rare_*mean biovolume.csv"))
        if f_rare:
            f_rare = [pick_best_threshold(f_rare, r"rare_(\d+)")]

    alias = ALIASES.get(folder_name)
    if alias == "uk":
        cand_total_uk = glob(os.path.join(cdir, "filtered_*uk*mean biovolume.csv"))
        if cand_total_uk:
            best_total_uk = pick_best_threshold(cand_total_uk, r"thr(\d+)")
            if not f_total or not f_total[0]:
                f_total = [best_total_uk]
        cand_rare_uk = glob(os.path.join(cdir, "rare_*uk*mean biovolume.csv"))
        if cand_rare_uk:
            best_rare_uk = pick_best_threshold(cand_rare_uk, r"rare_(\d+)")
            if not f_rare or not f_rare[0]:
                f_rare = [best_rare_uk]

    if not f_total or not f_total[0] or not f_rare or not f_rare[0]:
        print(f"   ⚠️ {folder_name}: missing thr/rare files, skip.")
        continue

    try:
        df_tot = load_filtered(f_total[0])
        df_rar = load_filtered(f_rare[0])

        tab16_tot, tab6_tot = aggregate_classes(df_tot)
        tab16_rar, tab6_rar = aggregate_classes(df_rar)

        out_prefix = folder_name if alias is None else alias

        save_csvs(out_dir, f"{out_prefix}_overall", tab16_tot, tab6_tot)
        save_csvs(out_dir, f"{out_prefix}_rare",    tab16_rar, tab6_rar)

        plot_dual_axis(tab6_tot, tab6_rar,
                       title=f"Size-class distribution ({folder_name})",
                       out_png=os.path.join(out_dir, f"size_classes_{out_prefix}_dualaxis.png"))

        print(f"   ✅ {folder_name}: saved CSV and graph in {out_dir}")
    except Exception as e:
        print(f"   ❌ {folder_name}: error {e}")

print("🏁 Done.")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
