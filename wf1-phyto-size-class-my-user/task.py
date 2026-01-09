import seaborn as sns
import os
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
import matplotlib.ticker as mticker
import textwrap
import statsmodels.formula.api as smf

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


conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF1/' + 'data'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

datain = filtered_file

XVar = 'carboncontent'      # there is a "biomass" column in the file

YMetric = 'density'

cluster = ['season', 'treatment']

base = 2

max_class = 14



action_name = "size_class"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]

    p = act.set_index("parameter")["value"]
    
    XVar = p.get("XVar", XVar)
    YMetric = p.get("YMetric", YMetric)
    base = float(p.get("base_log", base))
    max_class = int(p.get("max_class", max_class))
    
    if "cluster" in p:
        cluster = [x for x in p["cluster"].split(",")]

dataset = pd.read_csv(datain, sep=";", engine='python', decimal='.')

dataset.columns = (
    dataset.columns
    .str.replace('\ufeff', '', regex=False)  # rimuove BOM
    .str.strip()
    .str.lower()
)

required_cols = {XVar.lower()}
if YMetric == 'density':
    required_cols.add('density')
elif YMetric == 'organismquantity':
    required_cols.add('organismquantity')
elif YMetric == 'richness':
    required_cols.add('scientificname_accepted')
else:
    raise ValueError("YMetric deve essere 'density', 'organismquantity' o 'richness'.")

missing_req = [c for c in required_cols if c not in dataset.columns]
if missing_req:
    raise KeyError(f"The required columns are missing from the dataset: {missing_req}")

size_all = pd.to_numeric(dataset[XVar.lower()], errors='coerce')
valid_idx = size_all.dropna().index
dataset_valid = dataset.loc[valid_idx].copy()
size = size_all.loc[valid_idx].astype(float)

if YMetric == 'density':
    y_series = pd.to_numeric(dataset_valid['density'], errors='coerce').fillna(0).astype(float)
    y_label = 'Density'
    y_colname = 'density_sum'
elif YMetric == 'organismquantity':
    y_series = pd.to_numeric(dataset_valid['organismquantity'], errors='coerce').fillna(0).astype(float)
    y_label = 'OrganismQuantity'
    y_colname = 'organismQuantity_sum'
elif YMetric == 'richness':
    y_series = None  # gestito separatamente
    y_label = 'Richness'
    y_colname = 'richness_n_taxa'

if base in [2, 10]:
    xlabz = f"log{int(base)} {XVar}"
else:
    xlabz = f"ln {XVar}"


def sci_notation_tick(x, pos):
    """Returns strings like '1 × 10^5' directly on ticks."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exp)
    if abs(mantissa - round(mantissa)) < 1e-6:
        mantissa_str = f"{int(round(mantissa))}"
    else:
        mantissa_str = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    return rf"${mantissa_str} \times 10^{{{exp}}}$"


def compute_logbins_for_subset(sub_idx, mainz, xlb, subtitle, filegraph, base):
    """
    sub_idx: Index of the rows in the valid dataset belonging to the cluster
    mainz: main title of the graph (cluster name)
    xlb: X-axis label
    subtitle: subtitle (cluster info)
    filegraph: PdfPages object
    """

    size_sub = size.loc[sub_idx]

    logbin = np.round(np.log(size_sub) / np.log(base))

    df_bin = pd.DataFrame({'logbin': logbin})

    if YMetric in ['density', 'organismquantity']:
        df_bin['y'] = y_series.loc[sub_idx].values
        ttz = df_bin.groupby('logbin')['y'].sum().sort_index()
    elif YMetric == 'richness':
        taxa_sub = dataset_valid.loc[sub_idx, 'scientificname_accepted'].astype(str)
        df_bin['taxon'] = taxa_sub.values
        ttz = df_bin.groupby('logbin')['taxon'].nunique().sort_index()
    else:
        raise ValueError("YMetric non valido.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ttz.index, ttz.values)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation_tick))
    ax.get_yaxis().offsetText.set_visible(False)

    ax.set_xlabel(xlb)
    ax.set_ylabel(y_label)
    ax.set_title(mainz)

    if len(ttz.values) > 0:
        ax.set_ylim(0, max(ttz.values) * 1.1)

    ax.text(0, 1.1, subtitle, transform=ax.transAxes, fontsize=10)

    fig.savefig(filegraph, format='pdf')
    plt.close(fig)

    return ttz


if cluster[0].upper() == "WHOLE":

    subtitle = "Whole dataset"
    pdf_path = os.path.join(output_dir, f"SizeClassOutput_log{base}_{XVar}_{YMetric}.pdf")

    with PdfPages(pdf_path) as filegraph:
        ttz = compute_logbins_for_subset(
            sub_idx=dataset_valid.index,
            mainz="Whole dataset",
            xlb=xlabz,
            subtitle=subtitle,
            filegraph=filegraph,
            base=base
        )

    final = pd.DataFrame({
        f'log{base}_{XVar}': ttz.index,
        y_colname: ttz.values
    })

else:
    cluster_lower = [c.lower() for c in cluster]
    missing_cluster = [c for c in cluster_lower if c not in dataset_valid.columns]
    if missing_cluster:
        raise KeyError(f"The following cluster columns do not exist in the dataset: {missing_cluster}")

    if len(cluster_lower) > 1:
        dataset_valid['cluster_id'] = dataset_valid[cluster_lower].astype(str).agg('.'.join, axis=1)
        info = dataset_valid[cluster_lower].dropna().drop_duplicates().astype(str)
        info['cluster_id'] = info.astype(str).agg('.'.join, axis=1)
        info = info.set_index('cluster_id')
    else:
        cl = cluster_lower[0]
        dataset_valid['cluster_id'] = dataset_valid[cl].astype(str)
        info = dataset_valid[[cl]].drop_duplicates().astype(str)
        info['cluster_id'] = info[cl]
        info = info.set_index('cluster_id')

    subtitle = '\n'.join(textwrap.wrap(f'cluster: {", ".join(cluster_lower)}', width=50))
    pdf_path = os.path.join(output_dir, f"SizeClassOutput_log{base}_{XVar}_{YMetric}.pdf")

    with PdfPages(pdf_path) as filegraph:
        idz = dataset_valid['cluster_id'].astype(str).unique()
        cclist = {}
        for id_ in idz:
            sub_idx = dataset_valid.index[dataset_valid['cluster_id'].astype(str) == id_]
            ttz = compute_logbins_for_subset(
                sub_idx=sub_idx,
                mainz=id_,
                xlb=xlabz,
                subtitle=subtitle,
                filegraph=filegraph,
                base=base
            )
            cclist[id_] = ttz

    all_bins = sorted(set().union(*[ttz.index for ttz in cclist.values()]))
    data_rows = []
    for id_ in idz:
        row = pd.Series(index=all_bins, dtype=float)
        row.loc[cclist[id_].index] = cclist[id_].values
        data_rows.append(row)

    data_rbind = pd.DataFrame(data_rows)
    data_rbind.index = idz
    data_rbind = data_rbind.fillna(0).reset_index().rename(columns={'index': 'cluster_id'})

    info_reset = info.reset_index()
    final = pd.merge(info_reset, data_rbind, on='cluster_id', how='right')
    final = final.drop(columns=['cluster_id'], errors='ignore')
    final = final.fillna(0)

if base in [2, 10]:
    csv_path = os.path.join(output_dir, f"SizeClassOutput_log{base}_{XVar}_{YMetric}.csv")
else:
    csv_path = os.path.join(output_dir, f"SizeClassOutput_ln{XVar}_{YMetric}.csv")

final.to_csv(csv_path, sep=';', decimal='.', index=False, encoding='latin1')
print("Output file written to:", csv_path)
print("PDF written in:", pdf_path)





x_ticks = np.arange(1, max_class + 1)

df_panel = dataset_valid.copy()
df_panel = df_panel.dropna(subset=['season', 'treatment'])

season_order = ["Winter", "Spring", "Summer", "Autumn"]
unique_seasons = df_panel['season'].astype(str).unique()

seasons = [s for s in season_order if s in unique_seasons] + \
          sorted([s for s in unique_seasons if s not in season_order])

treatments = np.sort(df_panel['treatment'].astype(str).unique())

n_rows = len(treatments)
n_cols = len(seasons)

if n_rows == 0 or n_cols == 0:
    print("No valid data for season/treatment, no panel.")
else:
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharey=True
    )

    axes = np.atleast_2d(axes)

    fig.suptitle(f"{YMetric} vs log{base}({XVar}) – panel", fontsize=16)

    for i, tr in enumerate(treatments):
        for j, se in enumerate(seasons):
            ax = axes[i, j]

            sub_idx = df_panel.index[
                (df_panel['treatment'].astype(str) == tr) &
                (df_panel['season'].astype(str) == se)
            ]

            if len(sub_idx) == 0:
                ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                ax.set_title(f"{tr} – {se}", fontsize=10)
                continue

            size_sub = size.loc[sub_idx]
            size_sub = size_sub[size_sub > 0]

            if len(size_sub) == 0:
                ax.text(0.5, 0.5, "no valid size", ha='center', va='center', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                ax.set_title(f"{tr} – {se}", fontsize=10)
                continue

            logbin = np.round(np.log(size_sub) / np.log(base)).astype(int)

            logbin_clipped = np.minimum(logbin, max_class)

            if YMetric in ["density", "organismquantity"]:
                y_sub = y_series.loc[size_sub.index]
                df_b = pd.DataFrame({"logbin": logbin_clipped, "y": y_sub})
                ttz = df_b.groupby("logbin")["y"].sum().sort_index()
            else:  # richness
                taxa_sub = df_panel.loc[size_sub.index, "scientificname_accepted"].astype(str)
                df_b = pd.DataFrame({"logbin": logbin_clipped, "taxon": taxa_sub})
                ttz = df_b.groupby("logbin")["taxon"].nunique().sort_index()

            if len(ttz) == 0:
                ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                ax.set_title(f"{tr} – {se}", fontsize=10)
                continue

            ax.bar(ttz.index, ttz.values)
            ax.set_title(f"{tr} – {se}", fontsize=10)

            ax.grid(False)

            ax.set_xlim(0.5, max_class + 0.5)
            ax.set_xticks(x_ticks)

            if i == n_rows - 1:
                ax.set_xlabel(xlabz)
                labels = [str(x) if x < max_class else "≥14" for x in x_ticks]
                ax.set_xticklabels(labels)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation_tick))
            ax.get_yaxis().offsetText.set_visible(False)

            if j == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    svg_path = os.path.join(output_dir, f"Panel_{XVar}_{YMetric}.svg")
    png_path = os.path.join(output_dir, f"Panel_{XVar}_{YMetric}.png")

    plt.savefig(svg_path)
    plt.savefig(png_path, dpi=300)

    print("Panel saved in:")
    print("SVG:", svg_path)
    print("PNG:", png_path)







input_file = filtered_file
output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(input_file, sep=";", engine="python", decimal=".")

df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)

df = df.dropna(subset=[XVar, "scientificname_accepted", YMetric, "season", "treatment"])
df = df[df[XVar] > 0]


logbin = np.round(np.log(df[XVar]) / np.log(base)).astype(int)
df["size_class"] = np.minimum(logbin, max_class)


group_cols = ["season", "treatment", "size_class", "scientificname_accepted"]

df_grouped = (
    df.groupby(group_cols)[YMetric]
    .sum()
    .reset_index()
    .rename(columns={"scientificname_accepted": "species", YMetric: "density"})
)

totals = (
    df_grouped.groupby(["season", "treatment", "size_class"])["density"]
    .sum()
    .reset_index()
    .rename(columns={"density": "total_density"})
)

df_merged = df_grouped.merge(totals, on=["season", "treatment", "size_class"])
df_merged["percent"] = df_merged["density"] / df_merged["total_density"] * 100

df_merged = df_merged.sort_values(["season", "treatment", "size_class", "percent"], ascending=[1,1,1,0])


full_csv_path = os.path.join(output_dir, "SpeciesContributions_Full.csv")
df_merged.to_csv(full_csv_path, sep=";", index=False, encoding="utf-8")
print("CSV created: ", full_csv_path)


def top_species(group, n=5):
    return group.nlargest(n, "percent")

df_top = df_merged.groupby(["season", "treatment", "size_class"], group_keys=True).apply(top_species, n=5)
df_top = df_top.reset_index(drop=True)

top_csv_path = os.path.join(output_dir, "SpeciesContributions_TOP.csv")
df_top.to_csv(top_csv_path, sep=";", index=False, encoding="utf-8")
print("CSV ereated: ", top_csv_path)






output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

infile = csv_path #os.path.join(output_dir, "SizeClassOutput_log2_carboncontent_density.csv")

df = pd.read_csv(infile, sep=";", decimal=".")

bin_cols = [c for c in df.columns if c not in ["season", "treatment"]]
bin_vals = np.array([float(c) for c in bin_cols])  # log2 carbon-content classes

rows = []

for idx, row in df.iterrows():
    y = row[bin_cols].values.astype(float)

    mask = y > 0
    if mask.sum() < 2:
        continue

    x = bin_vals[mask]
    y_pos = y[mask]

    logy = np.log10(y_pos)
    X = sm.add_constant(x)
    model = sm.OLS(logy, X).fit()

    slope = model.params[1]
    intercept = model.params[0]
    r2 = model.rsquared
    p_slope = model.pvalues[1]

    sk = skew(y_pos, bias=False)
    ku = kurtosis(y_pos, fisher=False, bias=False)  # >3 = leptokurtic

    rows.append({
        "season": row["season"],
        "treatment": row["treatment"],
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "p_slope": p_slope,
        "n_bins_used": int(mask.sum()),
        "skewness": sk,
        "kurtosis": ku
    })

stats_df = pd.DataFrame(rows)

stats_path = os.path.join(output_dir, f"SizeSpectrum_STATS_log{base}_{XVar}_{YMetric}.csv")
stats_df.to_csv(stats_path, sep=";", decimal=".", index=False)
print("Statistics by season × treatment saved to:", stats_path)
print(stats_df)

stats_df["season"] = stats_df["season"].astype("category")
stats_df["treatment"] = stats_df["treatment"].astype("category")

model = smf.ols("slope ~ C(season) + C(treatment)", data=stats_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

anova_path = os.path.join(output_dir, f"SizeSpectrum_ANOVA_slope_log{base}_{XVar}_{YMetric}.csv")
anova_table.to_csv(anova_path, sep=";", decimal=".")
print("\nANOVA: slope ~ season + treatment")
print(anova_table)

try:
except ImportError:
    sns = None

if sns is not None:
    sns.set_theme(style="whitegrid")  # clean academic style

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        data=stats_df,
        x="season",
        y="slope",
        hue="treatment",
        palette="Set2",
        edgecolor="black",
        linewidth=1,
    )

    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.set_ylabel("Slope (log10 density vs log2 carbon content)", fontsize=11)
    ax.set_xlabel("Season", fontsize=11)
    ax.set_title("Size-spectrum slopes by season and treatment", fontsize=12)

    ax.legend(
        title="Treatment",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
        frameon=False,
    )

    plt.tight_layout()

    fig_path = os.path.join(output_dir, "SizeSpectrum_slopes_barplot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Slope plot saved to:", fig_path)

else:
    print("Seaborn is not installed. Skipping slope plot.")

