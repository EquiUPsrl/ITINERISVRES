import pandas as pd
import re
import os
import numpy as np
from kmodes.kprototypes import KPrototypes
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scikit_posthocs as sp
from scipy.stats import levene

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--abiotic_file', action='store', type=str, required=True, dest='abiotic_file')

arg_parser.add_argument('--biotic_file', action='store', type=str, required=True, dest='biotic_file')

arg_parser.add_argument('--input_traits', action='store', type=str, required=True, dest='input_traits')


args = arg_parser.parse_args()
print(args)

id = args.id

abiotic_file = args.abiotic_file.replace('"','')
biotic_file = args.biotic_file.replace('"','')
input_traits = args.input_traits.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/' + 'output'

bio_path = biotic_file 
traits_path = input_traits
env_path = abiotic_file

abundance_col = "density"  # colonna di densità

def clean_species(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s+(sp\.|spp\.|cf\.|aff\.).*", "", name)
    return name

phyto = pd.read_csv(bio_path, sep=";", encoding="utf-8", low_memory=False)
phyto.columns = phyto.columns.str.replace("ï»¿", "", regex=False).str.strip()

for col in ["locality", "locationID", "acceptedNameUsage"]:
    phyto[col] = phyto[col].astype(str).str.strip()

phyto["year"] = pd.to_numeric(phyto["year"], errors="coerce")
phyto[abundance_col] = pd.to_numeric(phyto[abundance_col], errors="coerce")
phyto = phyto.dropna(subset=["year", abundance_col])



env = None
if os.path.exists(env_path):
    env = pd.read_csv(env_path, sep=";", encoding="utf-8", low_memory=False)

    env["locality"] = env["locality"].astype(str).str.strip()
    env["locationID"] = env["locationID"].astype(str).str.strip()
    env["year"] = pd.to_numeric(env["year"], errors="coerce")
    env = env.dropna(subset=["year"])
    env["year"] = env["year"].astype(int)


traits = pd.read_csv(traits_path, sep=";", encoding="utf-8", low_memory=False)
traits.columns = traits.columns.str.replace("ï»¿", "", regex=False).str.strip()

traits["acceptedNameUsage"] = traits["acceptedNameUsage"].astype(str).str.strip()
traits["Species_clean"] = traits["acceptedNameUsage"].apply(clean_species)

traits_fun = traits.copy()

traits_fun["Motility"] = traits_fun["mobility"].astype(str).str.strip()
traits_fun["Motility"] = traits_fun["Motility"].replace({
    "non motile": "Non-motile",
    "non-motile": "Non-motile"
})
traits_fun["Motility"] = traits_fun["Motility"].apply(
    lambda x: "Motile" if str(x).lower().startswith("motile") else x
)

traits_fun["toxicity"] = pd.to_numeric(traits_fun["toxicity"], errors="coerce")
traits_fun["Toxin"] = traits_fun["toxicity"].replace({
    0: "Non-toxic",
    1: "Toxic"
}).astype(str)

traits_fun["Water_Trophy"] = traits_fun["trophicStrategy"].astype(str).str.strip()

traits_fun["Phytobs_Cell_Form"] = traits_fun["Shape"].astype(str).str.strip()

if "lifeForm" in traits_fun.columns:
    traits_fun["Life_Form"] = traits_fun["lifeForm"].astype(str).str.strip()
else:
    def derive_life_form(row):
        nc = row.get("nonColonial", np.nan)
        c  = row.get("colonial", np.nan)
        sc = row.get("solitary_and_colonial_possible", np.nan)
        if c == 1:
            if sc == 1:
                return "Solitary/Colonial"
            return "Colonial"
        if nc == 1 and (pd.isna(c) or c == 0):
            return "Non-colonial"
        if sc == 1:
            return "Solitary/Colonial"
        return "Unknown"

    for col in ["nonColonial", "colonial", "solitary_and_colonial_possible"]:
        if col in traits_fun.columns:
            traits_fun[col] = pd.to_numeric(traits_fun[col], errors="coerce")
    traits_fun["Life_Form"] = traits_fun.apply(derive_life_form, axis=1)

traits_cat = traits_fun[[
    "Species_clean",
    "Phytobs_Cell_Form",
    "Motility",
    "Life_Form",
    "Toxin",
    "Water_Trophy"
]].drop_duplicates()




lakes = phyto["locality"].unique()

for lake_name in lakes:
    lake_data = phyto[phyto["locality"] == lake_name].copy()
    lake_location_id = lake_data.iloc[0]["locationID"]

    out_dir = os.path.join(conf_output_path, lake_name + "_target_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    lake_data["Species_clean"] = lake_data["acceptedNameUsage"].apply(clean_species)
    
    print(f"Years available {lake_name} {lake_location_id}:", sorted(lake_data["year"].unique()))
    
    phyto_traits_trs = pd.merge(
        lake_data,
        traits_cat,
        on="Species_clean",
        how="left"
    )
    
    mask_missing = phyto_traits_trs[[
        "Phytobs_Cell_Form", "Motility", "Life_Form", "Toxin", "Water_Trophy"
    ]].isna().any(axis=1)
    
    missing_species = (
        phyto_traits_trs.loc[mask_missing, "acceptedNameUsage"]
        .dropna()
        .unique()
    )
    
    print(f"Total species {lake_name} {lake_location_id}: {phyto_traits_trs['acceptedNameUsage'].nunique()}")
    print(f"Species with at least one missing trait: {len(missing_species)}")
    
    pd.DataFrame({"Species_missing_traits_" + lake_name: missing_species}).to_csv(
        os.path.join(out_dir, f"species_missing_traits_{lake_name}.csv"),
        sep = ";",
        index=False
    )
    
    coverage_pct = round(
        (phyto_traits_trs["acceptedNameUsage"].nunique() - len(missing_species))
        / phyto_traits_trs["acceptedNameUsage"].nunique() * 100, 1
    )
    print(f"Coverage of functional traits ({lake_name}): {coverage_pct}%")
    
    phyto_traits_trs.to_csv(
        os.path.join(out_dir, f"phyto_{lake_name}_with_traits.csv"),
        sep = ";",
        index=False
    )
    
    if "Season" in phyto_traits_trs.columns:
        phyto_traits_trs["Sample_ID"] = (
            phyto_traits_trs["locality"].astype(str).str.strip()
            + "_" + phyto_traits_trs["Season"].astype(str).str.strip()
            + "_" + phyto_traits_trs["year"].astype(int).astype(str)
        )
    else:
        phyto_traits_trs["Sample_ID"] = (
            phyto_traits_trs["locality"].astype(str).str.strip()
            + "_" + phyto_traits_trs["year"].astype(int).astype(str)
        )
    
    group_cols = [
        "Sample_ID",
        "Species_clean",
        "acceptedNameUsage",
    ]
    
    for extra in ["phylum", "class"]:
        if extra in phyto_traits_trs.columns:
            group_cols.append(extra)
    
    group_cols += [
        "Motility",
        "Life_Form",
        "Phytobs_Cell_Form",
        "Toxin",
        "Water_Trophy"
    ]
    
    density_summary = (
        phyto_traits_trs
        .groupby(group_cols, dropna=False)
        .agg(Density=(abundance_col, "sum"))
        .reset_index()
    )

    env_vars = []
    if env is not None:
        
        env_trs = env[env["locality"] == lake_name].copy()
    
        if "Season" in env_trs.columns and "Season" in phyto_traits_trs.columns:
            env_trs["Sample_ID"] = (
                env_trs["locality"].astype(str).str.strip()
                + "_" + env_trs["Season"].astype(str).str.strip()
                + "_" + env_trs["year"].astype(int).astype(str)
            )
        else:
            env_trs["Sample_ID"] = (
                env_trs["locality"].astype(str).str.strip()
                + "_" + env_trs["year"].astype(int).astype(str)
            )
    
        exclude_cols = ["Sample_ID", "locality", "locationID", "year", "Season"]
        env_vars = [
            c for c in env_trs.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(env_trs[c])
        ]
    
        if env_vars:
            env_summary = (
                env_trs
                .groupby("Sample_ID", dropna=False)
                .agg({v: "mean" for v in env_vars})
                .reset_index()
            )
        else:
            env_summary = env_trs[["Sample_ID"]].drop_duplicates()
    else:
        print("Abiotic file not found, proceeding without environmental variables.")
        env_summary = pd.DataFrame({"Sample_ID": density_summary["Sample_ID"].unique()})
    
    classification_input_trs = pd.merge(
        density_summary,
        env_summary,
        on="Sample_ID",
        how="left"
    )
    
    classification_input_trs.to_csv(
        os.path.join(out_dir, f"classification_input_{lake_name}_traits_env.csv"),
        index=False
    )
    
    df = classification_input_trs.copy()
    
    categorical_cols = [
        "Motility",
        "Life_Form",
        "Phytobs_Cell_Form",
        "Toxin",
        "Water_Trophy"
    ]
    
    for c in categorical_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "NaN": np.nan, "": np.nan})
        df[c] = df[c].fillna("Unknown")
    
    env_vars = [v for v in env_vars if v in df.columns]
    
    df["Density_log10"] = np.log10(df["Density"].replace(0, np.nan))
    df["Density_log10"] = df["Density_log10"].fillna(df["Density_log10"].median())
    
    num_cols = ["Density_log10"] + env_vars
    
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    for c in list(num_cols):
        if df[c].notna().any():
            df[c] = df[c].fillna(df[c].median())
        else:
            df.drop(columns=[c], inplace=True)
            num_cols.remove(c)
    
    drop_for_clust = [
        "Species_clean",
        "acceptedNameUsage",
        "Sample_ID",
        "phylum",
        "class",
        "locality",
        "locationID",
        "year",
        "Density"   
    ]
    df_clust = df.drop(columns=[c for c in drop_for_clust if c in df.columns])
    
    df_clust = df_clust.drop_duplicates()
    
    categorical_cols = [c for c in categorical_cols if c in df_clust.columns]
    num_cols = [c for c in df_clust.columns if c not in categorical_cols]
    
    for c in categorical_cols:
        df_clust[c] = df_clust[c].astype(str).str.strip()
        df_clust[c] = df_clust[c].replace({"nan": "Unknown", "": "Unknown"})
    
    for c in list(num_cols):
        df_clust[c] = pd.to_numeric(df_clust[c], errors="coerce")
        if df_clust[c].notna().any():
            df_clust[c] = df_clust[c].fillna(df_clust[c].median())
        else:
            df_clust.drop(columns=[c], inplace=True)
            num_cols.remove(c)
    
    df_clust = df_clust.dropna()
    
    if df_clust.shape[0] < 2:
        print(f"Few records for K-Prototypes on {lake_name}: clustering not performed.")
    else:
        cat_indices = [df_clust.columns.get_loc(col) for col in categorical_cols]
        n_points = df_clust.shape[0]
        k = min(4, max(2, n_points - 1))
    
        print(f"\nRun K-Prototypes ({lake_name}) with k = {k}, n = {n_points}")
        print("Categorical columns:", categorical_cols)
        print("Numeric columns:", num_cols)
    
        kproto = KPrototypes(n_clusters=k, init="Huang", n_init=5, verbose=1)
        clusters = kproto.fit_predict(df_clust.to_numpy(), categorical=cat_indices)
    
        df_result = df.loc[df_clust.index].copy()
        df_result["Cluster"] = clusters + 1
        cluster_labels = {i + 1: chr(65 + i) for i in range(k)}
        df_result["Cluster_Label"] = df_result["Cluster"].map(cluster_labels)
    
        out_clusters = os.path.join(out_dir, f"classification_with_clusters_{lake_name}.csv")
        df_result.to_csv(out_clusters, index=False)
        print(f"Clustering completed, saved to: {out_clusters}")
    
        species_cluster_summary = (
            df_result
            .groupby(["Cluster_Label", "acceptedNameUsage"], as_index=False)
            .agg(
                n_samples=("Sample_ID", "nunique"),
                mean_density=("Density", "mean")
            )
            .sort_values(["Cluster_Label", "n_samples"], ascending=[True, False])
        )
    
        species_cluster_path = os.path.join(
            out_dir, f"species_by_cluster_{lake_name}.csv"
        )
        species_cluster_summary.to_csv(species_cluster_path, index=False)
        print(f"Species per cluster saved in: {species_cluster_path}")
    
        print("\nExample of species aggregation by cluster:")
        print(species_cluster_summary.head(20))
    
        if "class" in df_result.columns:
            tax_col = "class"
        elif "phylum" in df_result.columns:
            tax_col = "phylum"
        else:
            tax_col = None
    
        if tax_col is None:
            print("No 'class' or 'phylum' columns found: Sankey skip.")
        else:
            df_sankey_taxa = (
                df_result[[tax_col, "Cluster_Label"]]
                .dropna()
                .groupby([tax_col, "Cluster_Label"])
                .size()
                .reset_index(name="Freq")
            )
    
            top_n = 15
            top_taxa = (
                df_sankey_taxa.groupby(tax_col)["Freq"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .index
            )
            df_sankey_taxa = df_sankey_taxa[df_sankey_taxa[tax_col].isin(top_taxa)]
    
            label_taxa     = list(df_sankey_taxa[tax_col].unique())
            label_clusters = list(df_sankey_taxa["Cluster_Label"].unique())
            labels = label_taxa + label_clusters
    
            df_sankey_taxa["source"] = df_sankey_taxa[tax_col].apply(lambda x: labels.index(x))
            df_sankey_taxa["target"] = df_sankey_taxa["Cluster_Label"].apply(lambda x: labels.index(x))
    
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=labels),
                link=dict(
                    source=df_sankey_taxa["source"],
                    target=df_sankey_taxa["target"],
                    value=df_sankey_taxa["Freq"]
                )
            )])
    
            fig.update_layout(
                title_text=f"{tax_col.capitalize()} → Functional cluster ({lake_name} {lake_location_id})",
                font_size=10
            )
    
            sankey_png = os.path.join(out_dir, f"sankey_{tax_col}_cluster_{lake_name}.png")
            try:
                fig.write_image(sankey_png, scale=2)  # richiede 'kaleido'
                print(f"Sankey {tax_col}-cluster saved in PNG: {sankey_png}")
            except Exception as e:
                print(f"Unable to save Sankey to PNG (requires 'kaleido'). Error: {e}")
                sankey_html = os.path.join(out_dir, f"sankey_{tax_col}_cluster_{lake_name}.html")
                fig.write_html(sankey_html)
                print(f"Sankey {tax_col}-cluster saved in HTML: {sankey_html}")
    


        
            "waterTemperature",
            "transparency",
            "totalPhosphorous",
            "totalNitrogen",
            "pH",
            "alcalinity",
            "ammonium",
            "dissolvedOxygen"
        ]
    
        vars_env = [v for v in important_env_vars if v in df_result.columns]
    
        vars_env += ["Density"]
    
        vars_env = list(dict.fromkeys(vars_env))
    
        vars_env_clean = []
        for var in vars_env:
            serie = df_result[var]
            if serie.notna().sum() == 0:
                continue
            if np.allclose(serie.dropna(), serie.dropna().iloc[0]):
                continue
            vars_env_clean.append(var)
    
        if vars_env_clean:
            print(f"\nEnvironmental analyzes by cluster ({lake_name})...")
            print("Variables used for boxplots:", vars_env_clean)
    
            df_long = df_result.melt(
                id_vars=["Cluster_Label"],
                value_vars=vars_env_clean,
                var_name="Variable",
                value_name="Value"
            )
    
            cluster_order = [c for c in ["A", "B", "C", "D"]
                             if c in df_result["Cluster_Label"].unique()]
    
            g = sns.catplot(
                data=df_long,
                x="Cluster_Label",
                y="Value",
                col="Variable",
                kind="box",
                order=cluster_order,
                col_wrap=4,
                sharey=False,
                height=4
            )
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(
                f"Environmental variables by functional cluster ({lake_name} {lake_location_id})",
                fontsize=14
            )
            box_path = os.path.join(out_dir, f"boxplot_cluster_{lake_name}.tiff")
            plt.savefig(box_path, dpi=300)
            plt.close()
            print(f"Boxplot saved in: {box_path}")
    
            kruskal_results = []
            for var in vars_env_clean:
                tmp = df_long[df_long["Variable"] == var]
    
                groups = [
                    grp["Value"].dropna().values
                    for _, grp in tmp.groupby("Cluster_Label")
                ]
    
                groups = [g for g in groups if len(g) > 0]
    
                if len(groups) < 2:
                    continue
    
                all_vals = np.concatenate(groups)
                if np.allclose(all_vals, all_vals[0]):
                    print(f"Kruskal skipped for {var}: all identical values ​​in clusters.")
                    continue
    
                try:
                    stat, pval = kruskal(*groups)
                    kruskal_results.append({"Variable": var, "p_value": pval})
                except ValueError as e:
                    print(f"Kruskal not calculable for {var}: {e}")
                    continue
    
            df_kruskal = pd.DataFrame(kruskal_results)
            df_kruskal.to_csv(
                os.path.join(out_dir, f"kruskal_test_results_{lake_name}.csv"),
                index=False
            )
    
            sign_vars = df_kruskal[df_kruskal["p_value"] < 0.05]["Variable"]
            dunn_all = []
            for var in sign_vars:
                data_temp = df_result[["Cluster_Label", var]].dropna()
                if data_temp["Cluster_Label"].nunique() > 1:
                    dunn = sp.posthoc_dunn(
                        data_temp,
                        val_col=var,
                        group_col="Cluster_Label",
                        p_adjust="bonferroni"
                    )
                    dunn["Variable"] = var
                    dunn_all.append(dunn)
    
            if dunn_all:
                df_dunn = pd.concat(dunn_all)
                df_dunn.to_csv(
                    os.path.join(out_dir, f"dunn_posthoc_results_{lake_name}.csv"),
                    index=False
                )
    
            for var in ["TP", "TN", "DIN",
                        "totalNitrogen", "totalPhosphorous",
                        "orthophosphate", "waterTemperature",
                        "alcalinity", "pH", "ammonium", "dissolvedOxygen"]:
                if var in df_result.columns:
                    groups = [
                        df_result[df_result["Cluster_Label"] == g][var].dropna()
                        for g in df_result["Cluster_Label"].unique()
                    ]
                    groups = [gr for gr in groups if len(gr) > 0]
                    if len(groups) > 1:
                        stat, pval = levene(
                            *groups,
                            center="trimmed",
                            proportiontocut=0.1
                        )
                        print(f"Levene - {var}: p-value = {pval:.5f}")
        else:
            print(f"No useful environmental variables available for boxplots ({lake_name}).")

output_dir = conf_output_path

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
