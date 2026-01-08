from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--abiotic_file', action='store', type=str, required=True, dest='abiotic_file')

arg_parser.add_argument('--biotic_file', action='store', type=str, required=True, dest='biotic_file')


args = arg_parser.parse_args()
print(args)

id = args.id

abiotic_file = args.abiotic_file.replace('"','')
biotic_file = args.biotic_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/' + 'output'

output_dir = conf_output_path
pca_output_dir = os.path.join(output_dir, "PCA")
os.makedirs(pca_output_dir, exist_ok=True)

env_data = pd.read_csv(abiotic_file, sep=";")
phyto_data = pd.read_csv(biotic_file, sep=";")

phyto_data = phyto_data[["year", "density"]]

merged_data = pd.merge(env_data, phyto_data, on=["year"])
print(merged_data.columns)
merged_data = merged_data.dropna(axis=1, how="all")
print(merged_data.columns)

numeric_data = merged_data.select_dtypes(include=[np.number]).dropna(how='all').fillna(0)

print(numeric_data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
explained_var = pca.explained_variance_ratio_ * 100

individuals_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=individuals_df, x="PC1", y="PC2", s=50, palette="viridis")
plt.title(f"PCA - Individuals (PC1 {explained_var[0]:.1f}%, PC2 {explained_var[1]:.1f}%)")
plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pca_output_dir, "pca_ind_plot.png"), dpi=300)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
var_df = pd.DataFrame(loadings, index=numeric_data.columns, columns=["PC1", "PC2"])

plt.figure(figsize=(8, 6))
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
for i in var_df.index:
    plt.arrow(0, 0, var_df.loc[i, "PC1"], var_df.loc[i, "PC2"],
              color="red", alpha=0.7, head_width=0.03)
    plt.text(var_df.loc[i, "PC1"] * 1.1, var_df.loc[i, "PC2"] * 1.1, i,
             fontsize=8, ha='center', va='center')
plt.title("PCA - Variable Loadings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pca_output_dir, "pca_var_plot.png"), dpi=300)

plt.figure(figsize=(8, 6))
plt.scatter(individuals_df["PC1"], individuals_df["PC2"], alpha=0.5, label='Individuals')
for i in var_df.index:
    plt.arrow(0, 0, var_df.loc[i, "PC1"], var_df.loc[i, "PC2"],
              color="blue", alpha=0.5, head_width=0.03)
    plt.text(var_df.loc[i, "PC1"] * 1.1, var_df.loc[i, "PC2"] * 1.1, i,
             fontsize=8, color="blue", ha='center', va='center')
plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
plt.title("PCA Biplot")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pca_output_dir, "pca_biplot.png"), dpi=300)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
