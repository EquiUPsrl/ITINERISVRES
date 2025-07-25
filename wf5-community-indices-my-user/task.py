import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import log
from numpy import sqrt

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--output_file_sd', action='store', type=str, required=True, dest='output_file_sd')


args = arg_parser.parse_args()
print(args)

id = args.id

output_file_sd = args.output_file_sd.replace('"','')



def R(x): return len(x[x > 0])
def Shannon_H(x): 
    x = x[x > 0]
    p = x / x.sum()
    return -(p * np.log(p)).sum()
def Shannon_H_Eq(x): return Shannon_H(x) / log(len(x)) if len(x) > 0 else np.nan
def Simpson_D(x): return 1 - np.sum((x / x.sum())**2) if x.sum() > 0 else np.nan
def Simpson_D_Eq(x): return 1 / (np.sum((x / x.sum())**2)) if x.sum() > 0 else np.nan
def Menhinick_D(x): return len(x[x > 0]) / sqrt(x.sum()) if x.sum() > 0 else np.nan
def Margalef_D(x): return (len(x[x > 0]) - 1) / log(x.sum()) if x.sum() > 1 else np.nan
def Gleason_D(x): return (len(x[x > 0]) - 1) / log(10) if len(x[x > 0]) > 1 else np.nan
def McInthosh_M(x): return sqrt(np.sum(x**2)) / x.sum() if x.sum() > 0 else np.nan
def Hurlbert_PIE(x):
    N = x.sum()
    if N <= 1: return np.nan
    return (N / (N - 1)) * (1 - np.sum((x / N) ** 2))
def Pielou_J(x): 
    h = Shannon_H(x)
    s = len(x[x > 0])
    return h / log(s) if s > 1 else np.nan
def Sheldon_J(x): return x[x > 0].count() / len(x) if len(x) > 0 else np.nan
def LudwReyn_J(x): return (1 - Simpson_D(x)) / log(len(x)) if len(x) > 1 else np.nan
def BergerParker_B(x): return max(x) / x.sum() if x.sum() > 0 else np.nan
def McNaughton_Alpha(x): 
    if len(x[x > 0]) < 2: return np.nan
    return (x.sort_values(ascending=False).iloc[0] + x.sort_values(ascending=False).iloc[1]) / x.sum()
def Hulburt(x): return 1 - np.sum((x / x.sum())**2) if x.sum() > 0 else np.nan

def morisita_horn_index(a, b):
    a = np.array(a)
    b = np.array(b)
    Na, Nb = a.sum(), b.sum()
    if Na == 0 or Nb == 0: return np.nan
    Da = (a**2).sum() / Na**2
    Db = (b**2).sum() / Nb**2
    num = 2 * (a * b).sum() / (Na * Nb)
    denom = Da + Db
    return num / denom if denom > 0 else np.nan

def morisita_similarity_matrix(df, group_labels="Group"):
    labels = df.index.tolist()
    sim_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            sim = morisita_horn_index(df.iloc[i], df.iloc[j])
            sim_matrix.iloc[i, j] = sim
            sim_matrix.iloc[j, i] = sim
    sim_matrix.index.name = group_labels
    sim_matrix.columns.name = group_labels
    return sim_matrix

def plot_morisita_heatmap(sim_matrix, title="Morisita-Horn Similarity", save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"label": "Index values"})
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Heatmap salvata in: {save_path}")
    plt.close()

output_dir = '/tmp/data/output'

print("Creating folder " + output_dir)
os.makedirs(output_dir, exist_ok=True)

datain = output_file_sd

dataset = pd.read_csv(datain, sep=';', decimal='.')

cluster = ['country']
index_names = ["R", "Shannon_H", "Shannon_H_Eq", "Simpson_D", "Simpson_D_Eq", "Menhinick_D",
               "Margalef_D", "Gleason_D", "McInthosh_M", "Hurlbert_PIE",
               "Pielou_J", "Sheldon_J", "LudwReyn_J", "BergerParker_B",
               "McNaughton_Alpha", "Hulburt"]

index_functions = [R, Shannon_H, Shannon_H_Eq, Simpson_D, Simpson_D_Eq, Menhinick_D,
                   Margalef_D, Gleason_D, McInthosh_M, Hurlbert_PIE,
                   Pielou_J, Sheldon_J, LudwReyn_J, BergerParker_B,
                   McNaughton_Alpha, Hulburt]



dataset['Density'] = pd.to_numeric(dataset['Density'], errors='coerce')
dataset = dataset[dataset['Density'].notna() & (dataset['Density'] > 0)]

den_matz = dataset.groupby(['country', 'month', dataset['acceptedNameUsage']])['Density'].sum().unstack(fill_value=0)
den_matz = den_matz[den_matz.sum(axis=1) > 0]

if cluster[0] != "WHOLE":
    if len(cluster) > 1:
        ID = dataset[cluster].astype(str).apply('.'.join, axis=1)
        info = dataset[cluster].dropna().drop_duplicates().astype(str)
        info.index = info.apply('.'.join, axis=1)
    elif len(cluster) == 1:
        ID = dataset[cluster[0]].dropna().astype(str)
        info = dataset[cluster].drop_duplicates()
        info.set_index(info[cluster[0]], inplace=True)
else:
    ID = pd.Series(['all'] * len(dataset))

index_list = {}
if len(ID.unique()) > 1:
    filegraph = PdfPages(f"{output_dir}/Index_Graph_test.pdf")

for name, func in zip(index_names, index_functions):
    values = den_matz.apply(func, axis=1)
    index_list[name] = values

    if len(ID.unique()) > 1:
        valid = values.dropna()
        if len(valid) > 1:
            plt.figure(figsize=(10, 6))
            x_labels = ['.'.join(map(str, i)) if isinstance(i, tuple) else str(i) for i in valid.index]
            plt.bar(x_labels, valid.values)
            plt.title(name)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(filegraph, format='pdf')

sim_matrix = morisita_similarity_matrix(den_matz, group_labels="ID")
plot_morisita_heatmap(sim_matrix, title="Morisita-Horn Similarity", save_path=f"{output_dir}/Morisita_Horn_Heatmap.pdf")
sim_matrix.to_csv(f"{output_dir}/Morisita_Horn_Matrix.csv", sep=';', decimal='.')

ind = pd.concat(index_list, axis=1)
ind.to_csv(f"{output_dir}/Index_Output_test.csv", sep=';', decimal='.', index=True)
if len(ID.unique()) > 1:
    filegraph.close()

print("✅ WF5 completed!")

