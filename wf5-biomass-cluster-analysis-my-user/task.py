from sklearn.cluster import KMeans
from scipy import stats
import fiona
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import geopandas as gpd
import seaborn as sns
from shapely.geometry import Point

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')

arg_parser.add_argument('--parameters_file_csv', action='store', type=str, required=True, dest='parameters_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

final_input = args.final_input.replace('"','')
parameters_file_csv = args.parameters_file_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

                  



print("Fiona:", fiona.__version__)
print("GDAL:", fiona.env.get_gdal_release_name())

output_dir = conf_output_path

include_y_in_features_when_x_exist = False  # True = X+Y; False = only X or y if no X is selected

n_clusters = 3  # The Low/Medium/High nomenclature applies only if = 3

STANDARDIZE = False

result_dir = os.path.join(output_dir, "cluster_result")
os.makedirs(result_dir, exist_ok=True)
print(f"All results will be saved in: {result_dir}")

shp_dir = os.path.join(result_dir, 'shapefile')
os.makedirs(shp_dir, exist_ok=True)

fig_dir = os.path.join(result_dir, 'scatter_clusters')
os.makedirs(fig_dir, exist_ok=True)

custom_na = ['NA', 'n.c', 'n.a.', '', 'NaN', 'nan', 'NULL', 'null']

param_file = parameters_file_csv 
param = pd.read_csv(param_file, sep=None, engine='python', encoding='latin-1', na_values=custom_na)
param.columns = param.columns.str.strip()

print(f"\nColumns {param_file}:", param.columns.tolist())

tags = param.iloc[1]
col_names = param.columns

x_var = [col for col, tag in zip(col_names, tags) if str(tag) == 'X']
y_var = [col for col, tag in zip(col_names, tags) if str(tag) == 'Y']
vars_factor = [col for col, tag in zip(col_names, tags) if str(tag) == 'F']

vars_factor = [col for col in vars_factor if col not in ('LATITUDE', 'LONGITUDE')]

print(f"\n--- Tag riga 2 {param_file} letti:")
print(tags)
print("\nFactors used after filter:", vars_factor)

if len(y_var) != 1:
    raise ValueError(f"Make sure there is EXACTLY one column tagged 'Y' in the file {param_file}.")

Y_NAME = y_var[0]
print(f'Abscissa (X): {x_var if len(x_var)>0 else "— (no X)"}')
print(f'Ordinate (Y): {Y_NAME}')
print(f'Comparison factors: {vars_factor}')

data_file = final_input
data = pd.read_csv(data_file, sep=None, engine='python', encoding='latin-1', na_values=custom_na)
data.columns = data.columns.str.strip()

print(f"\nColumns {data_file}:", data.columns.tolist())

possible_coords = ['LATITUDE', 'LONGITUDE']
present_coords = [col for col in possible_coords if col in data.columns]

needed_cols = present_coords + [Y_NAME] + vars_factor + x_var  # <<< Y always included in the dataset
if 'YEAR' not in needed_cols and 'YEAR' in data.columns:
    needed_cols.append('YEAR')

needed_cols = list(dict.fromkeys(needed_cols))
print("\nRequired columns:", needed_cols)

for col in needed_cols:
    if col not in data.columns:
        raise ValueError(f"The requested column '{col}' is not present in the data file!")

for col in needed_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

df = data[needed_cols].dropna().copy()
print("Selected columns (no duplicates):", df.columns.tolist())

if len(x_var) == 0:
    feature_cols = [Y_NAME]
    print("Clustering ONLY on the Y variable:", feature_cols)
else:
    if include_y_in_features_when_x_exist:
        feature_cols = x_var + [Y_NAME]
        print("Clustering on X + Y:", feature_cols)
    else:
        feature_cols = x_var
        print("Clustering ONLY on X:", feature_cols)


if STANDARDIZE:
    scaler = StandardScaler()
    X_mat = scaler.fit_transform(df[feature_cols])
else:
    X_mat = df[feature_cols].values

kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
df['cluster_num'] = kmeans.fit_predict(X_mat)

cluster_mean_y = df.groupby('cluster_num')[Y_NAME].mean()
sorted_cluster = cluster_mean_y.sort_values().index.tolist()

if n_clusters == 3:
    cluster_names_map = {sorted_cluster[0]: "Low",
                         sorted_cluster[1]: "Medium",
                         sorted_cluster[2]: "High"}
else:
    cluster_names_map = {sorted_cluster[i]: f"C{i+1}" for i in range(n_clusters)}

df['cluster'] = df['cluster_num'].map(cluster_names_map)
cluster_display_order = [cluster_names_map[i] for i in sorted_cluster]

rank_map = {cl: i+1 for i, cl in enumerate(sorted_cluster)}
df['cluster_rank_by_Y'] = df['cluster_num'].map(rank_map)

df['cluster_mean_Y'] = df['cluster_num'].map(cluster_mean_y.to_dict())

cluster_size = df.groupby('cluster_num').size().to_dict()
df['cluster_size'] = df['cluster_num'].map(cluster_size)

output_file = os.path.join(result_dir, "cluster_result.csv")
df_out = df.copy()
df_out.to_csv(output_file, index=False)
print(f"\nFile CSV salvato in: {output_file}")
print("Additional columns included: 'cluster_rank_by_Y', 'cluster_mean_Y', 'cluster_size'.")

if all(col in df_out.columns for col in ['LATITUDE', 'LONGITUDE']):
    df_out['LATITUDE'] = pd.to_numeric(df_out['LATITUDE'], errors='coerce')
    df_out['LONGITUDE'] = pd.to_numeric(df_out['LONGITUDE'], errors='coerce')
    df_out = df_out.dropna(subset=['LATITUDE', 'LONGITUDE'])

    gdf = gpd.GeoDataFrame(
        df_out,
        geometry=[Point(xy) for xy in zip(df_out['LONGITUDE'], df_out['LATITUDE'])],
        crs='EPSG:4326'
    )
    shp_file = os.path.join(shp_dir, 'cluster_results.shp')
    gdf.to_file(shp_file, engine="fiona")
    print(f"Shapefile saved in: {shp_file}")

all_stats = []
for factor in vars_factor:
    print(f"\nFactor analysis '{factor}':")
    stats = df_out.groupby('cluster')[factor].agg(['mean', 'std', 'count'])
    print(stats)
    stats['factor'] = factor
    stats['cluster'] = stats.index
    all_stats.append(stats.reset_index(drop=True))

    plt.figure(figsize=(7, 5))
    df_out.boxplot(column=factor, by='cluster', grid=False)
    plt.xlabel('Cluster')
    plt.ylabel(factor)
    plt.title(f'Distribution {factor} per cluster')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'boxplot_{factor}_cluster.png'))
    plt.close()

plt.figure(figsize=(7, 5))
df_out.boxplot(column=Y_NAME, by='cluster', grid=False)
plt.xlabel('Cluster')
plt.ylabel(Y_NAME)
plt.title(f'Distribution {Y_NAME} per cluster')
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, f'boxplot_{Y_NAME}_cluster.png'))
plt.close()


if len(feature_cols) >= 2:
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_mat)
    df_out['PC1'] = Z[:, 0]
    df_out['PC2'] = Z[:, 1]

    plt.figure(figsize=(8, 6))
    for cname in cluster_display_order:
        sub = df_out[df_out['cluster'] == cname]
        plt.scatter(sub['PC1'], sub['PC2'], label=f'Cluster {cname}', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusterization (2D PCA of features)')
    plt.legend()
    plt.tight_layout()
    plt.close()
    print("PCA figure saved in:", os.path.join(fig_dir, 'scatter_clusters_pca.png'))

if len(x_var) >= 1:
    x_first = x_var[0]
    plt.figure(figsize=(8, 6))
    for cname in cluster_display_order:
        sub = df_out[df_out['cluster'] == cname]
        plt.scatter(sub[x_first], sub[Y_NAME], label=f'Cluster {cname}', alpha=0.6)
    plt.xlabel(x_first)
    plt.ylabel(Y_NAME)
    plt.title('Clusterization (first X vs Y)')
    plt.legend()
    plt.tight_layout()
    plt.close()

    for xcol in x_var:
        plt.figure(figsize=(8, 6))
        for cname in cluster_display_order:
            sub = df_out[df_out['cluster'] == cname]
            plt.scatter(sub[xcol], sub[Y_NAME], label=f'Cluster {cname}', alpha=0.6)
        plt.xlabel(xcol)
        plt.ylabel(Y_NAME)
        plt.title(f'Clusterization ({xcol} vs {Y_NAME})')
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f'scatter_{xcol}_vs_{Y_NAME}.png')
        plt.savefig(out_path)
        plt.close()
    print(f"Scatter for every X saved in: {fig_dir}")
else:
    print("No X present: X vs Y scatter skipped.")

try:
    if len(x_var) >= 2:
        pair_cols = list(x_var)
        plot_df = df_out[pair_cols + [Y_NAME, 'cluster']].copy()
        g = sns.pairplot(
            data=plot_df,
            vars=pair_cols,
            hue='cluster',
            corner=True,
            diag_kind='hist'
        )
        g.fig.suptitle('Pairplot of Xs with hue=cluster', y=1.02)
        pairplot_path = os.path.join(fig_dir, 'pairplot_X_hue_cluster.png')
        plt.close(g.fig)
        print(f"Pairplot saved in: {pairplot_path}")
    else:
        print("Less than 2 X: pairplot not generated.")
except ImportError:
    print("Seaborn is not installed: pairplot skipped. Install with `pip install seaborn` to enable it.")
except Exception as e:
    print(f"Pairplot not generated due to an error: {e}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
