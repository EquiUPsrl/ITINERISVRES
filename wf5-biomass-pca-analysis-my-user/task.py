import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

output_dir = conf_output_path


STANDARDIZE = True

pca_all_dir = os.path.join(output_dir, 'PCA')
os.makedirs(pca_all_dir, exist_ok=True)

input_path = final_input
param_path = parameters_file_csv

custom_na = ['NA', 'n.c', 'n.a.', '', 'NaN', 'nan', 'NULL', 'null']

df = pd.read_csv(input_path, engine='python', encoding='ISO-8859-1', na_values=custom_na, delimiter=';')
param_df = pd.read_csv(param_path, engine='python', encoding='ISO-8859-1', na_values=custom_na, delimiter=';')

roles = param_df.iloc[0]
target_col = roles[roles == 'Y'].index[0]
predictor_cols = roles[roles == 'X'].index.tolist()
level_cols = False #roles[roles == 'L'].index.tolist()

def save_variables_list(target, predictors, filepath):
    with open(filepath, 'w') as f:
        f.write(f"Target variable:\n{target}\n\n")
        f.write("Predictor variables:\n")
        for p in predictors:
            f.write(f"{p}\n")

def plot_pca_scatter(pca, pcs, target_values, folder, prefix, target_name):
    plt.figure(figsize=(10,8))
    var_exp = pca.explained_variance_ratio_
    xlabel = f'PC1 ({var_exp[0]*100:.1f}% variance explained)'
    ylabel = f'PC2 ({var_exp[1]*100:.1f}% variance explained)'

    scatter = plt.scatter(
        pcs[:,0], pcs[:,1],
        c=target_values,
        cmap='viridis',
        edgecolor='k',
        s=50,
        alpha=0.7
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'Sample distribution on first two principal components\n{prefix}', fontsize=16)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.grid(True, linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label(target_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{prefix}_PC1_vs_PC2.png'))
    plt.close()

def plot_loadings_correlation(loadings, folder, prefix):
    pcs_to_plot = ['PC1', 'PC2']
    for pc in pcs_to_plot:
        if pc in loadings.columns:
            plt.figure(figsize=(8, len(loadings)*0.4 + 1))
            sorted_loadings = loadings[pc].sort_values()
            ax = sorted_loadings.plot(kind='barh')
            ax.set_xlabel(f'Loading on {pc}')
            ax.set_title(f'Variable Loadings on {pc} - {prefix}')
            ax.axvline(0, color='black', linewidth=1)
            plt.tight_layout()
            plt.savefig(os.path.join(folder, f'{prefix}_loadings_{pc}.png'))
            plt.close()

def perform_pca(data, predictors, target, prefix='output', folder=output_dir, standardize=True):
    variable_predictors = [col for col in predictors if data[col].nunique() > 1]

    print("variable_predictors", variable_predictors)

    if len(variable_predictors) == 0:
        print(f"No variables with variation in {prefix}, PCA skipped.")
        return

    os.makedirs(folder, exist_ok=True)
    save_variables_list(target, variable_predictors, os.path.join(folder, f'{prefix}_variables_used.txt'))

    for col in variable_predictors + [target]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=variable_predictors + [target])

    X = data[variable_predictors]

    if standardize:
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X)
    else:
        X_transformed = X.values

    pca = PCA()
    pcs = pca.fit_transform(X_transformed)
    explained_variance = pca.explained_variance_ratio_

    loadings = pd.DataFrame(
        pca.components_.T,
        index=variable_predictors,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
    )

    pc_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    pc_df[target] = data[target].values

    corr = pc_df.corr()[target][:-1]

    ev_df = pd.DataFrame({
        'PCA': [f'PCA{i+1}' for i in range(len(explained_variance))],
        'Explained_Variance': explained_variance
    })
    ev_df.to_csv(os.path.join(folder, f'{prefix}_explained_variance.txt'), sep='\t', index=False)
    corr.to_csv(os.path.join(folder, f'{prefix}_correlation_with_target.txt'), sep='\t')
    loadings[['PC1', 'PC2']].to_csv(os.path.join(folder, f'{prefix}_loadings_PC1_PC2.txt'), sep='\t')

    plot_pca_scatter(pca, pcs, data[target].values, folder, prefix, target)
    plot_loadings_correlation(loadings, folder, prefix)

    print(f"Results and plots saved with prefix '{prefix}' in {folder} (standardize={standardize})")


print("### PCA on entire dataset ###")
perform_pca(df, predictor_cols, target_col, prefix='all_data', folder=pca_all_dir, standardize=STANDARDIZE)

if level_cols:
    level_col = level_cols[0]
    print(f"\n### PCA by levels of {level_col} ###")
    for lvl in df[level_col].dropna().unique():
        print(f"\nLevel: {lvl}")
        subset = df[df[level_col] == lvl].copy()  # Importante: .copy() per evitare SettingWithCopyWarning
        level_folder = os.path.join(pca_all_dir, f'level_{lvl}')
        perform_pca(subset, predictor_cols, target_col, prefix=f'level_{lvl}', folder=level_folder, standardize=STANDARDIZE)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
