import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filtered_input', action='store', type=str, required=True, dest='filtered_input')

arg_parser.add_argument('--parameters_csv', action='store', type=str, required=True, dest='parameters_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

filtered_input = args.filtered_input.replace('"','')
parameters_csv = args.parameters_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

data_path = filtered_input
param_path = parameters_csv
output_dir = conf_output_path

desc_dir = os.path.join(output_dir, "descriptive_plots")
os.makedirs(desc_dir, exist_ok=True)

def read_csv_auto_sep(filepath, guess_rows=5):
    possible_seps = [',', ';', '\t', '|']
    best_cols = 0
    best_sep = ','
    for sep in possible_seps:
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=guess_rows)
            if df.shape[1] > best_cols:
                best_cols = df.shape[1]
                best_sep = sep
        except Exception:
            continue
    return pd.read_csv(filepath, sep=best_sep)

df = read_csv_auto_sep(data_path)
param = read_csv_auto_sep(param_path)

param_row = param.iloc[2]  # third row
columns = param.columns
Y_vars = [col for col in columns if str(param_row[col]).strip().upper() == "Y"]
X_vars = [col for col in columns if str(param_row[col]).strip().upper() == "X"]
quant_vars = X_vars + Y_vars
group_cols = [col for col in columns if str(param_row[col]).strip().lower() == "f"]

if group_cols:
    desc = df.groupby(group_cols)[quant_vars].agg(['mean', 'std', 'min', 'max', 'count'])
    desc.to_csv(os.path.join(desc_dir, f"descriptive_statistics.csv"))
    print(desc)

for var in quant_vars:
    for group_col in group_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_col, y=var, data=df)
        plt.title(f"Boxplot of {var} by {group_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(desc_dir, f"boxplot_{var}_by_{group_col}.png"))
        plt.close()

equations = []
for group_col in group_cols:
    for x in X_vars:
        y = Y_vars[0]

        X_data = df[[x]].dropna()
        y_data = df.loc[X_data.index, y]

        model = LinearRegression()
        model.fit(X_data, y_data)

        slope = model.coef_[0]
        intercept = model.intercept_

        equation = f"{y} = {intercept:.4f} + {slope:.4f} * {x} (group: {group_col})"
        equations.append(equation)
        
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=x, y=y, hue=group_col, data=df, s=70)
        sns.regplot(x=x, y=y, data=df, scatter=False, color='black')
        plt.title(f"Scatterplot and regression {y} vs {x}")
        plt.tight_layout()
        plt.savefig(os.path.join(desc_dir, f"scatter_reg_{y}_vs_{x}_by_{group_col}.png"))
        plt.close()

print(f"All results and plots have been saved in: {desc_dir}")

output_path = os.path.join(desc_dir, "regression_equations.txt")
with open(output_path, "w") as f:
    for eq in equations:
        f.write(eq + "\n")

print(f"Equations saved in: {output_path}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
