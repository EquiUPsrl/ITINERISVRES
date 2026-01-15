from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from statsmodels.regression.linear_model import OLS
from sklearn.metrics import mean_squared_error

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_csv', action='store', type=str, required=True, dest='data_csv')

arg_parser.add_argument('--parameters_file_csv', action='store', type=str, required=True, dest='parameters_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

data_csv = args.data_csv.replace('"','')
parameters_file_csv = args.parameters_file_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1_0/' + 'output'

def safe_log(x):
    try:
        return np.log(float(x)) if float(x) > 0 else np.nan
    except:
        return np.nan

def linearRegression(dati_df, params_file, output_dir, transformations_dict, trasf=False):
    prefix = "_trasf" if trasf else ""

    SEED = 42
    np.random.seed(SEED)
    
    dati = dati_df.copy()
    
    params = pd.read_csv(params_file, delimiter=';')
    
    number = int(str(params.loc[params['Parameter'] == 'number', 'value'].values[0]).split()[0])
    
    target_var = params.loc[params['Parameter'] == 'Target variable', 'value'].values[0]
    
    transformations_dict["target_var"] = target_var
    transformations_dict["predictors"] = [c for c in dati.columns if c != target_var]
    transformations_dict["target_log"] = target_var in transformations_dict.get("log_features", [])
    
    print("\nDynamic Transformation Dictionary:")
    print(transformations_dict)

    if target_var not in dati.columns:
        raise ValueError(f"The target variable '{target_var}' is not present in the data!")
    X = dati.drop(columns=[target_var])
    y = dati[target_var]

    cv = KFold(n_splits=number, shuffle=True, random_state=SEED)
    
    r2_scores, rmse_scores, mae_scores = [], [], []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)
    
    final_model = LinearRegression()
    final_model.fit(X, y)
    
    output_lines = [
        "Linear regression model cross-validation results:",
        f"Mean R2: {mean_r2:.4f} (std: {std_r2:.4f})",
        f"Mean RMSE: {mean_rmse:.4f}",
        f"Mean MAE: {mean_mae:.4f}",
        "",
        f"Coefficients: {dict(zip(X.columns, final_model.coef_))}",
        f"Intercept: {final_model.intercept_:.4f}"
    ]
    
    for line in output_lines:
        print(line)
    
    output_base_dir = output_dir
    new_dir_lm = os.path.join(output_base_dir, "LinearModel")
    os.makedirs(new_dir_lm, exist_ok=True)
    
    output_path_lm = os.path.join(new_dir_lm, "risult_modell_lm" + prefix + ".txt")
    with open(output_path_lm, 'w') as f:
        f.write('\n'.join(output_lines))
    
    model_bundle = {
        "model": final_model,
        "transformations": transformations_dict
    }
    model_path_lm = os.path.join(new_dir_lm, "model_lm" + prefix + ".pkl")
    joblib.dump(model_bundle, model_path_lm)
    
    coeff_strs = [f"({coef:.4f} * {col})" for coef, col in zip(final_model.coef_, X.columns)]
    equation = " + ".join(coeff_strs)
    equation = f"y = {final_model.intercept_:.4f} + {equation}"
    
    output_lines.append("")
    output_lines.append("Linear regression equation:")
    output_lines.append(equation)
    with open(output_path_lm, 'a') as f:
        f.write('\n\nLinear regression equation:\n' + equation + '\n')
    
    print("\nLinear regression equation:")
    print(equation)

    X_const = add_constant(X)
    sm_model = OLS(y, X_const).fit()
    
    bp_test_stat, bp_pvalue, _, _ = het_breuschpagan(sm_model.resid, sm_model.model.exog)
    bp_output_file = os.path.join(new_dir_lm, "Breusch_Pagan_test_results" + prefix + ".txt")
    with open(bp_output_file, 'w') as f:
        f.write("Breusch-Pagan Test Results:\n")
        f.write(f"LM Statistic: {bp_test_stat:.4f}\n")
        f.write(f"p-value: {bp_pvalue:.4f}\n")
    print(f"Breusch-Pagan test results have been saved in: {bp_output_file}")
    
    shapiro_stat, shapiro_pvalue = shapiro(sm_model.resid)
    shapiro_output_file = os.path.join(new_dir_lm, "Shapiro_Wilk_test_results" + prefix + ".txt")
    with open(shapiro_output_file, 'w') as f:
        f.write("Shapiro-Wilk Test Results:\n")
        f.write(f"Test statistics: {shapiro_stat:.4f}\n")
        f.write(f"p-value: {shapiro_pvalue:.4f}\n")
    print(f"Shapiro-Wilk test results have been saved in: {shapiro_output_file}")


output_dir = conf_output_path

log_columns = []
with open(parameters_file_csv, encoding='utf-8') as f:
    for line in f:
        if line.strip().startswith("Log"):
            log_columns = line.strip().replace(';', ' ').split()[1:]
            break


transformations_dict = {
    "log_features": log_columns
}

df = pd.read_csv(data_csv, delimiter=';')

for col in log_columns:
    if col in df.columns:
        df[col] = df[col].map(safe_log)
    else:
        print(f"Column '{col}' not found in the data.")

df = df.dropna()

linearRegression(df, parameters_file_csv, output_dir, transformations_dict, trasf=True)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
