import pandas as pd
import os
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_file_csv', action='store', type=str, required=True, dest='data_file_csv')

arg_parser.add_argument('--parameters_file_csv', action='store', type=str, required=True, dest='parameters_file_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

data_file_csv = args.data_file_csv.replace('"','')
parameters_file_csv = args.parameters_file_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1_1/work/' + 'output'

df = pd.read_csv(data_file_csv, delimiter=';')

params = pd.read_csv(parameters_file_csv, delimiter=';')

def get_param(param):
    return params.loc[params['Parameter'] == param, 'value'].values[0]

log_columns = []
with open(parameters_file_csv, encoding='utf-8') as f:
    for line in f:
        if line.strip().startswith("Log"):
            log_columns = line.strip().replace(';', ' ').split()[1:]
            break

print("Columns to log: ", log_columns)

target_var = get_param('Target variable')

def safe_log(x):
    try:
        return np.log(float(x)) if float(x) > 0 else np.nan
    except:
        return np.nan

for col in log_columns:
    if col in df.columns:
        df[col] = df[col].map(safe_log)
    else:
        print(f"Column '{col}' not found in the data.")

df = df.dropna()

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

transformations_dict = {
    "log_features": log_columns,
    "target_var": target_var,
    "predictors": [c for c in df.columns if c != target_var],
    "target_log": target_var in log_columns   # <-- AUTOMATICO!
}

print("\nDynamic Transformation Dictionary:")
print(transformations_dict)


def randomForest(dati_df, params_file, output_dir, transformations_dict, trasf=False):
    prefix = "_trasf" if trasf else ""

    SEED = 42
    np.random.seed(SEED)
    
    dati = dati_df.copy()
    
    params = pd.read_csv(params_file, delimiter=';')
    def get_param(param):
        row = params.loc[params['Parameter'] == param].iloc[0]  # prendi la riga
        values = row.iloc[1:]                                     # escludi la prima colonna
        values = values.dropna()                                  # elimina eventuali NaN
        return values.tolist() 

    def to_int_list(lst):
        """Converts only the numeric values in the list to integers, ignoring the others."""
        result = []
        for v in lst:
            try:
                result.append(int(v))
            except (ValueError, TypeError):
                continue  # ignore non-convertible values
        return result

    number = int(get_param('number')[0])  # It only takes the first value
    ntree_list = to_int_list(get_param('ntree'))
    mtry_list = to_int_list(get_param('mtry'))
    target_var = get_param('Target variable')[0]

    print("ntree_list", ntree_list)

    if target_var not in dati.columns:
        raise ValueError(f"The target variable '{target_var}' is not present in the data!")

    X = dati.drop(columns=[target_var])
    y = dati[target_var]

    transformations_dict["predictors"] = list(X.columns)
    transformations_dict["target_var"] = target_var

    cv = KFold(n_splits=number, shuffle=True, random_state=SEED)
    
    results = []
    best_score = -np.inf
    best_model = None
    best_params = None

    param_combos = [(ntree, mtry) for ntree in ntree_list for mtry in mtry_list]
    for ntree, mtry in param_combos:
        model = RandomForestRegressor(
            n_estimators=ntree,
            max_features=mtry,
            random_state=SEED,
            n_jobs=-1
        )
        r2_scores, rmse_scores, mae_scores = [], [], []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_rmse = np.mean(rmse_scores)
        mean_mae = np.mean(mae_scores)
        results.append({
            'ntree': ntree,
            'mtry': mtry,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_rmse': mean_rmse,
            'mean_mae': mean_mae
        })
        if mean_r2 > best_score:
            best_score = mean_r2
            best_model = model
            best_params = {'ntree': ntree, 'mtry': mtry}
    
    best_model.fit(X, y)
    
    output_lines = []
    output_lines.append('Grid search results (mean CV metrics):')
    for res in results:
        output_lines.append(
            f"ntree: {res['ntree']}, mtry: {res['mtry']}, "
            f"mean R2: {res['mean_r2']:.4f} (std: {res['std_r2']:.4f}), "
            f"mean RMSE: {res['mean_rmse']:.4f}, mean MAE: {res['mean_mae']:.4f}"
        )
    output_lines.append(f"\nBest model parameters: {best_params}")
    output_lines.append(f"Best mean R2: {best_score:.4f}")
    for line in output_lines:
        print(line)
    
    output_base_dir = output_dir
    new_dir = os.path.join(output_base_dir, "RandomForest_Model")
    os.makedirs(new_dir, exist_ok=True)
    
    output_path = os.path.join(new_dir, "result_modell_rf" + prefix + ".txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    model_bundle = {
        "model": best_model,
        "transformations": transformations_dict
    }
    model_path = os.path.join(new_dir, "random_forest_model" + prefix + ".pkl")
    joblib.dump(model_bundle, model_path)

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(new_dir, "grid_search_results" + prefix + ".csv")
    results_df.to_csv(results_csv_path, index=False)

    return model_path

model_path = randomForest(df, parameters_file_csv, output_dir, transformations_dict, trasf=True)

file_model_path = open("/tmp/model_path_" + id + ".json", "w")
file_model_path.write(json.dumps(model_path))
file_model_path.close()
file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
