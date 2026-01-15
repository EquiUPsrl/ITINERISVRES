import pandas as pd
from sklearn.metrics import mean_absolute_error
import os
import numpy as np
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

def variable_comparison(data_file, params_file, trasf=False) :

    prefix = ""
    if trasf:
        prefix = "_trasf"

    data = pd.read_csv(data_file, delimiter=';')
    params = pd.read_csv(params_file, delimiter=';')

    target_var = params.loc[params['Parameter'] == 'Target variable', 'value'].values[0]
    if target_var not in data.columns:
        raise ValueError(f"The target variable '{target_var}' is not present in the data.")
    
    y = data[target_var]
    predictors = [col for col in data.columns if col != target_var]
    
    results = []
    for pred in predictors:
        y_pred = data[pred]
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results.append({
            'Predictor': pred,
            'MAE': mae,
            'RMSE': rmse
        })
        print(f"{pred}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    
    output_dir = os.path.join(conf_output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "mae_rmse_single_predictors" + prefix + ".txt")
    with open(output_file, 'w') as f:
        f.write("Predictor\tMAE\tRMSE\n")
        for row in results:
            f.write(f"{row['Predictor']}\t{row['MAE']:.4f}\t{row['RMSE']:.4f}\n")
    
    print(f"Results saved in: {output_file}")



variable_comparison(data_csv, parameters_file_csv)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
