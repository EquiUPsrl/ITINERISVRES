import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')


args = arg_parser.parse_args()
print(args)

id = args.id

final_input = args.final_input.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF6/' + 'data'

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True


output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)
print(f"Folder '{output_dir}' ready.")

csv_path = final_input
event_df = pd.read_csv(csv_path, low_memory=False)

cutoff_year = 2018
selected_features = ['waterTemperature', 'conductivity', 'totalNitrogen', 'totalPhosphorous', 'transparency', 'dissolvedOxygen']


action_name = "machine_learning"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    
    value = p.get("selected_features")
    cutoff_year = int(p.get("cutoff_year"))

    if isinstance(value, str) and value.strip():
        selected_features = [x.strip() for x in value.split(",")]


print("cutoff_year:", cutoff_year)
print("selected_features:", selected_features)


def prepare_ml_df(df, abiotic_cols=None, log_transform=False, single_lake=None, selected_features=None):
    """
    Prepare the DataFrame for ML: adds lag, time features, imputation, and one-hot encoding.
    Returns:
        ml_df: preprocessed DataFrame
        feature_cols: list of feature column names
        target_col: name of the target column
    """

    df['parsed_date'] = pd.to_datetime(
        df['parsed_date'],
        errors='coerce'
    )
    
    ml_df = df.copy()
    
    ml_df['year'] = ml_df['parsed_date'].dt.year
    ml_df['month'] = ml_df['parsed_date'].dt.month

    if single_lake:
        ml_df = ml_df[ml_df['locality'] == single_lake]

    target_col = 'density_log' if log_transform else 'density'
    ml_df[target_col] = np.log10(ml_df['density'] + 1) if log_transform else ml_df['density']

    lag_col = target_col + '_lag1'
    ml_df = ml_df.sort_values(['locality', 'parsed_date'])
    ml_df[lag_col] = ml_df.groupby('locality')[target_col].shift(1)
    ml_df = ml_df.dropna(subset=[lag_col])

    if abiotic_cols:
        for col in abiotic_cols:
            if col in ml_df.columns:
                ml_df[col] = ml_df.groupby('locality')[col].transform(lambda x: x.fillna(x.mean()))
                ml_df[col] = ml_df[col].fillna(ml_df[col].mean())
        print(f"Total NA in abiotic variables after imputation: {ml_df[abiotic_cols].isna().sum().sum()}")

    if selected_features:
        feature_cols = selected_features + ['year', 'month', lag_col]
    else:
        feature_cols = (abiotic_cols if abiotic_cols else []) + ['year', 'month', lag_col]

    if not single_lake:
        ml_df = pd.get_dummies(ml_df, columns=['locality'], drop_first=True)
        feature_cols += [c for c in ml_df.columns if c.startswith('locality_')]

    return ml_df, feature_cols, target_col


def plot_and_save_shap(shap_dir, model_name, shap_values, X_for_plot, feature_names):
    """Create and save SHAP summary plot and mean absolute SHAP barplot."""

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)


    try:
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_for_plot,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"{model_name} - SHAP summary ({selected_lake})")
        plt.tight_layout()
        plt.savefig(f"{shap_dir}/{model_name}_SHAP_summary_{selected_lake}.png", dpi=300)
        plt.savefig(f"{shap_dir}/{model_name}_SHAP_summary_{selected_lake}.svg")
        plt.close()
    except Exception as e:
        print(f"WARNING: SHAP summary plot failed for {model_name}: {e}")

    plt.figure(figsize=(6, 5))
    plt.barh(importance.index, importance.values)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"{model_name} - SHAP feature importance ({selected_lake})")
    plt.tight_layout()
    plt.savefig(f"{shap_dir}/{model_name}_SHAP_barplot_{selected_lake}.png", dpi=300)
    plt.savefig(f"{shap_dir}/{model_name}_SHAP_barplot_{selected_lake}.svg")
    plt.close()

    print(f"{model_name}: SHAP importance saved")


lakes = list(event_df['locality'].unique()) + ["All lakes"]

for selected_lake in lakes:

    print("Selected lake: " + selected_lake)

    lake_dir = os.path.join(output_dir, f"Machine_Learning/{selected_lake}")
    os.makedirs(lake_dir, exist_ok=True)

    ml_df, features, target_col = prepare_ml_df(
        event_df,
        log_transform=True,
        single_lake=None if selected_lake=="All lakes" else selected_lake,
        selected_features=selected_features
    )
    
    X = ml_df[features].replace([np.inf, -np.inf], np.nan).fillna(ml_df[features].mean())
    y = ml_df[target_col]
    
    train_mask = ml_df['parsed_date'].dt.year < cutoff_year
    test_mask = ~train_mask  # faster
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Train samples: {X_train.shape[0]} - years until {cutoff_year-1}")
    print(f"Test samples: {X_test.shape[0]} - years from {cutoff_year}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lin_model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred_lin = lin_model.predict(X_test_scaled)
    
    plt.figure()
    plt.scatter(y_test, y_pred_lin, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Observed density")
    plt.ylabel("Predicted density")
    plt.title(f"Linear Regression - Test set ({selected_lake})")
    plt.tight_layout()
    plt.savefig(os.path.join(lake_dir, "linear_regression_testset.png"), dpi=300)
    plt.savefig(os.path.join(lake_dir, "linear_regression_testset.svg"))
    plt.show()
    
    print(f"=== Linear Regression ({selected_lake}) ===")
    print(f"R2 (train): {lin_model.score(X_train_scaled, y_train):.4f}")
    print(f"R2 (test) : {lin_model.score(X_test_scaled, y_test):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_lin):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lin)):.4f}")
    
    ridge_alphas = [0.1, 0.5, 0.7, 1.0, 5.0, 10.0]
    ridge_cv_model = RidgeCV(alphas=ridge_alphas, scoring='r2').fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_cv_model.predict(X_test_scaled)
    
    plt.figure()
    plt.scatter(y_test, y_pred_ridge, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Observed density")
    plt.ylabel("Predicted density")
    plt.title(f"Ridge Regression - Test set ({selected_lake})")
    plt.tight_layout()
    plt.savefig(os.path.join(lake_dir, "ridge_regression_testset.png"), dpi=300)
    plt.savefig(os.path.join(lake_dir, "ridge_regression_testset.svg"))
    plt.show()
    
    print(f"\n=== Ridge Regression ({selected_lake}) ===")
    print(f"Best alpha: {ridge_cv_model.alpha_}")
    print(f"R2 train: {ridge_cv_model.score(X_train_scaled, y_train):.4f}")
    print(f"R2 test : {ridge_cv_model.score(X_test_scaled, y_test):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_ridge):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")
    
    param_grid_svr = {'C':[0.1,1,5,10,20],'epsilon':[0.01,0.1,1,5],'kernel':['rbf']}
    grid_svr = GridSearchCV(SVR(), param_grid_svr, cv=3, n_jobs=-1, scoring='r2')
    grid_svr.fit(X_train_scaled, y_train)
    y_pred_svr = grid_svr.predict(X_test_scaled)
    
    plt.figure()
    plt.scatter(y_test, y_pred_svr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Observed density")
    plt.ylabel("Predicted density")
    plt.title(f"SVR - Test set ({selected_lake})")
    plt.tight_layout()
    plt.savefig(os.path.join(lake_dir, "svr_testset.png"), dpi=300)
    plt.savefig(os.path.join(lake_dir, "svr_testset.svg"))
    plt.show()
    
    print(f"Best SVR parameters ({selected_lake}):", grid_svr.best_params_)
    print(f"R2 train: {grid_svr.score(X_train_scaled, y_train):.4f}")
    print(f"R2 test : {grid_svr.score(X_test_scaled, y_test):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.4f}")
    
    param_grid_rf = {
        'n_estimators':[100,200],
        'max_depth':[5,10,None],
        'max_features':['sqrt','log2',None]
    }
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, n_jobs=-1, scoring='r2')
    grid_rf.fit(X_train, y_train)
    y_pred_rf = grid_rf.predict(X_test)
    
    plt.figure()
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Observed density")
    plt.ylabel("Predicted density")
    plt.title(f"Random Forest - Test set ({selected_lake})")
    plt.tight_layout()
    plt.savefig(os.path.join(lake_dir, "random_forest_testset.png"), dpi=300)
    plt.savefig(os.path.join(lake_dir, "random_forest_testset.svg"))
    plt.show()
    
    print(f"Best RF parameters ({selected_lake}):", grid_rf.best_params_)
    print(f"R2 train: {grid_rf.score(X_train, y_train):.4f}")
    print(f"R2 test : {grid_rf.score(X_test, y_test):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
    
    results = [
        {"Model":"Linear Regression","R2_train":lin_model.score(X_train_scaled,y_train),
         "R2_test":lin_model.score(X_test_scaled,y_test),
         "MAE":mean_absolute_error(y_test,y_pred_lin),
         "RMSE":np.sqrt(mean_squared_error(y_test,y_pred_lin))},
        {"Model":f"Ridge Regression (alpha={ridge_cv_model.alpha_})",
         "R2_train":ridge_cv_model.score(X_train_scaled,y_train),
         "R2_test":ridge_cv_model.score(X_test_scaled,y_test),
         "MAE":mean_absolute_error(y_test,y_pred_ridge),
         "RMSE":np.sqrt(mean_squared_error(y_test,y_pred_ridge))},
        {"Model":f"SVR (best params: {grid_svr.best_params_})",
         "R2_train":grid_svr.score(X_train_scaled,y_train),
         "R2_test":grid_svr.score(X_test_scaled,y_test),
         "MAE":mean_absolute_error(y_test,y_pred_svr),
         "RMSE":np.sqrt(mean_squared_error(y_test,y_pred_svr))},
        {"Model":f"Random Forest (best params: {grid_rf.best_params_})",
         "R2_train":grid_rf.score(X_train,y_train),
         "R2_test":grid_rf.score(X_test,y_test),
         "MAE":mean_absolute_error(y_test,y_pred_rf),
         "RMSE":np.sqrt(mean_squared_error(y_test,y_pred_rf))}
    ]
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(lake_dir, "model_performance_summary.csv"), index=False)
    
    plt.figure(figsize=(8,5))
    plt.bar(results_df["Model"], results_df["R2_test"])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("R2 (test)")
    plt.title(f"Model comparison - R2 on test set ({selected_lake})")
    plt.tight_layout()
    plt.savefig(os.path.join(lake_dir, "model_R2_comparison.png"), dpi=300)
    plt.savefig(os.path.join(lake_dir, "model_R2_comparison.svg"))
    plt.show()
    
    if selected_lake=="All lakes":
        df_lake = event_df.sort_values("parsed_date")
    else:
        df_lake = event_df[event_df["locality"]==selected_lake].sort_values("parsed_date")

    df_lake["density_log"] = np.log10(df_lake["density"]+1)
    df_lake["density_log_lag1"] = df_lake.groupby("locality")["density_log"].shift(1)

    shap_features = [f for f in features if f in df_lake.columns]
    df_lake = df_lake.dropna(subset=shap_features + ["density_log"])
    
    X_lake = df_lake[shap_features]
    y_lake = df_lake["density_log"]
    X_scaled = StandardScaler().fit_transform(X_lake)
    
    shap_dir = os.path.join(lake_dir, "SHAP")
    os.makedirs(shap_dir, exist_ok=True)
    
    ridge_model = Ridge(alpha=10).fit(X_scaled, y_lake)
    explainer_ridge = shap.LinearExplainer(ridge_model, X_scaled)
    plot_and_save_shap(shap_dir, "Ridge", explainer_ridge.shap_values(X_scaled), X_lake, X_lake.columns)
    
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt', random_state=0).fit(X_lake, y_lake)
    explainer_rf = shap.TreeExplainer(rf_model)
    plot_and_save_shap(shap_dir, "RandomForest", explainer_rf.shap_values(X_lake), X_lake, X_lake.columns)
    
    svr_model = SVR(C=1, epsilon=0.01, kernel="rbf").fit(X_scaled, y_lake)
    n_samples = X_scaled.shape[0]
    bg = X_scaled[np.random.choice(n_samples, min(50,n_samples), replace=False)]
    eval_idx = min(150, n_samples)
    explainer_svr = shap.KernelExplainer(svr_model.predict, bg)
    plot_and_save_shap(shap_dir, "SVR", explainer_svr.shap_values(X_scaled[:eval_idx]), X_lake.iloc[:eval_idx], X_lake.columns)
    
    print(f"\nSHAP analysis completed for lake: {selected_lake}")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
