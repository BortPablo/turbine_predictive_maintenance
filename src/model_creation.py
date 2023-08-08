import os

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv()

import mlflow

mlflow.set_tracking_uri(os.getenv('MLFLOW_URL'))

import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor

def load_failures() -> pd.DataFrame:
    """Load the failures from the wind farm dataset."""
    df = pd.read_csv('../data/raw/htw-failures-2016.csv', sep=';')
    aux = pd.read_csv('../data/raw/htw-failures-2017.csv', sep=';')

    df = pd.concat([df, aux], axis=0).reset_index(drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()
    df = df[df.Turbine_ID != 'T09']
    df['Turbine_ID'] = df['Turbine_ID'].apply(lambda x: int(x[1:]))
    return df

def load_costs() -> pd.DataFrame:
    """Load the costs from the wind farm dataset."""
    return pd.read_csv('../data/raw/HTW_Costs.csv').set_index('Component')

def trb_per_failures() -> dict:
    """Return a dictionary with the turbine ID per component failure"""
    failures = load_failures()
    trb_per_comp = {}
    for comp in failures.Component.unique():
        trb_per_comp[comp] = failures[(failures.Component == comp)&(failures.index >= '2017-06-01')].Turbine_ID.unique().tolist()
    return trb_per_comp

def create_time_columns(df) -> pd.DataFrame:
    """Create time columns from the index of the dataframe."""
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    return df

def filter_FP(FP, resample_time) -> list:
    """Filter the consecutive false positives from the list of FP."""
    aux = []
    for i in range(len(FP)-1):
        if (FP[i+1]-FP[i]).total_seconds() == 60*60*resample_time:
            continue
        aux.append(FP[i])
        aux.append(FP[i+1])
    if len(aux) == 0:
        aux.append(FP[0])
    return aux

def test_params(resample_time, center, deviation, num_dev, xgb_params) -> None:
    """ Function to test the parameters of the model"""
    targets = {
    'GENERATOR': ['Gen_Phase1_Temp_Avg','Gen_Phase2_Temp_Avg','Gen_Phase3_Temp_Avg','Gen_SlipRing_Temp_Avg',],
    'HYDRAULIC_GROUP': ['Hyd_Oil_Temp_Avg'],
    'GENERATOR_BEARING': ['Gen_Bear_Temp_Avg','Gen_Bear2_Temp_Avg'],
    'TRANSFORMER': ['HVTrafo_Phase1_Temp_Avg','HVTrafo_Phase2_Temp_Avg','HVTrafo_Phase3_Temp_Avg'],
    'GEARBOX': ['Gear_Oil_Temp_Avg', 'Gear_Bear_Temp_Avg']
    }

    features = ['Gen_RPM_Avg', 'Nac_Temp_Avg','Rtr_RPM_Avg', 'Amb_WindSpeed_Avg', 'Amb_WindDir_Abs_Avg', 'Amb_Temp_Avg', 'Prod_LatestAvg_TotActPwr', 'Prod_LatestAvg_TotReactPwr',
        'Spin_Temp_Avg', 'Blds_PitchAngle_Avg', 'Grd_Busbar_Temp_Avg','Nac_Direction_Avg', 'theoretical_performance_ratio']
    
    for id in [1, 6, 7, 11]:
        for comp in targets.keys():
            if id not in trb_per_failures()[comp]:
                continue
            for col in targets[comp]:
                MODULE_NAME = os.getenv('MODULE_NAME')
                # Start mlflow run
                with mlflow.start_run(run_name=f'{MODULE_NAME}_trb{id}_{col}') as run:
                    # Set tags
                    mlflow.set_tags({
                        'MODULE': MODULE_NAME,
                        'trb_num': id,
                        'col': col,
                        'model': 'XGBRegressor'
                    })

                    # Load train/test datasets
                    train = pd.read_parquet(f'../data/processed/{id}/train.parquet')

                    # Select the target variable
                    y_train = train.pop(col)

                    # Select the features to be used in the model
                    train = train[features]
                    train = create_time_columns(train)

                    # Split the train dataset into train and validation
                    X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size=0.2, random_state=42)

                    # Scale the features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)

                    # Select the parameters for XGBRegressor model
                    mlflow.log_params(xgb_params)
                    mlflow.log_params({
                        'resample_time': resample_time,
                        'center': center,
                        'deviation': deviation,
                        'num_dev': num_dev
                    })

                    # Train the model
                    model = XGBRegressor(**xgb_params)
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=10, verbose=False)

                    # Save the scaler and model into mflow
                    mlflow.sklearn.log_model(scaler, 'scaler', pyfunc_predict_fn='transform')
                    mlflow.xgboost.log_model(model, 'model')

                    # Evaluate the model
                    y_pred = model.predict(X_val_scaled)
                    mlflow.log_metrics({
                        'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                        'mae': mean_absolute_error(y_val, y_pred),
                        'r2': r2_score(y_val, y_pred)
                    })

                    X_val = pd.DataFrame(X_val, columns=features + ['month', 'day', 'hour', 'dayofweek'])
                    X_val['y_true'] = y_val
                    X_val['y_pred'] = y_pred
                    X_val['residual'] = y_val - y_pred

                    threshold = {
                        'mean': str(X_val['residual'].mean()),
                        'median': str(X_val['residual'].median()),
                        'std': str(X_val['residual'].std()),
                        'mad': str((X_val['residual'] - X_val['residual'].median()).abs().median())
                    }
                    # Save json
                    with open(f'../data/interim/threshold.json', 'w') as f:
                        json.dump(threshold, f)

                    mlflow.log_artifact(f'../data/interim/threshold.json')

                    # Get costs metric
                    test = pd.read_parquet(f'../data/processed/{id}/test.parquet')

                    y_test = test.pop(col)

                    # Select the features to be used in the model
                    test = test[features]
                    test = create_time_columns(test)

                    X_test_scaled = scaler.transform(test)

                    test[col] = y_test
                    test['y_pred'] = model.predict(X_test_scaled)
                    test['residual'] = test[col] - test['y_pred']

                    test = test[[col, 'y_pred', 'residual']]

                    alarms = test.copy()
                    alarms = alarms.resample(f'{resample_time}H').mean()
                    alarms['anomalous_pred'] = np.where(
                        (alarms['residual'] >=  float(threshold[center]) + num_dev * float(threshold[deviation]))|
                        (alarms['residual'] <=  float(threshold[center]) - num_dev * float(threshold[deviation])),
                        1,
                        0
                    )
                    failures = load_failures()
                    failure_dates = failures[(failures['Turbine_ID'] == id) & (failures['Component'] == comp)].index

                    init = True
                    for date in failure_dates:
                        if init:
                            alarms['anomalous_real'] = np.where(
                                (alarms.index >= date - pd.Timedelta(days=90)) & (alarms.index <= date),
                                1,
                                0
                            )
                            init = False
                        else:
                            alarms['anomalous_real'] = np.where(
                                (alarms.index >= date - pd.Timedelta(days=90)) & (alarms.index <= date),
                                1,
                                alarms['anomalous_real']
                            )

                    alarms[['anomalous_real', 'anomalous_pred']].plot(figsize=(15, 5), title=f'Anomalous Real vs Anomalous Predicted (turbine {id}, column {col})')
                    for date in failure_dates:
                        plt.axvline(x=date, color='red', linestyle='--')
                    plt.savefig(f'../reports/check.png')
                    plt.clf()
                    mlflow.log_artifact(f'../reports/check.png')

                    costs = load_costs()

                    total_cost = 0
                    TP_cost = 0
                    FN_cost = 0
                    TP = alarms[(alarms['anomalous_real']&alarms['anomalous_pred']) == 1].index
                    if TP.shape[0] == 0:
                        FN_cost -= costs.loc[comp, 'Replacement_Cost'] * failure_dates.shape[0]
                        total_cost -= costs.loc[comp, 'Replacement_Cost'] * failure_dates.shape[0]
                    else:
                        for date in failure_dates:
                            aux = TP[(TP < date)&(TP > date - pd.Timedelta(days=90))]
                            if len(aux) == 0:
                                total_cost -= costs.loc[comp, 'Replacement_Cost']
                                FN_cost -= costs.loc[comp, 'Replacement_Cost']
                                continue
                            days = (date - aux[0]).days + (date - aux[0]).seconds/(24*60*60)
                            total_cost += costs.loc[comp, 'Replacement_Cost'] - (costs.loc[comp, 'Repair_Cost'] + (costs.loc[comp, 'Replacement_Cost'] - costs.loc[comp, 'Repair_Cost'])*(1 - days/90))
                            TP_cost += costs.loc[comp, 'Replacement_Cost'] - (costs.loc[comp, 'Repair_Cost'] + (costs.loc[comp, 'Replacement_Cost'] - costs.loc[comp, 'Repair_Cost'])*(1 - days/90))

                    FP = alarms[(alarms['anomalous_real'] == 0)&(alarms['anomalous_pred'] == 1)].index
                    if len(FP) > 1:
                        FP = filter_FP(FP, resample_time)
                    FP_cost = -len(FP) * costs.loc[comp, 'Inspection_cost']
                    total_cost += FP_cost

                    mlflow.log_metrics({
                        'total_cost': total_cost,
                        'TP_cost': TP_cost,
                        'FP_cost': FP_cost,
                        'FN_cost': FN_cost
                    })
                    print(f'Model: {MODULE_NAME}_trb{id}_{col}',
                                 f'Params: {xgb_params}',  f'Resample time: {resample_time}', f'Deviation: {deviation}',
                                 f'Total cost: {total_cost}')

if __name__ == '__main__':
    for n_estimators in [200, 500, 1000]:
        for max_depth in [3, 5, 7, 9]:
            for min_child_weight in [1, 3, 5]:
                for learning_rate in [0.01, 0.05, 0.1]:
                    for resample_time in [6, 12]:
                        for center in ['median']:
                            for deviation in ['std', 'mad']:
                                for num_dev in [3]:                              

                                    xgb_params = {
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth,
                                        'min_child_weight': min_child_weight,
                                        'learning_rate': learning_rate,
                                        'tree_method':'gpu_hist',
                                    }
                                    test_params(
                                        resample_time=resample_time,
                                        center=center,
                                        deviation=deviation,
                                        num_dev=num_dev,
                                        xgb_params=xgb_params
                                    )