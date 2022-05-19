from procan_connectome.model_training.loocv_wrapper import LOOCV_Wrapper
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from procan_connectome.data_processing.rf_importance_filter import RFImportanceFilter
from procan_connectome.data_processing.correlation_filter import CorrelationFilter
from procan_connectome.data_processing.powertransformer_wrapper import PowerTransformerWrapper
from procan_connectome.data_processing.select_k_best_filter import SelectKBestFilter
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np 
import os
import pandas as pd
import logging
import datetime
import xgboost as xgb
from procan_connectome.config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL
from procan_connectome.utils.load_dataset import get_rf_dataset, get_svc_dataset

if __name__ == "__main__": 
    # df = get_rf_dataset(threshold=0.001)
    df = pd.read_csv(os.path.join(DATA_PATH, 'combined_datasets.csv'))
    df = df.set_index('ID')
    X, y = df.drop(columns=['label']), df['label']
    # pipeline = Pipeline([
    #     ('cf', CorrelationFilter(threshold=0.9)),
    #     ('rf_filter', RFImportanceFilter(threshold=0.001)),

    # ])
    
    pipeline =  Pipeline([
        ('pt', PowerTransformerWrapper()),
        ('kbest', SelectKBestFilter(k=300))
    ])
    log_dir = os.path.join(DATA_PATH, 'logs')

    estimators = [
        # RandomForestClassifier(random_state=RANDOM_STATE),
        # LinearSVC(random_state=RANDOM_STATE),
        xgb.XGBClassifier(objective="multi:softmax",
                          random_state = RANDOM_STATE, 
                          eval_metric="logloss",
                          use_label_encoder=False,
                          num_class=5,
                          n_jobs=1)
    ]

    grids = [
        # {
        #     'n_estimators': [100, 200, 400, 500, 700],
        #     'criterion': ['gini', 'entropy'],
        #     'min_samples_split': [2, 3, 5],
        #     'min_samples_leaf': [1, 2, 5],
        #     'class_weight': ['balanced', None]
        # },
        # {
        #     'tol': [1E-7, 1E-6],
        #     'C': [1.0, 1.25, 1.5, 2.0],
        #     'fit_intercept': [True, False],
        #     'max_iter': [1000000]
        # }
        {
            'max_depth': [3, 5, 7, 10],
            "n_estimators": [10, 40, 80, 100, 200, 400],
            "gamma": [0, 0.1, 1.0, 10.0, 100],
            "reg_lambda": [0, 0.1, 1.0, 10, 100],
            "alpha": [0, 0.1, 1.0, 10, 100],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
        },
    ]

    log_names = [
        "XGB_LOOCV_logloss_balanced_eval_max_grid",
        # "LinearSVC_LOOCV",
    ]

    

    for estimator, grid, log_file_name in list(zip(estimators, grids, log_names)): 
        log_file_name = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}" + "_" + log_file_name

        logging.basicConfig(
            filename=os.path.join(DATA_PATH, 'logs', log_file_name + "_LOGS"),
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=LOGGER_LEVEL
        )


        loocv = LOOCV_Wrapper(
            X, 
            y, 
            estimator, 
            pipeline=pipeline, 
            param_grid=grid,
            perform_grid_search=True,
            label_col='label',
            log_file_name = log_file_name,
            log_dir=log_dir,
            balance_classes=True, 
            scoring='f1_weighted',
            verbose=2,
            n_samples=None,
            single_label_upsample=None,
            cv=None,
            select_features=False,
            encode_labels=True,
            feature_threshold=0.001
        )

        loocv.fit(X,y) 