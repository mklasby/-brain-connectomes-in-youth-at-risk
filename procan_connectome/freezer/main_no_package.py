from numpy.lib.function_base import place
from loocv_wrapper import LOOCV_Wrapper
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np 
import os
import pandas as pd
import logging
import datetime
from config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL

if __name__ == "__main__": 
    df = pd.read_csv(os.path.join(DATA_PATH, 'combined_datasets.csv'))
    df = df.set_index('ID')
    X, y = df.drop(columns=['label']), df['label']
    pipeline = Pipeline(steps = ( ('ss', StandardScaler()), ('pt', PowerTransformer()) ) )
    log_dir = os.path.join(DATA_PATH, 'logs')

    estimators = [
        # RandomForestClassifier(random_state=RANDOM_STATE),
        LinearSVC(random_state=RANDOM_STATE),
        GradientBoostingClassifier(random_state=RANDOM_STATE)
    ]

    grids = [
        # {
        #     'n_estimators': [10, 50, 100, 200, 400],
        #     'criterion': ['gini', 'entropy'],
        #     'min_samples_split': [2, 3, 5],
        #     'min_samples_leaf': [1, 2, 5],
        #     'class_weight': ['balanced', None]
        # },
        {
            'tol': [1E-7, 1E-6],
            'C': [1.0, 1.25, 1.5, 2.0],
            'fit_intercept': [True, False],
            'max_iter': [100000]
        }, 
        {
            'learning_rate': [0.1, 0.01, 1.0],
            'n_estimators': [10, 50, 100, 200, 400],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [2, 3, 5],
            'max_depth': [None, 3, 5]
        }
    ]

    log_names = [
        # "RF_LOOCV",
        "LinearSVC_LOOCV",
        "GBC_LOOCV",
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
            log_dir=log_dir, 
            label_col='label',
            log_file_name = log_file_name,
            balance_classes=True, 
            scoring='f1_weighted',
            verbose=2,
            n_samples=18,
            single_label_upsample="Transition"
        )

        loocv.fit(X,y) 