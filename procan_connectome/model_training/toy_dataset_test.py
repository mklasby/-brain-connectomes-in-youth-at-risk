# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: 'Python 3.8.5 64-bit (''ensf-ml'': conda)'
#     name: python3
# ---

from numpy.lib.function_base import place
from procan_connectome.model_training.loocv_wrapper import LOOCV_Wrapper
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np 
import os
import pandas as pd
import logging
import datetime
from procan_connectome.config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL
from procan_connectome.utils.load_dataset import get_rf_dataset, get_svc_dataset
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
df = load_breast_cancer(return_X_y=False, as_frame=True)

X, y = df.data, df.target

# +
pipeline = Pipeline(steps = ( ('ss', StandardScaler()), ('pt', PowerTransformer()) ) )
log_dir = os.path.join(DATA_PATH, 'logs')

estimators = [
    # RandomForestClassifier(random_state=RANDOM_STATE),
    LinearSVC(random_state=RANDOM_STATE),
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
        'tol': [1E-10, 1E-9, 1E-8],
        'C': [1.0],
        'fit_intercept': [True],
        'max_iter': [100000000]
    }
]

log_names = [
    # "RF_LOOCV",
    "LinearSVC_LOOCV_test_df",
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
        label_col='target',
        log_file_name = log_file_name,
        log_dir=log_dir,
        balance_classes=True, 
        scoring='f1_weighted',
        verbose=3,
        n_samples=None,
        single_label_upsample=None,
        cv=None,
        select_features=True,
        feature_threshold=0.01
    )
    loocv.fit(X,y) 
