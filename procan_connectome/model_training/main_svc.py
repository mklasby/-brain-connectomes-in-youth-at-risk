from numpy.lib.function_base import place
from scipy.sparse.construct import rand
from procan_connectome.model_training.loocv_wrapper import LOOCV_Wrapper
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from procan_connectome.data_processing.linear_svc_importance_filter import LinearSVCImportanceFilter
from procan_connectome.data_processing.correlation_filter import CorrelationFilter
from procan_connectome.data_processing.powertransformer_wrapper import PowerTransformerWrapper
from procan_connectome.data_processing.select_k_best_filter import SelectKBestFilter
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
import numpy as np 
import os
import pandas as pd
import logging
import datetime
from procan_connectome.config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL
from procan_connectome.utils.load_dataset import get_rf_dataset, get_svc_dataset

if __name__ == "__main__": 
    # df = get_svc_dataset(threshold=0.001, global_only=False)
    df = pd.read_csv(os.path.join(DATA_PATH, 'combined_datasets.csv'))
    df = df.set_index('ID')
    X, y = df.drop(columns=['label']), df['label']
    # pipeline =  Pipeline([
    #     ('cf', CorrelationFilter(threshold=0.9)),
    #     ('pt', PowerTransformerWrapper()),
    #     ('svc_filter', LinearSVCImportanceFilter(threshold=0.001)),

    # ])
    
    pipeline =  Pipeline([
        ('MinMax', MinMaxScaler()),
        ('SelectFromModel', SelectFromModel(LinearSVC(random_state=RANDOM_STATE), threshold=0.001)),
        ('FastICA', FastICA(n_components=80))
    ])
    
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
            'tol': [1E-10],
            'C': [0.1, 1.0, 1.5],
            'fit_intercept': [True, False],
            'max_iter': [100000000]
        }
    ]

    log_names = [
        # "RF_LOOCV",
        "LinearSVC_LOOCV_FINAL_TRIAL",
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
            balance_classes=False, 
            scoring='f1_weighted',
            verbose=2,
            n_samples=None,
            single_label_upsample=None,
            cv=None,
            select_features=False,
            feature_threshold=0.001,
            grid_search_feature_selection=False
        )
        loocv.fit(X,y) 