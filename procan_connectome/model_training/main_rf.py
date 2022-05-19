from numpy.lib.function_base import place
from procan_connectome.model_training.loocv_wrapper import LOOCV_Wrapper
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from procan_connectome.data_processing.rf_importance_filter import RFImportanceFilter
from procan_connectome.data_processing.correlation_filter import CorrelationFilter
from procan_connectome.data_processing.powertransformer_wrapper import PowerTransformerWrapper
from procan_connectome.data_processing.select_k_best_filter import SelectKBestFilter
from procan_connectome.data_processing.pca_transformer import PCATransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np 
import os
import pandas as pd
import logging
import datetime
from procan_connectome.config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL
from procan_connectome.utils.load_dataset import get_rf_dataset, get_svc_dataset

if __name__ == "__main__": 
    # df = get_rf_dataset(threshold=0.001)
    df = pd.read_csv(os.path.join(DATA_PATH, 'combined_datasets.csv'))
    df = df.set_index('ID')
    X, y = df.drop(columns=['label']), df['label']
    pipeline = Pipeline([
        ('pt', PowerTransformerWrapper()),
        ('cf', CorrelationFilter(threshold=0.95)),
        ("pca", PCATransformer(n_components=0.95)),

    ])
    
    # pipeline =  Pipeline([
    #     ('pt', PowerTransformerWrapper()),
    #     ('kbest', SelectKBestFilter(k=300))
    # ])
    log_dir = os.path.join(DATA_PATH, 'logs')

    estimators = [
        RandomForestClassifier(random_state=RANDOM_STATE),
        # LinearSVC(random_state=RANDOM_STATE),
    ]

    grids = [
        {
            'n_estimators': [100, 200, 400, 500, 700],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 5],
            'class_weight': ['balanced', None]
        },
        # {
        #     'tol': [1E-7, 1E-6],
        #     'C': [1.0, 1.25, 1.5, 2.0],
        #     'fit_intercept': [True, False],
        #     'max_iter': [1000000]
        # }
    ]

    log_names = [
        "RF_LOOCV_FINAL_TRIAL_cf+pt+pca+if",
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
            perform_grid_search=False,
            label_col='label',
            log_file_name = log_file_name,
            log_dir=log_dir,
            balance_classes=True, 
            scoring='f1_weighted',
            verbose=2,
            n_samples=None,
            single_label_upsample=None,
            cv=None,
            select_features=True,
            feature_threshold=0.001
        )

        loocv.fit(X,y) 