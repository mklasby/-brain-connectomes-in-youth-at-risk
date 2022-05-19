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
#     display_name: 'Python 3.7.10 64-bit (''ML'': conda)'
#     name: python3
# ---



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import LeaveOneOut, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import resample
from procan_connectome.config import DATA_PATH, RANDOM_STATE
from procan_connectome.data_processing.linear_svc_importance_filter import LinearSVCImportanceFilter
from procan_connectome.data_processing.rf_importance_filter import RFImportanceFilter
from procan_connectome.eda.EDA import rf_loovc
from procan_connectome.data_processing.powertransformer_wrapper import PowerTransformerWrapper
import logging
import datetime

if __name__ == "__main__": 
    df = pd.read_csv(os.path.join(DATA_PATH, "combined_datasets_global.csv"))
    df = df.set_index('ID')
    df

if __name__ == "__main__": 
    X, y = df.drop(columns='label'), df['label']
    pt = PowerTransformerWrapper() 
    X_trans = pt.fit_transform(X)
    X_trans


if __name__ == "__main__": 
    svc = LinearSVC(random_state=RANDOM_STATE, max_iter = 100000000)
    svcif = LinearSVCImportanceFilter(random_state=RANDOM_STATE, threshold=0.001, sort=False)
    svcif.fit(X_trans, y)


# +
def svc_loovc(X: pd.DataFrame, y: pd.Series, standard_scale: bool=False, power_transform:bool=False, threshold:float=0.001): 
  loo = LeaveOneOut()
  importances = []
  y_true = []
  y_predict = []
  counter = 0

  for train_idx, test_idx in loo.split(X):
    counter += 1 
    if counter % 10 == 0: 
      print(f"Iteration {counter} of {len(X)}")
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    if standard_scale: 
      std_scaler = StandardScaler()
      X_train = std_scaler.fit_transform(X_train)
      X_test = std_scaler.transform(X_test)
    if power_transform: 
      pt = PowerTransformer()
      X_train = pt.fit_transform(X_train)
      X_test = pt.transform(X_test)
    svc = LinearSVCImportanceFilter(random_state=RANDOM_STATE, sort=False, threshold=threshold)
    svc.fit(X_train,y_train)
    importances.append(svc.feature_importances_df_['Importance'].values)
    y_predict.append(svc.estimator_.predict(X_test)[0])
    y_true.append(y_test[0])

  acc = accuracy_score(y_true, y_predict)
  logging.debug(f"Feature Selection LOOCV Accuracy: {acc}")
  labels = X.columns
  feature_importances = np.array(importances).mean(axis=0)
  feature_importances_df = pd.DataFrame(list(zip(map(lambda x: round(x,6), feature_importances), labels)), columns=["Importance", "Feature"])
  logging.debug(f"First 10 Features: {feature_importances_df.iloc[:10]}")
  results_df = pd.DataFrame({
      "y_true": y_true,
      'y_pred': y_predict
  }).set_index(X.index)
  return results_df, feature_importances_df

if __name__ == "__main__": 
  X, y = df.drop(columns="label"), df['label']
  results_df, feature_importances_df = svc_loovc(X_trans, y, False, False)
  feature_importances_df.to_csv(os.path.join(DATA_PATH, 'SVC_feature_importances_global.csv'), index=False)
# -

if __name__ == "__main__": 
  X, y = df.drop(columns="label"), df['label']
  results_df, feature_importances_df = rf_loovc(X_trans, y, False, False)
  feature_importances_df.to_csv(os.path.join(DATA_PATH, 'rf_feature_importances_2.csv'), index=False)
