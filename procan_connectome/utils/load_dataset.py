import os
import pandas as pd 
from procan_connectome.config import DATA_PATH

# Feature Engineering Helpers
def _get_important_features(df, importance_df, threshold=0.001):
  important_features = importance_df.loc[importance_df['Importance'] >= threshold]['Feature']
  filtered_df = df[important_features.values.tolist()]
  filtered_df = filtered_df.merge(df['label'], left_index=True, right_index=True)
  return filtered_df

def get_svc_dataset(threshold:float=0.001, global_only:bool = False) -> pd.DataFrame: 
    if not global_only: 
        DATASET = 'SVC_feature_importances.csv'
    else: 
        DATASET = 'SVC_feature_importances_global.csv'
    return _get_filtered_dataset(DATASET, threshold)

def get_rf_dataset(threshold:float=0.001) -> pd.DataFrame: 
    DATASET = 'rf_feature_importances.csv'
    return _get_filtered_dataset(DATASET, threshold)

def _get_filtered_dataset(fname:str, threshold:float) -> pd.DataFrame: 
    df = pd.read_csv(os.path.join(DATA_PATH, "combined_datasets.csv"))
    df = df.set_index('ID')
    importance_df = pd.read_csv(os.path.join(DATA_PATH, fname))
    return _get_important_features(df, importance_df, threshold)