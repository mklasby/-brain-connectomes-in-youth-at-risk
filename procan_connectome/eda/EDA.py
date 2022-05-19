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


# + [markdown] id="Rx9h6-N7P-SN"
# # PROCAN Connectome and ML: Data structure and questions for analysis
# ## Exploratory Data Analysis
# * Catherine Lebel: Advisor (neuroimaging)
# * Paul Metzak: Functional imaging
# * Mohammed Shakeel: Structural imaging
# * Dr. Roberto Souza: Machine learning 
# * Mike Lasby: Machine learning 
#
# Last Updated: 2021-06-17<br>
# By: Mike Lasby
#

# + id="3ZfekDJ5P2q4"
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
# from google.colab import drive
import os
import h5py
import re
# import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from procan_connectome.config import PLOT_PATH, DATA_PATH

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1623966483529, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="HI9RWqADI2mC" outputId="573960e5-4c27-456e-ca45-538d88aa3beb"
# if __name__ == "__main__": 
#   ROOT_PLOT= "./drive/MyDrive/research/structural-connectivity/plots/"
#   ROOT_RESULTS= "./drive/MyDrive/research/structural-connectivity/results/"
#   drive.mount('/content/drive/')
# -

ROOT_PLOT = PLOT_PATH
ROOT_RESULTS = DATA_PATH

# + colab={"base_uri": "https://localhost:8080/", "height": 137} executionInfo={"elapsed": 43365, "status": "ok", "timestamp": 1621100606581, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="WeJ1ZiQwt9Ny" outputId="98d5c748-7eca-4ca6-b649-9640d7d484ac"
"""Structural connectivity

Group folders: represent each of the 5 groups of interest:
HC 
Stage_0
Stage_1a
Stage_1b
Transition

Sub-folders in each group:
ID: contains participant ID in same order as all remaining files in the folder 
Site_age: contains covariates site and age (in that order)
Global_metrics: contains global metrics
Modular_interactions: contains modular interactions
Nodal_metrics: contains nodal metrics. There are 90 columns in each of these files. Each column represents one AAL region. 
See corresponding names of region in <AAL_Label_Names.txt> (please ignore leading 1's)

Variables to use:
Global_metrics: 
	Connectome density: Represents density of the entire connectome
	Connectome Intensity: Represents intensity of the entire connectome
	Assortativity: standardized value only <rzscore.txt>
	Hierarchy: standardized value only <bzscore.txt>
	Network efficiency: 
		Global efficiency <Eg.txt>
		Local efficiency <Eloc.txt>
	SmallWorld: 
		Clustering coefficient <Cp.txt>
		Gamma <Gamma.txt>
		Lambda <Lambda.txt>
		Shortest path length <Lp.txt>
		Sigma <Sigma.txt>
	Synchronization: standardized value only <szscore.txt>	
Modular_interactions: All variables. Labels represent module name and connections. e.g, <SumEdgeNum_Between_Module01_02.txt> means modular interaction between module 1 and module 2
	Module labels are given in <Module_labels.txt>
Nodal metrics:
	Betweenness Centrality <Bc.txt>
	Degree Centrality <Dc.txt>
	Nodal Cluster Coefficient <NCp.txt>
	Nodal Efficiency <Ne.txt>
	Nodal Local efficiency <NLe.txt>  		
	Nodal Shortest path <NLp.txt>
	Participant coefficient: standardized value only <CustomPc_normalized.txt>

Additional notes: 
The Matlab Data file is retained to make it easier to import the data.
Except site in <Site_age.txt> all are continuous variables
Please get in touch in case you have any questions: <mohammed.kalathil.ucalgary.ca>
"""


# + id="XLYLn__K1Dev"
"""Helper methods to compile datasets from .txt files."""


def get_ids(file_dir, fname="ID.txt"): 
  full_path = os.path.join(file_dir, fname)
  df = pd.read_csv(full_path, header=None, names=["ID"])
  df['ID'] = df['ID'].astype("object")
  return df

def get_site_age(df, file_dir, fname="Site_age.txt"):
  full_path = os.path.join(file_dir, fname)
  new_df = pd.read_csv(full_path, header=None, names = ["Site", "Age"], sep="\t")
  df = df.join(new_df)
  return df

def parse_all_txt_in_dir(df, file_dir, nodal_metrics = False, NODAL_COLUMNS=[]): 
  for root, subdirs, fnames in os.walk(file_dir): 
    for subdir in subdirs: 
      df = parse_all_txt_in_dir(df, os.path.join(root, subdir), nodal_metrics, NODAL_COLUMNS)
    for fname in fnames: 
      if fname.split(".")[1] == "txt" or fname.split(".")[1] == "csv":  
        # print(f"Parsing file {fname}...")
        path = os.path.join(root, fname)
        if nodal_metrics: 
          pre = fname.split(".")[0] + "-"
          names = [pre + label for label in NODAL_COLUMNS]
          new_df = pd.read_csv(path, header=None, names=names, sep="\t", index_col=False)
        else: 
          new_df = pd.read_csv(path, header=None, names=[fname.split(".")[0]])
        df = df.join(new_df)
    return df
  
def prefix_columns(prefix, df, exclude=["ID", "label"]): 
  cols = [col_name for col_name in df.columns if col_name not in exclude]
  new_names = {col_name: prefix + col_name for col_name in cols}
  df = df.rename(columns=new_names)
  return df

def id_matcher(s):
  """Function to match id's from different datasets"""
  if s[0] == "S":
    s_prime = s.split("_")[0][:-1] + "_"+ s.split("_")[1][1:]
  else: 
    s_prime = s.split("_")[0] + "_"+ s.split("_")[1][1:]
  return s_prime


# + id="gS6FTRt2c98j"
def get_functional_df(prefix_col:bool=True, global_only:bool=False): 
  """Gets functional dataset from .txt files. 

  Args: 
    prefix_col: If true, prefix all features columns with "fun"
  
  Returns: 
    dataframe of combined data functional data. 
  """
  ROOT_LOCATION = os.path.join(ROOT_RESULTS, "functional_connectome_data")
  LABEL_DICT = {
          1: "Stage_0",
          0 :"HC",
          3: "Stage_1b",
          2: "Stage_1a",
          4: "Transition"
  }

  # No need to drop '1 ' for this set
  NODAL_COLUMNS = pd.read_csv(os.path.join(ROOT_LOCATION, "AAL_Label_Names.txt"), header=None, names=["AAL_label"])["AAL_label"].values

  VARS_TO_DROP = [
                  "r",
                  "b",
                  "s"
  ]

  VARS_TO_DROP_REGEX =[
                      "CustomPc-"
  ]

  COVARIATES = [
                # "Site",
                # "Age",
                # "HeadMovement",
                "Group_BL"
  ]

  RENAME_COLS = {
      "fmri_density": "Connectome_density",
      "fmri_intensity": "Connectome_intensity"
  }


  df = pd.read_csv(os.path.join(ROOT_LOCATION, "fmri_IDs_stages_covariates.csv"), sep="\t")
  df['ID'] = df['ID'].apply(lambda x: id_matcher(x)).astype("object")
  df['label'] = df['Group_BL'].map(LABEL_DICT)


  for fname in os.listdir(ROOT_LOCATION): 
    if len(fname.split(".")) == 1:  # It's a subdir
        file_dir = os.path.join(ROOT_LOCATION, fname)
        if global_only: 
          if not re.search("Nodal_metrics|Modular_interactions", file_dir): 
            print(f"Parsing features in {file_dir}...")
            df = parse_all_txt_in_dir(df, file_dir)
        else: 
          # Get global features, modular features
          if not re.search("Nodal_metrics", file_dir): 
            print(f"Parsing features in {file_dir}...")
            df = parse_all_txt_in_dir(df, file_dir)
          else: 
          # Get Nodal_metrics
            print(f"Parsing features in {file_dir}...")
            df = parse_all_txt_in_dir(df,file_dir, nodal_metrics=True, NODAL_COLUMNS=NODAL_COLUMNS) 
          # df.join(feature_df)

  df = df.drop(columns=VARS_TO_DROP)
  df = df.drop(columns=COVARIATES)
  for var in VARS_TO_DROP_REGEX: 
    df = df.drop(columns=list(df.filter(regex=var)))
  df = df.rename(columns=RENAME_COLS)
  if prefix_col:
    df = prefix_columns("fun-", df)
  return df

if __name__ == "__main__": 
  fun_df = get_functional_df(prefix_col=False, global_only=True)
  display(fun_df)


# + id="_KibMB7CsjwR"
def get_structural_df(prefix_col=True, global_only:bool=False): 
  """Gets structural dataset from .txt files. 

  Args: 
    prefix_col: If true, prefix all features columns with "str"
  
  Returns: 
    dataframe of combined data structural data. 
  """


  # ROOT_LOCATION = "./drive/MyDrive/research/structural-connectivity/Structural_connectivity/"
  ROOT_LOCATION = os.path.join(ROOT_RESULTS, "Structural_connectivity")
  LABELS = [
            "Stage_0",
            "HC",
            "Stage_1b",
            "Stage_1a",
            "Transition"
  ]

  NODAL_COLUMNS = pd.read_csv(os.path.join(ROOT_LOCATION, "AAL_Label_Names.txt"), header=None, names=["AAL_label"])['AAL_label'].apply(lambda x: x[2:]).values

  VARS_TO_DROP = [
                  "r",
                  "b",
                  "s"
  ]

  VARS_TO_DROP_REGEX =[
                      "CustomPc-"
  ]

  COVARIATES = [
                # "Site",
                # "Age",
  ]

  dfs = []

  for label_folder in os.listdir(ROOT_LOCATION): 
    if (label_folder in LABELS):
      file_dir = os.path.join(ROOT_LOCATION, label_folder)
      df = get_ids(file_dir)
      df = get_site_age(df, file_dir)
      # # Get global features
      print(f"Parsing features for {file_dir}...")
      df = parse_all_txt_in_dir(df, os.path.join(file_dir, "Global_metrics"))
      if not global_only: 
        # # Get Modular_interactions
        print(f"Parsing features for {file_dir}...")
        df = parse_all_txt_in_dir(df, os.path.join(file_dir, "Modular_interactions"))
        # Get Nodal_metrics
        print(f"Parsing features for {file_dir}...")
        df = parse_all_txt_in_dir(df, os.path.join(file_dir, "Nodal_metrics"), nodal_metrics=True, NODAL_COLUMNS=NODAL_COLUMNS)
      df['label'] = label_folder
      dfs.append(df)
      # print("\n\n")

  df = pd.concat(dfs)
  df = df.drop(columns=VARS_TO_DROP)
  df = df.drop(columns=COVARIATES)
  for var in VARS_TO_DROP_REGEX: 
    df = df.drop(columns=list(df.filter(regex=var)))
  if prefix_col:
    df = prefix_columns("str-", df)
  return df


if __name__ == "__main__": 
  struct_df = get_structural_df(False, True)
  display(struct_df)


# + colab={"base_uri": "https://localhost:8080/", "height": 776} executionInfo={"elapsed": 111194, "status": "ok", "timestamp": 1623964753913, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="Aonjc_CVmfyl" outputId="e7784820-275e-429d-f2bd-22bb93f7cf82"
def get_dataset(functional=True, structural=True, global_only:bool=False): 
  if functional and structural: 
    fun_df = get_functional_df(prefix_col=True, global_only=global_only)
    fun_df = fun_df.set_index("ID")
    struct_df = get_structural_df(prefix_col=True, global_only=global_only)
    struct_df = struct_df.set_index("ID").drop(columns='label')
    df = fun_df.join(struct_df)
    df = df.drop(columns=['fun-Age', 'fun-Site'])
    df = df.rename(columns={"str-Age": "Age", 'str-Site': "Site"})
    return df
  if functional and not structural: 
    return get_functional_df(prefix_col=False, global_only=global_only)
  if structural and not functional: 
    return get_structural_df(prefix_col=False, global_only=global_only)

if __name__ == "__main__": 
  global_only=True
  df = get_dataset(global_only=global_only)
  df.to_csv(os.path.join(ROOT_RESULTS, 'combined_datasets_global.csv'))
  # functional_df = get_dataset(True, False)
  # functional_df.to_csv("./drive/MyDrive/research/structural-connectivity/data/functional_dataset.csv")
  # structural_df = get_dataset(False, True)
  # structural_df.to_csv("./drive/MyDrive/research/structural-connectivity/data/structural_dataset.csv")
  display(df)

# + colab={"base_uri": "https://localhost:8080/", "height": 473} executionInfo={"elapsed": 838, "status": "ok", "timestamp": 1623966491983, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="eyPFTMsZ8eJr" outputId="0dfc439c-84e2-4f30-e3c3-f688fe398214"
# df = pd.read_csv("./drive/MyDrive/research/structural-connectivity/data/combined_datasets.csv")
# df

# + [markdown] id="Kqx49nPbPy4A"
# # Exploratory Data Analysis

# + id="mEx5YOZVRaQJ"
from sklearn.metrics import confusion_matrix
# Plotting Methods
def save_fig(plot_path, fig): 
    fig.savefig(os.path.join(plot_path, fig.axes[0].title.get_text()+'.png'), bbox_inches='tight')
    print(f"Figured saved to {os.path.join(plot_path, fig.axes[0].title.get_text()+'.png')}")
    return 

def plot_clusters_2D(df, x, y, hue, label_col_name=None, fig_title=None, plot_path=None, custom_label_map = None, custom_label_name=None):
    if custom_label_map != None:
      df[custom_label_name] = df[label_col_name].map(custom_label_map)
    if plot_path != None: 
      plt.style.use(os.path.join(plot_path, "plt_plot_style.mplstyle")) 
    DIMS = [15,10]
    fig, ax = plt.subplots(figsize=DIMS)
    sns.scatterplot(x=x, y=y, hue=hue, data=df, ax=ax)
    title_string = (fig_title)
    ax.set_title(title_string)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if plot_path != None: 
        save_fig(plot_path, fig)
    return fig

def confusion_matrix_plot(cm, label_names, fig_title=None, plot_path = None):
    plt.style.use(os.path.join(plot_path, 'plt_plot_style.mplstyle'))
    DIMS = [15,10]
    fig, ax = plt.subplots(figsize=DIMS)
    if plot_path != None: 
      plt.style.use(os.path.join(plot_path, "plt_plot_style.mplstyle"))
    sns.heatmap(cm, annot=True, ax=ax, fmt='.2f', cmap='Greens')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    if fig_title==None: 
      ax.set_title('Confusion Matrix')
    else: 
      ax.set_title(fig_title)
    ax.xaxis.set_ticklabels(label_names, rotation=90)
    ax.yaxis.set_ticklabels(label_names, rotation=360)
    if plot_path != None: 
      save_fig(plot_path, fig)
    return fig 

def plot_feature_importance(df, plot_title, feature_col='Feature', importance_col = 'Importance', n_features=None, plot_path=None):
    df.sort_values(by=[importance_col], ascending=False, inplace=True)
    DIMS = [15, 10]
    fig = plt.figure(figsize=DIMS)
    if n_features == None:
        x=df[importance_col]
        y=df[feature_col]
    else: 
        x=df[importance_col].iloc[:n_features]
        y=df[feature_col].iloc[:n_features]
    ax = sns.barplot(x=x, y=y, color='darkturquoise')
    fig.add_subplot(ax)
    plt.title(plot_title)
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    if n_features == None: 
        plt.legend(y, bbox_to_anchor=(1.05, 1),
                   loc='upper left', fancybox=False)
        ax.set_yticklabels([], size=10)
    if plot_path != None: 
        save_fig(plot_path, fig)
    return fig


# + id="odauK1IMWykd"
# Histogram plot helpers
def get_counts_split_by_feature(df:pd.DataFrame, agg_col:str, count_col:str, normalize:bool=True): 
    """Get count of instances of count_col when df is grouped by agg_col
    
    Args: 
        df: df to count/aggregate
        agg_col: col to aggregate with. We will groupby this column when calculating the counts for count col
        count_col: column to count distinct values of. We will calculate the proportion for distinct value in this feature
        normalize: if True, counts are normalized based on total count. Raw counts if False. 
   
   Returns: 
        df indexed by distinct values in agg_col and columns consisting of distinct value counts of count_col. 
    """
    
    targets = df[count_col].unique().tolist()
    agg_series = []

    for target in targets: 
        count = df.loc[df[count_col] == target].groupby(agg_col).count()[count_col].rename(f"{target}_count")
        agg_series.append(count)
    agg_df = pd.concat(agg_series, axis=1)

    if normalize: 
        # Normalize features by row sums 
        agg_df = agg_df.div(agg_df.sum(axis=1), axis=0)
    return agg_df

def get_label_ids(df: pd.DataFrame, label:str, label_col:str="label"): 
    return df.loc[df[label_col] == label].index.tolist()
    
def get_one_vs_rest_labels(df: pd.DataFrame, sample_indices:list, sample_name:str, hue_col:str="Hue"): 
    dis_df = df.copy(deep=True)
    dis_df[hue_col] = dis_df.index.map(lambda x: sample_name if x in sample_indices else "Others")
    return dis_df

def generate_feature_dist_plot(df: pd.DataFrame, sample_indices:list, feature:str,
                               sample_name:str,one_vs_rest:bool=True, plot_path=None,title_string=None, **kwargs):
    if one_vs_rest:
      dis_df = get_one_vs_rest_labels(df, sample_indices, sample_name, hue_col=kwargs['hue'])
    else: 
      dis_df = df.copy(deep=True)
    if plot_path != None: 
        plt.style.use(os.path.join(plot_path, "plt_plot_style.mplstyle"))
    DIMS = [15,10]
    fig, ax = plt.subplots(figsize=DIMS)
    sns.histplot(data=dis_df, x=feature,ax=ax,**kwargs)
    ax.set_title(title_string, fontsize=24)
    plt.xticks(np.arange(df['Age'].min(), df['Age'].max(), 1))
    if plot_path != None: 
        save_fig(plot_path, fig)
    return fig


# + id="G24ZwICTYs-M"
# Feature Engineering Helpers
def get_important_features(df, importance_df, threshold=0.001):
  important_features = importance_df.loc[importance_df['Importance'] >= threshold]['Feature']
  filtered_df = df[important_features.values.tolist()]
  filtered_df = filtered_df.merge(df['label'], left_index=True, right_index=True)
  return filtered_df


# + colab={"base_uri": "https://localhost:8080/", "height": 686} executionInfo={"elapsed": 10927, "status": "ok", "timestamp": 1621053857463, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="li9ao3w9EfFc" outputId="7c88fb9a-f926-471f-9672-bf06ea657b8e"
# Class Balance Analysis

def get_class_balance_plot(df, label_col, fig_title=None, plot_path=None): 
  label_counts = df.groupby(label_col).count().iloc[:, 0].rename("counts")
  if plot_path != None: 
    plt.style.use(os.path.join(plot_path, "plt_plot_style.mplstyle"))
  DIMS = [15,10]
  fig, ax = plt.subplots(figsize=DIMS)
  sns.barplot(x=label_counts.index, y=label_counts.values,ax=ax, color='darkturquoise')
  if fig_title == None: 
    fig_title = 'Label Frequency vs. Labels'
  ax.set_title(fig_title)
  ax.set_xlabel('Labels')
  ax.set_ylabel('Label Frequency')
  if plot_path != None: 
      save_fig(plot_path, fig)
  return fig
if __name__ == "__main__": 
  fig = get_class_balance_plot(df, 'label', plot_path=ROOT_PLOT)


# + id="lWOzXN_tKMfT"
def get_balanced_df(df, label_column='label', sample_size=None): 
  if sample_size == None: 
    sample_size = df.groupby(label_column).count().iloc[:,0].rename(label_column).min()
  balanced_dfs = []
  for label in df[label_column].unique():
     balanced_dfs.append(df.loc[df[label_column] == label].sample(sample_size))
  balanced_df = pd.concat(balanced_dfs)
  return balanced_df
if __name__ == "__main__": 
  balanced_df = get_balanced_df(df)

# + [markdown] id="SK68DHaMMniH"
# # Dimensionality Reduction with UMAP and PCA

# + colab={"base_uri": "https://localhost:8080/", "height": 686} executionInfo={"elapsed": 13516, "status": "ok", "timestamp": 1621053860066, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="39DRRm17PyHg" outputId="e8429f7d-a00b-4284-fa0c-96e72e2d2820"
# UMAP Projection to 2D
if __name__ == "__main__": 
  features = df.loc[:, df.columns != "label"]
  umap_proj_arr = umap.UMAP(random_state=42).fit_transform(features)
  umap_proj_df = pd.DataFrame(umap_proj_arr, columns=["dim_0", "dim_1"])
  umap_proj_df = umap_proj_df.set_index(features.index)
  clusters_df = pd.merge(umap_proj_df, features, left_index=True, right_index=True)
  clusters_df = pd.merge(df['label'], clusters_df, left_index=True, right_index=True)
  fig = plot_clusters_2D(clusters_df, 'dim_0', 'dim_1', 'label',fig_title="UMAP projection of combined datasets - global only", plot_path=ROOT_PLOT )

# + colab={"base_uri": "https://localhost:8080/", "height": 686} executionInfo={"elapsed": 1089, "status": "ok", "timestamp": 1621054196611, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="N1ghNTGlKeDh" outputId="332ecdab-662c-4f47-db07-6fe9f39aa190"
# PCA to 2 or 3 dimensions
from sklearn.decomposition import PCA
if __name__ == "__main__": 
  features = df.loc[:, df.columns != "label"]
  pca = PCA(n_components=2, random_state=42)
  pca_df = pd.DataFrame(pca.fit_transform(features), columns=["dim_0", "dim_1"]).set_index(features.index)
  pca_df = pd.merge(df['label'], pca_df, left_index=True, right_index=True)
  fig = plot_clusters_2D(pca_df, 'dim_0', 'dim_1', 'label',fig_title="PCA 2D decomposition of combined datasets - global only", plot_path=ROOT_PLOT )

# + [markdown] id="J18EGglEMhxT"
# # Feature Analysis with all features

# + colab={"base_uri": "https://localhost:8080/", "height": 453} executionInfo={"elapsed": 62665, "status": "ok", "timestamp": 1621053929446, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="Q9JtLyT4nX24" outputId="e0de1220-4c96-4e7d-fb7c-4144407a1f70"
# Random Forest Wrapper for LOOCV

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

def rf_loovc(X: pd.DataFrame, y: pd.Series, standard_scale: bool=False, power_transform:bool=False): 
  loo = LeaveOneOut()
  importances = []
  y_true = []
  y_predict = []

  for train_idx, test_idx in loo.split(X):
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
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train,y_train)
    importances.append(rf.feature_importances_)
    y_predict.append(rf.predict(X_test)[0])
    y_true.append(y_test[0])

  acc = accuracy_score(y_true, y_predict)
  print(f"Accuracy: {acc}")
  labels = X.columns
  feature_importances = np.array(importances).mean(axis=0)
  feature_importances_df = pd.DataFrame(sorted(zip(map(lambda x: round(x,6), feature_importances), labels), reverse=True), columns=["Importance", "Feature"])
  print("Features sorted by score:")
  display(feature_importances_df)
  results_df = pd.DataFrame({
      "y_true": y_true,
      'y_pred': y_predict
  }).set_index(X.index)
  return results_df, feature_importances_df
  
if __name__ == "__main__": 
  X, y = df.drop(columns="label"), df['label']
  results_df, feature_importances_df = rf_loovc(X, y, False, False)
  feature_importances_df.to_csv(f"{ROOT_RESULTS}feature_importances.csv", index=False)


# + colab={"base_uri": "https://localhost:8080/", "height": 747} executionInfo={"elapsed": 1106, "status": "ok", "timestamp": 1621053946713, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="d84NsCPINRzV" outputId="fb95584a-c8b0-4e82-ea24-e75f20cc3a28"
# Confusion matrix 
if __name__ == "__main__": 
  labels = results_df['y_true'].unique()
  cm = confusion_matrix(results_df['y_true'], results_df['y_pred'], labels=labels)
  fig = confusion_matrix_plot(cm, labels, fig_title= "All Feature Confusion Matrix", plot_path=ROOT_PLOT)

# + id="7G5A3c7qzmDo"
if __name__ == "__main__": 
  fig = plot_feature_importance(feature_importances_df, "Full Data Feature Importances", feature_col='Feature', importance_col='Importance', plot_path=ROOT_PLOT)

# + colab={"base_uri": "https://localhost:8080/", "height": 630} executionInfo={"elapsed": 1034, "status": "ok", "timestamp": 1621053951209, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="sabcyV1q0Edm" outputId="84ce06f1-9b26-4df8-9238-cec218f06cc9"
if __name__ == "__main__": 
  fig = plot_feature_importance(feature_importances_df, "TOP 10 Full Data Feature Importances", feature_col='Feature', importance_col='Importance', n_features=10, plot_path=ROOT_PLOT)

# + colab={"base_uri": "https://localhost:8080/", "height": 686} executionInfo={"elapsed": 2748, "status": "ok", "timestamp": 1621054089451, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="PJKoacNBT-d8" outputId="705c5892-7a1a-4571-9cf0-676bf2992418"
# UMAP Projection to 2D
if __name__ == "__main__": 
  THRESHOLD = 0.002
  df_top_features_002 = get_important_features(df, feature_importances_df, threshold=THRESHOLD)
  features = df_top_features_002.loc[:, df_top_features_002.columns != "label"]
  umap_proj_arr = umap.UMAP(random_state=42).fit_transform(features)
  umap_proj_df = pd.DataFrame(umap_proj_arr, columns=["dim_0", "dim_1"])
  umap_proj_df = umap_proj_df.set_index(features.index)
  clusters_df = pd.merge(umap_proj_df, features, left_index=True, right_index=True)
  clusters_df = pd.merge(df['label'], clusters_df, left_index=True, right_index=True)
  fig = plot_clusters_2D(clusters_df, 'dim_0', 'dim_1', 'label',fig_title="UMAP projection of 0.02% features", plot_path=ROOT_PLOT )

# + colab={"base_uri": "https://localhost:8080/", "height": 686} executionInfo={"elapsed": 872, "status": "ok", "timestamp": 1621054212128, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="9AoLN_pUW7Dr" outputId="cb06763d-2253-491a-9b50-8b967e4d204d"
# PCA to 2 or 3 dimensions
from sklearn.decomposition import PCA
if __name__ == "__main__": 
  df_top_features_002 = get_important_features(df, feature_importances_df, threshold=THRESHOLD)
  features = df_top_features_002.loc[:, df_top_features_002.columns != "label"]
  pca = PCA(n_components=2, random_state=42)
  pca_df = pd.DataFrame(pca.fit_transform(features), columns=["dim_0", "dim_1"]).set_index(features.index)
  pca_df = pd.merge(df['label'], pca_df, left_index=True, right_index=True)
  fig = plot_clusters_2D(pca_df, 'dim_0', 'dim_1', 'label',fig_title="PCA 2D decomposition of 0.02% features", plot_path=ROOT_PLOT )

# + colab={"base_uri": "https://localhost:8080/", "height": 720} executionInfo={"elapsed": 2803, "status": "ok", "timestamp": 1621055339903, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="ab_Oad_fZ0dt" outputId="cf0cf3fc-856e-48c7-b026-ed71431cac6f"
# PCA to 95% and then UMAP to 2D
from sklearn.decomposition import PCA
if __name__ == "__main__": 
  # df_top_features_002 = get_important_features(df, feature_importances_df, threshold=THRESHOLD)
  features = df.loc[:, df.columns != "label"]
  pca = PCA(n_components=0.95,svd_solver='full', random_state=42)
  features = pca.fit_transform(features)
  print(f"Shape features {features.shape}")
  umap_proj_arr = umap.UMAP(random_state=42).fit_transform(features)
  print(f"Shape umap_proj_arr {umap_proj_arr.shape}")
  umap_proj_df = pd.DataFrame(umap_proj_arr, columns=["dim_0", "dim_1"])
  umap_proj_df = umap_proj_df.set_index(df.index)
  # clusters_df = pd.merge(umap_proj_df, features, left_index=True, right_index=True)
  clusters_df = pd.merge(df['label'], umap_proj_df, left_index=True, right_index=True)
  fig = plot_clusters_2D(clusters_df, 'dim_0', 'dim_1', 'label',fig_title="UMAP projection of PCA decomposition datasets", plot_path=ROOT_PLOT )

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 14354, "status": "ok", "timestamp": 1620957539879, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="AItfFVZimNPc" outputId="9772d6e0-5693-44eb-b7c9-88d088cd3a25"
if __name__ == "__main__": 
  top_features = feature_importances_df[:3]['Feature'].values.tolist()
  dist_plot_df = df.copy(deep=True)

  label_dict = {}
  for label in dist_plot_df.label.unique(): 
        label_dict[label] = get_label_ids(df, label)

  for label in label_dict:
      for feature in top_features:
          sample_name = f"{label}"
          fig_title = f"{feature}_Histogram-of-{label}-vs-Rest_Combined-Dataset"
          fig = generate_feature_dist_plot(dist_plot_df, label_dict[label], feature, sample_name, hue_col=f"{label} vs. rest", title_string=fig_title, plot_path=ROOT_PLOT)

  for feature in top_features:
    fig_title = f"{feature}_Histogram_Combined-Dataset"
    fig = generate_feature_dist_plot(dist_plot_df, [], feature, "", hue_col="label", title_string=fig_title, plot_path=ROOT_PLOT, one_vs_rest=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 689} executionInfo={"elapsed": 1167, "status": "ok", "timestamp": 1620957002327, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="iwwaZgXljAxS" outputId="9c31f2ab-00d0-4e13-e8b2-8d0cb2a97e00"
if __name__ == "__main__": 
  feature = 'Age'
  fig_title = f"{feature}_Histogram_Combined-Dataset_stack"
  fig = generate_feature_dist_plot(dist_plot_df, [], feature, "", hue_col="label",
                                  title_string=fig_title, plot_path=ROOT_PLOT,
                                  one_vs_rest=False, kde=False, multiple="stack",
                                  stat='count')

# + colab={"base_uri": "https://localhost:8080/", "height": 689} executionInfo={"elapsed": 2700, "status": "ok", "timestamp": 1623966564357, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="PgQqayAnkYOv" outputId="f3d80a06-fd6e-4343-91ae-d3b4ee5bb387"
if __name__ == "__main__": 
  # ages = [age for age in range(df['Age'].unique().min(), df['Age'].unique().max())]
  ages = np.arange(df['Age'].min(), df['Age'].max(), 1)
  feature = 'Age'
  fig_title = f"{feature} Histogram"
  dist_plot_df = df.copy(deep=True)
  # plt.xticks(np.arange(df['Age'].min(), df['Age'].max(), 1))
  fig = generate_feature_dist_plot(dist_plot_df, [], feature, "Age", hue="label",
                                  title_string=fig_title, plot_path=ROOT_PLOT,
                                  one_vs_rest=False, kde=False, multiple="dodge",
                                  stat='count', shrink=0.9, bins=ages, common_norm=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 16981, "status": "ok", "timestamp": 1621100990420, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="iCbYDC2ZIT_P" outputId="3cd5f0d2-1eed-466b-8e18-f9cf431e0aef"
# EDA after preprocessing
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

if __name__ == "__main__": 
  df = pd.read_csv("./drive/MyDrive/research/structural-connectivity/data/combined_datasets.csv")
  feature_importances_df = pd.read_csv("./drive/MyDrive/research/structural-connectivity/results/feature_importances.csv")

  pipe = Pipeline(steps=[
                       ("Standard_Scaler", StandardScaler()),
                       ("Power Transform", PowerTransformer()),
  ])

  top_features = feature_importances_df[:3]['Feature'].values.tolist()
  dist_plot_df = df.copy(deep=True)
  display(dist_plot_df[top_features])
  dist_plot_df[top_features]=pipe.fit_transform(dist_plot_df[top_features])
  display(dist_plot_df[top_features])

  label_dict = {}
  for label in dist_plot_df.label.unique(): 
        label_dict[label] = get_label_ids(df, label)

  for label in label_dict:
      for feature in top_features:
          sample_name = f"{label}"
          fig_title = f"{feature}-Transformed_Histogram-of-{label}-vs-Rest_Combined-Dataset"
          fig = generate_feature_dist_plot(dist_plot_df, label_dict[label], feature, sample_name, hue_col=f"{label} vs. rest", title_string=fig_title, plot_path=ROOT_PLOT)

  for feature in top_features:
    fig_title = f"{feature}-Transformed_Histogram_Combined-Dataset"
    fig = generate_feature_dist_plot(dist_plot_df, [], feature, "", hue_col="label", title_string=fig_title, plot_path=ROOT_PLOT, one_vs_rest=False)

# + [markdown] id="ToWvaoNCmt_3"
# # END OF EDA

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8614, "status": "ok", "timestamp": 1623966944483, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="5PO7gGcqnBPP" outputId="7634ba94-6c16-4d20-c977-c9131a9ec268" language="bash"
# pip install jupytext
# cd drive/MyDrive/research/structural-connectivity
# jupytext --to py EDA.ipynb

# + colab={"base_uri": "https://localhost:8080/", "height": 390} executionInfo={"elapsed": 1045, "status": "ok", "timestamp": 1620838814483, "user": {"displayName": "Mike Lasby", "photoUrl": "", "userId": "01259677206324830508"}, "user_tz": 360} id="BqhpKc2mr9ni" outputId="f9901c80-c968-4fcb-8276-1f513e693172"
# # Experiment summary
# experiments = {
#     "Trial" :[1, 2, 3,
#              4, 5, 6,
#              7, 8, 9,
#               10, "11 - GBF"],
#     "Scaling": ['None', 'None', 'None',
#                 'Standard Scaler', 'Power Transform', 'None',
#                 'Standard Scaler', 'Power Transform', 'None',
#                 'None', 'None'],
#     "Features used" : ["All","UMAP","PCA",
#                        "All", 'All', '0.001 importance',
#                       '0.001 importance', '0.001 importance', '0.002 importance', 'Balanced Sample - All', "0.002 importance"],
#     "Num Features": [1356, 2, 2,
#                      1356, 1356, 190,
#                      190, 190, 12, 1356, 12],
#     'Accuracy': [0.2982456140350877, 0.22807017543859648, 0.2222222222222222,
#                  0.29239766081871343, 0.30409356725146197, 0.3508771929824561,
#                  0.3508771929824561, 0.34502923976608185, 0.38011695906432746,
#                  0.23809523809523808, 0.39766081871345027]
# }
# experiment_results = pd.DataFrame(experiments).sort_values(by='Accuracy', ascending=False)
# experiment_results.to_csv(f"{ROOT_RESULTS}experiment_results.csv", index=False)
# display(experiment_results)
