import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import os
import h5py
import re
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from procan_connectome.config import PLOT_PATH, DATA_PATH

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

def get_functional_df(prefix_col:bool=True, global_only:bool=False): 
  """Gets functional dataset from .txt files. 

  Args: 
    prefix_col: If true, prefix all features columns with "fun"
  
  Returns: 
    dataframe of combined data functional data. 
  """
  ROOT_LOCATION = os.path.join(DATA_PATH, "functional_connectome_data")
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

def get_structural_df(prefix_col=True, global_only:bool=False): 
  """Gets structural dataset from .txt files. 

  Args: 
    prefix_col: If true, prefix all features columns with "str"
  
  Returns: 
    dataframe of combined data structural data. 
  """


  # ROOT_LOCATION = "./drive/MyDrive/research/structural-connectivity/Structural_connectivity/"
  ROOT_LOCATION = os.path.join(DATA_PATH, "Structural_connectivity")
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

def parse_dataset(functional=True, structural=True, global_only:bool=False): 
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

def get_dataset(fname:str, index_col:str = 'ID'):       
    df = pd.read_csv(os.path.join(DATA_PATH, fname))    
    df = df.set_index(index_col)
    return df  

if __name__ == "__main__": 
  global_only=False
  df = parse_dataset(global_only=global_only)
  df.to_csv(os.path.join(DATA_PATH, 'combined_datasets.csv'))