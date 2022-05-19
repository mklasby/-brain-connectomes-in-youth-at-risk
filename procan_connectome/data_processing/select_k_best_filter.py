"""Transformer to obtain the most important features according to an sklearn.ensemble model. 

Typical usage example: 
    rfif = rf_importance_filter.RFImportanceFilter('classifier', threshold=0.001)
    rfif.fit(X_train)
    X_test = rfif.transform(X_test)
"""
from scipy.sparse.construct import random
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
import numpy as np 
import pandas as pd
import logging
from sklearn.feature_selection import SelectKBest


class SelectKBestFilter(BaseEstimator, TransformerMixin): 
    """Transformer to obtain the most important features according to an sklearn.feature_selection.SelectKBest instance. 
    
    Attributes: 
        mode: Type of task, "classifier" or "regression". 
        threshold: Minimum importance required for feature to be kept. 
        ignore_features: Features to keep regardless of importance. 
        copy: 
        rf: 
            Ensemble model used to find importances. Use this parameter if you want to use a forest with 
            custom hyperparameters. 
    """
    
    def __init__(self, k:int=10, ignore_features:list=None, copy:bool=True,
                 ):
        self.ignore_features = ignore_features
        self.copy = copy
        self.k = k
    
    def fit(self, X: pd.DataFrame, y:pd.Series):
        """Public method to fit transformer to X. 
        
        Args: 
            X: A dataframe to fit to. 
            y: Target vector to fit to, required to find importances.  
        """
        self._fit(X, y)
        return self
    
    def _reset(self): 
        """Private method to reset fit parameters."""
        if hasattr(self, "kbest_"): 
            del self.support_
            del self.kbest_

    def _fit(self, X: pd.DataFrame, y:pd.Series): 
        """Private method to implement fit."""
        self._reset()
        self.kbest_ = SelectKBest(k=self.k)
        self.kbest_.fit(X, y)
        self.support_ = self.kbest_.get_support()            
        
    def transform(self, X: pd.DataFrame, y=None, copy:bool=None)-> pd.DataFrame:
        """Transforms X by dropping features that match regex.  
        
        Args: 
            X: 
                A dataframe to transform
            y: 
                Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
                Defaults to None. 
            copy: 
                Optional parameter to use instead of self.copy. Defaults to None.
        
        Returns: 
            Transformed dataframe with dropped features.
        
        Raises: 
            NotFittedError: If the transformer has not been fit. 
        """
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # TODO: Check input with _validate_data or sim? 
        if copy: 
            X = X.copy(deep=True)
        features_to_keep = X.columns[self.support_]
        if self.ignore_features != None: 
            for feature in self.ignore_features: 
                if feature not in features_to_keep: 
                    features_to_keep.append(feature)
        X = X[features_to_keep]
        return X

    