"""Transformer to power transform a dataframe and maintain column names.  

Typical usage example: 
    
    pt = PowerTransform(ignore_features=['label_col', 'index_col'])
    X_train = pt.fit_transform(X_train)
    X_test = oc.transform(X_test)
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted
import numpy as np 
import pandas as pd
import logging


class PowerTransformerWrapper(BaseEstimator, TransformerMixin): 
    """Wrapper for power transformer which returns as pd.DataFrame rather than np.arr
    
    Intent is to maintain column names and still ensure that we can use the powerTransformer in a single pipeline.  
    
    Attributes: 
        ignore_features: List of feature names to ignore. 
        copy: A bool indicating if the dataframes passed to transform should be copied before being transformed. 
    """
    
    def __init__(self, ignore_features:list=None, copy:bool=True):
        self.copy = copy
        self.ignore_features = ignore_features
    
    def fit(self, X: pd.DataFrame, y=None):
        """Public method to fit transformer to X. 
        
        Args: 
            X: A dataframe to fit to. 
            y: Ignored. Defaults to None. 
            
        
        Note that y is ignored, but noted as arg to maintain compatibility with TransformerMixin.
        """
        self._fit(X, y=y)
        return self
    
    def _reset(self): 
        """Private method to reset fit parameters."""
        if hasattr(self, "quantiles_"): 
            del self.pt_
            del self.transform_features_

    def _fit(self, X: pd.DataFrame, y=None): 
        """Private method to implement fit."""
        self._reset()
        # TODO: Check input with _validate_data or sim? 
        self.pt_ = PowerTransformer()
        if self.ignore_features == None: 
            self.transform_features_ = X.columns
        else: 
            self.transform_features_ = [col for col in X.columns if col not in self.ignore_features]
        self.pt_.fit(X[self.transform_features_])
        logging.info(f"PowerTransformer fit to {len(self.transform_features_)} features.")
            
        
    def transform(self, X: pd.DataFrame, y=None, copy:bool=None):
        """Transforms X by power transforming all columns not listed in ignore_cols. 
        
        Args: 
            X: 
                A dataframe to transform
            y: 
                Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
                Defaults to None. 
            copy: 
                Optional parameter to use instead of self.copy. Defaults to None.
        
        Returns: 
            Transformed dataframe with power transformed features.           
    
        Raises: 
            NotFittedError: If the transformer has not been fit. 
        """
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # TODO: Check input with _validate_data or sim? 
        if copy: 
            X = X.copy(deep=True)
        X[self.transform_features_] = self.pt_.transform(X[self.transform_features_])
        return X
    