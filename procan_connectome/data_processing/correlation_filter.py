"""Transformer to drop correlated columns. 

Typical usage example: 
    
    cf = CorrelationFilter(threshold=0.8, columns_to_ignore=['label_col'])
    cf.fit(X_train)
    X_test = cf.transform(X_test)
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np 
import pandas as pd
import logging


class CorrelationFilter(BaseEstimator, TransformerMixin): 
    """Custom transformer to remove columns that are correlated. 
    
    Only one columns is dropped for each correlated pair.  
    
    Attributes: 
        copy: A bool indicating if the dataframes passed to transform should be copied before being transformed. 
        threshold: 
            Threshold at which a column is considered correlated. 
            (eg., 0.9 means drop all columsn with >= 90% correlation)
        columns_to_ignore: 
            list of columns names to ignore. These columns will not be considered when calculating correlations nor 
            when dropping columns. 
    """
    
    
    def __init__(self, threshold:float = 0.9, columns_to_ignore:list=None, copy:bool=True):
        """Inits a CF object."""
        self.threshold = threshold
        self.columns_to_ignore = columns_to_ignore
        self.copy = copy
    
    def fit(self, X: pd.DataFrame, y=None):
        """Public method to fit transformer to X. 
        
        Finds columns where correlations > threshold and stores them in self.to_drop_ attribute. 
        
        Args: 
            X: A dataframe to fit to. 
            y: Ignored. Defaults to None. 
            
        
        Note that y is ignored, but noted as arg to maintain compatibility with TransformerMixin.
        """
        self._fit(X, y=y)
        return self
    
    def _reset(self): 
        """Private method to reset fit parameters."""
        if hasattr(self, "to_drop_"): 
            del self.to_drop_

    def _fit(self, X: pd.DataFrame, y=None): 
        """Private method to implement fit."""
        self._reset()
        self.to_drop_ = []
        # TODO: Check input with _validate_data or sim? 
        corrdata = X.copy(deep=True)
        if self.columns_to_ignore is not None: 
            corrdata = corrdata.drop(columns=self.columns_to_ignore)
        corr_matrix = corrdata.corr().abs()
        upper_diagonal = pd.DataFrame(np.triu(corr_matrix, k=1), columns=corr_matrix.columns, index=corr_matrix.index)
        self.to_drop_ = [column for column in upper_diagonal.columns if any(upper_diagonal[column] >= self.threshold)]
        logging.info(f"Found {len(self.to_drop_)} correlated features to drop")
        
    def transform(self, X: pd.DataFrame, y=None, copy:bool=None): 
        """Transforms X by dropping one of each correlated column pair. 
        
        Args: 
            X: 
                A dataframe to transform
            y: 
                Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
                Defaults to None. 
            copy: 
                Optional parameter to use instead of self.copy. Defaults to None.
        
        Returns: 
            Transformed dataframe with one of each corrleated column pair dropped.           
    
        Raises: 
            NotFittedError: If the transformer has not been fit. 
        """
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        if copy: 
            X = X.copy(deep=True)
        X = X.drop(columns=self.to_drop_)
        return X
