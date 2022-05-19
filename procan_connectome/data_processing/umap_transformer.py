"""Custom transformer to wrap umap dimensionality reduction.  

Typical usage example: 
    umap = UmapTransformer(copy=True)
    X_train = umap.fit_transform(X_train)
    X_test = umap.transform(X_test)
"""


from sklearn.base import BaseEstimator, TransformerMixin
import umap.umap_ as umap 
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd
import logging


class UmapTransformer(BaseEstimator, TransformerMixin): 
    """Transformer wrapper for UMAP
    
    Reduce dimensions of df_train features to 2D using UMAP from umap-learn. Option to append umap dimensions to 
    original dataframe instead of dropping. 
    
    Attributes: 
        standardize: If True perform standardScaler() on dataset before transform. 
        random_state: random_state int to use. 
        copy: A bool indicating if the dataframes passed to transform should be copied before being transformed.
        append_cols: If True, append umap dims to original features instead of dropping original features. 
    """
    
    def __init__(self, standardize:bool=True, random_state:int = 42, copy:bool=True, append_cols:bool=True):
        self.copy = copy
        self.standardize = standardize
        self.random_state = random_state
        self.append_cols = append_cols
    
    def fit(self, X: pd.DataFrame, y=None, **kwargs):
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
        if hasattr(self, "umap_"): 
            del self.umap_
            del self.pipe_

    def _fit(self, X: pd.DataFrame, y=None, **kwargs): 
        """Private method to implement fit."""
        self._reset()
        steps = []
        if self.standardize:
            steps.append(('ss', StandardScaler()))
        steps.append(('umap', umap.UMAP(random_state=self.random_state, **kwargs)))
        steps = tuple(steps)
        
        self.pipe_ = Pipeline(steps=steps)
        self.pipe_.fit(X)     
        
    def transform(self, X: pd.DataFrame, y=None, copy:bool=None, dim_0_name:str='dim_0',
                  dim_1_name:str='dim_1') -> pd.DataFrame: 
        """Transforms X by projecting into 2D with UMAP.  
        
        Args: 
            X: 
                A dataframe to transform
            y: 
                Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
                Defaults to None. 
            copy: 
                Optional parameter to use instead of self.copy. Defaults to None.
            dim_0_name:
                string for first umap dimension name. 
            dim_1_name:
                string for second umap dimension name. 
        
        Returns: 
            Transformed dataframe with projected features.            
    
        Raises: 
            NotFittedError: If the transformer has not been fit. 
        """
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # TODO: Check input with _validate_data or sim? 
        if copy: 
            X = X.copy(deep=True)
        X_transformed = self.pipe_.transform(X)
        X_transformed = pd.DataFrame(data={dim_0_name: X_transformed[:,0], dim_1_name: X_transformed[:,1]},
                                     index=X.index)
        if self.append_cols: 
            X = X.join(X_transformed)
        else: 
            X = X_transformed
        return X
    