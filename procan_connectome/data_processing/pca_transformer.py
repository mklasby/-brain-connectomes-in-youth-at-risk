"""Todo"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd
import logging


class PCATransformer(BaseEstimator, TransformerMixin): 
    """Transformer wrapper for PCA
    Attributes: 
        random_state: random_state int to use. 
        copy: A bool indicating if the dataframes passed to transform should be copied before being transformed.
        append_cols: If True, append umap dims to original features instead of dropping original features. 
        n_components: n_components for PCA. See sklearn docs.
    """
    
    def __init__(self, random_state:int = 42, copy:bool=True, append_cols:bool=False, n_components: float = 0.95):
        self.copy = copy
        self.random_state = random_state
        self.append_cols = append_cols
        self.n_components = n_components
    
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
        if hasattr(self, "pca_"): 
            del self.pca_

    def _fit(self, X: pd.DataFrame, y=None): 
        """Private method to implement fit."""
        self._reset()
        self.pca_ = PCA(n_components=self.n_components, svd_solver="full")
        self.pca_.fit(X)
        
    def transform(self, X: pd.DataFrame, y=None, copy:bool=None) -> pd.DataFrame: 
        """Transforms X with PCA. 
        
        Args: 
            X: 
                A dataframe to transform
            y: 
                Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
                Defaults to None. 
            copy: 
                Optional parameter to use instead of self.copy. Defaults to None.
        
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
        X_transformed = self.pca_.transform(X)
        X_transformed = pd.DataFrame(data=X_transformed,
                                     index=X.index)
        if self.append_cols: 
            X = X.join(X_transformed)
        else: 
            X = X_transformed
        return X
    