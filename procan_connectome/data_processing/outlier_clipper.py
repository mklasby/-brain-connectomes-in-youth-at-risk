"""Transformer to clip outliers. 

Typical usage example: 
    
    oc = OutlierClipper(lower_limit=0.05, upper_limit=0.95, regexs=['Fee', 'pay'], copy=True, case=True)
    oc.fit(X_train)
    X_test = oc.transform(X_test)
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np 
import pandas as pd
import logging


class OutlierClipper(BaseEstimator, TransformerMixin): 
    """Customer transformer to clip outliers to lower and upper limits. 
    
    Can target specific columns using regexs parameter or apply to all columns. 
    
    Attributes: 
        lower_limit: Lower quantile to clip to, eg. clip values <= 0.01 (1%) Defaults to 0.01
        upper_limit: Upper quantile to clip to, eg., clip values >= 0.99 (99%) Defaults to 0.99
        regexs: List of strings to use to filter column names. If None, all columns considered valid targets for clipping. 
        case: Boolean, defaults to False. If False, regex filter ignores case. 
        copy: A bool indicating if the dataframes passed to transform should be copied before being transformed. 
    """
    
    def __init__(self, lower_limit:float=0.01, upper_limit:float=0.99, regexs:list=None, case:bool=False,
                 copy:bool=True):
        """Init a new OutlierClipper object"""
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.regexs = regexs
        self.copy = copy
        self.case = case
        #TODO: Add a whitelist for specific columns? 
    
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
            del self.quantiles_
            del self.to_clip_

    def _fit(self, X: pd.DataFrame, y=None): 
        """Private method to implement fit."""
        self._reset()
        self.quantiles_ = {}
        self.to_clip_ = []
        # TODO: Check input with _validate_data or sim? 
        
        if self.regexs == None: 
            self.to_clip_.extend(X.columns.tolist())
            logging.info(f"No regexs passed. All features added to to_clip_")
        else: 
            for regex in self.regexs: 
                if not self.case: 
                    regex = f"(?i){regex}"
                self.to_clip_.extend(X.filter(regex=regex).columns.tolist())     
            logging.info(f"Found {len(self.to_clip_)} features matching the regexs for outlier clipper")
        
        for feature in self.to_clip_: 
            self.quantiles_[feature] = X[feature].quantile([self.lower_limit, self.upper_limit]).values
            
        
    def transform(self, X: pd.DataFrame, y=None, copy:bool=None): 
        """Transforms X by clipping outliers based on fit dataframe. 
        
        Args: 
            X: 
                A dataframe to transform
            y: 
                Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
                Defaults to None. 
            copy: 
                Optional parameter to use instead of self.copy. Defaults to None.
        
        Returns: 
            Transformed dataframe with outliers clipped based on values in fit() dataframe.           
    
        Raises: 
            NotFittedError: If the transformer has not been fit. 
        """
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # TODO: Check input with _validate_data or sim? 
        if copy: 
            X = X.copy(deep=True)
        for feature in self.to_clip_:
            X[feature] = np.clip(X[feature], self.quantiles_[feature][0], self.quantiles_[feature][1])
        return X
    