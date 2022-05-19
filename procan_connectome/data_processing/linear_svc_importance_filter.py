"""Transformer to obtain the most important features according to an sklearn.svm.LinearSVC model. 

Typical usage example: 
    lsif = estimator_importance_filter.estimatorImportanceFilter('classifier', threshold=0.001)
    lsif.fit(X_train)
    X_test = lsif.transform(X_test)
"""
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
import numpy as np 
import pandas as pd
import logging
from sklearn.svm import LinearSVC


class LinearSVCImportanceFilter(BaseEstimator, TransformerMixin): 
    """Transformer to obtain the most important features according to an sklearn.ensemble model. 
    
    Attributes: 
        mode: Type of task, "classifier" or "regression". 
        threshold: Minimum importance required for feature to be kept. 
        ignore_features: Features to keep regardless of importance. 
        copy: 
        estimator: 
            Model used to find importances. Use this parameter if you want to use a forest with 
            custom hyperparameters. 
    """
    
    def __init__(self, threshold:float = 0.01, ignore_features:list=None, copy:bool=True,
                estimator: LinearSVC=None, random_state:float=42, sort:bool=False):
        self.threshold = threshold
        self.ignore_features = ignore_features
        self.copy = copy
        self.estimator = estimator
        self.random_state=random_state
        self.sort = sort
    
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
        if hasattr(self, "estimator_"): 
            del self.feature_importances_df_
            del self.important_features_
            del self.estimator_
            del self.coefs_
            
    def _fit(self, X: pd.DataFrame, y:pd.Series): 
        """Private method to implement fit."""
        self._reset()
        if self.estimator == None:
                self.estimator_ = LinearSVC(random_state=self.random_state)
        else: 
            self.estimator_ = clone(self.estimator)
        
        self.estimator_.fit(X, y)
        self._get_feature_importance_df(X)
        logging.info(f"Found {len(self.important_features_.values.tolist())} important features.")
            
        
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
        features_to_keep = self.important_features_.values.tolist()
        if self.ignore_features != None: 
            for feature in self.ignore_features: 
                if feature not in features_to_keep: 
                    features_to_keep.append(feature)
        X = X[features_to_keep]
        return X
    
    
    def _get_feature_importance_df(self, X:pd.DataFrame):
        """Private method to select features with importances > threshold."""
        coefs = np.sum(np.abs(self.estimator_.coef_), axis=0)
        if self.sort: 
            self.coefs_ = pd.DataFrame(sorted(zip(map(lambda x: round(x,6), coefs), X.columns),reverse=True),
                                    columns=["Coef", "Feature"])
        else: 
            self.coefs_ = pd.DataFrame(list(zip(map(lambda x: round(x,6), coefs), X.columns)),
                                    columns=["Coef", "Feature"])
        coef_sum = self.coefs_['Coef'].sum()
        self.feature_importances_df_ = self.coefs_.copy()
        self.feature_importances_df_['Importance'] = self.feature_importances_df_['Coef'].apply(lambda x: x/coef_sum)
        self.important_features_ = self.feature_importances_df_.loc[self.feature_importances_df_['Importance']>=
                                                                   self.threshold]['Feature']

    