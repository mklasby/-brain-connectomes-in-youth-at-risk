from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline 
from sklearn.base import BaseEstimator
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import logging 
from sklearn.metrics import get_scorer

class PipelineEvaluator(BaseEstimator):
    """PipelineEvaluator evaluates several pipelines and provides results on best performing candidate. 
    """
    
    def __init__(self,
                 pipelines: list, 
                 estimator: BaseEstimator,
                 scorer:str = 'accuracy',
                 test_size:float=0.33,
                 random_state:float=42,
                 include_dummy_case:bool=True): 
        self.pipelines = pipelines
        self.estimator = estimator
        self.scorer = scorer
        self.test_size=test_size
        self.random_state=random_state
        self.include_dummy_case = include_dummy_case
        
    def _reset(self): 
        """Private method to reset fit parameters."""
        if hasattr(self, 'names_'): 
            del self.scores_
            del self.names_
            del self.pipe_dict_
    
    def fit(self, X:pd.DataFrame, y: pd.Series): 
        self._fit(X,y)
    
    def _fit(self, X:pd.DataFrame, y: pd.Series): 
        self._reset()
        self.scores_ = []
        self.names_ = []
        self.pipe_dict_ = {}
        
        if self.include_dummy_case: 
            self._process_dummy_case(X,y)
        
        for pipe in self.pipelines:
            self._process_pipe(pipe, X, y)
        logging.info(f"Evaluation complete on {len(self.pipelines)} pipelines.")
        
    def _process_dummy_case(self, X:pd.DataFrame, y:pd.Series):
        X = X.copy(deep=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size, random_state=self.random_state)
        pipe_name = 'No_pipeline'
        self.names_.append(pipe_name)
        logging.info(f"Evaluating {pipe_name} pipeline...")
        self._score_pipeline(X_train, X_test, y_train, y_test)
        
            
    def _process_pipe(self, pipe: Pipeline, X:pd.DataFrame, y:pd.Series):
        X = X.copy(deep=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size, random_state=self.random_state)
        pipe_name = self._get_pipeline_name(pipe)
        logging.info(f"Evaluating {pipe_name} pipeline...")
        self.names_.append(pipe_name)
        X_train = pipe.fit_transform(X_train, y_train)
        X_test = pipe.transform(X_test)
        self.pipe_dict_[pipe_name] = pipe 
        self._score_pipeline(X_train, X_test, y_train, y_test)      
        
    def _get_pipeline_name(self, pipe:Pipeline) -> str:
        pipe_name = "" 
        for name, transformer in pipe.steps: 
            pipe_name = pipe_name + name + "-"
        pipe_name = pipe_name[:-1]
        return pipe_name
    
    def _score_pipeline(self,
                        X_train:pd.DataFrame,
                        X_test:pd.DataFrame,
                        y_train:pd.DataFrame,
                        y_test:pd.DataFrame): 
        estimator = clone(self.estimator) 
        estimator.fit(X_train, y_train)
        scorer = get_scorer(self.scorer)
        self.scores_.append(scorer(estimator, X_test, y_test))
        
    def get_results(self) -> pd.DataFrame: 
        check_is_fitted(self)
        df = pd.DataFrame(
            {
                "Pipeline_name": self.names_,
                "Score": self.scores_
            }
        )
        return df 
        
    def get_fitted_pipes(self) -> dict: 
        check_is_fitted(self)
        return self.pipe_dict_