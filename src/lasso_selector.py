
# src/lasso_selector.py

from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base_selector import BaseFeatureSelector

class LassoFeatureSelector(BaseFeatureSelector):
    def __init__(self, cv=3, random_state=101, max_iter=10000):
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter

    def select(self, X, y):
        pipeline = make_pipeline(
            StandardScaler(),
            LassoCV(cv=self.cv, random_state=self.random_state, max_iter=self.max_iter)
        )
        pipeline.fit(X, y)
        mask = pipeline.named_steps['lassocv'].coef_ != 0
        return X.loc[:, mask]
