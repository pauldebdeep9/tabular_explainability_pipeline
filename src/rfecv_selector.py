
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor
from .base_selector import BaseFeatureSelector

class RFECVFeatureSelector(BaseFeatureSelector):
    def __init__(self, step=0.05, cv=3, scoring="neg_mean_absolute_error", random_state=101):
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def select(self, X, y):
        model = GradientBoostingRegressor(random_state=self.random_state)
        rfecv = RFECV(estimator=model, step=self.step, cv=self.cv, scoring=self.scoring)
        rfecv.fit(X, y)
        return X.loc[:, rfecv.support_]
