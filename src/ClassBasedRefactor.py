import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from interpret.glassbox import ExplainableBoostingRegressor

class FeatureExplainer:
    def __init__(self, df, method="lasso", n_train=40, drop_cols=None, max_display=5, sample_no=33):
        self.df = df.dropna(axis=1)
        self.method = method
        self.n_train = n_train
        self.drop_cols = drop_cols or []
        self.max_display = max_display
        self.sample_no = sample_no
        self.X, self.y = self._prepare_data()
        self.X_selected = self._feature_selection()

    def _prepare_data(self):
        df = self.df.drop(columns=self.drop_cols, errors='ignore')
        X = df.drop(['date', 'CP score'], axis=1)
        y = df['CP score']
        return X.iloc[:self.n_train], y.iloc[:self.n_train]

    def _feature_selection(self):
        if self.method == "lasso":
            pipe = make_pipeline(StandardScaler(), LassoCV(cv=3, random_state=101))
            pipe.fit(self.X, self.y)
            mask = pipe.named_steps['lassocv'].coef_ != 0
        elif self.method == "rfecv":
            rfecv = RFECV(GradientBoostingRegressor(random_state=101), step=0.05, cv=3, scoring='neg_mean_absolute_error')
            rfecv.fit(self.X, self.y)
            mask = rfecv.support_
        else:
            return self.X  # No FS
        return self.X.loc[:, mask]

    def _save_plot(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def explain_global(self):
        model = ExplainableBoostingRegressor()
        model.fit(self.X_selected, self.y)
        X_sample = shap.utils.sample(self.X_selected, 100)
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(self.X_selected)

        shap.plots.bar(shap_values, max_display=self.max_display, show=False)
        self._save_plot(f"shap_bar_{self.method}.png")

        shap.summary_plot(shap_values, self.X_selected, max_display=self.max_display, show=False)
        self._save_plot(f"shap_summary_{self.method}.png")

        return shap_values

    def explain_local(self):
        model = GradientBoostingRegressor().fit(self.X_selected, self.y)
        explainer = shap.Explainer(model.predict, self.X_selected)
        shap_values = explainer(self.X_selected)

        shap.plots.waterfall(shap_values[self.sample_no], max_display=self.max_display)
        self._save_plot(f"shap_local_{self.method}.png")

        return shap_values[self.sample_no], self.X_selected.iloc[self.sample_no]
