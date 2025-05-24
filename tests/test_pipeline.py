import pandas as pd
import numpy as np
from src.feature_explainer import FeatureExplainer
from src.lasso_selector import LassoFeatureSelector
from src.rfecv_selector import RFECVFeatureSelector

def make_dummy_data(n_samples=50, n_features=10):
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(np.random.rand(n_samples) * 100, name="CP score")
    df = X.copy()
    df["CP score"] = y
    df["date"] = pd.date_range(start="2021-01-01", periods=n_samples)
    return df

def test_lasso_selector_runs():
    df = make_dummy_data()
    selector = LassoFeatureSelector(cv=3, random_state=42)
    X = df.drop(columns=["CP score", "date"])
    y = df["CP score"]
    X_selected = selector.select(X, y)
    assert not X_selected.empty
    assert X_selected.shape[1] <= X.shape[1]

def test_rfecv_selector_runs():
    df = make_dummy_data()
    selector = RFECVFeatureSelector(step=0.1, cv=3)
    X = df.drop(columns=["CP score", "date"])
    y = df["CP score"]
    X_selected = selector.select(X, y)
    assert not X_selected.empty
    assert X_selected.shape[1] <= X.shape[1]

def test_feature_explainer_global_local():
    df = make_dummy_data()
    selector = LassoFeatureSelector(cv=3)
    explainer = FeatureExplainer(df, selector, n_train=40, drop_cols=[], max_display=5, sample_no=10)

    shap_values = explainer.explain_global()
    assert shap_values.values.shape[0] == 40

    local_shap, instance = explainer.explain_local()
    assert local_shap.values.shape[0] == instance.shape[0]
