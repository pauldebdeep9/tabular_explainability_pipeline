
# src/__init__.py

from .base_selector import BaseFeatureSelector
from .lasso_selector import LassoFeatureSelector
from .feature_explainer import FeatureExplainer
from .rfecv_selector import RFECVFeatureSelector
# You can later add more like:


__all__ = [
    "base_selector",
    "lasso_selector",
    "feature_explainer",
    "rfecv_selector"
]

