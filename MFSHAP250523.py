# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:41:56 2021

@author: 70K9734
"""


import pandas as pd
import numpy as np
import interpret.glassbox
import os
import time
import shap
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, chi2
from lassonet import LassoNetRegressor
import matplotlib.pyplot as plt


# model = make_pipeline(StandardScaler(with_mean=False), Lasso())


n_train= 40
max_display= 5
mode= 1

df= pd.read_csv('Data/CPExplain221014.csv')
df= df.iloc[:72]

sample_no= 33

method= 'lasso'
# method= 'rfecv'

drop_feat= ['Man_pmi_mx', 'Man_pmi_tr']

def clean(df):
    df = df.dropna(axis=1)  # Drop columns with NaNs
    X = df.drop(['date', 'CP score'], axis=1)
    X = X.drop(columns=drop_feat, errors='ignore')  # Drop additional specified features
    total_local_points = X.shape[0]
    return X, total_local_points


def select_training(df):
    X, total_local_points= clean(df)
    X_train= X.iloc[:n_train]
    y_train= df['CP score'][:n_train]
    return X_train, y_train



def feature_selection(df, method):
    X_train, y_train = select_training(df)

    if method == "rfecv":
        rfc_rf = GradientBoostingRegressor(random_state=101)
        rfecv_rf = RFECV(estimator=rfc_rf, step=0.05, cv=3, scoring='neg_mean_absolute_error')
        rfecv_rf.fit(X_train, y_train)
        support_mask = rfecv_rf.support_
    
    elif method == "lasso":
    # Create a pipeline with scaling + LassoCV
        model = make_pipeline(StandardScaler(), LassoCV(cv=3, random_state=101))
            # Fit the model
        model.fit(X_train, y_train)
            # Extract coefficients from the Lasso model
        lasso_coef = model.named_steps['lassocv'].coef_
            # Create a support mask where non-zero coefficients indicate selected features
        support_mask = lasso_coef != 0

    else:
        raise ValueError("Invalid method. Choose 'rfecv' or 'lassonet'.")

    selected_features = X_train.columns[support_mask]
    X_train = X_train[selected_features]
    print('Selected features:', list(selected_features))

    return X_train

# test= feature_selection(df, method)

def with_fs1(df, method):
    X_train= feature_selection(df, method)
    _, y_train = select_training(df)
    return X_train, y_train

def shap_value(df, method, mode):
    if mode==0:
        X_train, y_train = select_training(df)
        X100 = shap.utils.sample(X_train, 100)

    elif mode==1:
        X_train, y_train = with_fs1(df, method)
        X100 = shap.utils.sample(X_train, 100)
    return X100



def interpret_glassbox(df,
                       n_train,
                       method, 
                       max_display= max_display,
                       feature_selection_mode= mode):

    if feature_selection_mode== 0:
        X_train, y_train= select_training(df)
        X100= shap_value(df, method, mode)
        model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
        model_ebm.fit(X_train, y_train)

# explain the GAM model with SHAP
        explainer_ebm = shap.Explainer(model_ebm.predict, X100)
        shap_values_ebm = explainer_ebm(X_train)
        shap.plots.bar(shap_values_ebm, max_display= max_display)
        filename = f"shap_waterfall_{method}.png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print('Global features ebm')
        shap.summary_plot(shap_values_ebm, X_train, max_display=max_display)
        filename = f"shap_summary_{method}.png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    elif feature_selection_mode== 1:
        X_train, y_train = with_fs1(df, method)
        X100 = shap_value(df, method, mode)
        model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
        model_ebm.fit(X_train, y_train)

        explainer_ebm = shap.Explainer(model_ebm.predict, X100)
        shap_values_ebm = explainer_ebm(X_train)
        filename = f"shap_waterfall_{method}.png"
        # plt.savefig(filename, dpi=300)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print('Global features ebm')
        shap.summary_plot(shap_values_ebm, X_train, max_display=6, show=False)
        plt.show()
        filename = f"shap_summary_{method}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    return shap_values_ebm


# shap_values_ebm= interpret_glassbox(df)

def global_feature_ebm(df, method):

    shap_values_ebm = interpret_glassbox(df, n_train, method, feature_selection_mode= mode)
    vals = np.abs(shap_values_ebm.values).mean(0)
    X_train, y_train = select_training(df)
    feature_names = X_train.columns
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                          columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                       ascending=False, inplace=True)
    global_features_ebm = feature_importance[0: max_display - 1]['col_name']
        # global_features_ebm= feature_importance[0:n]['col_name']
        # print('Important features on training set are:', global_features_ebm)
    return global_features_ebm, shap_values_ebm


def local_feature_importance(df, instance_index= sample_no):
    X_train, y_train = select_training(df)

    # Train model (you can swap in your preferred model here)
    model = GradientBoostingRegressor(random_state=101)
    model.fit(X_train, y_train)

    # SHAP explanation
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    # Plot local explanation
    shap.plots.waterfall(shap_values[instance_index], max_display=max_display)
    filename = f"shap_local_{method}.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    return shap_values[instance_index], X_train.iloc[instance_index]


global_features_ebm, shap_values_ebm = global_feature_ebm(df, method)
_shap_value, _X_train = local_feature_importance(df, sample_no)




