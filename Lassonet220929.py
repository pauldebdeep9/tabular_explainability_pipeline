# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:16:03 2022

@author: 70K9734
"""


import pandas as pd
import numpy as np
from lassonet import LassoNetRegressor, plot_path
from sklearn.preprocessing import StandardScaler, scale
import interpret.glassbox
import os
import time
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


df= pd.read_csv('Data/CPExplain221014.csv')
n_train= 40
mode= 0
max_display= 6
top_n= 10
sample_ind= 7

def drop_null(df):
    df= df.dropna(axis= 1)
    return df


df= drop_null(df)

# print(df.shape)

def clean(df):
    df= df.dropna(axis=1)
    # for mode=0
    # X= df.drop(['date', 'CP score'], axis= 1)
    
    # for mode=1
    X= df.drop(['date', 'CP score', 'Man_pmi_th'], axis= 1)
    if mode==1:
        X.drop(['Man_pmi_my', 'Man_pmi_vn'], axis= 1)
    X = StandardScaler().fit_transform(X)
    # X= pd.DataFrame(data= X, 
    #                 columns= list(df.drop(['date', 'CP score'], axis= 1).columns))
    y = df['CP score'].values
    y= scale(y)

    total_local_points= X.shape[0]
    # X100 = shap.utils.sample(X, 100)
    return X, total_local_points

def standardize(X, y):
    X = StandardScaler().fit_transform(X)
    # y = scale(y)
    return X, y

def select_training(df):
    X, total_local_points= clean(df)
    # X_train= X.iloc[:n_train]
    X_train = X[:n_train]
    y_train= df['CP score'][:n_train]
    y_train= np.array(y_train)
    return X_train, y_train, X

def feature_selection(df):
    X_train, y_train, X = select_training(df)

    model = LassoNetRegressor(
        hidden_dims=(10,),
        verbose=True,
    )
    path = model.path(X_train, y_train)

    n_selected = []
    mse = []
    lambda_ = []

    for save in path:
        model.load(save.state_dict)
        y_pred = model.predict(X_train)
        n_selected.append(save.selected.sum())
        mse.append(mean_squared_error(y_train, y_pred))
        lambda_.append(save.lambda_)

    fig = plt.figure(figsize=(8, 12))
    fig.show()

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, mse, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("MSE")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, mse, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("MSE")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")
    plt.figure(figsize=(10, 20))

    feature_names = list(df.columns)
    importances = model.feature_importances_.numpy()
    order = np.argsort(importances)[::-1][:top_n]
    importances = importances[order]
    ordered_feature_names = [feature_names[i] for i in order]

    def select_subset(ordered_feature_names, df):
        df = df.dropna(axis=1)
        X = df.drop(['date', 'CP score'], axis=1)
        # X = StandardScaler().fit_transform(X)
        X_train_fs= X[X.columns.intersection(ordered_feature_names)]
        X_full= X_train_fs
        X_train_fs_red= X_train_fs.iloc[:n_train]

        # X_train_fs= X_train_fs.drop()
        return X_train_fs_red, X_full

    X_train_fs, X_full= select_subset(ordered_feature_names, df)


    return X_train_fs, y_train, X_full


X_train_fs, y_train, X_full= feature_selection(df)


def shap_beeswarm(shap_values_ebm, max_display):
    # shap.plots.beeswarm(shap_values_ebm, max_display= max_display)
    shap.plots.beeswarm(shap_values_ebm, order=shap_values_ebm.abs.max(0))


def shap_value(df, mode):
    if mode==0:
        X_train, y_train, X = select_training(df)
        X100 = shap.utils.sample(X, 100)

    elif mode==1:
        X_train, y_train, X = feature_selection(df)
        X100 = shap.utils.sample(X, 100)
    return X100

X100= shap_value(df, mode)

# print(X100)


def interpret_glassbox(df,
                       n_train,
                       max_display= max_display,
                       feature_selection_mode= mode):

    if feature_selection_mode== 0:
        X_train, y_train, X= select_training(df)
        X= pd.DataFrame(X, columns= list(df.drop(['date', 'CP score'], axis= 1).columns))
        X100= shap_value(df, mode)
        model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
        model_ebm.fit(X_train, y_train)

# explain the GAM model with SHAP
        explainer_ebm = shap.TreeExplainer(model_ebm.predict, X100)
        shap_values_ebm = explainer_ebm(X)
        shap.plots.bar(shap_values_ebm, max_display= max_display)
        shap.summary_plot(shap_values_ebm,
                          plot_type= 'bar',
                          plot_size='auto')
        shap_beeswarm(shap_values_ebm, max_display)
        print('Global features ebm')

    elif  feature_selection_mode== 1:
        X_train, y_train, X = feature_selection(df)
        X100 = shap_value(df, mode)
        model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
        model_ebm.fit(X_train, y_train)

        explainer_ebm = shap.Explainer(model_ebm.predict, X)
        shap_values_ebm = explainer_ebm(X)
        shap.plots.bar(shap_values_ebm, max_display=max_display)
        shap_beeswarm(shap_values_ebm, max_display)
        print('Global features ebm')

    return shap_values_ebm


shap_values_ebm= interpret_glassbox(df,
                                    n_train)


# df_X_full= pd.DataFrame(data= X_full, columns= ordered_feature_names)


def global_feature_ebm(df, mode= mode):
    shap_values_ebm = interpret_glassbox(df, n_train, feature_selection_mode= mode)
    vals = np.abs(shap_values_ebm.values).mean(0)
    X_train, y_train, X = feature_selection(df)
    feature_names = X_train.columns
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                          columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                       ascending=False, inplace=True)
    global_features_ebm = feature_importance[0: max_display - 1]['col_name']
        # global_features_ebm= feature_importance[0:n]['col_name']
        # print('Important features on training set are:', global_features_ebm)
    return global_features_ebm, shap_values_ebm

global_features_ebm, shap_values_ebm = global_feature_ebm(df)

print('Globally important features are:', global_features_ebm)



def plot_par_dep(feature, sample_ind):
    if mode==1:
        X_train, y_train, charano_X= feature_selection(df)
    else:
        X_train, y_train, charano_X = select_training(df)

    global_features_ebm, shap_values_ebm= global_feature_ebm(df)
    model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
    model_ebm.fit(X_train, y_train)
    shap.partial_dependence_plot(
    feature, model_ebm.predict, charano_X, model_expected_value=True,
    feature_expected_value=True, show= True, ice=False,
    shap_values=shap_values_ebm[sample_ind:sample_ind+1,:],
    xmin="percentile(1)",
    xmax="percentile(99)")
    return global_features_ebm


def waterfall_shap_ebm (shap_values_ebm, sample_ind, max_display):
    shap_values_ebm_sample_index= shap_values_ebm[sample_ind]
    shap.plots.waterfall(shap_values_ebm_sample_index, max_display= max_display)
    print('Waterfall plot for {} EBM'.format(sample_ind))

waterfall_shap_ebm (shap_values_ebm, sample_ind, max_display)

def mean_var (shap_values_ebm, top_n):
    var_shap_value= 100*np.var(shap_values_ebm.abs.max(0).values[:top_n])
    mean_shap_value= 100*np.mean(shap_values_ebm.abs.max(0).values[:top_n])
    return var_shap_value, mean_shap_value


var_shap_value, mean_shap_value= mean_var (shap_values_ebm, top_n)

a_values= shap_values_ebm.abs.max(0).values[:top_n]


print('Mean=', mean_shap_value)
print('Var=', var_shap_value)
# print('The selected features are:', returned_features)
# print('# of seatures selected=', len(returned_features))

sns.ecdfplot(data= shap_values_ebm.abs.max(0).values[:top_n])


# to do:
# some issues with standard scalar
# make sure everything is in standard scalar
# seems the top important features are varying- try to use random seed


df_cdf= pd.read_excel('shap221017.xlsx')
df_cdf= df_cdf.dropna(axis= 1)
sns.ecdfplot(data= df_cdf)


