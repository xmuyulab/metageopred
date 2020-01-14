# -*- coding:utf-8 -*-

import random
import warnings
import collections
import numpy as np
import pandas as pd
import sys
import os

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.neighbors.nearest_centroid import NearestCentroid
from pyproj import Proj, transform
from multiprocessing import Pool

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from data_process import tsne_scatter
from data_process import coord_transf, affine_transform
from data_process import binning_data,calculate_centroids 

# feature selected by Recursive Feature Elimination
def feature_selection_wrapper(data, label, n=50,step=5,verbose=5,feature_return='log'):
    """
    data: pandas DataFrame,load from train_data.csv
    label: pandas DataFrame, sample city label
    feature_return: optional, feature selected model {log:LogisticRegression, rf:RandomForestClassifier}
    n: int or None (default=5) The number of features to select.
    step: int or float, optional (default=1) If greater than or equal to 1, then step corresponds to 
          the (integer) number of features to remove at each iteration. 
          If within (0.0, 1.0), then step corresponds to the percentage (rounded down) 
          of features to remove at each iteration.
    verbose: int (default=5) Controls verbosity of output.
    return: list,feature list
    """
    tmp = pd.merge(label, data, left_index=True, right_index=True)
    X, y = tmp[data.columns].values, tmp["city"].values
    random.seed(10)
    print(feature_return,feature_return=='log')
    if feature_return=='rf':
        rfe_selector = RFE(estimator=RandomForestClassifier(n_jobs=-1, 
                                                            random_state=np.random.seed(13),
                                                            class_weight="balanced"),
                           n_features_to_select=n,
                           step=step,
                           verbose=verbose)
    elif feature_return=='log':
        rfe_selector = RFE(estimator=LogisticRegression(penalty="l1",
                                                        C=1,
                                                        multi_class="ovr",
                                                        solver='liblinear',
                                                        class_weight="balanced"),
                           n_features_to_select=n,
                           step=step,
                           verbose=verbose)
    else:
        raise ValueError
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = data.loc[:, rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')

    return rfe_feature


# feature selected by Random Forest
def feature_selection_embeded(data, label, feature_return='embeded_rf_feature'):
    """
    data: pandas DataFrame,load from train_data.csv
    label: pandas DataFrame, sample city label
    feature_return: optional, feature selected model: 
        ['embeded_rf_feature','embeded_lr_selector', 'embeded_lgb_selector']
    return: feature list
    """
    tmp = pd.merge(label, data, left_index=True, right_index=True)
    X, y = tmp[data.columns].values, tmp["city"].values

    assert feature_return in ['embeded_rf_feature','embeded_lr_selector', 'embeded_lgb_selector']
    # feature selected by Random Forest model
    if feature_return == 'embeded_rf_feature':
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(criterion='gini',
                                                                     max_features='auto',
                                                                     random_state=np.random.seed(13),
                                                                     n_jobs=-1,
                                                                     class_weight='balanced',
                                                                     n_estimators=500), threshold='3.7*mean')
                                                                     #1.5
        embeded_rf_selector.fit(X, y)
        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = data.loc[:, embeded_rf_support].columns.tolist()
        print(str(len(embeded_rf_feature)),'RandomForestClassifier selected features')
        return embeded_rf_feature
    # feature selected by Logistic Regression model
    elif feature_return == 'embeded_lr_selector':

        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty='l1',
                                                                 C=0.6,
                                                                 class_weight='balanced'), threshold='2*mean')
        #X, y = tmp[embeded_rf_feature].values, tmp["city"].values
        embeded_lr_selector.fit(X, y)
        embeded_lr_selector = embeded_lr_selector.get_support()
        embeded_lr_selector = data.loc[:, embeded_lr_selector].columns.tolist()

        print(str(len(embeded_lr_selector)),'LogisticRegression selected features')
        return embeded_lr_selector
    # feature selected by LGBM model
    else:
        lgbc = LGBMClassifier(n_estimators=500,
                              learning_rate=0.05,
                              num_leaves=32,
                              colsample_bytree=0.2,
                              reg_alpha=3,
                              reg_lambda=1,
                              min_split_gain=0.01,
                              min_child_weight=40)

        embeded_lgb_selector = SelectFromModel(lgbc, threshold='mean')
        embeded_lgb_selector.fit(X, y)
        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = data.loc[:, embeded_lgb_support].columns.tolist()
        print(str(len(embeded_lgb_feature)),'LGBMClassifier selected features')
        return embeded_lgb_selector
    return None


# feature selection function(oldest)
def feature_selection(data, label):
    """
    Log + RF
    data: pandas DataFrame,load from train_data.csv
    label: pandas DataFrame, sample city label
    return: pandas DataFrame, feature importance
    """
    clf1 = LogisticRegression(penalty='l1', C=1, class_weight='balanced')
    clf2 = RandomForestClassifier(criterion='entropy', max_features='auto', n_jobs=-1, class_weight='balanced')

    tmp = pd.merge(label, data, left_index=True, right_index=True)
    x, y = tmp[data.columns].values, tmp["city"].values
    clf1.fit(x, y)
    if len(clf1.classes_) == 2:
        # clf1 = LogisticRegression(penalty='l2')
        # clf1.fit(x,y)
        feature_log = pd.DataFrame(
            index=data.columns, columns=["coef_"], data=clf1.coef_.T
        )
        city_key = feature_log[feature_log.coef_ != 0].index
        if len(city_key) == 0:
            city_key = data.columns
    else:
        feature_log = pd.DataFrame(
            index=data.columns, columns=clf1.classes_, data=clf1.coef_.T
        )
        feature_log = abs(feature_log)
        city_key = []

        for i in feature_log.columns:
            # display(HTML(feature_imp.sort_values(i,ascending=False)[[i]][:10].to_html()))
            city_key += list(feature_log.sort_values(i,
                                                     ascending=False)[[i]][:20].index)
    # key = list(set(city_key))

    key = list(data.columns)
    # print(len(key),tmp[key].values.shape,tmp["city"].shape)

    clf2.fit(tmp[key].values, tmp["city"].values)

    feature_rf = pd.DataFrame(
        index=key, columns=["importance"], data=clf2.feature_importances_
    )
    return feature_rf


if  __name__ == "__main__":
    # load train data
    train_data = pd.read_csv("../data/train_data.csv", index_col=0)
    # load test data
    test_data = pd.read_csv('../data/test_data.csv', index_col=0)
    # binning data
    all_data = pd.merge(train_data, test_data, left_index=True, right_index=True)
    all_data = binning_data(all_data.T)
    
    train_data = all_data.loc[train_data.columns]
    test_data = all_data.loc[test_data.columns]
    
    # extracted city label from sample name
    label = pd.DataFrame(
        index=train_data.index,
        columns=["city"],
        data=[i.split("_")[3].split("-")[0] for i in train_data.index])

    label = label.reset_index()
    label = label.rename(columns={"index": "sample"})

    tmp = pd.merge(label, train_data, left_on="sample", right_index=True)
    
    def multipro(i):
        x, y = tmp.values, tmp["city"].values                         
        n = 10  # Number of re-shuffling & splitting iterations
        # Stratified ShuffleSplit cross-validator
        sp = StratifiedShuffleSplit(n_splits=n, test_size=0.2, random_state=0)
        acc_list = []
        acc = 0
        for train_sample, test_sample in sp.split(x, y):

            clf = LogisticRegression(penalty="l2", 
                                    C=0.5,
                                    multi_class="ovr",
                                    solver='liblinear', 
                                    class_weight="balanced")
            
            key = feature_selection_wrapper(tmp.iloc[train_sample][train_data.columns], tmp.iloc[train_sample][['city']],i)

            X_train, y_train = tmp[key].iloc[train_sample].values, tmp['city'].iloc[train_sample].values
            X_test, y_test = tmp[key].iloc[test_sample].values, tmp['city'].iloc[test_sample].values

            clf.fit(X_train, y_train)
            print(clf.score(X_test, y_test))
            acc += clf.score(X_test, y_test)
            acc_list.append(clf.score(X_test, y_test))
        
        with open('../feature_extration_result/feature_extration_result.txt','a') as f:
            f.write('{}\t{}\t{}\n'.format(i,acc / n, np.min(acc_list)))
        print(i)
        print("feature number:{}".format(len(key)))
        print("average acc score:{}".format(acc / n))
        print("minimum acc score:{}".format(np.min(acc_list)))
        return None
    multipro(50)
    #with Pool(72) as p:
    #     result = p.map(multipro, [i for i in range(1,100)])