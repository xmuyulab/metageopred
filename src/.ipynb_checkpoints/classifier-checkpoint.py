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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.neighbors.nearest_centroid import NearestCentroid
from pyproj import Proj, transform
from multiprocessing import Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from feature_selection import feature_selection_wrapper
from feature_selection import feature_selection_embeded
from feature_selection import feature_selection

from data_process import coord_transf, affine_transform
from data_process import binning_data,calculate_centroids     

if  __name__ == "__main__":
    tag = 'kraken_data'
    # load train data
    train_data = pd.read_csv("../data/{}/train_data.csv".format(tag), index_col=0)
    train_data = train_data[train_data.index!='unclassified']
    # load test data
    test_data = pd.read_csv('../data/{}/test_data.csv'.format(tag), index_col=0)
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
    
    #Feature Selection
    # Feature selection from train_data
    # key_enrf = feature_selection_embeded(tmp[train_data.columns], tmp[['city']], feature_return='embeded_rf_feature')
    # key_walr = feature_selection_wrapper(tmp[train_data.columns], tmp[['city']])
    # RF_feature + REF_feature
    # key = list(set(key_enrf+key_walr))
    # key = list(set(key_walr))
    # save feature in feature list
    # with open('feature_list.txt','w') as f:
    #    for i in key:
    #        f.write('{}\n'.format(i))
    # key = pd.read_table('../feature_extration_result/{}_feature_list.txt'.format(tag),header=None)[0].tolist()
    
    #Model
    # clf = LogisticRegression(penalty="l2", 
    #                          C=0.5, 
    #                          multi_class="multinomial", 
    #                          solver='sag',
    #                          class_weight="balanced")
    #clf = RandomForestClassifier()
    x, y = tmp.values, tmp["city"].values                         
    # training model
    # clf.fit(x, y)
    # predict test data
    # test_pre_proba = pd.DataFrame(index=test_data.index,
    #                               columns=clf.classes_,
    #                               data=clf.predict_proba(test_data[key].values))
    # test_pre_result = pd.DataFrame(index=test_data.index,
    #                                columns=['predict_result'],
    #                                data=clf.predict(test_data[key].values))
    #test_pre_proba.to_csv('test_pre_proba.csv')
    #test_pre_result.to_csv('tset_pre_label.csv')
    # save feature data (calculate city bio-distance)
    #tmp.to_csv('feature_bin_data.csv')
    # Validation
    n = 100  # Number of re-shuffling & splitting iterations
    # Stratified ShuffleSplit cross-validator
    sp = StratifiedShuffleSplit(n_splits=n, test_size=0.3, random_state=0)
    # acc_list = []
    # acc = 0

    def performance(train_sample,test_sample):
        
        lgr = LogisticRegression(penalty="l2", 
                                 C=0.5,
                                 multi_class="ovr",
                                 solver='liblinear', 
                                 class_weight="balanced")
        rdf = RandomForestClassifier()
        xgb = XGBClassifier()
        knn = KNeighborsClassifier()
        key = feature_selection_wrapper(tmp.iloc[train_sample][train_data.columns], tmp.iloc[train_sample][['city']],50)
        X_train, y_train = tmp[key].iloc[train_sample].values, tmp['city'].iloc[train_sample].values
        X_test, y_test = tmp[key].iloc[test_sample].values, tmp['city'].iloc[test_sample].values
        
        lgr.fit(X_train, y_train)
        rdf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        with open('../classifier_performance.txt','a+') as f:
            f.write('lgr\t{}\n'.format(lgr.score(X_test, y_test)))
            f.write('rdf\t{}\n'.format(rdf.score(X_test, y_test)))
            f.write('xgb\t{}\n'.format(xgb.score(X_test, y_test)))
            f.write('knn\t{}\n'.format(knn.score(X_test, y_test)))
        
    data_split = [[train_sample, test_sample] for train_sample, test_sample in sp.split(x, y)]
    with Pool(72) as p:
        result = p.starmap(performance, data_split)
        # acc += clf.score(X_test, y_test)
        # acc_list.append(clf.score(X_test, y_test))

    # print("feature number:{}".format(len(key)))
    # print("average acc score:{}".format(acc / n))
    # print("minimum acc score:{}".format(np.min(acc_list)))
    