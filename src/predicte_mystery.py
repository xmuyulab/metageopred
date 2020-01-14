# -*- coding:utf-8 -*-

import random
import warnings
import collections
import numpy as np
import pandas as pd
import sys
import os

import random
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

#from pyKriging import kriging
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from feature_selection import feature_selection_wrapper
from feature_selection import feature_selection_embeded
from feature_selection import feature_selection

from data_process import tsne_scatter
from data_process import coord_transf, affine_transform
from data_process import binning_data,calculate_centroids     

world_city = pd.read_csv('../data/worldcities_new.csv')
europe = world_city[(world_city['y']<8500000)&(world_city['y']>4500000)&(world_city['x']>-1000000)&(world_city['x']<3000000)]
europe = europe[~europe.population.isna()]

all_sample = pd.read_csv('../data/train_data.csv',index_col=0).T
all_feature = list(all_sample.columns)
# Binning in all train set
all_sample = binning_data(all_sample)
all_sample_label = [i.split('_')[3].split('-')[0] for i in all_sample.index]
all_sample['city'] = all_sample_label

europe_sample = all_sample[all_sample.city.isin(['BER','LON','MAR','PXO','SOF','STO'])]
city_name = {'BER':'Berlin','LON':'London','MAR':'Marseille','PXO':'Porto','SOF':'Sofia','STO':'Stockholm'}

test_city = 'STO'

train_data = europe_sample[europe_sample['city']!=test_city]
test_data = europe_sample[europe_sample['city']==test_city]

#key_enrf = feature_selection_embeded(train_data[all_feature], train_data[['city']], feature_return='embeded_rf_feature')
#key_walr = feature_selection_wrapper(train_data[all_feature], train_data[['city']])

#key = list(set(key_enrf+key_walr))
key = list(pd.read_table('../feature_extration_result/feature_list.txt',header=None)[0])
print(len(key))
# model
clf = LogisticRegression(penalty="l2", 
                         C=0.5, 
                         multi_class="ovr", 
                         solver='liblinear', 
                         class_weight="balanced")
x, y = train_data[key].values, train_data["city"].values                         
# training model
clf.fit(x, y)
# predict test data
test_pre_proba = pd.DataFrame(index=test_data.index,
                                columns=clf.classes_,
                                data=clf.predict_proba(test_data[key].values))
test_pre_result = pd.DataFrame(index=test_data.index,
                                columns=['predict_result'],
                                data=clf.predict(test_data[key].values))

# save feature data (calculate city bio-distance)
#tmp.to_csv('feature_bin_data.csv')
# ============================================Validation==============================================================
"""
n = 1000  # Number of re-shuffling & splitting iterations
# Stratified ShuffleSplit cross-validator
sp = StratifiedShuffleSplit(n_splits=n, test_size=0.3, random_state=0)
acc_list = []
acc = 0
for train_sample, test_sample in sp.split(x, y):
    clf = LogisticRegression(penalty="l2", 
                                C=0.5,
                                multi_class="ovr",
                                solver='liblinear', 
                                class_weight="balanced")

    X_train, y_train = train_data[key].iloc[train_sample].values, train_data['city'].iloc[train_sample].values
    X_test, y_test = train_data[key].iloc[test_sample].values, train_data['city'].iloc[test_sample].values

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    acc += clf.score(X_test, y_test)
    acc_list.append(clf.score(X_test, y_test))

print("feature number:{}".format(len(key)))
print("average acc score:{}".format(acc / n))
print("minimum acc score:{}".format(np.min(acc_list)))
"""
# =============================================affine transform============================================================
tsne_data = tsne_scatter(train_data,key,'city')
# calculate bio-centroids point
city_label, city_centroids = calculate_centroids(tsne_data)
# transform geographic point into biological point
to_pts = city_centroids[[list(city_label).index(i) for i in city_label]]
from_pts = europe[europe.city_ascii.isin([city_name[i] for i in city_label])][['x','y']].values

src_pts = europe[['x','y']].values
bio_pts = affine_transform(from_pts, to_pts,src_pts)

europe['bio_x'] = [i[0] for i in bio_pts]
europe['bio_y'] = [i[1] for i in bio_pts]

# =========================================-kriging interpolation===========================================================

"""

for i in test_pre_proba.index:
    tmp_map = europe.copy()
    tmp_map['GEOprob'] = 0
    tmp_map['BIOprob'] = 0
    BIOtrain_point = []
    GEOtrain_point = []
    train_y = []
    for c,p in list(test_pre_proba.T[i].to_dict().items()):
        BIOtrain_point.append(list(tmp_map[tmp_map.city_ascii==city_name[c]][['bio_x','bio_y']].values[0]))
        GEOtrain_point.append(list(tmp_map[tmp_map.city_ascii==city_name[c]][['x','y']].values[0]))
        train_y.append(p)

    BIOtrain_point = np.array(BIOtrain_point)
    GEOtrain_point = np.array(GEOtrain_point)
    
    bio_kriging = kriging(BIOtrain_point, train_y)
    bio_kriging.train()
    
    geo_kriging = kriging(GEOtrain_point, train_y)
    geo_kriging.train()

    for ind in range(tmp_map.shape[0]):
        tmp_map.BIOprob.iloc[ind] = bio_kriging.predict([tmp_map.iloc[ind]['bio_x'],tmp_map.iloc[ind]['bio_y']])
        tmp_map.GEOprob.iloc[ind] = geo_kriging.predict([tmp_map.iloc[ind]['x'],tmp_map.iloc[ind]['y']])
    tmp_map[['city_ascii','GEOprob','BIOprob','x','y','bio_x','bio_y']].to_csv('../kriging_result/{}.csv'.format(i.split('_')[3]),index=False)
"""

for i in test_pre_proba.index:
    tmp_map = europe.copy()
    tmp_map['GEOprob'] = 0
    tmp_map['BIOprob'] = 0
    BIOtrain_point = []
    GEOtrain_point = []
    train_y = []
    for c,p in list(test_pre_proba.T[i].to_dict().items()):
        BIOtrain_point.append(list(tmp_map[tmp_map.city_ascii==city_name[c]][['bio_x','bio_y']].values[0]))
        GEOtrain_point.append(list(tmp_map[tmp_map.city_ascii==city_name[c]][['x','y']].values[0]))
        train_y.append(p)

    BIOtrain_point = np.array(BIOtrain_point)
    GEOtrain_point = np.array(GEOtrain_point)
    
    bio_kriging = OrdinaryKriging(BIOtrain_point[:, 0],
                                  BIOtrain_point[:, 1], 
                                  train_y, 
                                  variogram_model='gaussian',
                                  verbose=False, 
                                  enable_plotting=False)
    geo_kriging = OrdinaryKriging(GEOtrain_point[:, 0],
                                  GEOtrain_point[:, 1], 
                                  train_y, 
                                  variogram_model='gaussian',
                                  verbose=False, 
                                  enable_plotting=False)

    bio_z, _ = bio_kriging.execute('points', 
                                    tmp_map['bio_x'].values,
                                    tmp_map['bio_y'].values)
    tmp_map['BIOprob'] = (bio_z.data-bio_z.data.min())/(bio_z.data.max()-bio_z.data.min()) 

    geo_z, _ = geo_kriging.execute('points', 
                                    tmp_map['x'].values, 
                                    tmp_map['y'].values)
    tmp_map['GEOprob'] = (geo_z.data-geo_z.data.min())/(geo_z.data.max()-geo_z.data.min()) 

    tmp_map[['city_ascii','GEOprob','BIOprob','x','y','bio_x','bio_y']].to_csv('../kriging_result/{}.csv'.format(i.split('_')[3]),index=False)
 


a = []
for i in os.listdir('../kriging_result/'):
    data = pd.read_csv("../kriging_result/{}".format(i))[['city_ascii', 'GEOprob', 'BIOprob']]
    a.append(data[data['city_ascii']=='Berlin'])