# -*- coding:utf-8 -*-

import random
import warnings
import collections
import numpy as np
import pandas as pd
import sys
import os

import random
from scipy.stats import zscore
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
from sklearn.neighbors.nearest_centroid import NearestCentroid
from pyproj import Proj, transform
from pyKriging import kriging
from multiprocessing import Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from feature_selection import feature_selection_wrapper
from feature_selection import feature_selection_embeded
from feature_selection import feature_selection
    
from data_process import pca_scatter
from data_process import coord_transf, affine_transform
from data_process import binning_data,calculate_centroids     


world_city = pd.read_csv('../data/worldcities_new.csv')
europe = world_city[(world_city['y']<9000000)&(world_city['y']>4500000)&(world_city['x']>-1000000)&(world_city['x']<3500000)]
europe = europe[~europe.population.isna()]

tag = 'kraken_data'

all_sample = pd.read_csv('../data/{}/train_data.csv'.format(tag),index_col=0).T
all_feature = list(all_sample.columns)
all_sample_label = [i.split('_')[3].split('-')[0] for i in all_sample.index]
all_sample['city'] = all_sample_label
# test data
expand_sample = pd.read_csv('../data/{}/test_data.csv'.format(tag),index_col=0).T
expand_feature = list(expand_sample.columns)

expand_sample = pd.merge(pd.read_csv('../data/test_true_label.csv',index_col=1),
                         expand_sample,
                         left_index=True,
                         right_index=True)

all_sample = pd.concat([all_sample,expand_sample])
all_data_label = all_sample['city'].copy()

# Binning in all train set
all_sample = binning_data(all_sample[all_feature])
all_sample['city'] = all_data_label

europe_sample = all_sample[all_sample.city.isin(['BER','LON','MAR','PXO','SOF','STO','Kiev', 'Oslo', 'Paris', 'Vienna'])]

def random_test(Number_test,test_city):
    
    #city_name  = {i[1]:i[0] for i in zip(random.sample(list(europe.city_ascii),10),['BER','LON','MAR','PXO','SOF','STO','Kiev', 'Oslo', 'Paris', 'Vienna'])}
    
    city_name = {'BER':'Berlin','LON':'London','MAR':'Marseille',
                'PXO':'Porto','SOF':'Sofia','STO':'Stockholm',
                'Kiev':'Kiev','Oslo':'Oslo','Paris':'Paris','Vienna':'Vienna'}
    
    #test_city = 'Vienna'
    train_data = europe_sample[europe_sample['city']!=test_city]
    #sample = list(train_data.index)
    #random.shuffle(sample)
    #print(sample[0])
    #train_data = pd.DataFrame(index=sample,columns=train_data.columns,data=train_data.values)
    #train_data = train_data.rename(index={train_data.index[i]:sample[i] for i in range(len(train_data.index))})
    #print(train_data.index[0])
    test_data = europe_sample[europe_sample['city']==test_city]

    #key_enrf = feature_selection_embeded(train_data[all_feature], train_data[['city']], feature_return='embeded_rf_feature')
    #key_walr = feature_selection_wrapper(train_data[all_feature], train_data[['city']])

    #key = list(set(key_enrf+key_walr))
    key = list(pd.read_table('../feature_extration_result/feature_list_{}.txt'.format(tag),header=None)[0])
    # model
    clf = LogisticRegression(penalty="l2", 
                            C=0.5, 
                            multi_class="ovr", 
                            solver='liblinear', 
                            class_weight="balanced")
    tmp = train_data["city"].values.copy()
    random.shuffle(tmp)
    x, y = train_data[key].values, tmp                         
    #with open('tmp.txt','a') as f:
    #    f.write('{}\n'.format(y))
    # training model
    clf.fit(x, y)
    # predict test data
    test_pre_proba = pd.DataFrame(index=test_data.index,
                                    columns=clf.classes_,
                                    data=clf.predict_proba(test_data[key].values))
    test_pre_result = pd.DataFrame(index=test_data.index,
                                    columns=['predict_result'],
                                    data=clf.predict(test_data[key].values))
    #test_pre_proba.to_csv('prob_{}.csv'.format(Number_test))
    # save feature data (calculate city bio-distance)
    #tmp.to_csv('feature_bin_data.csv')

    print(Number_test)
    
    # affine transform
    pca_data = pca_scatter(train_data,key,'city')
    # calculate bio-centroids point 
    city_label, city_centroids = calculate_centroids(pca_data)
    # transform geographic point into biological point
    to_pts = city_centroids[[list(city_label).index(i) for i in city_label]]
    from_pts = europe[europe.city_ascii.isin([city_name[i] for i in city_label])][['x','y']].values

    src_pts = europe[['x','y']].values
    bio_pts = affine_transform(from_pts, to_pts,src_pts)

    europe['bio_x'] = [i[0] for i in bio_pts]
    europe['bio_y'] = [i[1] for i in bio_pts]

    # kriging interpolation
    bio_result = []
    geo_rseult = []
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
        
        #for ind in range(tmp_map.shape[0]):
        #    tmp_map.BIOprob.iloc[ind] = bio_kriging.predict([tmp_map.iloc[ind]['bio_x'],tmp_map.iloc[ind]['bio_y']])
        #    tmp_map.GEOprob.iloc[ind] = geo_kriging.predict([tmp_map.iloc[ind]['x'],tmp_map.iloc[ind]['y']])
        #tmp_map[['city_ascii','GEOprob','BIOprob','x','y','bio_x','bio_y']].to_csv('../kriging_result/{}.csv'.format(i.split('_')[3]),index=False)
        #tmp_map['BIOprob'] = zscore(tmp_map['BIOprob'])
        #tmp_map['GEOprob'] = zscore(tmp_map['GEOprob'])
        
        #bio_result.append(tmp_map[tmp_map.city_ascii==test_city]['BIOprob'].values[0])
        #geo_rseult.append(tmp_map[tmp_map.city_ascii==test_city]['GEOprob'].values[0])
        bio_result.append(bio_kriging.predict([tmp_map[tmp_map.city_ascii==city_name[test_city]]['bio_x'].values[0],tmp_map[tmp_map.city_ascii==city_name[test_city]]['bio_y'].values[0]]))
        geo_rseult.append(geo_kriging.predict([tmp_map[tmp_map.city_ascii==city_name[test_city]]['x'].values[0],tmp_map[tmp_map.city_ascii==city_name[test_city]]['y'].values[0]]))
    if not os.path.isdir('../random_test/{}'.format(tag)):
        os.makedirs('../random_test/{}'.format(tag))
    with open('../random_test/{}/bio_shuffle_{}_result.txt'.format(tag,test_city),'a') as f:
        f.write('{}\n'.format(bio_result))
    with open('../random_test/{}/geo_shuffle_{}_result.txt'.format(tag,test_city),'a') as f:
        f.write('{}\n'.format(geo_rseult))
    return None

inp = [(j,i) for i in ['BER','LON','MAR','PXO','SOF','STO','Kiev', 'Oslo', 'Paris', 'Vienna'] for j in range(1000)]
with Pool(72) as p:
    result = p.starmap(random_test, inp)