# -*- coding:utf-8 -*-
import random
import warnings
import collections
import numpy as np
import pandas as pd
import sys
import os

from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from sklearn.neighbors.nearest_centroid import NearestCentroid
from pyproj import Proj, transform
from multiprocessing import Pool

warnings.filterwarnings("ignore")


# Series binning function
def binner(seq):
    """
    Using quantile to bin the series to 3 ordinal sequences -1 0 1
    seq: pandas Series
    return: pandas Series after binning
    """
    q1 = seq.quantile(0.25)
    # q2 = seq.quantile(0.5)
    q3 = seq.quantile(0.75)

    return pd.Series(index=seq.index, data=[[-1, [1, 0][i < q3]][i > q1] for i in seq])
    

# Data binning function
def binning_data(data):
    """
    Using quantile to bin the DataFrame to 3 ordinal sequences -1 0 1
    seq: pandas DataFrame
    return: pandas DataFrame after binning
    """
    for i in data.columns:
        data[i] = binner(data[i])
    return data


def affine_fit(from_pts, to_pts):
    """
    Fit an affine transformation to given point sets.
    More precisely: solve (least squares fit) matrix 'A'and 't' from
    'p ~= A*q+t', given vectors 'p' and 'q'.
    Works with arbitrary dimensional vectors (2d, 3d, 4d...).
    Written by Jarno Elonen <elonen@iki.fi> in 2007.
    Placed in Public Domain.
    Based on paper "Fitting affine and orthogonal transformations
    between two sets of points, by Helmuth Späth (2003).
    """
    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q) < 1:
        print("error: source points must equal target points")
        return False

    dim = len(q[0])  # dimension
    if len(q) < dim:
        print("point number < dimension")
        return False

    # dim x (dim+1)  matrix
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]
    for j in range(dim):
        for k in range(dim+1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]  # extend a colum with value 1
                c[k][j] += qt[k] * p[i][j]

    # (dim+1) x (dim+1)  matrix
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim+1):
            for j in range(dim+1):
                Q[i][j] += qt[i] * qt[j]

    M = [Q[i] + c[i] for i in range(dim+1)]

    # if the source point is colinear with target point
    def gauss_jordan(m, eps=1.0/(10**10)):
        
        (h, w) = (len(m), len(m[0]))
        for y in range(0, h):
            maxrow = y
            for y2 in range(y+1, h):
                if abs(m[y2][y]) > abs(m[maxrow][y]):
                    maxrow = y2
            (m[y], m[maxrow]) = (m[maxrow], m[y])
            if abs(m[y][y]) <= eps:
                return False
            for y2 in range(y+1, h):
                c = m[y2][y] / m[y][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
        for y in range(h-1, 0-1, -1):
            c = m[y][y]
            for y2 in range(0, y):
                for x in range(w-1, y-1, -1):
                    m[y2][x] -= m[y][x] * m[y2][y] / c
            m[y][y] /= c
            for x in range(h, w):
                m[y][x] /= c
        return True

    if not gauss_jordan(M):
        print("error: colinear")
        return False

    class transformation:
        def __init__(self):
            pass

        def To_Str(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    str +=" %f*x%d + " % (M[i][j+dim+1], i)
                str += "%f" % M[dim][j+dim+1]
                res += str + "\n"
            return res

        def transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j+dim+1]
                res[j] += M[dim][j+dim+1]
            return res
    return transformation()


def calculate_centroids(pca_data):
    """
    Calculate centorids
    """
    assert 'pc_1' in pca_data.columns
    assert 'pc_2' in pca_data.columns
    assert 'city' in pca_data.columns
    X = pca_data[['pc_1','pc_2']].values
    y = pca_data['city'].values
    clf = NearestCentroid()
    clf.fit(X, y)
    return clf.classes_, clf.centroids_


def coord_transf(lat,lon):
    """
    Performs cartographic transformations between 
    geographic (lat/lon) and map projection (x/y) coordinates
    """
    inProj  = Proj("+init=EPSG:4326")
    outProj = Proj("+init=EPSG:3857")
    x,y =  (lon,lat)#lng lat
    return transform(inProj,outProj,x,y)


def affine_transform(from_pt, to_pt,src_pt):
    """
    transform geographic point into biological point.
    from_pt: list of point
    to_pt: list of point
    src_pt: list of point
    """
    trn = affine_fit(from_pt, to_pt)
    #print("tranformation equation:")
    #print(trn.To_Str())
    tar_pt = [trn.transform(i) for i in src_pt]
    return tar_pt


def pca_scatter(data,feature,traget,draw=False):
    """
    input:
        data: data is a dataframe with sample index and feature is a list
        feature: feature is list of key dimension that you fit into embedded space
        traget: str, traget information with sample id index
    return:
        scatterplot
    e.g.
        tsne_scatter(data,[feature1,feature2,feature3],traget)
    """
    assert len(feature)>2
#     X_tsne = TSNE(n_components=2,
#                   learning_rate=30,
#                   random_state=np.random.seed(10),
#                   metric='euclidean').fit_transform(data[feature].values)
    X_pca = PCA(n_components=2).fit_transform(data[feature].values)
    sample_pca = pd.DataFrame(index=data.index,columns=['pc_1','pc_2'],data=X_pca)
    
    pca_data = pd.merge(data[[traget]],sample_pca,right_index=True,left_index=True)
    if draw:
       plt.figure(figsize=(13,9))
       ax = sns.scatterplot(x="pc_1", y="pc_2", hue=traget,data=pca_data)
    return pca_data