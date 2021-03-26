import pandas as pd
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage import feature
import os

def over_sampling(df): # prend le trainset
    sm = SMOTE(random_state=42,k_neighbors=6)
    X_res, y_res = sm.fit_resample(df.iloc[:,1:-1], df.iloc[:,-1])
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res

def index_features_select(X,Y,c=1):
    rf = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=30)
    rf.fit(X, Y)
    len_feat = X.values.shape[1]

    #print(labels.columns[1:])

    
    #print(len_feat)
    
    return np.where(rf.feature_importances_>=(c/len_feat)) 