
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("features_train.csv",sep=",")
X,Y = data.iloc[:,:-1] ,data.iloc[:,-1]

splits_ = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)


def over_sampling(X,Y): # prend le trainset
    sm = SMOTE(random_state=42,k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, Y)
    # print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res

def index_features_select(X,Y,c=1):
    rf = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=30)
    rf.fit(X, Y)
    len_feat = X.values.shape[1]
    return np.where(rf.feature_importances_>=(c/len_feat)) 




for train_index, test_index in splits_.split(X,Y):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
X_train , Y_train = over_sampling(X_train,Y_train)




svc1=SVC(kernel="linear",probability=True,random_state=42)
NN1=KNeighborsClassifier(n_neighbors=1,algorithm="kd_tree")
forest1=RandomForestClassifier(n_estimators=150,criterion="gini",max_depth=40,min_samples_split = 7 ,random_state=42)


ensemble_model_2 = VotingClassifier(estimators=[
         ('svc1', svc1), 
          ('knn', NN1), 
         ('fr', forest1)], voting='soft')
ensemble_model_2.fit(X_train,Y_train)
y_preds = ensemble_model_2.predict(X_test)
print(ensemble_model_2.score(X_test,Y_test),"  ",ensemble_model_2.score(X_train,Y_train))
print("-----")
print(confusion_matrix(Y_test,y_preds)) # le soft est en général meilleur
'''
voting = "soft" a de meilleures performances que voting = "hard"
Le KNN, le SVC et le randomForest s'avèrent être les modèles les plus performants pour notre modèle
C'est ainsi que nous avons choisi d'utiliser une méthode d'ensemble pour regrouper ces 3 classifieurs avec des performances assez 
satisfisantes individuellement mais plus beaucoup plus intéressantes mises ensemble
'''

Pkl_Filename = "model.pickle"  

with open(Pkl_Filename, 'wb') as file:  # pour enregistre le modèle sous format .pickle
    pickle.dump(ensemble_model_2, file)


