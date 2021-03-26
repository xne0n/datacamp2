import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("features_train.csv",sep=",")
X,Y = data.iloc[:,:-1] ,data.iloc[:,-1]
#print(data)

splits_ = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
# splits_ = StratifiedKFold(n_splits=3,shuffle=True,random_state=4)

# for train_index, test_index in splits_.split(X,Y):
#     X_train,X_test = X.iloc[train_index],X.iloc[test_index]
#     Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
#     print(Counter(Y_train),"  ",Counter(Y_test))
# X_train , Y_train = over_sampling(X_train,Y_train)
# print(Counter(Y_train))

def over_sampling(X,Y): # prend le trainset
    sm = SMOTE(random_state=42,k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, Y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res

def index_features_select(X,Y,c=1):
    rf = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=30)
    rf.fit(X, Y)
    len_feat = X.values.shape[1]
    #print(len_feat)
    print(np.where(rf.feature_importances_>=(c/len_feat)))
    return np.where(rf.feature_importances_>=(c/len_feat)) 

indexes= index_features_select(X,Y,c=1)[0]
# print(indexes)
# print(X.iloc[:,indexes])
X=X.iloc[:,indexes]


for train_index, test_index in splits_.split(X,Y):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]
    # print(Counter(Y_train),"  ",Counter(Y_test))
X_train , Y_train = over_sampling(X_train,Y_train)

# clf= SVC(kernel="linear")
# clf.fit(X_train,Y_train)
# print(clf.score(X_train,Y_train),"  ",clf.score(X_test,Y_test))

# print(Counter(Y_train))

# models = [
#   SVC(kernel="linear"),
#   SVC(kernel="poly"),
#   SVC(kernel="rbf"),
#   MLPClassifier(hidden_layer_sizes=(60),activation="relu",learning_rate_init=0.001,max_iter=200,random_state=0),
#   KNeighborsClassifier(n_neighbors=1,algorithm="brute"),
#   KNeighborsClassifier(n_neighbors=1,algorithm="kd_tree"),
#   RandomForestClassifier(n_estimators=120,criterion="gini",max_depth=25,min_samples_split = 5 ,random_state=42)]
# for model in models:
#     model.fit(X_train,Y_train)
#     print(model.score(X_train,Y_train),"  ",model.score(X_test,Y_test))
# print("-----------")    
# y_preds = model.predict(X_test)
# print(confusion_matrix(Y_test,y_preds))

svc1=SVC(kernel="linear",probability=True,random_state=42)
NN1=KNeighborsClassifier(n_neighbors=2,algorithm="kd_tree")
forest1=RandomForestClassifier(n_estimators=120,criterion="gini",max_depth=30,min_samples_split = 5 ,random_state=42)
ensemble_model_1 = VotingClassifier(estimators=[
         ('lr', svc1), ('rf', NN1), ('gnb', forest1)], voting='hard')
ensemble_model_1.fit(X_train,Y_train)
y_preds = ensemble_model_1.predict(X_test)
print(ensemble_model_1.score(X_test,Y_test),"  ",ensemble_model_1.score(X_train,Y_train))
print("-----")
print(confusion_matrix(Y_test,y_preds))

ensemble_model_2 = VotingClassifier(estimators=[
         ('lr', svc1), ('rf', NN1), ('gnb', forest1)], voting='soft')
ensemble_model_2.fit(X_train,Y_train)
y_preds = ensemble_model_2.predict(X_test)
print(ensemble_model_2.score(X_test,Y_test),"  ",ensemble_model_2.score(X_train,Y_train))
print("-----")
print(confusion_matrix(Y_test,y_preds)) # le soft est en général meilleur

Pkl_Filename = "model.pickle"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(ensemble_model_2, file)

# svc =SVC()
# parameters_svm = {'kernel':('linear', 'rbf',"poly"), 'C':range(1,10)}
# clf = GridSearchCV(svc, parameters_svm)
# clf.fit(X, Y)
# print(clf.best_params_)
# y_preds = clf.predict(X_test)
# print(clf.score(X_test,Y_test)," ",clf.score(X_train,Y_train))
# print(confusion_matrix(Y_test,y_preds))

# forest = RandomForestClassifier()
# parameters_forest = {"n_estimators":range(100,120),
# "criterion":["gini", "entropy"],
#  "max_depth":range(20,30),
#  "min_samples_split" : range(3,10),
#   "min_samples_leaf" : range(1,10)
# }
# clf2 = GridSearchCV(forest,parameters_forest)
# clf2.fit(X, Y)
# print(clf2.best_params_)
# y_preds = clf2.predict(X_test)
# print(clf2.score(X_test,Y_test)," ",clf2.score(X_train,Y_train))
# print(confusion_matrix(Y_test,y_preds))
