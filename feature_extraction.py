import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier

labels = pd.read_csv("Dataset\\trainset\\trainset.csv")


def over_sampling(df):
    sm = SMOTE(random_state=42,k_neighbors=6)
    X_res, y_res = sm.fit_resample(df.iloc[:,1:-1], df.iloc[:,-1])
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res


def add_feature(df): # ceci fonctionne avec X et Y pas juste avec le DataFrame
    df["h_Nez_ment"]=df.iloc[:,68+8]-df.iloc[:,68+33]
    df["h_bouche_int"]=df.iloc[:,68+66]-df.iloc[:,68+62]

    
    df["oeil_droit_sourcil"]=df.iloc[:,68+19]-df.iloc[:,68+37]
    df["oeil_gauche_sourcil"]=df.iloc[:,68+24]-df.iloc[:,68+44]

    
    df["inter_sourcil"]=df.iloc[:,68+24]-df.iloc[:,68+44]

    #normalization=(df.iloc[:,42]-df.iloc[:,39])
    #print(normalization)
    #df=df.iloc[:,:]/normalization
    
    df["relat_oeild_1"]=(-df.iloc[:,42]+df.iloc[:,45])/(-df.iloc[:,68+43]+df.iloc[:,68+47])#ici c'est largeur/hauteur
    df["relat_oeild_2"]=(-df.iloc[:,42]+df.iloc[:,45])/(-df.iloc[:,68+44]+df.iloc[:,68+46])
    df["relat_oeilg_1"]=(-df.iloc[:,36]+df.iloc[:,39])/(-df.iloc[:,68+37]+df.iloc[:,68+41])
    df["relat_oeilg_2"]=(-df.iloc[:,36]+df.iloc[:,39])/(-df.iloc[:,68+38]+df.iloc[:,68+40])

    df["larg_sur_haut_bouche_int"]= (-df.iloc[:,68+62]+df.iloc[:,68+66])/(-df.iloc[:,60]+df.iloc[:,64])
    df["larg_sur_haut_bouche_ext"]= (-df.iloc[:,68+51]+df.iloc[:,68+57])/(-df.iloc[:,48]+df.iloc[:,54])
    return df

def index_features_select(X,Y,c=1):
    rf = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=30)
    rf.fit(X, Y)
    len_feat = X.values.shape[1]

    #print(labels.columns[1:])
    print(len_feat)
    return np.where(rf.feature_importances_>=(c/len_feat)) 



def cos_radius(df,a,b,c):
    # a est le sommet de l'angle
    ab_vec_x = ( df.iloc[:,1+b].values-df.iloc[:,1+a].values )#/(df.iloc[:,1+42].values-df.iloc[:,1+39].values)
    ab_vec_y = -( df.iloc[:,69+b].values-df.iloc[:,69+a].values )#/(df.iloc[:,69+30].values-df.iloc[:,69+27].values)

    cd_vec_x = ( df.iloc[:,1+c].values-df.iloc[:,1+a].values )#/(df.iloc[:,1+42].values-df.iloc[:,1+39].values)
    cd_vec_y = -( df.iloc[:,69+c].values-df.iloc[:,69+a].values)#/(df.iloc[:,69+30].values-df.iloc[:,69+27].values)

    prod_scal= ab_vec_x*cd_vec_x + ab_vec_y*cd_vec_y
    ab_norm = np.sqrt( ab_vec_x**2 + ab_vec_y**2)
    cd_norm = np.sqrt( cd_vec_x**2 + cd_vec_y**2)


    
    # pour avoir la valeur en degrés faire np.degrees(np.arcos(prod_scal/(ab_norm*cd_norm)))
    # en radiant juste np.arcos(prod_scal/(ab_norm*cd_norm))
    return (prod_scal)/(ab_norm*cd_norm)

test = labels.copy()
X_over,Y_over = over_sampling(test)
X_new = add_feature(X_over)
print(X_new)
print(index_features_select(X_new,Y_over,c=0.9))
