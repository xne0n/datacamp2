import pandas as pd
import numpy as np
#import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier

class importcsv:
    def __init__(self,path,train): 
        df=pd.read_csv(path, sep=',',header=0)
        self.filename=df.iloc[:,:1]
        self.size=len(df)
        self.landmarks=np.empty([self.size,68,2])
        for i in range(self.size):
            for j in range(0,68):
                self.landmarks[i,j,0] = df.iloc[i,j+1]
                self.landmarks[i,j,1] = df.iloc[i,j+69]
        if train :
            self.target = df.iloc[:,-1:]

data = importcsv("./Dataset/trainset/trainset.csv",1)

labels = pd.read_csv("Dataset\\trainset\\trainset.csv")

'''rf = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=30)
rf.fit(labels.iloc[:,1:-1], labels.iloc[:,-1])
#print(labels.columns[1:])
print(labels.iloc[:,1:-1].values.shape[1])
print(np.where(rf.feature_importances_>=(1.5/136))) # ici 136 = nb de features, il faut rajouter 1 à ces index'''

'''feature_importances = pd.DataFrame(rf.feature_importances_, index =labels.columns[1:-1],  
columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)'''

'''sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(labels.iloc[:,1:-1], labels.iloc[:,-1])
print('Resampled dataset shape %s' % Counter(y_res))'''




def over_sampling(df):
    sm = SMOTE(random_state=42,k_neighbors=6)
    X_res, y_res = sm.fit_resample(df.iloc[:,1:-1], df.iloc[:,-1])
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res


def add_feature(df): # ceci fonctionne avec X et Y pas juste avec le DataFrame
    df["h_Nez_ment"]=df.iloc[:,68+8]-df.iloc[:,68+33]
    df["h_bouche_int"]=df.iloc[:,68+66]-df.iloc[:,68+62]
   
    df["larg_bouche_int"]=df.iloc[:,60]-df.iloc[:,64]
    df["relat_oeild_1"]=(-df.iloc[:,42]+df.iloc[:,45])/(-df.iloc[:,68+43]+df.iloc[:,68+47])#ici c'est largeur/hauteur
    df["relat_oeild_2"]=(-df.iloc[:,42]+df.iloc[:,45])/(-df.iloc[:,68+44]+df.iloc[:,68+46])
    df["relat_oeilg_1"]=(-df.iloc[:,36]+df.iloc[:,39])/(-df.iloc[:,68+37]+df.iloc[:,68+41])
    df["relat_oeilg_2"]=(-df.iloc[:,36]+df.iloc[:,39])/(-df.iloc[:,68+38]+df.iloc[:,68+40])

    df["larg_sur_haut_bouche_int"]= (-df.iloc[:,68+62]+df.iloc[:,68+66])/(-df.iloc[:,60]+df.iloc[:,64])
    df["larg_sur_haut_bouche_ext"]= (-df.iloc[:,68+51]+df.iloc[:,68+57])/(-df.iloc[:,48]+df.iloc[:,54])
    #df["dist"]
 

    return df

def add_features(df):
    labels = df["label"]
    df.drop(['label'], axis=1)
    df["h_Nez_ment"]=df.iloc[:,69+8]-df.iloc[:,69+33]
    df["h_bouche_int"]=df.iloc[:,69+66]-df.iloc[:,69+62]
   
    df["larg_bouche_int"]=df.iloc[:,1+60]-df.iloc[:,1+64]
    df["relat_oeild_1"]=(-df.iloc[:,1+42]+df.iloc[:,1+45])/(-df.iloc[:,69+43]+df.iloc[:,69+47])#ici c'est largeur/hauteur
    df["relat_oeild_2"]=(-df.iloc[:,1+42]+df.iloc[:,1+45])/(-df.iloc[:,69+44]+df.iloc[:,69+46])
    df["relat_oeilg_1"]=(-df.iloc[:,1+36]+df.iloc[:,1+39])/(-df.iloc[:,69+37]+df.iloc[:,69+41])
    df["relat_oeilg_2"]=(-df.iloc[:,1+36]+df.iloc[:,1+39])/(-df.iloc[:,69+38]+df.iloc[:,69+40])

    df["haut_sur_larg_bouche_int"]= (-df.iloc[:,69+62]+df.iloc[:,69+66]) / (-df.iloc[:,1+60]+df.iloc[:,1+64])
    df["haut_sur_larg_bouche_ext"]= (-df.iloc[:,69+51]+df.iloc[:,69+57])/(-df.iloc[:,1+48]+df.iloc[:,1+54])
    #df["dist"]
 
    # angle 24,23,25 et 19,18,20

    df["labels"]=labels
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

print(index_features_select(X_new,Y_over,c=0.9))









