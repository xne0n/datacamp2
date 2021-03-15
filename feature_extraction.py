import pandas as pd
import numpy as np
import imblearn


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
import pandas as pd
import numpy as np
labels = pd.read_csv("Dataset\\trainset\\trainset.csv")
def add_features(df):
    labels = df["label"]
    df.drop(['label'], axis=1)
    df["h_Nez_ment"]=df.iloc[:,69+8]-df.iloc[:,69+33]
    df["h_bouche_int"]=df.iloc[:,69+66]-df.iloc[:,69+62]
   
    df["larg_bouche_int"]=df.iloc[:,1+60]-df.iloc[:,1+64]
    df["relat_oeild_1"]=(-df.iloc[:,1+42]+df.iloc[:,1+45])/(-df.iloc[:,69+43]+df.iloc[:,69+47])#ici c'est largeur/hauteur
    df["relat_oeild_2"]=(-df.iloc[:,1+42]+df.iloc[:,1+45])/(-df.iloc[:,69+44]+df.iloc[:,69+46])
    df["relat_oeilg_1"]=(-df.iloc[:,1+36]+df.iloc[:,1+39])/(-df.iloc[:,69+37]+df.iloc[:,69+38])
    df["relat_oeilg_2"]=(-df.iloc[:,1+36]+df.iloc[:,1+39])/(-df.iloc[:,69+41]+df.iloc[:,69+40])

    df["larg_sur_haut_bouche_int"]= (-df.iloc[:,1+60]+df.iloc[:,1+64])/(-df.iloc[:,69+62]+df.iloc[:,69+66])
    df["larg_sur_haut_bouche_ext"]= (-df.iloc[:,1+48]+df.iloc[:,1+54])/(-df.iloc[:,69+51]+df.iloc[:,69+57])
    df["dist"]
 
    # angle 24,23,25 et 19,18,20

    df["labels"]=labels
    return df
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
#print(add_features(labels))
#print(cos_radius(labels,60,61,67))
#cos_radius(labels,60,51,57)

'''print(np.degrees(np.arccos(cos_radius(labels,60,61,67)[[0,1,4,10]])))
print("-----------")
print(np.degrees(np.arccos(cos_radius(labels,60,62,66)[[0,1,4,10]])))
print("-----------")'''
#print(np.degrees(np.arccos(cos_radius(labels,48,51,57)[[0,1,4,10]])))
#print("Neutre","Neutre-degout","Etonne","Joie")

#print(cos_radius(labels,60,61,67).values)








#print(np.unique(labels.iloc[:,-1].values,return_counts=True))
#print((data.landmarks[0,0,:]))
'''print(labels.iloc[0,[1+33,69+33]],"\n",labels.iloc[0,[1+8,69+8]])
print(labels.iloc[0,[1+51,69+51]],"\n",labels.iloc[0,[1+57,69+57]])
print("--------------")
print(labels.iloc[1,[1+33,69+33]],"\n",labels.iloc[1,[1+8,69+8]])
print(labels.iloc[1,[1+51,69+51]],"\n",labels.iloc[1,[1+57,69+57]])
print("--------------")
print(labels.iloc[4,[1+33,69+33]],"\n",labels.iloc[4,[1+8,69+8]])
print(labels.iloc[4,[1+51,69+51]],"\n",labels.iloc[4,[1+57,69+57]])
print("--------------")
print(labels.iloc[10,[1+33,69+33]],"\n",labels.iloc[10,[1+8,69+8]])
print(labels.iloc[10,[1+51,69+51]],"\n",labels.iloc[10,[1+57,69+57]])'''

'''
print(labels.iloc[0,[1+42,69+42]],"\n",labels.iloc[0,[1+45,69+45]])
print(labels.iloc[0,[1+43,69+43]],"\n",labels.iloc[0,[1+47,69+47]])
print(labels.iloc[0,[1+44,69+44]],"\n",labels.iloc[0,[1+46,69+46]])'''

'''print((-labels.iloc[0,1+42]+labels.iloc[0,1+45])/(-labels.iloc[0,69+43]+labels.iloc[0,69+47]))
print((-labels.iloc[1,1+42]+labels.iloc[1,1+45])/(-labels.iloc[1,69+43]+labels.iloc[1,69+47]))
print((-labels.iloc[2,1+42]+labels.iloc[2,1+45])/(-labels.iloc[2,69+43]+labels.iloc[2,69+47]))
print((-labels.iloc[-2,1+42]+labels.iloc[-2,1+45])/(-labels.iloc[-2,69+43]+labels.iloc[-2,69+47]))
print((-labels.iloc[-3,1+42]+labels.iloc[-3,1+45])/(-labels.iloc[-3,69+43]+labels.iloc[-3,69+47]))
print((-labels.iloc[10,1+42]+labels.iloc[10,1+45])/(-labels.iloc[10,69+43]+labels.iloc[10,69+47]))
print((-labels.iloc[4,1+42]+labels.iloc[4,1+45])/(-labels.iloc[4,69+43]+labels.iloc[4,69+47]))'''

print(labels.iloc[:,-1])
## Faire l'angle



