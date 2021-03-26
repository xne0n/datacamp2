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

class importcsv:
    def __init__(self,path,train): 
        df=pd.read_csv(path, sep=',',header=0)
        self.filename=df["filename"].astype(str).values.tolist()
        self.size=len(df)
        self.landmarks=np.empty([self.size,68,2],dtype=np.longdouble)
        for i in range(self.size):
            for j in range(0,68):
                self.landmarks[i,j,0] = df.iloc[i,j+1]
                self.landmarks[i,j,1] = df.iloc[i,j+69]
        if train :
            self.target = df.iloc[:,-1:]

class normalization:
    def __init__(self,data):
        self.landmarks=np.empty([data.size,68,2],dtype=np.longdouble)
        self.normFactor=[]
        reference=np.max(distance(data,39,42))
        for i in range(data.size):
            factor=reference/distance(data,39,42,i)
            self.normFactor.append(factor)
            for j in range(0,68):
                self.landmarks[i,j,0] = data.landmarks[i,j,0]*factor
                self.landmarks[i,j,1] = data.landmarks[i,j,1]*factor


def rotateImage(img, angle, origin):
    rotationMatrix = cv2.getRotationMatrix2D(origin, angle, 1.0)
    return cv2.warpAffine(img, rotationMatrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def rotateLandmarks(origin, point, angle):
    ox, oy = origin
    px, py = point    
    return ( ox+math.cos(angle)*(px-ox)-math.sin(angle)*(py-oy) ),( oy+math.sin(angle)*(px-ox)+math.cos(angle)*(py-oy) )

def getHeadTilt(data,img):
    return  math.degrees(math.atan2(data.landmarks[img,42,1] - data.landmarks[img,39,1], data.landmarks[img,42,0] - data.landmarks[img,39,0]))

def distance(data,A,B,specific=-1):
    if specific>=0 :
        a=np.array((data.landmarks[specific,A,0],data.landmarks[specific,A,1]))
        b=np.array((data.landmarks[specific,B,0],data.landmarks[specific,B,1]))
        return np.linalg.norm(a-b) # calcul distance euclidienne
    else:
        dist=[]
        for i in range(originalData.size):
            a=np.array((data.landmarks[i,A,0],data.landmarks[i,A,1]))
            b=np.array((data.landmarks[i,B,0],data.landmarks[i,B,1]))
            dist.append(np.linalg.norm(a-b))
        return np.array(dist)

def extractTextureFeatures():
    
    columns = ['texInterSourcil1', 'texInterSourcil2','texCoinBoucheD1', 'texCoinBoucheD2', 'texCoinBoucheG1', 'texCoinBoucheG2', 'texPaupiereG1', 'texPaupiereG2', 'texPaupiereD1', 'texPaupiereD2']
    
    df = pd.DataFrame(index=range(originalData.size),columns=columns)

    maxHeight= int ( np.max(distance(normalizedData,33,8)) +3 )
    maxWidth= int ( np.max(distance(normalizedData,33,16)) +3 )
    
    for file in range(originalData.size):
        # print(file,end='')
        headTilt=getHeadTilt(normalizedData,file)
        factor=normalizedData.normFactor[file]

        img = cv2.imread("./Dataset/trainset/"+originalData.filename[file]+".png", cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)), interpolation = cv2.INTER_AREA)
                
        origin=tuple(np.array(img.shape[1::-1]) / 2)
        img = rotateImage(img,headTilt,origin)
        img = img[int(round(normalizedData.landmarks[file,33,1]))-maxHeight:int(round(normalizedData.landmarks[file,33,1]))+maxHeight, int(round(normalizedData.landmarks[file,33,0]))-maxWidth:int(round(normalizedData.landmarks[file,33,0]))+maxWidth]
        
        noseX,noseY=normalizedData.landmarks[file,33,0],normalizedData.landmarks[file,33,1]
        
        for i in range(len(normalizedData.landmarks[file,:,:])):

            rotatedX,rotatedY = rotateLandmarks(origin,(normalizedData.landmarks[file,i,0],normalizedData.landmarks[file,i,1]),-(math.radians(headTilt)))
            normalizedData.landmarks[file,i,0]=rotatedX-noseX+maxWidth
            normalizedData.landmarks[file,i,1]=rotatedY-noseY+maxHeight
            # plt.scatter(np.round(normalizedData.landmarks[file,i,0]),np.round(normalizedData.landmarks[file,i,1]),c="red",s=1)
        
        '''
        ZONE ENTRE LES SOURCILS
        '''
        height=int((normalizedData.landmarks[file,27,1]-normalizedData.landmarks[file,21,1])/2)
        width=int((normalizedData.landmarks[file,22,0]-normalizedData.landmarks[file,21,0])/2)
        centerY=int(normalizedData.landmarks[file,27,1]-height)
        centerX=int(normalizedData.landmarks[file,27,0])
        
        df.iloc[file,0], df.iloc[file,1] = LBP(img[centerY-height:centerY+height,centerX-width:centerX+width])

        '''
        ZONE COIN DROIT DE LA BOUCHE
        '''
        height=int((normalizedData.landmarks[file,59,1]-normalizedData.landmarks[file,49,1]))
        width=int((normalizedData.landmarks[file,49,0]-normalizedData.landmarks[file,48,0]))
        centerY=int(normalizedData.landmarks[file,48,1])
        centerX=int(normalizedData.landmarks[file,48,0])
        
        df.iloc[file,2], df.iloc[file,3] = LBP(img[centerY-height:centerY+height,centerX-width:centerX+width])

        '''
        ZONE COIN GAUCHE DE LA BOUCHE
        '''

        height=int((normalizedData.landmarks[file,55,1]-normalizedData.landmarks[file,52,1]))
        width=int((normalizedData.landmarks[file,54,0]-normalizedData.landmarks[file,53,0]))
        centerY=int(normalizedData.landmarks[file,54,1])
        centerX=int(normalizedData.landmarks[file,54,0])
        
        df.iloc[file,4], df.iloc[file,5] = LBP(img[centerY-height:centerY+height,centerX-width:centerX+width])

        '''
        ZONE PAUPIERE GAUCHE
        '''

        height=int((normalizedData.landmarks[file,1,1]-normalizedData.landmarks[file,36,1])/2)
        width=int((normalizedData.landmarks[file,39,0]-normalizedData.landmarks[file,36,0])/2)
        centerY=int(normalizedData.landmarks[file,41,1]+height)
        centerX=int(normalizedData.landmarks[file,36,0]+width)

        df.iloc[file,6], df.iloc[file,7] = LBP(img[centerY-height:centerY+height,centerX-width:centerX+width])

        '''
        ZONE PAUPIERE DROITE
        '''

        height=int((normalizedData.landmarks[file,1,1]-normalizedData.landmarks[file,42,1])/2)
        width=int((normalizedData.landmarks[file,45,0]-normalizedData.landmarks[file,42,0])/2)
        centerY=int(normalizedData.landmarks[file,47,1]+height)
        centerX=int(normalizedData.landmarks[file,42,0]+width)

        df.iloc[file,8], df.iloc[file,9] = LBP(img[centerY-height:centerY+height,centerX-width:centerX+width])
        
    return df
    # plt.imshow(cv2.cvtColor(img[centerY-height:centerY+height,centerX-width:centerX+width], cv2.COLOR_BGR2RGB))
    # plt.imshow(img[centerY-height:centerY+height,centerX-width:centerX+width])
    # plt.imshow(img)
    # plt.scatter(centerX,centerY,c="blue",s=1)
    
    # plt.savefig('test'+str(file)+'.png')

'''
On fait l'Upsampling après le split, et dans le split nos données de test doivent avoir la même distribution
'''
def LBP(img):

    ret_s, th_s = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th_s=feature.local_binary_pattern(th_s, 24,8, method="uniform")
    th_s[th_s==24]=0
    rate_nblack_pix = (th_s[th_s!=0].size) / th_s.size
    rate_nblack_on_bl = (th_s[th_s!=0].size)/(th_s[th_s==0].size)

    return rate_nblack_pix, rate_nblack_on_bl

def landmarksDataFrame():

    dfx = pd.DataFrame(normalizedData.landmarks[:,:,0], columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67'])
    dfy = pd.DataFrame(normalizedData.landmarks[:,:,1], columns = ['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22', 'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 'y_30', 'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36', 'y_37', 'y_38', 'y_39', 'y_40', 'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47', 'y_48', 'y_49', 'y_50', 'y_51', 'y_52', 'y_53', 'y_54', 'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 'y_61', 'y_62', 'y_63', 'y_64', 'y_65', 'y_66', 'y_67'])
    return pd.concat([dfx, dfy], axis=1)

def extractGeoFeatures():

    columns = ['h_nez_ment','h_bouche_int', 'oeil_d_sourcil', 'oeil_g_sourcil', 'inter_sourcil', 'relat_oeil_d_1',
           'relat_oeil_d_2', 'relat_oeil_g_1', 'relat_oeil_g_2', 'relat_bouche_int',

           'labels']
    df = pd.DataFrame(index=range(originalData.size),columns=columns)
    df["h_nez_ment"]=distance(normalizedData,8,33)
    df["h_bouche_int"]=distance(normalizedData,62,66)
    df["oeil_d_sourcil"]=distance(normalizedData,19,37)
    df["oeil_g_sourcil"]=distance(normalizedData,24,44)
    df["inter_sourcil"]=distance(normalizedData,21,22)
    df["relat_oeil_d_1"]=(distance(normalizedData,42,45)/distance(normalizedData,43,47))
    df["relat_oeil_d_2"]=(distance(normalizedData,42,45)/distance(normalizedData,44,46))
    df["relat_oeil_g_1"]=(distance(normalizedData,36,39)/distance(normalizedData,37,41))
    df["relat_oeil_g_2"]=(distance(normalizedData,36,39)/distance(normalizedData,38,40))
    df["relat_bouche_int"]= (distance(normalizedData,62,66)/distance(normalizedData,60,64))

    df["labels"]=originalData.target
    return df

def index_features_select(X,Y,c=1):
    rf = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=50)
    rf.fit(X, Y)
    len_feat = X.values.shape[1]
    return np.where(rf.feature_importances_>=(c/len_feat))[0] 


originalData = importcsv("./Dataset/trainset/trainset.csv",1)
normalizedData = normalization(originalData)

texFeatures=extractTextureFeatures()
landFeatures=landmarksDataFrame()
geoFeatures=extractGeoFeatures()
labels = geoFeatures.iloc[:,-1]

fullFeatures = pd.concat([landFeatures,texFeatures,geoFeatures], axis=1)

finalFeatures = fullFeatures.iloc[:,index_features_select(fullFeatures.iloc[:,:-1],fullFeatures.iloc[:,-1],c=1)]
finalFeatures = pd.concat([finalFeatures,labels],axis=1)

finalFeatures.to_csv("features_train.csv",index=False)

# new_features = extract_features("./Dataset/trainset/")


# train_set = pd.read_csv("./Dataset/trainset/trainset.csv", sep=',',header=0)
# train_set = train_set.drop(columns=['filename', 'label'],axis = 1,errors='ignore') # pour rendre automatique même avec le test.csv

# for i in range(len(os.listdir("./Dataset/trainset/"))-1):
#     train_set.iloc[i,:68] = normalizedData.landmarks[i,:,0]
#     train_set.iloc[i,68:] = normalizedData.landmarks[i,:,1]

# trainset = pd.concat([train_set,new_features],axis=1)
# trainset.to_csv("features_train.csv",index=False)