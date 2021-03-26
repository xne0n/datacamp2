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

def extractTextureFeatures(file):

    maxHeight= math.ceil((np.max(distance(normalizedData,33,8))/10))*10 
    maxWidth= math.ceil((np.max(distance(normalizedData,33,16))/10))*10

    '''
    ICI IL FAUDRA BOUCLER SUR TOUT LES IMAGES 
    ET APPLIQUER LA METHODE D'EXTRACTION DE FEATURES DE TEXTURE QU'ON VEUT
    '''

    # for file in range(originalData.size):

    #file=450

    headTilt=getHeadTilt(normalizedData,file)
    factor=normalizedData.normFactor[file]

    img = cv2.imread("./Dataset/trainset/"+originalData.filename[file]+".png", cv2.IMREAD_UNCHANGED)
    # img = plt.imread("./Dataset/trainset/"+originalData.filename[file]+".png")

    #print(len(os.listdir("./Dataset/trainset/")))

    img = cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)), interpolation = cv2.INTER_AREA)
            
    origin=tuple(np.array(img.shape[1::-1]) / 2)
    img = rotateImage(img,headTilt,origin)
    img = img[int(round(normalizedData.landmarks[file,33,1]))-maxHeight:int(round(normalizedData.landmarks[file,33,1]))+maxHeight, int(round(normalizedData.landmarks[file,33,0]))-maxWidth:int(round(normalizedData.landmarks[file,33,0]))+maxWidth]
    
    noseX,noseY=normalizedData.landmarks[file,33,0],normalizedData.landmarks[file,33,1]
    
    for i in range(len(normalizedData.landmarks[file,:,:])):
            rotatedX,rotatedY = rotateLandmarks(origin,(normalizedData.landmarks[file,i,0],normalizedData.landmarks[file,i,1]),-(math.radians(headTilt)))
            normalizedData.landmarks[file,i,0]=rotatedX-noseX+maxWidth
            normalizedData.landmarks[file,i,1]=rotatedY-noseY+maxHeight
    #plt.scatter(np.round(normalizedData.landmarks[file,:,0]),np.round(normalizedData.landmarks[file,:,1]),c="red",s=1)
    #plt.imshow(img)
    '''
    Pour les images pour la représentation il faut faire un imshow(Y,X) et non (X,Y)
    '''
    #plt.imshow(img[int(normalizedData.landmarks[file,48,0]):int(normalizedData.landmarks[file,54,0]),int(normalizedData.landmarks[file,51,1]):int(normalizedData.landmarks[file,57,1])])
    #mouth=img[int(normalizedData.landmarks[file,51,1]):int(normalizedData.landmarks[file,57,1])+1,int(normalizedData.landmarks[file,48,0]):int(normalizedData.landmarks[file,54,0])+1]
    
    mouth=img[int(normalizedData.landmarks[file,29,1]):int(normalizedData.landmarks[file,35,1])+25,int(normalizedData.landmarks[file,31,0])-10:int(normalizedData.landmarks[file,35,0])+10]
    
    #mouth=img[int(normalizedData.landmarks[file,24,1])-25:int(normalizedData.landmarks[file,24,1])+35,int(normalizedData.landmarks[file,17,0]):int(normalizedData.landmarks[file,26,0])+1]
    
    # g_kernel = cv2.getGaborKernel((5, 5), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    # mouth1 = cv2.filter2D(mouth, cv2.CV_8UC3, g_kernel)
    # plt.figure()
    # plt.imshow(mouth1)
    # plt.savefig("test_2.png")

    #print(LBP(mouth))


    # plt.figure()
    # plt.imshow(mouth)
    # plt.savefig("test_.png")
    # ret_s, th_s = cv2.threshold(mouth,127,255,cv2.THRESH_BINARY)

    # th_s=feature.local_binary_pattern(th_s, 24,
    #         8, method="uniform")
    # th_s[th_s==24]=0
    #print(np.unique(th_s))


    # plt.imshow(th_s)
    # plt.savefig('test1.png')
    # plt.figure()
    # plt.scatter(np.round(normalizedData.landmarks[file,:,0]),np.round(normalizedData.landmarks[file,:,1]),c="red",s=1)
    # plt.imshow(img)
    # plt.savefig('test2.png')



    '''
    ICI AJOUTER LA METHODE D'EXTRACTION DE FEATURES DE TEXTURE QU'ON VEUT
    AVANT LA FIN DU FOR
    '''
'''
On fait l'Upsampling après le split, et dans le split nos données de test doivent avoir la même distribution
'''
def LBP(path,img_sector):
    
    nb_images= len(os.listdir(path)) - 1
    output = np.empty(shape = (nb_images,2),dtype=np.float)

    i=0
    for image in os.listdir(path):
        if image[-1]=="v":
            continue
        else:
            path_im=os.path.join(path,image)
            img = cv2.imread(path_im,0)
            img = img[img_sector[0]-5:img_sector[1]+5, img_sector[2]-5:img_sector[3]+5]
            ret_s, th_s = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            th_s=feature.local_binary_pattern(th_s, 24,
                    8, method="uniform")
            th_s[th_s==24]=0
            rate_nblack_pix = (th_s[th_s!=0].size) / th_s.size
            rate_nblack_on_bl = (th_s[th_s!=0].size)/(th_s[th_s==0].size)
            output[i,:]= np.array([rate_nblack_pix,rate_nblack_on_bl])
            i+=1

    return output #rate_nblack_pix,rate_nblack_on_bl
    '''
    La fonction renvoie un numpy array
    '''




def extract_features(path):
    columns = ['h_nez_ment','h_bouche_int', 'oeil_d_sourcil', 'oeil_g_sourcil', 'inter_sourcil', 'relat_oeil_d_1',
           'relat_oeil_d_2', 'relat_oeil_g_1', 'relat_oeil_g_2', 'relat_bouche_int',
            'LBP_mouth_1','LBP_mouth_2',

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
    #df["relat_bouche_int"]= (distance(normalizedData,60,64)/distance(normalizedData,62,66))
    df["relat_bouche_int"]= (distance(normalizedData,62,66)/distance(normalizedData,60,64))
    a,b= int(np.max(normalizedData.landmarks[:,51,1],axis=0)),int(np.max(normalizedData.landmarks[:,57,1],axis=0))
    c,d= int(np.max(normalizedData.landmarks[:,48,0],axis=0)),int(np.max(normalizedData.landmarks[:,54,0],axis=0))
    
    img_sec = (a,b,c,d)
    mouth = LBP(path,img_sec)
    df["LBP_mouth_1"] = mouth[:,0]
    df["LBP_mouth_2"] = mouth[:,1]
    
    df["labels"]=originalData.target
    return df



originalData = importcsv("./Dataset/trainset/trainset.csv",1)
normalizedData = normalization(originalData)

for file in range(len(os.listdir("./Dataset/trainset/"))-1): # on ne prend pas le fichier csv
    extractTextureFeatures(file) # ceci nous donne de nouveaux landmarks

new_features = extract_features("./Dataset/trainset/")
train_set = pd.read_csv("./Dataset/trainset/trainset.csv", sep=',',header=0)
train_set = train_set.drop(columns=['filename', 'label'],axis = 1,errors='ignore') # pour rendre automatique même avec le test.csv
print(normalizedData.landmarks[450,:,:].shape)
for i in range(len(os.listdir("./Dataset/trainset/"))-1):
    train_set.iloc[i,:68] = normalizedData.landmarks[i,:,0]
    train_set.iloc[i,68:] = normalizedData.landmarks[i,:,1]

trainset = pd.concat([train_set,new_features],axis=1)
trainset.to_csv("features_train.csv",index=False)

# testData = importcsv("./Dataset/testset/testset.csv",0)
# testDataNorm= normalization(testData)

# for file in range(len(os.listdir("./Dataset/testset/"))-1): # on ne prend pas le fichier csv
#     extractTextureFeatures(file) # ceci nous donne de nouveaux landmarks

# new_features = extract_features("./Dataset/trainset/")
# train_set = pd.read_csv("./Dataset/trainset/trainset.csv", sep=',',header=0)
# train_set = train_set.drop(columns=['filename', 'label'],axis = 1,errors='ignore') # pour rendre automatique même avec le test.csv
# print(normalizedData.landmarks[450,:,:].shape)
# for i in range(len(os.listdir("./Dataset/trainset/"))-1):
#     train_set.iloc[i,:68] = normalizedData.landmarks[i,:,0]
#     train_set.iloc[i,68:] = normalizedData.landmarks[i,:,1]

# trainset = pd.concat([train_set,new_features],axis=1)
# trainset.to_csv("features_train.csv",index=False)




#file=450
#img_sec= (int(normalizedData.landmarks[file,29,1]),int(normalizedData.landmarks[file,35,1]),int(normalizedData.landmarks[file,31,0]),int(normalizedData.landmarks[file,35,0]))

# a= int(np.max(normalizedData.landmarks[:,51,1],axis=0))
# b= int(np.max(normalizedData.landmarks[:,57,1],axis=0))
# c= int(np.max(normalizedData.landmarks[:,48,0],axis=0))
# d= int(np.max(normalizedData.landmarks[:,54,0],axis=0))

# img_sec = (a,b,c,d)

# print(LBP("./Dataset/trainset/",img_sec))





# test = labels.copy()
# X_over,Y_over = over_sampling(test)
# X_new = add_feature(X_over)
# print(X_new)
# print(index_features_select(X_new,Y_over,c=0.9))