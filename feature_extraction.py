import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

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


def showLandmaksOnImage(filename,landmarks):
    im=cv2.imread("./Dataset/trainset/"+filename+".png",-1)
    implot = plt.imshow(im)
    for i in range(len(landmarks)):
        plt.scatter(round(landmarks[i,0]),round(landmarks[i,1]),c="red",s=0.1)
    plt.savefig("./Dataset/trainset_withlandmarks/"+filename+"_withlandmarks.png")
    plt.clf()

data = importcsv("./Dataset/trainset/trainset.csv",1)
