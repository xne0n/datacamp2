import pandas as pd
import numpy as np

class importcsv:
    def __init__(self,path,train): 
        df=pd.read_csv(path, sep=',',header=0)
        self.filename=df.iloc[:,:1]
        self.size=len(df)
        self.landmarks=np.empty([self.size,67,2])
        for i in range(self.size):
            for j in range(0,67):
                self.landmarks[i,j,0] = df.iloc[i,j+1]
                self.landmarks[i,j,1] = df.iloc[i,j+68]
        if train :
            self.target = df.iloc[:,-1:]

data = importcsv("./Dataset/trainset/trainset.csv",1)