import pickle
import pandas as pd
import numpy as np

Pkl_Filename = "model.pickle"
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

data = pd.read_csv("features_test.csv",sep=",")
preds =Pickled_LR_Model.predict(data)
output = pd.DataFrame({"labels":preds})
output.to_csv("predictions.csv",header=False,index=False)



