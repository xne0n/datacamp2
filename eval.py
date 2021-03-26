import pickle

Pkl_Filename = "model.pickle"
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

print(Pickled_LR_Model)
