# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os  
from sklearn.preprocessing import MinMaxScaler
################################### Labels : 
# Files 
Ninerals = [
    'Wolframite', 'Tourmaline', 'Sphalerite','Rutile','Quartz', 'Pyrite', 'Orthose', 'Muscovite', 'Molybdenite', 'Ilmenite',
    'Hematite', 'Fluorite', 'Chlorite', 'Chalcopyrite', 'Cassiterite', 'Biotite', 'Arsenopyrite', 'Apatite', 'Albite']
# Variables :
Elemnts = ['Si', 'Al', 'K', 'Ca', 'Fe', 'Mg', 'Mn', 'CaF', 'S', 'Ti',
                'Sn', 'W', 'Cu', 'Zn', 'Ba', 'Rb', 'Sr', 'Li', 'As', 'Mo', 'Na', 'P']
################################### Preprocessing the Data :
"""
        > Reading data 
        > Labelizing data
        > Collecting data in one dataframe (data base)
"""



#### Training set :
scaler = MinMaxScaler()
DF = pd.DataFrame()
for i in range(len(Ninerals)):
    directory = os.getcwd() + os.path.sep + "DATA3"  + os.path.sep + "{}.xlsx".format(Ninerals[i])
    df = pd.read_excel(directory)
    df_data = df[df.columns[1:]]
    df_data = df_data.fillna(0) 
    df_data["Target"] =  [int(i)] *len(df_data)
    DF = DF.append(df_data)

### Validation set :
df_ = pd.read_excel(os.getcwd() + os.path.sep + "DATA3"  + os.path.sep + "Dataset4_All_RefMinerals_Insitudata.xlsx")
df_ = df_.iloc[6224::]
df_ = df_[Elemnts]
df_ = df_.fillna(0) 
df_["Target"] =  [1000] *len(df_)
""" Saving the validation data set"""
df_.to_csv(os.getcwd() + os.path.sep + "DATA3"  + os.path.sep +"Dataset_validation_ref.csv", index = None)
###### Processing : 
""" Mixing the training data set"""
DF=DF.sample(frac=1).reset_index(drop=True)
""" Reading the validation data set"""
df_Validation = pd.read_csv(os.getcwd() + os.path.sep + "DATA3"  + os.path.sep +"Dataset_validation_ref.csv")
DF = DF.append(df_Validation)
DF[DF < 0] = 0
""" Scaling all the data set"""
DF[Elemnts] = scaler.fit_transform(DF[Elemnts])
""" Separating the  Training data set and the  """
DDFF_train_test = DF.iloc[0:6224]
print(len(DDFF_train_test))
""" Saving the Training set (train and test) """
DDFF_train_test.to_csv("Train_Test.csv", index=False )
""" Saving the Validation set (train and test) """
DDFF_Validation = DF.iloc[6224::]
DDFF_Validation.to_csv("Validation.csv", index=False )
print(len(DDFF_Validation))
