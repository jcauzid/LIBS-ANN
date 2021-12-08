# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plot

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
import joblib
from joblib import Parallel, delayed

Ninerals = [
    'Wolframite', 'Tourmaline', 'Sphalerite','Rutile','Quartz', 'Pyrite', 'Orthose', 'Muscovite', 'Molybdenite', 'Ilmenite',
    'Hematite', 'Fluorite', 'Chlorite', 'Chalcopyrite', 'Cassiterite', 'Biotite', 'Arsenopyrite', 'Apatite', 'Albite']
Elemnts = ['Si', 'Al', 'K', 'Ca', 'Fe', 'Mg', 'Mn', 'CaF', 'S', 'Ti',
                'Sn', 'W', 'Cu', 'Zn', 'Ba', 'Rb', 'Sr', 'Li', 'As', 'Mo', 'Na', 'P']


################################### Preprocessing the Data :
"""Reading Validation Data """
# Inputs
DF_Validation = pd.read_csv("Validation.csv" )
DF_Validation = DF_Validation.iloc[0:2000]
Input_Set_Validation = DF_Validation[Elemnts]
INPUTS_Validation = Input_Set_Validation.to_numpy()


""" fonction to get the maximum value in a list of proability """
def get_res(list_res):
    max_value = None
    max_idx = None
    for idx, num in enumerate(list_res):
        if (max_value is None or num > max_value):
            max_value = num
            max_idx = idx
    return [max_idx, max_value]
################################### Predicting with the neural network :


def ProcessInput(k):
    with tf.compat.v1.Session() as sess:
        new_saver = tf.compat.v1.train.import_meta_graph( os.getcwd() + os.path.sep + "Checkpoints"+ os.path.sep + "ckpt.meta"  )
        new_saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + os.path.sep + "Checkpoints"))
        graph = tf.compat.v1.get_default_graph()
        Input = graph.get_tensor_by_name("ph_input:0")
        print("Evaluation on the sample number : {}".format(k))
        feed_dict ={Input : INPUTS_Validation[k:k+1] }
        Prediction = graph.get_tensor_by_name("Prediction:0")
        r = sess.run(Prediction,feed_dict)
    return r 


#for k in range(len(DF_Validation)):
#   ProcessInput(k)
from multiprocessing import Pool
pool = multiprocessing.Pool(processes=4)
results = pool.map(ProcessInput, range(len(DF_Validation)) )

#Saving Results
R_Minerals = []
R_Probability = []
for w in results:
    idx, elmnt  = get_res(list_res =w[0])
    R_Minerals.append(Ninerals[idx])
    R_Probability.append(elmnt)
Minls = {'Minirals': R_Minerals, 'Prob': R_Probability}
df = pd.DataFrame(data=Minls)

df.to_csv("Results.csv")