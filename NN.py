#/*##########################################################################
# Copyright (C) 2020-2021 The University of Lorraine - France
#
# This file is part of the LIBS-ANN toolkit developed at the GeoRessources
# Laboratory of the University of Lorraine, France.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plot
Ninerals = [
    'Wolframite', 'Tourmaline', 'Sphalerite','Rutile','Quartz', 'Pyrite', 'Orthose', 'Muscovite', 'Molybdenite', 'Ilmenite',
    'Hematite', 'Fluorite', 'Chlorite', 'Chalcopyrite', 'Cassiterite', 'Biotite', 'Arsenopyrite', 'Apatite', 'Albite']

Elemnts = ['Si', 'Al', 'K', 'Ca', 'Fe', 'Mg', 'Mn', 'CaF', 'S', 'Ti',
                'Sn', 'W', 'Cu', 'Zn', 'Ba', 'Rb', 'Sr', 'Li', 'As', 'Mo', 'Na', 'P']
############################## import Data :
DF_ = pd.read_csv("Train_Test.csv" )
DF_ = DF_.iloc[0:6000]

######## Inputs :
Input_Set = DF_[Elemnts]
Vector_Input_Set = Input_Set.to_numpy()
INPUTS = Vector_Input_Set
######## Outputs :
Output_Set = DF_['Target']
Vector_Output_Set = Output_Set.to_numpy()
def id_to_vector(vector):
    IdVector = []
    for elemnt in vector:
        IdVector.append(np.eye(N = elemnt+1, M = len(Ninerals))[elemnt:][0].tolist())
    return np.array(IdVector)
OUPUTS = id_to_vector(vector = Vector_Output_Set)
############################## Train and Test set :
X_train, X_test, y_train, y_test = train_test_split(INPUTS, OUPUTS, test_size=0.2, random_state=42)

m_train, n_train = np.shape(y_train)
Y_train = []
for i in range(m_train):
    Y_train.append(y_train[i])

m_test, n_test = np.shape(y_test)
Y_test = []
for i in range(m_test):
    Y_test.append(y_test[i])
################################################################################## Neural Networ #############################################################""
############## Parameters :
nbr_ni = 200
learning_rate = 0.0001
taille_batch = 60
nbr_entraimnemt = 200

NumMinls = len(Ninerals)
NumElements = len(Elemnts)
############## Neural Networks Architecture :
tf.compat.v1.disable_eager_execution()
# Input Layer : Placeholders 
ph_input = tf.compat.v1.placeholder(shape = (None,NumElements), dtype = tf.float32, name = "ph_input")
ph_output = tf.compat.v1.placeholder(shape = (None,NumMinls), dtype = tf.float32, name = "ph_output")
# Hidden Layer 1 :
wci = tf.compat.v1.Variable(tf.random.truncated_normal(shape = (NumElements, nbr_ni)), dtype = tf.float32, name = "wci")
bci = tf.compat.v1.Variable(np.zeros(shape = (nbr_ni)), dtype = tf.float32, name = "bci")
sci = tf.linalg.matmul( ph_input , wci)+ bci
sci = tf.math.sigmoid(sci)

# Hidden Layer 2 :
wci2 = tf.compat.v1.Variable(tf.random.truncated_normal(shape = (nbr_ni, nbr_ni)), dtype = tf.float32, name = "wci2")
bci2 = tf.compat.v1.Variable(np.zeros(shape = (nbr_ni)), dtype = tf.float32, name = "bci2")
sci2 = tf.linalg.matmul( sci , wci2)+ bci2
sci2 = tf.math.sigmoid(sci2)
# Hidden Layer 3 :
wcs = tf.compat.v1.Variable(tf.random.truncated_normal(shape = (nbr_ni, NumMinls)), dtype = tf.float32, name = "wcs")
bcs = tf.compat.v1.Variable(np.zeros(shape = (NumMinls)), dtype = tf.float32, name = "bcs")
scs = tf.linalg.matmul( sci2 , wcs)+ bcs
# Output Layer :
scso = tf.nn.softmax(scs, name = "Prediction") 
# objective function :
loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels = ph_output, logits = scs)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
accuracy = tf.math.reduce_mean( tf.cast( tf.math.equal( tf.math.argmax( scso,1 ), tf.math.argmax( ph_output,1 )), dtype =  tf.float32) )
# saving the model :
checkpoint_directory = os.getcwd() + os.path.sep + "Checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")



saver=tf.compat.v1.train.Saver(save_relative_paths = True)
# Training and Testing :
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    tab_acc_train = []
    tab_acc_test = []

    for id_entrainement in range(nbr_entraimnemt):
        print("ID entrainement ", id_entrainement )

        ### Training :
        for batch in range(0,len(X_train), taille_batch):
            res = sess.run(  train, feed_dict = { ph_input  : X_train[batch : batch + taille_batch],   ph_output : Y_train[batch : batch + taille_batch] })

        ### Accuracy Training :
        tab_acc = []
        for batch in range(0,len(X_train), taille_batch):
            acc = sess.run(  accuracy, feed_dict = { ph_input  : X_train[batch : batch + taille_batch],   ph_output : Y_train[batch : batch + taille_batch] })
            tab_acc.append(acc)

        print("Accuracy Training  ", np.mean(tab_acc) )
        tab_acc_train.append(1-np.mean(tab_acc))

        ### Accuracy Testing :
        tab_acc = []
        for batch in range(0,len(X_test), taille_batch):
            acc = sess.run(  accuracy, feed_dict = { ph_input  : X_test[batch : batch + taille_batch],   ph_output : Y_test[batch : batch + taille_batch] })
            tab_acc.append(acc)

        print("Accuracy Testing  ", np.mean(tab_acc) )
        tab_acc_test.append(1-np.mean(tab_acc))
    
    save_path = saver.save(sess, checkpoint_prefix)

    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_acc_train, label = "Train error")
    plot.plot(tab_acc_test, label  = "Test error")
    plot.legend(loc="upper right")
    plot.show()
