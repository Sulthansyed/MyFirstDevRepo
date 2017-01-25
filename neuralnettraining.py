# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 06:21:45 2017

@author: 548857
"""

#My First Neural Net Practice

#Downloading the needed python libraries

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import metrics

#fix random no seed for reproductiblity

seed = 7
np.random.seed(seed)

#loading the dataset
dataset = np.loadtxt("C:\\Users\\548857\\Desktop\\pima-indians-diabetes.csv" , delimiter = ",")

#Split into X and Y matrix for easier multiplication

X_train = dataset[350:,0:8]
#print(X_train)
Y_train = dataset[350:,8]
#print(Y_train)
X_test = dataset[351:,0:8]
#print(X_test)
Y_test = dataset[351:,8]
#print(Y_test)

#createModel

model = Sequential()
model.add(Dense(12,input_dim=8,init= 'uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#fit the model
model.fit(X_train,Y_train,nb_epoch = 150,batch_size=10)

#evaluate the model
scores = model.evaluate(X_train,Y_train)
print("%s: %.2f%%" %(model.metrics_names[1],scores[1]*100))
#chk = np.array([2,141,58,34,128,25.4,0.699,24]).reshape(1,-1)
prediction=model.predict(X_test)
rounded = np.around(prediction)
print(rounded)
print(metrics.confusion_matrix(Y_test,rounded))