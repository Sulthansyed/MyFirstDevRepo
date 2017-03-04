
# coding: utf-8

# In[ ]:

#importing the needed dependencies
import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical,pad_sequences
from tflearn.datasets import imdb


# In[ ]:

#loading the dataset
train,test,_ = imdb.load_data(path='imdb.pkl', n_words = 10000, valid_portion=0.1)

#split of train,test data
trainX,trainY = train
testX,testY = test


# In[ ]:

trainX


# In[ ]:

trainY[0:3]


# In[ ]:

#Curating the dataset to feed to the neural net
trainX = pad_sequences(trainX, maxlen=100,value=0.)#convert the input review into matrix from 0to100 with padding
testX = pad_sequences(testX, maxlen=100,value=0.)


# In[ ]:

trainX[1]


# In[ ]:

testX[1]


# In[ ]:

#Converting output into categorical matrix
trainY = to_categorical(trainY,nb_classes=2)
testY = to_categorical(testY,nb_classes=2)


# In[ ]:

trainY[1]


# In[ ]:

testY[-1]


# In[ ]:

#preparing the network
#clearing the pre intialised graph values
tf.reset_default_graph()
#Input Layer
net = tflearn.input_data([None,100])#size of each review matrix
#Hidden Layer
net = tflearn.embedding(net, input_dim=10000,output_dim=128)
#Recurrent Neural Network uses LSTM(LongShortTermMemory) feature which remembers the input from previous layers
net = tflearn.lstm(net, 128, dropout=0.8)
#Outputlayer
net = tflearn.fully_connected(net, 2, activation='softmax')#softmax works well than sigmoid for multiclass classfication
#performance optimizer
net = tflearn.regression(net,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy')


# In[ ]:

#training the model
model = tflearn.DNN(net,tensorboard_verbose=0)


# In[ ]:

#train the network
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True)


# In[ ]:



