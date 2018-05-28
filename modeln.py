import numpy
import numpy as py
from pandas import read_csv
import math
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#load data


#trainX=
#trainY=
#model
kersize=4
inputlength=10
nfilters=1
nb_output=1


model = Sequential()
model.add(Conv2D(nfilters,
                 input_shape=(4,300,1),#channel last
                 kernel_size=(4,3),
                 strides=1,
                 activation='relu'))
print(model.output_shape)
model.add(Reshape((model.output_shape[2],model.output_shape[3])))
model.add(LSTM(32))
model.add(Dense(nb_output,activation='relu'))
model.add(Dense(nb_output,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam')


#model.fit(trainX, trainY, epochs =100, batch_size=1, verbose = 2)
