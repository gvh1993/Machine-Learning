# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:14:58 2017

@author: Gert-Jan
"""
import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import matplotlib.pyplot as plt

##### PRE-PROCESSING #####
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)# (60000, 28, 28)
plt.imshow(X_train[0])

#(n, width, height) to (n, width, height, depth).
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#The final preprocessing step for the input data is to 
#convert our data type to float32 and normalize our data values to the range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


#Hmm... that may be problematic. 
#We should have 10 different classes, one for each digit, 
#but it looks like we only have a 1-dimensional array. 
#Let's take a look at the labels for the first 10 training samples:

print(y_train[:10])# [5 0 4 1 9 2 1 3 1 4]
#And there's the problem. The y_train and y_test data are not split into 10 distinct class labels, 
#but rather are represented as a single array with the class values.
#We can fix this easily:

# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)



##### MODEL #####
model = Sequential()

# 32 = filters: Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
# 3 = kernel_size: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
# 3 = strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the width and height. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
# The input shape parameter should be the shape of 1 sample. In this case, it's the same (28, 28, 1)
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))


print(model.output_shape) # (None, 32, 26, 26)


model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))#MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter.
model.add(Dropout(0.25))# to prevent overfitting -> regularization 


#Fully connected Dense layers
model.add(Flatten())#Also note that the weights from the Convolution layers must be flattened (made 1-dimensional) before passing them to the fully connected Dense layer.
model.add(Dense(128, activation='relu'))#Dense(output size, activation = activation)
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))#output layer must be softmax


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

##### FIT & EVALUATE #####
model.fit(X_train, y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)

##### save model #####
model.save("convolutional_model.h5")