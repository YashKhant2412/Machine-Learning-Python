# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:41:01 2020

@author: Yash
"""

#Importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(128,128,3),activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding 2nd convolution layer
classifier.add(Convolution2D(64,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Flattening
classifier.add(Flatten()) 

#Full Connection
classifier.add(Dense(units = 128,activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units = 128,activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units = 1,activation='sigmoid', kernel_initializer='uniform'))

#Compile CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting image data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 16,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/16,
                         epochs=50,
                         validation_data = test_set,
                         validation_steps = 2000/16)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('single_prediction/dog.jpg', target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)
