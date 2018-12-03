# importing the necessary functions
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# read in csv file with simulated driving on Track 1 in the 
# counter-clockwise direction
lines1 = []
with open('../firstSample/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines1.append(line)

# read in csv file with simulated driving on Track 1 in the 
# clockwise direction
lines2 = []        
with open('../secondSample/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines2.append(line)

# creating an empty to load *ALL* images and *ALL* steering angles
images = []
measurements = []
# correction to be applied to the left and right camera steering
# angles
correction = 0.2

# looping through the first set of simulator data
for line in lines1:
    for i in range(3):
        sourcepath = line[i]
        filename = sourcepath.split('/')[-1]
        current_path = '../firstSample/IMG/'+ filename
        # center camera image
        if i == 0 :
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
        # left camera image
        elif i == 1 :
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3]) + correction
            measurements.append(measurement)
        # right camera image
        else: 
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3]) - correction
            measurements.append(measurement)

# looping through the second set of simulator data
for line in lines2:
    for i in range(3):
        sourcepath = line[i]
        filename = sourcepath.split('/')[-1]
        current_path = '../secondSample/IMG/'+ filename
        # center camera image
        if i == 0 :
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
        # left camera image
        elif i == 1 :
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3]) + correction
            measurements.append(measurement)
        # right camera image
        else: 
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3]) - correction
            measurements.append(measurement)

# augmenting images by flipping them vertically
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # flip each image
    augmented_images.append(np.fliplr(image))
    # adjust the steering angle to reflect the image being flipped
    augmented_measurements.append(measurement*-1.0)

# convert the list of images to numpy arrays
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# the neural network
model = Sequential()
# lambda function to normalize the images
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3)))
# crop the image to only use the road aspects
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
# convolution Layer 1
model.add(Convolution2D(24,5,5,subsample=(2,2)))
# Dropout Layer
model.add(Dropout(0.5))
# Activation Layer
model.add(Activation('relu'))
# convolution Layer 2
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# conolution Layer 3
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# convolution Layer 4
model.add(Convolution2D(64,3,3,subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# convolution Layer 5
model.add(Convolution2D(24,3,3,subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# flatten to single 1-D array
model.add(Flatten())
# fully connected Layer 1
model.add(Dense(100))
# fully connected Layer 2
model.add(Dense(50))
# fully connected Layer 3
model.add(Dense(10))
# fully connected Layer 4
model.add(Dense(1))

# compile the model using the mean squared error and the adam optimizer
model.compile(loss='mse',optimizer = 'adam')
# fit the model and set the number of epochs
history_object = model.fit(X_train,y_train,validation_split = 0.2,shuffle = True, nb_epoch =10, verbose=1)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('Images/Loss_Diagram',bbox_inches='tight')

# save the model after fitting
model.save('model.h5')
