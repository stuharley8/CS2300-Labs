#!/usr/bin/env python
"""
Steering angle prediction model

Written by Jay Urbain, updates and modifications made by Derek Riley

# Building a model
# --data - subdirectory of images for training
# --h5modelout - learned output model .h5
# --model - machine learning model name - see top of model.py source for different models to select

python model.py --data data --h5modeloutput model_commaAiModelPrime_e06252018.h5 --model commaAiModelPrime


"""
import cv2
import csv
import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, ELU
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2
from keras.optimizers import Adam

import sklearn

import sys
import time
import argparse
import os

# basic net
def getModel(model="nVidiaModel"):
    if model == "basicModel":
        return basicModel()
    elif model == "nVidiaModel":
        return nVidiaModel()
    elif model == "nVidiaModelRegularization":
        return nVidiaModelRegularization()
    elif model == "commaAiModel":
        return commaAiModel()
    elif model == "commaAiModelPrime":
        return commaAiModelPrime()

def basicModel(time_len=1):
    """
    Creates basic single layer autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten(input_shape=(160,230,3)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def nVidiaModel():
    """
    Creates nVidia autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def nVidiaModelRegularization():
    """
    Creates nVidia autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def commaAiModel(time_len=1):
    """
    Creates comma.ai autonomous car  model
    Reduce dropout
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def commaAiModelPrime(time_len=1):
    """
    Creates comma.ai enhanced autonomous car  model
    Replace dropout with regularization
    Add 3 additional convolution layers
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    # Add three 5x5 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())

    # model.add(Dropout(.2))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())

    # model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())

    # model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(1))

    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model


def getDrivingLogs(path, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def getImages(path):
    """
    Get all training images on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    directories = [x[0] for x in os.walk(path)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataDirectories:
        lines = getDrivingLogs(directory, skipHeader=True)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineCenterLeftRightImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)

def generator(samples, batch_size=32, flip = False):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                if(flip):
                    images.append(cv2.flip(image,1))
                    angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--h5modeloutput', type=str, default='')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--batch_size', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    return parser.parse_args()

def main():
    args = parse_args()
    data = args.data
    h5modeloutput = args.h5modeloutput
    modelname = args.model
    my_batch_size = int(args.batch_size)
    my_epochs = int(args.epochs)

    print(args)

    # Reading images locations.
    centerPaths, leftPaths, rightPaths, measurements = getImages(data)
    imagePaths, measurements = combineCenterLeftRightImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
    print('Total Images: {}'.format(len(imagePaths)))

    # Splitting samples and creating generators.
    from sklearn.model_selection import train_test_split

    samples = list(zip(imagePaths, measurements))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    train_generator = generator(train_samples, batch_size=my_batch_size)
    validation_generator = generator(validation_samples, batch_size=my_batch_size)

    # Model creation
    model = getModel(model=modelname)

    # Train the model
    history_object = model.fit_generator(train_generator, steps_per_epoch= \
        len(train_samples)/my_batch_size, validation_data=validation_generator, \
        validation_steps=len(validation_samples)/my_batch_size, epochs=my_epochs, verbose=2)

    model.save(h5modeloutput)

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

if __name__ == "__main__":
    main()


