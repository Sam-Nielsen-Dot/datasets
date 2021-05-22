import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import csv
import random
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras

asian = 2
caucasion = 0
black = 4
african = 1

EPOCHS = 10
IMG_WIDTH = 48
IMG_HEIGHT = 48
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from PIL import Image
import numpy as np

def id_generator(size=20, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def load___images():
    df = pd.read_csv('age_gender.csv', delimiter=',')
    df.dataframeName = 'age_gender.csv'
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')

    df=df.sample(frac=1) # shuffle

    X=df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32")) #converting data to numpy array
    X=np.array(X)/255.0 #normalization

    X_t = []
    for i in range(X.shape[0]):
        X_t.append(X[i].reshape(48,48,1)) #reshaping the data to (n,48,48,1)
    X = np.array(X_t)

    age=df['age'].values
    gender=df['gender'].values
    ethnicity = df['ethnicity'].values

    #plot(X, ethnicity)

    y = np.array(df['gender'])
    y = keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    model = get_model(y_train)

    model.fit(X_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(X_test,  y_test, verbose=2)


    model.save("model_gender.h5")
    print(f"Model saved to model_gender.h5")


def get_model(y):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            50, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 2nd Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            50, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # 2nd Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output unit for all categories
        tf.keras.layers.Dense(y.shape[1], activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



def plot(X,y):
    for i in range(5):
        plt.title(y[i],)
        plt.imshow(X[i].reshape(48,48))
        plt.show()

load___images()

