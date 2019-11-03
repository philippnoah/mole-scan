from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# random
import random

import os
from PIL import Image
import glob
import cv2
import pandas as pd
import csv
import sys

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# read args
file_path = sys.argv[1]
im = Image.open(file_path).convert('1')
img = mpimg.imread(file_path)
gray = rgb2gray(img)
test_file = np.array(gray).reshape(1, 100, 75)

# ::PARSE DATA::

# sanity
print("TF.version ", tf.__version__)

# get dataset
fashion_mnist = keras.datasets.fashion_mnist

# set dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']

training_file = r'../skin-cancer-mnist-ham10000/hmnist_28_28_L_training.csv'
df = pd.read_csv(training_file)
df_train = df.sample(frac=1)
training_data = np.array(df_train.iloc[1:,:-1], dtype=np.float).reshape(-1, 28, 28)
training_labels = np.array(df_train.iloc[1:,-1], dtype=np.int)

testing_file = r'../skin-cancer-mnist-ham10000/hmnist_28_28_L_testing.csv'
df = pd.read_csv(testing_file)
df_test = df.sample(frac=1)
testing_data = np.array(df_train.iloc[1:,:-1], dtype=np.float).reshape(-1, 28, 28)
testing_labels = np.array(df_train.iloc[1:,-1], dtype=np.int)

# display a random image
plt.figure()
plt.imshow(training_data[random.randint(0, 100)], cmap='gray', vmin=0, vmax=255, interpolation='none')
plt.colorbar()
plt.grid(False)
# plt.show()

# normalize data
training_data = training_data / 255.0
testing_data = testing_data / 255.0

# sanity
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_data[i], cmap=plt.cm.binary)
    # plt.xlabel(class_names[label_data[i]])
# plt.show()

# ::BUILD MODEL::
filters = 100
kernel_size = 10

print(training_data.shape)
# init layers
model = keras.Sequential([
    # input layer
    # hidden layer (single) with 128 nodes
    # conv layer
    # keras.layers.Conv2D(14, 4,
    #       activation='relu',
    #       input_shape=(None,28,28)),
    keras.layers.Flatten(),

    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),

    # output layer with 10 P(X=class), where Sum(classes)=1
    keras.layers.Dense(7, activation='softmax')
])

# specify loss function, learning method (optimizer), metrics to keep track of
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ::TRAIN MODEL::

# start training
# l = 100 # speed up learning process
l = len(training_data)
model.fit(training_data[:l], training_labels[:l], epochs=100)

# evaluate
test_loss, test_acc = model.evaluate(testing_data,  testing_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print(model.summary())

# ::PREDICT::
exit()
# declare predictions
predictions = model.predict(testing_data)
for prediction in predictions[:100]:
    print(prediction[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
