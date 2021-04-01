#!/usr/bin/env python
# coding: utf-8


import sys
get_ipython().system('{sys.executable} -m pip install sklearn')

# imports and declarations
import datetime
import errno
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import collections
import pathlib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D, Activation, DepthwiseConv2D
from tensorflow.keras.applications import MobileNetV2

# necessary for my 3080 GPU to ensure it performs as expected and doesn't weirdly run out of memory

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def plot_images(x_train):
  for i in range(16):
    plt.subplot(4,4,1+i)
    plt.axis('off')
    plt.imshow(x_train[i, :, :, 0], cmap='gray')
  plt.savefig('images.png')
  plt.close()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert from uint to floats, and normalize to [0,1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print('x_train.shape = ', x_train.shape, ' y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape, ' y_test.shape = ', y_test.shape)

print('data type: ', type(x_train[0][0][0]))
print('label type: ', type(y_train[0][0]))

plot_images(x_train)

def LeNet_impl():
  """
  LeNet 5 implementation. Noise level is inputted as an integer to make saving the models easier.
  """
  data_in = keras.Input(shape=(28,28,1))
  kernel_size=(5,5)

  x = Conv2D(6, kernel_size, activation='relu', strides = (1,1), input_shape= (28,28,1), padding='same')(data_in)
  x = AveragePooling2D(pool_size=(2,2), strides = (2,2))(x)
  x = Conv2D(16, kernel_size, activation='relu', strides = (1,1))(x)
  x = AveragePooling2D(pool_size=(2,2), strides = (2,2))(x)
  x = Conv2D(120, kernel_size, activation='relu', strides = (1,1))(x)
  x = Flatten()(x)
  x = Dense(120, activation='relu')(x)
  x = Dense(84, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(10, activation='softmax')(x)
  
  lenet_model = keras.Model(inputs = data_in, outputs = x, name = "lenet_model")
  lenet_model.compile(optimizer = optimizers.SGD(learning_rate=0.01), loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
  return lenet_model

def depthwise_sep_conv(x, filters, alpha, strides = (1, 1)):
    y = DepthwiseConv2D((3, 3), padding = 'same', strides = strides)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(int(filters * alpha), (1, 1), padding = 'same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def MobileNet_impl(alpha = 1):
  data_in = Input(shape = (28, 28, 1))
  x = ZeroPadding2D(padding = (2, 2))(data_in)

  x = Conv2D(int(32 * alpha), (3, 3), padding = 'same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = depthwise_sep_conv(x, 64, alpha)
  x = depthwise_sep_conv(x, 128, alpha, strides= (2, 2))
  x = depthwise_sep_conv(x, 128, alpha)
  x = depthwise_sep_conv(x, 256, alpha, strides= (2, 2))
  x = depthwise_sep_conv(x, 256, alpha)
  x = depthwise_sep_conv(x, 512, alpha, strides= (2, 2))
  for _ in range(5):
    x = depthwise_sep_conv(x, 512, alpha)
  x = depthwise_sep_conv(x, 1024, alpha, strides= (2, 2))
  x = depthwise_sep_conv(x, 1024, alpha)
  x = GlobalAveragePooling2D()(x)
  x = Dense(units = 10)(x)
  x = Activation('softmax')(x)

  mobilenet_model = keras.Model(inputs = data_in, outputs = x, name = "mobilenet_model")
  mobilenet_model.compile(optimizer = optimizers.RMSprop(lr = 0.01), loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
  
  return mobilenet_model

lenet_weights_dict = {}
lenet_1 = LeNet_impl()
lenet_weight_callback = tf.keras.callbacks.LambdaCallback( on_epoch_end=lambda epoch, logs: lenet_weights_dict.update({epoch:lenet_1.layers[1].get_weights()}))
lenet_fitness = lenet_1.fit(x=x_train, y=y_train, steps_per_epoch=128, epochs=15, verbose=1, validation_data=(x_test, y_test), callbacks=lenet_weight_callback)

mobilenet_weights_dict = {}

mobilenet_1 = MobileNet_impl()
mobilenet_weight_callback = tf.keras.callbacks.LambdaCallback( on_epoch_end=lambda epoch, logs: mobilenet_weights_dict.update({epoch:mobilenet_1.layers[2].get_weights()}))
mobilenet_fitness = mobilenet_1.fit(x=x_train, y=y_train, steps_per_epoch=128, epochs=15, verbose=1, validation_data=(x_test, y_test), callbacks=mobilenet_weight_callback)
mobilenet_1.save('./models/mobilenet/alpha-1-callback')
with open('./histories/mobilenet/alpha-1-callback.txt', 'w') as default_history_file:
    default_history_file.write(json.dumps(mobilenet_fitness.history))

with open(pathlib.Path('histories/mobilenet/alpha-1-callback.txt'), 'r') as default_model_history_file:
  default_history_file_contents = json.loads(default_model_history_file.readlines()[0])
  plt.plot(default_history_file_contents['loss'])
  plt.plot(default_history_file_contents['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
# plt.savefig('accuracy.png', bbox_inches='tight')
plt.show()

def combine_models(alpha, theta_0_model, theta_star_model, new_model):
    for index, weights in enumerate(theta_0_model.trainable_weights):
        new_model.trainable_weights[index].assign((1 - alpha) * weights)

    print(len(theta_star_model.trainable_weights))
    for index, weights in enumerate(theta_star_model.trainable_weights):
        new_model.trainable_weights[index].assign_add(alpha * weights)
    return new_model

lenet_j_theta = []
mobilenet_j_theta = []

alphas = [ x * 0.2 for x in range (0, 10)]
lenet_init = LeNet_impl() # untrained model (load this)
lenet_final = keras.models.load_model('./models/lenet/default') # trained model (load this)

mobilenet_init = MobileNet_impl() # untrained model (load this)
mobilenet_final = keras.models.load_model('./models/mobilenet/alpha-1') # trained model (load this)

for alpha in alphas:
  lenet_combined_model = combine_models(alpha, lenet_init, lenet_final, LeNet_impl())
  lenet_combined_score = lenet_combined_model.evaluate(x_test, y_test, verbose = 0)
  lenet_j_theta.append(lenet_combined_score[0])

  mobilenet_combined_model = combine_models(alpha, mobilenet_init, mobilenet_final, MobileNet_impl())
  mobilenet_combined_score = mobilenet_combined_model.evaluate(x_test, y_test, verbose = 0)
  mobilenet_j_theta.append(mobilenet_combined_score[0])

plt.plot(alphas, lenet_j_theta)
plt.plot(alphas, mobilenet_j_theta)
plt.title('error surface experiment combined')
plt.ylabel('theta')
plt.xlabel('alpha value')
plt.legend(['LeNet', 'MobileNet'], loc='center left')
plt.savefig('exp-3.png', bbox_inches='tight')
plt.show()

def plot_pca(x, y, loss):
    plt.scatter(x, y)
    plt.yscale('log')
    plt.xscale('log')
    i = 0
    for a, b in zip(x, y):
        plt.text(a, b, "%.2f" % loss[i])
        # print("a: {}, b: {}, loss: {}".format(a, b, "%.2f" % loss[i]))
        i += 1
    plt.show()
    plt.save

pca_w = []
for epoch, weights in lenet_weights_dict.items():
    pca_w.append([weights[0][0][0][0], weights[1]])

variance = []
variance_x = []
variance_y = []
for w in pca_w:
    pca = PCA(n_components=2)
    pca.fit_transform(w)
    variance.append(pca.explained_variance_)


for v in range(len(variance)):
    variance_x.append(variance[v][0])
    variance_y.append(variance[v][1])


plt.scatter(variance_x, variance_y)
plt.yscale('log')
plt.xscale('log')
i = 0
for a, b in zip(variance_x, variance_y):
    plt.text(a, b, "%.2f" % lenet_fitness.history['loss'][i])
    i += 1
plt.show()
plt.savefig('exp-4-lenet.png', bbox_inches='tight')

pca_w = []
for epoch, weights in mobilenet_weights_dict.items():
    print("weights at 2nd layer for epoch {}: {}".format(epoch, weights))
    pca_w.append([weights[0][0][0][0], weights[1]])

variance_x = []
variance_y = []
for w in pca_w:
    pca = PCA(n_components=2)
    pca.fit_transform(w)
    variance.append(pca.explained_variance_)


print("varience len: ", len(variance))
for v in range(len(variance)):
    variance_x.append(variance[v][0])
    variance_y.append(variance[v][1])

print("variencex len: ", len(variance_x))
print("variencey len: ", len(variance_y))

plt.scatter(variance_x, variance_y)
plt.yscale('log')
plt.xscale('log')
i = 0
for a, b in zip(variance_x, variance_y):
    plt.text(a, b, "%.2f" % mobilenet_fitness.history['loss'][i])
    i += 1
plt.show()
plt.savefig('exp-4-lenet.png', bbox_inches='tight')
