import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding, multiply, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from sklearn import metrics


class Part2Generator():
    def __init__(self, batch_size=128):
      self.batch_size = batch_size

    def build(self, input_shape, input_size, conf_shape=1, layer_size=128, dropout=1):
      noise_in = Input(shape=(input_shape,), batch_size=self.batch_size)
      conf_in = Input(shape=(conf_shape, ))
      x = Concatenate()([noise_in, conf_in])
      x = Dense(layer_size, activation='relu')(noise_in)
      if dropout < 1:
          x = Dropout(dropout)(x)
      x = Dense(layer_size * 2, activation='relu')(x)
      if dropout < 1:
          x = Dropout(dropout)(x)
      x = Dense(layer_size * 4, activation='relu')(x)
      x = Dense(input_size)(x)
      x = Concatenate()([x, conf_in])
      return Model(inputs=[noise_in, conf_in],  outputs=x)


def confidence_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred[:, -1])
