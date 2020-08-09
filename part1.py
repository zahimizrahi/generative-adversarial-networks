import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding, multiply, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from sklearn import metrics
from utils import load_data_from_arff

