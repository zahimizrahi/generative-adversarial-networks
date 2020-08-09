import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding, multiply, Concatenate, LeakyReLU, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from sklearn import metrics
import matplotlib.pyplot as plt


class Part2Generator():
    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def build(self, input_shape, input_size, conf_shape=1, layer_size=128, dropout=1):
        noise_in = Input(shape=(input_shape,), batch_size=self.batch_size)
        conf_in = Input(shape=(conf_shape,))
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
        return Model(inputs=[noise_in, conf_in], outputs=x)


def plot_confidence_graph(y_hat):
    plt.hist(y_hat[:, 1], histtype='step', label='1')
    plt.xlabel('Label prediction')
    plt.ylabel('Number of samples')
    plt.legend(loc='upper left')
    plt.show()


def evaluate_random_forest(rf_model, x_test, y_test):
    y_pred = rf_model.predict(x_test)
    y_hat = rf_model.predict_proba(x_test)
    plot_confidence_graph(y_hat)
    y_confidence = np.empty(len(y_pred))

    for i in range(len(y_pred)):
        pred = int(y_pred[i])
        conf = y_hat[i][pred]
        y_confidence = conf
    y_confidence = y_hat[:, 1]
    min_conf = y_confidence.min(0)
    max_conf = y_confidence.max(0)
    mean_conf = y_confidence.mean(0)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("min confidence: %f, max confidence: %f, mean confidence: %f" % (min_conf, max_conf,
                                                                           mean_conf))


def evaluate_generator(generator, rf_model, x_test, y_test, generated_size=128, conf_size=1, noise_size=128,
                       save_path='part2'):
    generator.load_weights(os.path.join(save_path, 'generator_weights.h5'))

    # evaluate rf_model
    evaluate_random_forest(rf_model, x_test, y_test)

    # evaluate generator
    noise = tf.random.normal((generated_size, noise_size))
    random_conf = np.random.uniform(0, 1, (generated_size, conf_size))
    random_conf = np.sort(random_conf, axis=0)

    generator_in = [noise, random_conf]
    gen_records = generator.predict(generator_in)

    generator_samples_x = gen_records[:, :-1]
    generator_samples_y = gen_records[:, -1]

    y_hat = rf_model.predict_proba(generator_samples_x)
    plot_confidence_graph(y_hat)

    rf_conf = y_hat[:, 1]
    rf_conf = np.reshape(rf_conf, generated_size)

    random_conf = np.reshape(random_conf, generated_size)
    diff_conf = np.abs(random_conf - rf_conf)

    bucket_difference_confidence = []
    bucket_max = 0.1
    difference_sum = 0
    difference_count = 0
    for i in range(len(random_conf)):
        if random_conf[i] < bucket_max:
            difference_count += 1
            difference_sum += diff_conf[i]
        else:
            bucket_difference_confidence.append(difference_sum / difference_count)
            bucket_max += 0.1
            difference_count = 1
            difference_sum = diff_conf[i]
    bucket_difference_confidence.append(difference_sum / difference_count)

    plt.plot(bucket_difference_confidence)
    plt.title('Avg. Diff per confidence buckets')
    plt.ylabel('Abs. Diff')
    plt.xlabel('Confidence Input')
    plt.xticks(np.arange(0, 10), labels=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
    plt.show()
