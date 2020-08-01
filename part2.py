import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding, multiply, Concatenate, LeakyReLU, BatchNormalization


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
      #x = Concatenate()([x, conf_in])
      return Model(inputs=[noise_in, conf_in],  outputs=x)


def train_generator_part2(generator, rf_model, df, epochs=500, conf_size=1, batch_size=128, noise_size=128):
    history = []
    x_df = df.iloc[:, :-1]
    y_hat = rf_model.predict_proba(x_df)
    for epoch in range(epochs):
        noise = tf.random.normal((batch_size, noise_size))
        random_conf = np.random.uniform(0, 1, (batch_size, conf_size))
        generator_in = [noise, random_conf]
        gen_records = generator.predict(generator_in)

        generator_samples_x = gen_records[:, :-1]
        generator_samples_y = gen_records[:, -1]

        # get confidence samples by random forest
        confs = y_hat[: , 1]
        conf_samples = []
        for i in range(len(random_conf)):
            index = (np.abs(confs - random_conf[i])).argmin()
            sample = x_df.iloc[index]
            sample = np.append(sample, generator_samples_y[i])
            conf_samples.append(sample)
        conf_samples = np.array(conf_samples)
        g_loss = generator.train_on_batch(generator_in, conf_samples)
        history.append(g_loss)
        print("%d [G loss: %f]" %
                          (epoch, g_loss))
    return history
