from tensorflow import keras
import numpy as np

batch_size = 100
original_dim = 28*28
intermediate_dim = 256
latent_dim = 2
nb_epoch = 5

x = keras.layers.Input(shape=(original_dim,), name="input")
h = keras.layers.Dense(intermediate_dim, activation='relu', name="encoding")(x)
z_mean = keras.layers.Dense(latent_dim, name="mean")(h)
z_log_var = keras.layers.Dense(latent_dim, name="log-variance")(h)
z = keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = keras.models.Model(x, [z_mean, z_log_var, z], name="encoder")


