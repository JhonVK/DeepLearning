import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = tf.constant([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = tf.constant([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

input_data=tf.constant([10.0])
print(model.predict(input_data))