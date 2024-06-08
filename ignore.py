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

#operações
tensor= tf.constant([[1,2], [3,4]])
tensor_2= tf.constant([[2,3], [7,8]])
print(tensor)
print(tensor+2)
print(tensor*10)
print(np.square(tensor)) #potencia 2
print(np.sqrt(tensor)) #raiz quadrada
print(np.dot(tensor, tensor_2)) #produto escalar

#strings
stringtf= tf.constant("TensorFlow")
print(stringtf)
print(tf.strings.length(stringtf))
tf.strings.unicode_decode(stringtf, "UTF8")
string_array=tf.constant(["TensorFlow", "Deep Learning", "AI"])

# Iterating through the TF string array
for string in string_array:
  print(string)

