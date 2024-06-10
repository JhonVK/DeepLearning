import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.
X_test = X_test / 255.
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.fit(X_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

# Convertendo o modelo para o TensorFlow Lite

#Criando o TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#Convertendo o modelo
tflite_model = converter.convert()

#Salvando a versão TFLite
with open("tflite_model", "wb") as f:
  f.write(tflite_model)