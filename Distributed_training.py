import os
import time
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#normalizando
X_train = X_train / 255.
X_test = X_test / 255.

#mudando dimensão das imagens
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

#Definição do modelo normal (não distribuído)
model_normal = tf.keras.models.Sequential()
model_normal.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model_normal.add(tf.keras.layers.Dropout(rate=0.2))
model_normal.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model_normal.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#Definição da estratégia distribuída
distribute = tf.distribute.MirroredStrategy()

#Definindo um modelo distribuído
with distribute.scope():
  model_distributed = tf.keras.models.Sequential()
  model_distributed.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
  model_distributed.add(tf.keras.layers.Dropout(rate=0.2))
  model_distributed.add(tf.keras.layers.Dense(units=10, activation='softmax'))
  model_distributed.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


#Comparação de velocidade entre o treinamento normal e o treinamento distribuído
starting_time = time.time()
model_distributed.fit(X_train, y_train, epochs=20, batch_size=128)
print("Distributed training: {}".format(time.time() - starting_time))

starting_time = time.time()
model_normal.fit(X_train, y_train, epochs=20, batch_size=128)
print("Normal training: {}".format(time.time() - starting_time))