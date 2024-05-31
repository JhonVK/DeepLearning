import tensorflow as tf
import numpy as np
from tensorflow import keras

number_of_words = 20000
max_len = 100

imdb = keras.datasets.imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

#limitando o tamanho das listas para max_len
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

#definindo o modelo
model = tf.keras.Sequential()

#embedding
model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))

#camada LSTM
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))


#camada de saida
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compilando
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#treinando
model.fit(X_train, y_train, epochs=20, batch_size=128)

#avaliando o modelo
test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))
print(test_loss)