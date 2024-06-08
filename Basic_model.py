import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train_normalized = X_train.copy()  # Crie uma cópia da matriz original
X_test_normalized = X_test.copy()

X_train_normalized = X_train_normalized.astype('float64')
X_test_normalized = X_test_normalized.astype('float64')



#normalizando (0-255) para valores entre(0-1) isso é util pois deixa o treinamento mais rapido



X_train_normalized/=255.0
X_test_normalized/=255.0

X_train_normalized=X_train_normalized.reshape(-1, 28*28)#transformando a matriz em uma vetor(-1 significa todas as imagens)

X_test_normalized=X_test_normalized.reshape(-1, 28*28)#como a dimensão de cada imagem é 28x28, mudamos toda a base de dados para o formato [-1 (todos os elementos), altura * largura]

print(X_train_normalized.shape)
print(X_test_normalized.shape)

model = tf.keras.Sequential()
#adicionando a primeira camada densa
#Hyper-parâmetros da camada:

#número de units/neurônios: 128
#função de ativação: ReLU
#input_shape (camada de entrada): (784, )

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))


#dropout(reduz a chance de overfitting)
model.add(tf.keras.layers.Dropout(0.2))

#Adicionando a camada de saída
#units: número de classes (10 na base de dados Fashion MNIST)
#função de ativação: softmax(mais de 2 classes)(apenas 2 seria o sigmoid)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))


#Compilando o modelo:
#Optimizer (otimizador): Adam
#Loss (função de erro): Sparse softmax (categorical) crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print(model.summary())

#treinando o modelo
model.fit(X_train_normalized, y_train, epochs=30)

test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test)

print("Test accuracy: {}".format(test_accuracy))

print(test_loss)

#salvando modelo
model_json = model.to_json()
with open("My_trained_models/fashion_model.json", "w") as json_file:
    json_file.write(model_json)

#salvando os pesos da rede
model.save_weights("My_trained_models/fashion_model.weights.h5")
