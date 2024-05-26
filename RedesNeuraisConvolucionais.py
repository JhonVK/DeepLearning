import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


cifar10 = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train_normalized = X_train.copy() 
X_test_normalized = X_test.copy()

X_train_normalized = X_train_normalized.astype('float64')
X_test_normalized = X_test_normalized.astype('float64')

X_train_normalized/=255.0
X_test_normalized/=255.0

plt.imshow(X_test[1])
plt.show()

#definindo o modelo
model = tf.keras.models.Sequential()


#Adicionado a primeira camada de convolução
#Hyper-parâmetros da camada de convolução:

#filters (filtros): 32
#kernel_size (tamanho do kernel): 3
#padding (preenchimento): same
#função de ativação: relu
#input_shape (camada de entrada): (32, 32, 3)
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

#Adicionando a segunda camada de convolução e a camada de max-pooling
#Hyper-parâmetros da camada de convolução:

#filters (filtros): 32
#kernel_size (tamanho do kernel):3
#padding (preenchimento): same
#função de ativação: relu
#Hyper-parâmetros da camada de max-pooling:

#pool_size: 2
#strides: 2
#padding: valid
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#Adicionando a terceira camada de convolução
#Hyper-parâmetros da camada de convolução:

#filters: 64
#kernel_size:3
#padding: same
#activation: relu
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

#Adicionando a quarta camada de convolução e a camada de max pooling
#Hyper-parâmetros da camada de convolução:

#filters: 64
#kernel_size:3
#padding: same
#activation: relu
#Hyper-parâmetros da camada de max pooling:

#pool_size: 2
#strides: 2
#padding: valid
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#Adicionando a camada de flattening
model.add(tf.keras.layers.Flatten())

#Adicionando a primeira camada densa (fully-connected)
#Hyper-parâmetros da camada densa:

#units/neurônios: 128
#função de ativação: relu
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Adicionando a camada de saída
#Hyper-parâmetros da camada de saída:

#units/neurônios: 10 (número de classes)
#activation: softmax
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.summary()

#compilando
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])
model.fit(X_train, y_train, epochs=5)

#avaliando
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
print(test_loss)