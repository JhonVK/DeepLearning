import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images_normalized = train_images.copy()  # Crie uma cópia da matriz original
test_images_normalized = test_images.copy()

train_images_normalized = train_images_normalized.astype('float64')
test_images_normalized = test_images_normalized.astype('float64')



#normalizando (0-255) para valores entre(0-1) isso é util pois deixa o treinamento mais rapido



train_images_normalized/=255.0
test_images_normalized/=255.0

train_images_normalized=train_images_normalized.reshape(-1, 28*28)#transformando a matriz em uma vetor(-1 significa todas as imagens)

test_images_normalized=test_images_normalized.reshape(-1, 28*28)#como a dimensão de cada imagem é 28x28, mudamos toda a base de dados para o formato [-1 (todos os elementos), altura * largura]

