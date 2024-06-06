import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from keras.preprocessing.image import ImageDataGenerator

#importando pasta
dataset_path_new = "./cats_and_dogs_filtered"

#setando train e validation
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

#carregando modelo pre-treinado
img_shape = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape, include_top = False, weights = "imagenet") #false(cabeçalho personalizado, (nao importar saida))
base_model.summary()

#congelando o modelo base(nao ira atualizar os pesos)
base_model.trainable = False

#ver dimensoes da ultima camada(4, 4, 1280)
print(base_model.output)

#reduzindo dimensionalidade dos dados(da ultima camada) (reduzindo para (none, 1280))
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

print(global_average_layer)

#camada saida
prediction_layer = tf.keras.layers.Dense(units = 1, activation = "sigmoid")(global_average_layer) #une com global averege layer

#definindo modelo
model = tf.keras.models.Model(inputs = base_model.input, outputs = prediction_layer)
model.summary()

#compilando o modelo
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.001), loss="binary_crossentropy", metrics = ["accuracy"])


data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

#criando gerador de imagens
train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_train.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

#treinando
model.fit(train_generator, epochs=5, validation_data=valid_generator)

valid_loss, valid_accuracy = model.evaluate(valid_generator)

print(valid_loss)
print(valid_accuracy)


#Fine tuning

#NÃO USE Fine Tuning em toda a rede neural, pois somente em algumas camadas já é suficiente.
#A ideia é adotar parte específica da rede neural para o problema específico
#Inicie o Fine Tuning DEPOIS que você finalizou a transferência de aprendizagem. 
#Se você tentar o Fine Tuning imediatamente, os gradientes serão muito diferentes entre o cabeçalho personalizado e algumas camadas descongeladas do modelo base

#descongelando algumas camadas do fim do modelo importado:
base_model.trainable = True
print(len(base_model.layers))

fine_tuning_at = 100  #vamos fazer o fine tuning a partir da camada 100

#congelando até a 100
for layer in base_model.layers[:fine_tuning_at]:
  layer.trainable = False

#compilando para o fine tuning
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.0001), loss="binary_crossentropy", metrics=["accuracy"])

#fine tuning
model.fit(train_generator, epochs=5, validation_data=valid_generator)

#avaliando
valid_loss, valid_accuracy = model.evaluate(valid_generator)
print(valid_accuracy)

#CONCLUSAO: neste caso o fine tuning nao faz tanta diferença