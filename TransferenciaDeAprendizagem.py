import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from keras.src.legacy.preprocessing.image import ImageDataGenerator

#importando pasta
dataset_path_new = "./cats_and_dogs_filtered"

#setando train e validation
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

#carregando modelo pre-treinado
img_shape = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape, include_top = False, weights = "imagenet") #false(cabeçalho personalizado, (nao importar saida))
base_model.summary()

#congelando o modelo base
base_model.trainable = False