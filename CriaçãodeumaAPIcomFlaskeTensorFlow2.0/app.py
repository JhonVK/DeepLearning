import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras

import imageio

from keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

#setando ambiente atual
os.chdir("C:\\Users\\joaov\\OneDrive\\DeepLearning\\CriaçãodeumaAPIcomFlaskeTensorFlow2.0")
#carregamento do modelo pre-treinado

with open("fashion_model.json", "r") as f:
    model_json=f.read()

model=tf.keras.models.model_from_json(model_json)

#carregando pesos
model.load_weights("fashion_model.weights.h5")