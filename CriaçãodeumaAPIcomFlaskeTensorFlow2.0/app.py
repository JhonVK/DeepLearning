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

#criação da API com Flask
app=Flask(__name__)
@app.route("/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    upload_dir="uploads/"
    image = imageio.v2.imread(upload_dir + img_name)


    classes=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    #[1,28,28]->[1, 784]
  
    prediction = model.predict([image.reshape(1, 28*28)])
    np_prediction = np.array(prediction)
    return jsonify({"object_indentifies": classes[np.argmax(np_prediction[0])]})

app.run(debug=False)