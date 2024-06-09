import requests
import json
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
image=X_test[1]
# Prepare os dados da imagem
data = {
    "signature_name": "serving_default",
    "instances": [image.tolist()],  # 'image' é sua imagem de entrada
}

headers = {"content-type": "application/json"}

# Converta os dados em JSON
data = json.dumps(data)

# Envie a requisição POST
response = requests.post('http://localhost:8501/v1/models/cifar10_model:predict', data=data)

predictions = json.loads(response.text)['predictions']

plt.imshow(X_test[1])
plt.show()

print(class_names[np.argmax(predictions[0])])

