from tensorflow import keras
import imageio
import os

#setando ambiente atual
os.chdir("C:\\Users\\joaov\\OneDrive\\DeepLearning\\CriaçãodeumaAPIcomFlaskeTensorFlow2.0")


fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imageio.imwrite("uploads/{}.png".format(i), X_test[i])
