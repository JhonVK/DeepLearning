import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

base_house= pd.read_csv(r'C:\Users\joaov\Desktop\OneDrive\CSV\house_prices.csv')


x=base_house.iloc[:, 3:19].values

#coluna 3 e todas as linhas, ALVO
y=base_house.iloc[:, 2].values

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

x_escalonado=StandardScaler()
x_treinamento_escalonado=x_escalonado.fit_transform(X_casas_treinamento)
y_escalonado=StandardScaler()
y_treinamento_escalonado=y_escalonado.fit_transform(y_casas_treinamento.reshape(-1,1))


x_teste_escalonado=x_escalonado.transform(X_casas_teste)
y_teste_escalonado=y_escalonado.transform(y_casas_teste.reshape(-1,1))

model = tf.keras.Sequential([ tf.keras.layers.Dense(128, activation='relu', input_shape=(x_treinamento_escalonado.shape[1],)),
                           tf.keras.layers.Dense(64, activation='relu'),
                           tf.keras.layers.Dense(1) 
                           ])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_treinamento_escalonado, y_treinamento_escalonado, epochs=50)

# Avaliar o modelo 
test_loss = model.evaluate(x_teste_escalonado, y_teste_escalonado) 
print("Test Loss: {}".format(test_loss)) # Fazer previsões 
predictions_escalonadas = model.predict(x_teste_escalonado) 

predictions = y_escalonado.inverse_transform(predictions_escalonadas) 
print(y_casas_teste[:5])
print(predictions[:5]) # Exibir as primeiras previsões # Avaliar as previsões em termos de erro absoluto médio (MAE) ou outra métrica relevante 
mae = np.mean(np.abs(predictions - y_casas_teste)) 
print(f"Mean Absolute Error: {mae}")