import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
from pandas.testing import assert_frame_equal #import alterado

from tqdm import tqdm_notebook, tqdm
from collections import deque

class AI_Trader():

  def __init__(self, state_size, action_space = 3, model_name = "AITrader"):
    self.state_size = state_size
    self.action_space = action_space
    self.memory = deque(maxlen = 2000)#atualiza os pesos de 2000 em 2000 pasos
    self.model_name = model_name

    self.gamma = 0.95 #fator de desconta da equação de bellman
    self.epsilon = 1.0 #porcentagem de randomicidade inicial(100#% nesse caso)
    self.epsilon_final = 0.01 #randomicidade final (1%) nao pode ser zero pois o agente pode ficar preso no minimo local
    self.epsilon_decay = 0.995 #taxa de edcay da randomicidade
    self.model = self.model_builder()

  def model_builder(self):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(self.state_size,))) #quantidade de neuronios na camada de entrada(preço das ações)
    model.add(tf.keras.layers.Dense(units = 32, activation = "relu"))#camada oculta
    model.add(tf.keras.layers.Dense(units = 64, activation = "relu"))#camada oculta
    model.add(tf.keras.layers.Dense(units = 128, activation = "relu"))#camada oculta
    model.add(tf.keras.layers.Dense(units = self.action_space, activation = "linear"))#camada de saida(primeiramente saira valores numericos(função linear)) 
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))#compilando o modelo
    return model

  def trade(self, state):
    #caso a condição for verdadeira uma ação aleatoria sera tomada(temos 3 ações)
    #no começo isso vai sempre acontecer pois o self.epsilon é alto
    if random.random() <= self.epsilon:
      return random.randrange(self.action_space)
    #Caso a condição for falsa entao vamos buscar as ações por meio da rede neural:
    #o retur retornara a ação a ser tomada
    actions = self.model.predict(state[0])
    return np.argmax(actions[0])

  def batch_train(self, batch_size):
    batch = [] #Onde vai ficar os nossos registros
    #buscando os ultimos registros e colocando no batch
    for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
      batch.append(self.memory[i])

    for state, action, reward, next_state, done in batch:
      if not done:
        #aplicando a equação de bellman, 
        reward = reward + self.gamma * np.amax(self.model.predict(next_state[0]))

      target = self.model.predict(state[0])
      target[0][action] = reward
    #treinando
      self.model.fit(state[0], target, epochs=1, verbose=0)

    #atualizando randomicidade
    if self.epsilon > self.epsilon_final:
      self.epsilon *= self.epsilon_decay

#Funções auxiliares

#vai ser util para normalizar os dados
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#formatação dos preços
#(lucro ou nao das negociações)
def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))
  
import yfinance as yf

#yfinance foi criada especificamente para o Yahoo Finance 
#e pode oferecer mais funcionalidades específicas do Yahoo Finance.



def dataset_loader(stock_name):
  #dataset = data_reader.DataReader(stock_name, data_source = "yahoo")
  dataset = yf.download(stock_name, start='2023-06-02')
  start_date = str(dataset.index[0]).split()[0] #data inicio
  end_date = str(dataset.index[-1]).split()[0] #data final
  close = dataset['Close'] #close é o valor de fechamento da ação
  return close

#criador de estados:
def state_creator(data, timestep, window_size):
  starting_id = timestep - window_size + 1

  if starting_id >= 0:
    # windowed_data = data[starting_id:timestep + 1] 
    windowed_data = np.array(data[starting_id:timestep + 1])
  else:
    # windowed_data = - starting_id * [data[0]] + list(data[0:timestep + 1]) 
    windowed_data = np.array(- starting_id * [data[0]] + list(data[0:timestep + 1])) 

  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

  return np.array([state]), windowed_data

#carregando a base de dados
stock_name = "PETR4.SA"
data = dataset_loader(stock_name)


#Treinando a IA

#hiperparâmetros:
window_size = 10
episodes = 1000
batch_size = 32
data_samples = len(data) - 1

#definição do modelo
trader = AI_Trader(window_size)

#loop de treinamento:
for episode in range(1, episodes + 1):
  print("Episode: {}/{}".format(episode, episodes))
  state = state_creator(data, 0, window_size + 1)
  total_profit = 0
  trader.inventory = []
  for t in tqdm(range(data_samples)):
    action = trader.trade(state)
    next_state = state_creator(data, t + 1, window_size + 1)
    reward = 0

    if action == 1: # Comprando uma ação
      trader.inventory.append(data[t])
      print("AI Trader bought: ", stocks_price_format(data[t]))
    elif action == 2 and len(trader.inventory) > 0: # Vendendo uma ação
      buy_price = trader.inventory.pop(0)

      reward = max(data[t] - buy_price, 0)
      total_profit += data[t] - buy_price
      print("AI Trader sold: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))

    if t == data_samples - 1:
      done = True
    else:
      done = False

    trader.memory.append((state, action, reward, next_state, done))

    state = next_state

    if done:
      print("########################")
      print("Total profit: {}".format(total_profit))
      print("########################")

    if len(trader.memory) > batch_size:
      trader.batch_train(batch_size)

  if episode % 10 == 0:
    trader.model.save("ai_trader_{}.h5".format(episode))