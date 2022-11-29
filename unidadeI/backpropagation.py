# Algoritmo backpropagation

import numpy as np


# Função de ativação
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada da função de ativação
def sigmoid_derivada(x):
    return x * (1 - x)

# Entrada
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])

# Saída
saidas = np.array([[0],[1],[1],[0]])

# Pesos
pesos0 = 2*np.random.random((2,3)) - 1
pesos1 = 2*np.random.random((3,1)) - 1

# Treinamento
epocas = 100000
taxa_aprendizado = 0.6
momento = 1

for j in range(epocas):
    # Camada de entrada
    camada_entrada = entradas

    # Camada oculta
    soma_sinapse0 = np.dot(camada_entrada, pesos0)
    camada_oculta = sigmoid(soma_sinapse0)

    # Camada de saída
    soma_sinapse1 = np.dot(camada_oculta, pesos1)
    camada_saida = sigmoid(soma_sinapse1)

    # Erro
    erro_camada_saida = saidas - camada_saida
    media_absoluta = np.mean(np.abs(erro_camada_saida))
    if (j % 10000) == 0:
        print("Erro: " + str(media_absoluta))

    # Derivada da função de ativação
    derivada_saida = sigmoid_derivada(camada_saida)

    # Delta da camada de saída
    delta_saida = erro_camada_saida * derivada_saida

    # Erro da camada oculta
    pesos1_transposta = pesos1.T
    delta_saida_x_peso = delta_saida.dot(pesos1_transposta)
    delta_camada_oculta = delta_saida_x_peso * sigmoid_derivada(camada_oculta)

    # Ajuste dos pesos
    camada_entrada_transposta = camada_entrada.T
    pesos0 = camada_entrada_transposta.dot(delta_camada_oculta)
    camada_oculta_transposta = camada_oculta.T
    pesos1 = camada_oculta_transposta.dot(delta_saida)




