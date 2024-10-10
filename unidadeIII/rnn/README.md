# Recurrent Neural Networks

As redes neurais recorrentes (RNN) são uma classe de redes neurais artificiais que são usadas para processar sequências de dados, como texto, áudio e vídeo. Elas são usadas para reconhecer padrões em sequências de dados, como reconhecer palavras em um texto, reconhecer a fala em um áudio, reconhecer ações em um vídeo, etc.

## O que são redes neurais recorrentes?

Diferentemente das redes feedforward, as redes neurais recorrentes possuem um loop interno que permite que a rede armazene informações sobre uma sequência de dados. A rede neural recorrente é composta por um conjunto de neurônios que são repetidos várias vezes ao longo da rede. Cada neurônio recebe como entrada o valor de saída do neurônio anterior e o valor de entrada atual. A figura abaixo mostra um exemplo de uma rede neural recorrente.

![RNN](https://www.simplilearn.com/ice9/free_resources_article_thumb/Simple_Recurrent_Neural_Network.png)


## Treinamento de uma rede neural recorrente

O treinamento de uma rede neural recorrente é feito de forma semelhante ao treinamento de uma rede neural feedforward. A diferença é que a rede neural recorrente possui um loop interno que permite que a rede armazene informações sobre uma sequência de dados. Para o treinamento, fazemos o uso do algoritmo de backpropagation, que é um algoritmo de otimização que calcula o gradiente da função de perda em relação aos parâmetros da rede neural. O algoritmo de backpropagation é aplicado em cada neurônio da rede, de trás para frente, calculando o gradiente da função de perda em relação aos parâmetros de cada neurônio. O gradiente é então usado para atualizar os parâmetros da rede neural. O algoritmo de backpropagation é aplicado várias vezes até que a rede neural atinja um nível de acurácia desejado.

A saída da rede neural recorrente é calculada da seguinte forma:

$$ h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$

$$ y_t = f(W_{hy}h_t + b_y) $$

onde $h_t$ é a saída do neurônio na camada oculta na iteração $t$, $y_t$ é a saída do neurônio na camada de saída na iteração $t$, $x_t$ é a entrada na iteração $t$, $W_{hh}$ é a matriz de pesos da camada oculta, $W_{xh}$ é a matriz de pesos da camada de entrada, $W_{hy}$ é a matriz de pesos da camada de saída, $b_h$ é o bias da camada oculta, $b_y$ é o bias da camada de saída e $f$ é a função de ativação.


Queremos minimizar a função de perda, que é calculada da seguinte forma:

$$ L = \sum_{t=1}^T L(y_t, \hat{y}_t) $$

$$ L(y_t, \hat{y}_t) = \frac{1}{2}(\hat{y}_t - y_t)^2 $$

$$ \hat{y}_t = \text{softmax}(y_t) $$

$$ y_t = W_{hy}h_t + b_y $$

$$ h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$


onde $L$ é a função de perda, $T$ é o número de iterações, $y_t$ é a saída do neurônio na camada de saída na iteração $t$, $\hat{y}_t$ é a saída do neurônio na camada de saída na iteração $t$ após a aplicação da função softmax, $x_t$ é a entrada na iteração $t$, $W_{hh}$ é a matriz de pesos da camada oculta, $W_{xh}$ é a matriz de pesos da camada de entrada, $W_{hy}$ é a matriz de pesos da camada de saída, $b_h$ é o bias da camada oculta, $b_y$ é o bias da camada de saída e $f$ é a função de ativação.

A função de perda é calculada para cada iteração e a soma de todas as funções de perda é usada para calcular o gradiente da função de perda em relação aos parâmetros da rede neural. O gradiente é então usado para atualizar os parâmetros da rede neural. O algoritmo de backpropagation é aplicado várias vezes até que a rede neural atinja um nível de acurácia desejado.

## Backpropagation Through Time

Considerando uma rede neural recorrente com $T$ iterações, o algoritmo de backpropagation é aplicado em cada iteração, de trás para frente, calculando o gradiente da função de perda em relação aos parâmetros de cada neurônio. O gradiente é então usado para atualizar os parâmetros da rede neural. 

Podemos escrever o algoritmo de backpropagation da seguinte forma:

$$ \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}} $$

$$ \frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{xh}} $$

$$ \frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial W_{hy}} $$

$$ \frac{\partial L}{\partial b_h} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial b_h} $$

$$ \frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial b_y} $$

$$ \frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial h_t} $$

$$ \frac{\partial L}{\partial y_t} = \frac{\partial L}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial y_t} $$

$$ \frac{\partial L}{\partial \hat{y}_t} = \frac{1}{2}(\hat{y}_t - y_t) $$

$$ \frac{\partial \hat{y}_t}{\partial y_t} = \text{softmax}'(y_t) $$

$$ \frac{\partial y_t}{\partial h_t} = W_{hy} $$

$$ \frac{\partial h_t}{\partial W_{hh}} = h_{t-1} $$

$$ \frac{\partial h_t}{\partial W_{xh}} = x_t $$

$$ \frac{\partial h_t}{\partial b_h} = 1 $$

$$ \frac{\partial y_t}{\partial b_y} = 1 $$

$$ \frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} W_{hy} $$

$$ \frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} W_{hy} f'(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$

$$ \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} W_{hy} f'(W_{hh}h_{t-1} + W_{xh}x_t + b_h) h_{t-1} $$

$$ \frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} W_{hy} f'(W_{hh}h_{t-1} + W_{xh}x_t + b_h) x_t $$

$$ \frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} h_t $$

$$ \frac{\partial L}{\partial b_h} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} W_{hy} f'(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$

$$ \frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial L}{\partial y_t} $$
$$ \frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} W_{hy} f'(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$

## Implementação com o Tensorflow 

Podemos fazer o uso do framework Tensorflow para implementar uma rede neural recorrente. Para isso, vamos usar o dataset MNIST, que contém imagens de dígitos escritos à mão. Cada imagem é de tamanho 28x28 pixels e cada pixel é representado por um valor entre 0 e 255. O dataset MNIST contém 60.000 imagens de treinamento e 10.000 imagens de teste.

Primeiramente, vamos importar as bibliotecas necessárias para a implementação.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

Agora, vamos carregar o dataset MNIST.

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

O dataset MNIST contém 60.000 imagens de treinamento e 10.000 imagens de teste. Cada imagem é de tamanho 28x28 pixels e cada pixel é representado por um valor entre 0 e 255. Vamos visualizar algumas imagens do dataset.

```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()
```


Agora, vamos normalizar os dados de treinamento e teste. Para isso, vamos dividir cada pixel por 255.

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

Agora, vamos criar o modelo da rede neural recorrente. Para isso, vamos usar o modelo Sequential do Tensorflow. O modelo Sequential é uma pilha linear de camadas. Vamos usar a camada Flatten para transformar as imagens de tamanho 28x28 pixels em um vetor de tamanho 784. Vamos usar a camada Dense para criar a camada de saída da rede neural recorrente. A camada Dense é uma camada totalmente conectada, ou seja, cada neurônio da camada de entrada está conectado a todos os neurônios da camada de saída. A camada Dense possui 128 neurônios e a função de ativação ReLU. A camada Dense possui 10 neurônios e a função de ativação softmax.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Agora, vamos compilar o modelo. Para isso, vamos usar a função de perda categorical_crossentropy e o otimizador Adam.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

Agora, vamos treinar o modelo. Para isso, vamos usar 5 épocas e o batch size de 32.

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Agora, vamos avaliar o modelo.

```python
model.evaluate(x_test, y_test)
```


    10000/10000 [==============================] - 0s 36us/sample - loss: 0.0969 - acc: 0.9710

## Conclusão

Neste artigo, vimos o que é uma rede neural recorrente e como ela funciona. Vimos também como implementar uma rede neural recorrente com o Tensorflow.

