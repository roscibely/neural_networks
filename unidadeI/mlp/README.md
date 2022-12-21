# Multilayer Perceptron (MLP) or Deep Feedforward Neural Network (DFFNN) 

Uma MLP é uma rede neural artificial composta por camadas de neurônios, onde cada camada é totalmente conectada com a camada seguinte. 

As MLPs são redes neurais artificiais que possuem uma arquitetura de rede feedforward, ou seja, a informação flui apenas na direção da frente, da camada de entrada até a camada de saída.

## Arquitetura

A arquitetura de uma MLP é composta por camadas de neurônios, onde cada camada é totalmente conectada com a camada seguinte.

A camada de entrada é composta por um número de neurônios igual ao número de atributos do conjunto de dados de entrada. A camada de saída é composta por um neurônio para cada classe do conjunto de dados de saída. As camadas intermediárias são compostas por um número arbitrário de neurônios.

A definição da quantidade de neurônios em cada camada é uma tarefa difícil, pois depende de diversos fatores, como o número de atributos do conjunto de dados de entrada, o número de classes do conjunto de dados de saída, o número de exemplos de treinamento, o número de neurônios na camada anterior, etc.

## Função de ativação

A função de ativação é responsável por ativar ou não um neurônio. A função de ativação é aplicada a cada neurônio da rede, e o resultado da função de ativação é o valor de saída do neurônio. 

Existem algumas funções de ativação, como a função degrau, a função sigmóide, a função tangente hiperbólica, a função ReLU, etc.

## Função de ativação sigmóide

A função sigmóide é uma função de ativação que possui uma curva em forma de "S". A função sigmóide é definida por:

$$f(x) = \frac{1}{1 + e^{-x}}$$

A função sigmóide é uma função contínua e diferenciável, e possui um valor mínimo de 0 e um valor máximo de 1.

A função sigmóide é muito utilizada em redes neurais artificiais, pois possui uma derivada simples, que é utilizada no algoritmo de treinamento da rede neural.

Em python, a função sigmóide pode ser implementada da seguinte forma:

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

## Função de ativação tangente hiperbólica

A função tangente hiperbólica é uma função de ativação que possui uma curva em forma de "S". A função tangente hiperbólica é definida por:

$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

A função tangente hiperbólica é uma função contínua e diferenciável, e possui um valor mínimo de -1 e um valor máximo de 1.

A diferença entre a função tangente hiperbólica e a função sigmóide é que a função tangente hiperbólica possui um valor mínimo de -1 e um valor máximo de 1, enquanto a função sigmóide possui um valor mínimo de 0 e um valor máximo de 1.

Em python, a função tangente hiperbólica pode ser implementada da seguinte forma:

```python
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
```

## Função de ativação ReLU

A função ReLU é uma função de ativação definida por:

$$f(x) = \begin{cases} 0 & x < 0 \\ x & x \geq 0 \end{cases}$$

A função ReLU é uma função não contínua e não diferenciável, e possui um valor mínimo de 0 e um valor máximo de infinito.

Em python, a função ReLU pode ser implementada da seguinte forma:

```python
def relu(x):
    return max(0, x)
```

## Função de ativação softmax
 

A função softmax é uma generalização da função sigmóide para múltiplas classes. A função softmax é definida por:

$$f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

A função softmax é uma função contínua e diferenciável, e possui um valor mínimo de 0 e um valor máximo de 1.

Matematicamente, a partir da função softmax podemos chegar na função sigmóide, pois:



$$f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} = \frac{e^{x_i}}{e^{x_i} + \sum_{j=1}^{n-1} e^{x_j}} = \frac{e^{x_i}}{e^{x_i} + \sum_{j=1}^{n-1} e^{x_j} + e^{0}} = \frac{e^{x_i}}{e^{x_i} + \sum_{j=1}^{n-1} e^{x_j} + 1} = \frac{e^{x_i}}{e^{x_i} + \sum_{j=1}^{n-1} e^{x_j} + e^0} = \frac{e^{x_i}}{e^{x_i} + \sum_{j=1}^{n-1} e^{x_j} + e^0} = \frac{e^{x_i}}{e^{x_i} + \sum_{j=1}^{n-1} e^{x_j} + 1}$$

Dessa forma, 

$$f(x) = \frac{e^x}{e^x + 1}$$

dividindo a função $f(x)$ por exponencial de x, temos:

$$f(x) = \frac{1}{1 + e^{-x}}$$

Em python, a função softmax pode ser implementada da seguinte forma:

```python
def softmax(x):
    return math.exp(x) / sum([math.exp(x) for x in x])
```



