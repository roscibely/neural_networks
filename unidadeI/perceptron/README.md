# Rede Perceptron de uma única camada

Rede proposta por Frank Rosenblatt em 1957. A rede perceptron é uma das redes neurais mais simples. Ela é composta por uma camada de entrada e uma camada de saída. A camada de entrada recebe os dados de entrada e a camada de saída retorna a saída da rede.

Matemáticamente o modelo do neurônio é dado por:

$$
\begin{align}
\mathbf{y} = f(\sum_{i=1}^{n} w_i x_i + b)
\end{align}
$$

onde:

* $\mathbf{y}$ é a saída da rede neural
* $\mathbf{x}$ é a entrada da rede neural
* $\mathbf{w}$ é o vetor de pesos
* $\mathbf{b}$ é o bias
* $n$ é o número de entradas
* $f$ é a função de ativação

Veja que precisamos definir os pesos e o bias. Os pesos são os parâmetros que serão ajustados durante o treinamento da rede. O bias é um parâmetro que também será ajustado durante o treinamento da rede.

## Função de ativação

A função de ativação é uma função que recebe um valor e retorna um valor. A função de ativação é responsável por transformar a saída do neurônio em uma saída desejada. A função de ativação mais utilizada é a função degrau. A função degrau é dada por:

$$
\begin{align}
f(x) = \left\{
\begin{array}{ll}
1 & se \: x \geq 0 \\
0 & se \: x < 0
\end{array}
\right.
\end{align}
$$

A função degrau é uma função não diferenciável. Isso significa que não é possível calcular a derivada da função degrau. Isso pode ser um problema para o treinamento da rede. Para contornar esse problema, podemos utilizar a função sigmóide. A função sigmóide é dada por:

$$
\begin{align}
f(x) = \frac{1}{1 + e^{-x}}
\end{align}
$$

A função sigmóide é uma função diferenciável. Isso significa que é possível calcular a derivada da função sigmóide. A função sigmóide é uma função contínua e crescente. Isso significa que a função sigmóide é uma função que mapeia qualquer valor real para um valor real entre 0 e 1. 

## Treinamento da rede

O treinamento da rede é realizado através do algoritmo de aprendizado supervisionado. O algoritmo de aprendizado supervisionado é um algoritmo que recebe um conjunto de dados de entrada e um conjunto de dados de saída. O algoritmo de aprendizado supervisionado é responsável por ajustar os pesos e o bias da rede de forma que a rede consiga mapear os dados de entrada para os dados de saída.

O algoritmo de aprendizado supervisionado é composto por três etapas:

1. Inicialização dos pesos e do bias
2. Cálculo da saída da rede
3. Atualização dos pesos e do bias

### Inicialização dos pesos e do bias

Os pesos e o bias são inicializados com valores aleatórios. Os valores aleatórios são escolhidos de forma que a rede consiga aprender os dados de entrada.

### Cálculo da saída da rede

A saída da rede é calculada através da função de ativação. A função de ativação é responsável por transformar a saída do neurônio em uma saída desejada.

Exemplo: Classificação binária 

Considere os valores iniciais para entrada x dados por: 

$$
\begin{align}
x = \begin{bmatrix}
0 & 0 \\
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}
\end{align}
$$

e a saida desejada y dada por:

$$
\begin{align}
y = \begin{bmatrix}
0 \\
0 \\
0 \\
1
\end{bmatrix}

\end{align}
$$

Precisamos atualizar os pesos. A regra de atualização dos pesos é dada por:

$$
\begin{align}
w_i = w_i + \eta (y - \hat{y}) x_i
\end{align}
$$

em que 

* $\eta$ é a taxa de aprendizado
* $y$ é a saída desejada
* $\hat{y}$ é a saída da rede
* $x_i$ é a entrada da rede





[Implementação em Python](https://github.com/roscibely/neural_networks/tree/main/unidadeI/perceptron/perceptron.py)
[Implementação em Python](https://github.com/roscibely/neural_networks/blob/develop/unidadeI/perceptron/perceptron.py)
