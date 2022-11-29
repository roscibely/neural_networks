# Definição de redes neurais artificiais

Uma rede neural artificial é um modelo matemático inspirado no funcionamento do cérebro humano. A ideia é que a rede possa aprender a partir de dados de entrada e saída, e assim, ser capaz de realizar tarefas que antes eram impossíveis de serem realizadas por computadores.

Dessa forma, uma rede neural artificial é um modelo matemático que recebe dados de entrada (x), processa esses dados e retorna uma saída (y). A rede é composta por camadas de neurônios, que são unidades de processamento que recebem dados de entrada, processam esses dados e retornam uma saída.


![Figure](https://neigrando.files.wordpress.com/2022/03/neuronio-e-rede-neural.png)


Matematicamente, podemos escrever a equação como:

$$
\begin{align}
\mathbf{y} = f(\mathbf{W} \mathbf{x} + \mathbf{b})
\end{align}
$$

onde:

* $\mathbf{y}$ é a saída da rede neural
* $\mathbf{x}$ é a entrada da rede neural
* $\mathbf{W}$ é a matriz de pesos
* $\mathbf{b}$ é o vetor de bias
* $f$ é a função de ativação



## Aplicações

As redes neurais artificiais são utilizadas em diversas áreas, como:

* Reconhecimento de padrões
* Reconhecimento de voz
* Reconhecimento de imagens (rostos, carros, emoções, etc)
* Reconhecimento de texto
* Reconhecimento de movimentos
* Reconhecimento de sentimentos
* Previsão de séries temporais (previsão do tempo, previsão de vendas, etc)
* Classificação de dados
* Classificação de textos
* Classificação de movimentos
* Classificação de sentimentos
* Classificação de padrões

## Tipos de redes neurais

As redes neurais podem ser divididas em dois grupo: Redes Feedforward e Redes Recorrentes.

### Redes Feedforward

As redes feedforward são redes neurais que possuem apenas uma camada de entrada, uma camada de saída e uma ou mais camadas intermediárias. As camadas intermediárias são chamadas de camadas ocultas.  Exemplo de redes feedforward:

* Perceptron
* Multilayer Perceptron
* Radial Basis Function Network
* Adaline
* Convolucional Neural Network
* Gated Attention Networks

*Exemplo de rede feedforward:*

![Rede feedforward](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png)

### Redes Recorrentes

As redes recorrentes são redes neurais que possuem uma camada de entrada, uma camada de saída e uma ou mais camadas intermediárias. As camadas intermediárias são chamadas de camadas ocultas. As camadas ocultas possuem conexões que se repetem ao longo do tempo. Exemplo de redes recorrentes:

* Recurrent Neural Network
* Long Short-Term Memory
* Gated Recurrent Unit

*Exemplo de rede recorrente:*

![Rede recorrente](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/300px-Recurrent_neural_network_unfold.svg.png)


Nesta primeira parte da disciplina, iremos estudar: 

* Algoritmo backpropagation
* Perceptron
* Multilayer Perceptron
* Radial Basis Function Network