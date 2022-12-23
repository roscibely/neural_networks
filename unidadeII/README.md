# Deep Neural Networks 


## 1. Introdução

Neste capítulo, vamos estudar redes neurais profundas, que são redes neurais com muitas camadas. Vamos ver como elas são treinadas e como podemos usá-las para resolver problemas de classificação e regressão.

## 2. Redes Neurais Profundas

Uma rede neural profunda é uma rede neural com muitas camadas. 

### 2.1. Redes Neurais Feedforward

A figura abaixo mostra uma rede neural profunda feedforward.

![Figure](https://onlinelibrary.wiley.com/cms/asset/cd64dc27-9e14-4ed6-8b99-f3a10e0cf24b/rnc5399-fig-0002-m.png)

Matematicamente, uma rede neural profunda é uma função $f$ que recebe um vetor de entrada $x$ e retorna um vetor de saída $y$:

$$y = f(x)$$

De acordo com a figura acima, a saída de uma rede neural profunda é dada pela seguinte expressão:

$$ V(x) = \sigma(\sum_{i=1}^{n} W_i \cdot f_i(x)) $$

onde $f_i(x)$ é a saída da $i$-ésima camada da rede neural, $W_i$ é o vetor de pesos da $i$-ésima camada e $\sigma$ é a função de ativação da última camada.

### 2.2. Redes Neurais Convolucionais

Uma rede neural convolucional é uma rede neural profunda que usa camadas convolucionais. A figura abaixo mostra uma rede neural convolucional.

![Figure](https://d3i71xaburhd42.cloudfront.net/cfee6a5245806b0d3fbb9ca23e993bad64fb9b2e/4-Figure2-1.png)

As redes neurais convolucionais são usadas para processar imagens, mas também podem ser usadas para processar outros tipos de dados, como texto e áudio.

### 2.3. Redes Neurais Recorrentes

Uma rede neural recorrente é uma rede neural profunda que usa camadas recorrentes. A figura abaixo mostra uma rede neural recorrente.

![Figure](https://cdn.ttgtmedia.com/rms/onlineimages/enterpriseai-recurrent_neural_network-f_mobile.png)


As redes neurais recorrentes são usadas para processar sequências, como texto e áudio.






