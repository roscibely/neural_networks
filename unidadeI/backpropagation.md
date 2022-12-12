# Algoritmo Backpropagation

O algoritmo Backpropagation é um algoritmo de aprendizado supervisionado, que tem como objetivo a minimização de uma função de custo. O algoritmo Backpropagation torna o processo de aprendizado de uma rede neural (seja ela de múltiplas camadas ou não) mais eficiente, pois ele é capaz de calcular o gradiente da função de custo em relação aos pesos da rede, e assim, atualizar os pesos da rede de forma eficiente.

## Funcionamento

No processo de treinamento de uma rede, o algoritmo backpropagation inicialmente calcula o erro da rede, e então, calcula o gradiente da função de custo em relação aos pesos da rede. O gradiente é calculado através da regra da cadeia, e é calculado para cada camada da rede, da camada de saída até a camada de entrada. Após o cálculo do gradiente, os pesos da rede são atualizados, e o processo é repetido até que o erro da rede seja menor que um valor pré-definido.

## Cálculo do erro

O erro da rede é calculado através da função de custo, que é uma função que mede o quão boa é a rede neural. A função de custo é definida pelo usuário, e pode ser qualquer função que seja diferenciável. A função de custo mais comum é a função de custo quadrática, que é definida por:

$$
E = \frac{1}{2} \sum_{i=1}^{n} (y_i - t_i)^2
$$

onde $y_i$ é o valor de saída da rede para a amostra $i$, e $t_i$ é o valor desejado de saída para a amostra $i$. 

## Cálculo do gradiente

O gradiente da função de custo em relação aos pesos da rede é calculado através da regra da cadeia. A regra da cadeia é uma regra que permite calcular o gradiente de uma função composta, ou seja, uma função que é composta por outras funções. A regra da cadeia é definida por:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial y} \frac{\partial y}{\partial x}
$$

onde $f$ é a função composta, $y$ é a função interna, e $x$ é a variável externa. A regra da cadeia é aplicada para cada camada da rede, da camada de saída até a camada de entrada. O gradiente da função de custo em relação aos pesos da rede é calculado da seguinte forma:

$$
\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial y_j} \frac{\partial y_j}{\partial w_{ij}}
$$

onde $w_{ij}$ é o peso da conexão entre a unidade $i$ da camada $l$ e a unidade $j$ da camada $l+1$, $y_j$ é a saída da unidade $j$ da camada $l+1$, e $E$ é a função de custo.

## Atualização dos pesos

Após o cálculo do gradiente, os pesos da rede são atualizados da seguinte forma:

$$
w_{ij} = w_{ij} - \eta \frac{\partial E}{\partial w_{ij}}
$$

onde $\eta$ é a taxa de aprendizado, que é um valor pré-definido pelo usuário. A atualização dos pesos é feita para cada peso da rede, e é feita para cada camada da rede, da camada de saída até a camada de entrada.



