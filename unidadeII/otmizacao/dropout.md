# Dropout 

## Introdução

O Dropout é uma técnica de regularização que consiste em desativar aleatoriamente algumas unidades de uma camada durante o treinamento. Essa técnica é muito útil para evitar overfitting, pois evita que o modelo memorize os dados de treinamento.

## Como usar?

A classe Dropout recebe como parâmetro:

* rate: fração das unidades que serão desativadas

Exemplo: 0.5 significa que 50% das unidades/neurônios serão desativadas.

Normalmente, o valor de rate é definido entre 0.2 e 0.5.

A classe Dropout pode ser utilizada para desativar aleatoriamente algumas unidades de uma camada durante o treinamento. Para isso, basta adicionar a instância da classe Dropout como uma camada do modelo. O framework TensorFlow possui uma classe chamada Dropout que implementa essa técnica.


A seguir, um exemplo de uso da técnica Dropout:

```python
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(32,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```


