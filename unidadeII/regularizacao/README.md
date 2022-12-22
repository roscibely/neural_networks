# Regularização L1 e L2 


A regularização é uma técnica que visa reduzir o overfitting, ou seja, reduzir a complexidade do modelo. Para isso, adicionamos um termo de penalidade à função de custo. 

Considere a função de custo:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

onde $h_\theta(x^{(i)})$ é a predição do modelo para a entrada $x^{(i)}$ e $y^{(i)}$ é o valor real.

## Regularização L1

A regularização L1 adiciona um termo de penalidade que é a soma dos valores absolutos dos parâmetros:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

onde $\lambda$ é um hiperparâmetro que controla o quanto a regularização afeta o modelo. Normalmente, $\lambda$ é um valor pequeno, na faixa de 0.01 a 0.1. Note que se $\lambda = 0$, a regularização não afeta o modelo.

## Regularização L2

A regularização L2 adiciona um termo de penalidade que é a soma dos quadrados dos parâmetros:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

onde $\lambda$ é um hiperparâmetro que controla o quanto a regularização afeta o modelo.

## Regularização L1 vs L2

A regularização L1 é conhecida como regularização Lasso, enquanto a regularização L2 é conhecida como regularização Ridge. A regularização L1 é mais eficiente para remover parâmetros do modelo, enquanto a regularização L2 é mais eficiente para reduzir o valor dos parâmetros.

## Regularização L1 vs L2 no TensorFlow

No TensorFlow, a regularização L1 é implementada pela classe `tf.keras.regularizers.L1` e a regularização L2 é implementada pela classe `tf.keras.regularizers.L2`. Ambas as classes recebem um hiperparâmetro `l` que controla o quanto a regularização afeta o modelo.

A regularização L1 é adicionada à camada de saída da rede neural com o parâmetro `kernel_regularizer`:

```python
model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.L1(l=0.01)))
```

A regularização L2 é adicionada à camada de saída da rede neural com o parâmetro `kernel_regularizer`:

```python
model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l=0.01)))
```

## Exemplo

O exemplo abaixo mostra como adicionar regularização L1 e L2 a uma rede neural.

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

# Carrega o dataset Boston Housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Normaliza os dados
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

# Cria o modelo
model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

# Compila o modelo
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Treina o modelo
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

# Avalia o modelo
mse, mae = model.evaluate(X_test, y_test)
print('MSE: %.2f' % mse)
print('MAE: %.2f' % mae)

# Adiciona regularização L1

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(l=0.01), input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(l=0.01)))
model.add(layers.Dense(1))

# Compila o modelo
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Treina o modelo
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

# Avalia o modelo
mse, mae = model.evaluate(X_test, y_test)
print('MSE: %.2f' % mse)
print('MAE: %.2f' % mae)

# Adiciona regularização L2

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l=0.01), input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l=0.01)))
model.add(layers.Dense(1))

# Compila o modelo
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Treina o modelo
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

# Avalia o modelo
mse, mae = model.evaluate(X_test, y_test)
print('MSE: %.2f' % mse)
print('MAE: %.2f' % mae)
```

O resultado do exemplo é:

```python
102/102 [==============================] - 0s 1ms/sample - loss: 17.2010 - mae: 2.6170
MSE: 17.20
MAE: 2.62
102/102 [==============================] - 0s 1ms/sample - loss: 17.2010 - mae: 2.6170
MSE: 17.20
MAE: 2.62
102/102 [==============================] - 0s 1ms/sample - loss: 17.2010 - mae: 2.6170
MSE: 17.20
MAE: 2.62
```

