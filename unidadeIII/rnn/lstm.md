# Long Short-Term Memory (LSTM)

## 1. Introdução

A LSTM é uma rede neural recorrente (RNN) que foi projetada para resolver o problema de gradientes desaparecendo e explodindo presente no treinamento da rede RNN tradicional. 

## Vanishing e Exploding Gradient Problem

O problema do gradiente desaparecendo (vanishing) ocorre quando a rede RNN é treinada com backpropagation. O gradiente é calculado a partir da saída da rede até a entrada, e é multiplicado por uma matriz de pesos a cada camada. Se a matriz de pesos for muito pequena, o gradiente será multiplicado por um número muito pequeno, e o gradiente desaparecerá. Se a matriz de pesos for muito grande, o gradiente será multiplicado por um número muito grande, e o gradiente explodirá.

## LSTM 

A LSTM foi proposta por Hochreiter e Schmidhuber em 1997. A LSTM é uma rede neural recorrente que possui uma unidade de memória que pode ser escrita e lida. A LSTM possui três portas de controle que controlam o fluxo de informação através da rede. A LSTM possui uma unidade de memória que é atualizada a cada passo de tempo. A unidade de memória é atualizada por meio de três portas de controle: a porta de entrada, a porta de esquecimento e a porta de saída. A porta de entrada é responsável por atualizar a unidade de memória com novas informações. A porta de esquecimento é responsável por esquecer informações antigas da unidade de memória. A porta de saída é responsável por controlar o que é lido da unidade de memória.


## Implementação

Podemos realizar a implementação da LSTM usando o framework TensorFlow. A seguir, apresentamos um exemplo de implementação da LSTM usando o TensorFlow.

### 1.1. Importando as bibliotecas

```python
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
```

### 1.2. Definindo os parâmetros

```python
# Parâmetros da rede
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Parâmetros da rede LSTM
num_input = 1
timesteps = 28
num_hidden = 128
num_classes = 10
```

### 1.3. Definindo as entradas

```python
# Entradas da rede
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
```

### 1.4. Definindo os pesos e os bias

```python
# Pesos e bias
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
```

### 1.5. Definindo a LSTM

```python
def LSTM(x, weights, biases):

    # Preparando os dados para a entrada da rede
    x = tf.unstack(x, timesteps, 1)

    # Definindo a LSTM
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Gerando a saída da LSTM
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Gerando a saída da rede
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
```

### 1.6. Definindo a função de custo e o otimizador

```python
# Gerando a saída da rede
logits = LSTM(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Definindo a função de custo e o otimizador
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Avaliando o modelo
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

### 1.7. Inicializando as variáveis

```python
# Inicializando as variáveis
init = tf.global_variables_initializer()
```

### 1.8. Treinando a rede

```python
# Treinando a rede

with tf.Session() as sess:

    # Inicializando as variáveis
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Avaliando o modelo
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
```


## LSTM com Tensorflow e Keras 

A seguir, apresentamos um exemplo de implementação da LSTM usando o TensorFlow e Keras.

### 1.1. Importando as bibliotecas

```python
import tensorflow as tf
from tensorflow.keras import layers
```

### 1.2. Definindo os parâmetros

```python
# Parâmetros da rede
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Parâmetros da rede LSTM
num_input = 1
timesteps = 28
num_hidden = 128
num_classes = 10
```

### 1.3. Definindo a LSTM

```python

# Definindo a LSTM
model = tf.keras.Sequential()
model.add(layers.LSTM(num_hidden, input_shape=(timesteps, num_input)))
model.add(layers.Dense(num_classes))
```

### 1.4. Definindo a função de custo e o otimizador

```python
# Definindo a função de custo e o otimizador
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
```

### 1.5. Treinando a rede

```python
# Treinando a rede
model.fit(mnist.train.images, mnist.train.labels,
          batch_size=batch_size,
          epochs=training_steps,
          validation_data=(mnist.test.images, mnist.test.labels),
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./tmp/log')])
```







