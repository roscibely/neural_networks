# Gated Recurrent Unit (GRU)

A Gated Recurrent Unit (GRU) é uma unidade de memória que é usada para manter informações relevantes sobre o passado. Ela é uma extensão da LSTM, que é uma unidade de memória mais simples. 

A GRU é composta por duas portas: uma porta de atualização e uma porta de reset. A porta de atualização é responsável por decidir quais informações serão mantidas na memória. A porta de reset é responsável por decidir quais informações serão esquecidas.

A GRU é composta por três vetores de pesos: $W_z$, $W_r$ e $W_h$. O vetor $W_z$ é responsável por decidir quais informações serão mantidas na memória. O vetor $W_r$ é responsável por decidir quais informações serão esquecidas. O vetor $W_h$ é responsável por decidir quais informações serão adicionadas à memória. 

![Figure](https://www.researchgate.net/publication/328462205/figure/fig4/AS:684914898923521@1540307845043/Gated-Recurrent-Unit-GRU.ppm)

## Atualização da memória

A atualização da memória é feita da seguinte forma:

$$
\begin{align}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \\
\tilde{h}_t &= \tanh(W_h \cdot [r_t \cdot h_{t-1}, x_t] + b_h) \\
h_t &= (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
\end{align}
$$

em que $z_t$ é a porta de atualização, $r_t$ é a porta de reset, $\tilde{h}_t$ é o vetor de atualização e $h_t$ é o vetor de memória atualizado.

## Porta de atualização

A porta de atualização é responsável por decidir quais informações serão mantidas na memória. Ela é calculada da seguinte forma:

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

em que $W_z$ é o vetor de pesos da porta de atualização, $h_{t-1}$ é o vetor de memória anterior, $x_t$ é o vetor de entrada atual e $b_z$ é o vetor de bias da porta de atualização.

## Porta de reset

A porta de reset é responsável por decidir quais informações serão esquecidas. Ela é calculada da seguinte forma:

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

em que $W_r$ é o vetor de pesos da porta de reset, $h_{t-1}$ é o vetor de memória anterior, $x_t$ é o vetor de entrada atual e $b_r$ é o vetor de bias da porta de reset.

## Vetor de atualização

O vetor de atualização é responsável por decidir quais informações serão adicionadas à memória. Ele é calculado da seguinte forma:

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \cdot h_{t-1}, x_t] + b_h)
$$

em que $W_h$ é o vetor de pesos do vetor de atualização, $r_t \cdot h_{t-1}$ é o vetor de memória anterior multiplicado pela porta de reset, $x_t$ é o vetor de entrada atual e $b_h$ é o vetor de bias do vetor de atualização.

## Vetor de memória atualizado

O vetor de memória atualizado é calculado da seguinte forma:

$$
h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
$$

em que $z_t$ é a porta de atualização, $h_{t-1}$ é o vetor de memória anterior e $\tilde{h}_t$ é o vetor de atualização.


## Implementação com Keras e TensorFlow 

A implementação da GRU com Keras e TensorFlow é feita da seguinte forma:

```python
from tensorflow.keras.layers import GRU

gru = GRU(units=10, return_sequences=True, return_state=True)
```

em que `units` é o número de unidades da GRU, `return_sequences` é um booleano que indica se a saída da GRU deve ser retornada para cada instante de tempo ou apenas para o último instante de tempo, `return_state` é um booleano que indica se o estado da GRU deve ser retornado.

## Exemplo 

O exemplo a seguir mostra como a GRU é usada para prever o próximo caractere de uma sequência de caracteres.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```

```python
# Caracteres
chars = 'abcdefghijklmnopqrstuvwxyz'

# Número de caracteres
num_chars = len(chars)

# Mapeamento de caracteres para índices
char_to_index = {char: index for index, char in enumerate(chars)}

# Mapeamento de índices para caracteres
index_to_char = {index: char for index, char in enumerate(chars)}
```

```python
# Sequência de caracteres
sequence = 'the quick brown fox jumps over the lazy dog'

# Tamanho da sequência
sequence_length = len(sequence)

# Tamanho da janela
window_size = 3

# Tamanho do passo
step_size = 1

# Sequências de entrada
input_sequences = []

# Sequências de saída
output_sequences = []

# Cria as sequências de entrada e saída
for i in range(0, sequence_length - window_size, step_size):
    input_sequences.append(sequence[i:i + window_size])
    output_sequences.append(sequence[i + window_size])

# Número de sequências
num_sequences = len(input_sequences)

# Tamanho do vocabulário
vocab_size = len(chars)

# Tamanho da sequência de entrada
input_sequence_length = len(input_sequences[0])

# Tamanho da sequência de saída
output_sequence_length = len(output_sequences[0])
```

```python
# Sequências de entrada
input_sequences

# Sequências de saída
output_sequences
```

```python
# Vetor de entrada
input_vectors = np.zeros((num_sequences, input_sequence_length, vocab_size), dtype=np.bool)

# Vetor de saída
output_vectors = np.zeros((num_sequences, vocab_size), dtype=np.bool)
```

```python
# Preenche os vetores de entrada e saída
for i, input_sequence in enumerate(input_sequences):
    for j, char in enumerate(input_sequence):
        input_vectors[i, j, char_to_index[char]] = 1
    output_vectors[i, char_to_index[output_sequences[i]]] = 1
```

```python
# Cria o modelo
model = Sequential()

# Adiciona a camada GRU
model.add(GRU(units=128, input_shape=(input_sequence_length, vocab_size)))

# Adiciona a camada densa
model.add(Dense(units=vocab_size, activation='softmax'))

# Compila o modelo
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Resumo do modelo
model.summary()
```

```python
# Treina o modelo
model.fit(input_vectors, output_vectors, epochs=100, batch_size=128)
```

```python
# Função para gerar texto
def generate_text(seed, num_chars):
    # Vetor de entrada
    input_vector = np.zeros((1, input_sequence_length, vocab_size))

    # Preenche o vetor de entrada
    for i, char in enumerate(seed):
        input_vector[0, i, char_to_index[char]] = 1

    # Gera o texto
    for i in range(num_chars):
        # Predição
        prediction = model.predict(input_vector, verbose=0)[0]

        # Índice do próximo caractere
        index = np.argmax(prediction)

        # Caractere
        char = index_to_char[index]

        # Adiciona o caractere ao texto
        seed += char

        # Atualiza o vetor de entrada
        input_vector[0, 0:input_sequence_length - 1, :] = input_vector[0, 1:, :]
        input_vector[0, input_sequence_length - 1, :] = 0
        input_vector[0, input_sequence_length - 1, index] = 1

    return seed
```

```python
# Gera o texto
generate_text('the quick brown fox jumps over the lazy dog', 100)
```



