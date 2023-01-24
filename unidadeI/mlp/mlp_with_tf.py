"""
Neste exemplo, vamos usar o conjunto de dados Iris para classificar as flores em três classes diferentes.

O conjunto de dados Iris contém 150 amostras de três espécies de Iris (Iris setosa, Iris virginica e Iris versicolor). Quatro recursos foram medidas a partir de cada amostra: o comprimento e a largura das sépalas e pétalas, em centímetros.

https://archive.ics.uci.edu/ml/datasets/Iris
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

# Split dataset
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

# Standardize features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Convert class labels into one-hot encoding
y_train_onehot = pd.get_dummies(y_train).values
y_test_onehot = pd.get_dummies(y_test).values

# Create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['mse'])
model.summary()

# Train model
history = model.fit(X_train_std, y_train_onehot, validation_split=0.2, epochs=100, batch_size=10)

# Plot training & validation accuracy values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Função Perda')
plt.ylabel('MSE')
plt.xlabel('Epocas')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()
