# Early Stopping 

## O que é?

Early Stopping é uma técnica de parada de treinamento de modelos de aprendizado de máquina que consiste em parar o treinamento de um modelo quando a métrica de avaliação não melhora mais. Tal técnica é muito útil para evitar overfitting, pois evita que o modelo continue treinando após atingir um ponto de saturação.

A métrica de avaliação pode ser a acurácia, a perda, o erro quadrático médio, etc.

Imagem que ilustra o Early Stopping

![Figure](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-28_at_12.59.56_PM_1D7lrVF.png)

O framework TensorFlow possui uma classe chamada [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) que implementa essa técnica. Essa classe pode ser utilizada para parar o treinamento de um modelo quando a métrica de avaliação não melhora mais.

## Como usar?

A classe EarlyStopping recebe como parâmetros:

* monitor: métrica de avaliação que será monitorada
* patience: número de épocas sem melhora da métrica de avaliação para que o treinamento seja interrompido
* verbose: se 0, não exibe mensagens de aviso; se 1, exibe mensagens de aviso
* mode: se 'auto', o modo é inferido a partir do nome da métrica de avaliação; se 'min', o treinamento será interrompido quando a métrica de avaliação não melhorar mais; se 'max', o treinamento será interrompido quando a métrica de avaliação não piorar mais
* baseline: valor de referência para a métrica de avaliação. O treinamento será interrompido se a métrica de avaliação não melhorar mais que o valor de referência
* restore_best_weights: se True, os pesos do modelo no melhor ponto de avaliação serão restaurados

A seguir, um exemplo de uso da classe EarlyStopping:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)
```

A classe EarlyStopping pode ser utilizada para parar o treinamento de um modelo quando a métrica de avaliação não melhora mais. Para isso, basta passar a instância da classe EarlyStopping como parâmetro do método fit do modelo.

```python
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
```





