# Métricas para avaliação de modelos de aprendizado de máquina

## Introdução

Este arquivo contém métricas para avaliação de modelos de aprendizado de máquina. As métricas são implementadas em Python e podem ser utilizadas em conjunto com bibliotecas como o [scikit-learn](http://scikit-learn.org/stable/).

## Métricas 

### Regressão

* Mean Absolute Error (MAE): 

O erro médio absoluto calcula a média do valor absoluto dos erros. É uma medida de erro de regressão que dá uma ideia de quão erradas as previsões são. No entanto, o MAE não fornece uma ideia da direção (sobrestimado ou subestimado) do erro. 

Sua equação é:

$$MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

em que $y_i$ é o valor real e $\hat{y}_i$ é o valor predito.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o MAE é:

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)
```
em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

* Mean Squared Error (MSE): 

O erro médio quadrático calcula a média dos erros quadrados. É uma medida de erro de regressão que dá uma ideia de quão erradas as previsões são. Quanto maior o erro quadrático, maior a penalidade. O MSE é mais popular do que o MAE porque o MSE "castiga" erros maiores, o que tende a ser útil no mundo real. Quanto mais próximo de zero, melhor é o modelo.

Sua equação é:

$$MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

em que $y_i$ é o valor real e $\hat{y}_i$ é o valor predito.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o MSE é:

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

* Root Mean Squared Error (RMSE): 

O erro médio quadrático calcula a raiz quadrada da média dos erros quadrados.  Diferente do MSE, o RMSE é expresso na mesma unidade da variável alvo. Quanto mais próximo de zero, melhor é o modelo. 

Sua equação é:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

em que $y_i$ é o valor real e $\hat{y}_i$ é o valor predito.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o RMSE é:

```python
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

* R-squared (R²): 

O R² é uma medida de quão bem os dados se ajustam a uma linha de regressão. Ele fornece uma medida de quão bem os dados se ajustam a uma linha de regressão. O R² é sempre entre 0 e 100%:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

em que $SS_{res}$ é a soma dos quadrados dos resíduos e $SS_{tot}$ é a soma dos quadrados totais.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o R² é:

```python
from sklearn.metrics import r2_score

r2_score(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

### Classificação

* Accuracy: 

A acurácia é a proporção de observações que foram classificadas corretamente. É uma medida de quão bem um modelo de classificação é capaz de prever ou classificar as classes. Quanto mais próximo de 1, melhor é o modelo.

Sua equação é:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

em que $TP$ é o número de verdadeiros positivos, $TN$ é o número de verdadeiros negativos, $FP$ é o número de falsos positivos e $FN$ é o número de falsos negativos.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular a acurácia é:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

* Precision: 

A precisão, também taxa de verdadeiros positivos, é a proporção de observações positivas que foram classificadas corretamente. É uma medida de quão bem um modelo de classificação é capaz de prever ou classificar as classes. Quanto mais próximo de 1, melhor é o modelo.

Sua equação é:

$$Precision = \frac{TP}{TP + FP}$$

em que $TP$ é o número de verdadeiros positivos e $FP$ é o número de falsos positivos.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular a precisão é:

```python
from sklearn.metrics import precision_score

precision_score(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.


* Recall: 

também conhecido como sensibilidade, é a proporção de observações positivas que foram classificadas corretamente. É uma medida de quão bem um modelo de classificação é capaz de prever as classes positivas. Quanto mais próximo de 1, melhor é o modelo. 

Sua equação é:

$$Recall = \frac{TP}{TP + FN}$$

em que $TP$ é o número de verdadeiros positivos e $FN$ é o número de falsos negativos.

Diferente da precisão, que é a proporção de observações classificadas como positivos e que são realmente positivos, o recall é a proporção de observações positivas que foram classificadas como positivos.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o recall é:

```python
from sklearn.metrics import recall_score

recall_score(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

* Speciﬁcity: 

também conhecido como taxa de verdadeiros negativos, é a proporção de observações negativas que foram classificadas corretamente. É uma medida de quão bem um modelo de classificação é capaz de prever as classes negativas. Quanto mais próximo de 1, melhor é o modelo.

Sua equação é:

$$Especiﬁcity = \frac{TN}{TN + FP}$$

em que $TN$ é o número de verdadeiros negativos e $FP$ é o número de falsos positivos.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular a especificidade é:

```python
from sklearn.metrics import recall_score

recall_score(y_true, y_pred, pos_label=0)
```



* F1-Score: 

O F1-Score é a média harmônica entre a precisão e o recall. Quanto mais próximo de 1, melhor é o modelo.

Sua equação é:

$$F1-Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

em que $Precision$ é a precisão e $Recall$ é o recall.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o F1-Score é:

```python
from sklearn.metrics import f1_score

f1_score(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

* Confusion Matrix: 

A matriz de confusão é uma tabela que é usada para descrever o desempenho de um modelo de classificação (ou "classificador") em um conjunto de dados para o qual os valores de verdade são conhecidos. Cada linha da matriz de confusão representa a instância de uma classe real e cada coluna representa a instância de uma classe prevista. O nome vem do fato de que é muito fácil confundir as linhas com as colunas.

A matriz de confusão é uma tabela que mostra a frequência com que um classificador classifica cada classe. Cada linha da matriz de confusão representa a instância de uma classe real e cada coluna representa a instância de uma classe prevista. O nome vem do fato de que é muito fácil confundir as linhas com as colunas.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular a matriz de confusão é:

```python

from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.


* Curva ROC (Receiver Operating Characteristic): 

A curva ROC é uma representação gráfica da sensibilidade (ou recall) em função da especificidade. Quanto mais próximo de 1 (no eixo da taxa de positivos), melhor é o modelo.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular a curva ROC é:

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_score)
```

em que `y_true` é o vetor de valores reais e `y_score` é o vetor de valores preditos.

Exemplo: Plotando a curva ROC

```python
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```


* AUC (Area Under the Curve): 

AUC é a área abaixo da curva ROC. Quanto mais próximo de 1, melhor é o modelo.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular a AUC é:

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_true, y_score)
```

em que `y_true` é o vetor de valores reais e `y_score` é o vetor de valores preditos.


* Log Loss:

Log Loss é uma métrica de avaliação para problemas de classificação. Quanto mais próximo de 0, melhor é o modelo.

Sua equação é:

$$LogLoss = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij})$$

em que $N$ é o número de observações, $M$ é o número de classes, $y_{ij}$ é 1 se a observação $i$ pertence à classe $j$ e 0 caso contrário, e $p_{ij}$ é a probabilidade de que a observação $i$ pertença à classe $j$.

Em Python, com a biblioteca [scikit-learn](http://scikit-learn.org/stable/), a função para calcular o Log Loss é:

```python
from sklearn.metrics import log_loss

log_loss(y_true, y_pred)
```

em que `y_true` é o vetor de valores reais e `y_pred` é o vetor de valores preditos.

Exemplo: 

```python

from sklearn.metrics import log_loss

y_true = [0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.35, 0.8]

log_loss(y_true, y_pred)
```

Saída:

```python
0.21616030584380992
```

* Curva Error Missclassification:

A curva Error Missclassification é uma representação gráfica da taxa de erro em função do limiar de classificação. Quanto mais próximo de 0, melhor é o modelo.

Em Python, com a biblioteca [imblearn](http://contrib.scikit-learn.org/imbalanced-learn/stable/), a função para calcular a curva Error Missclassification é:

```python
from imblearn.metrics import plot_learning_curve

plot_learning_curve(y_true, y_pred, scoring='Missclassification error')
```







