import numpy as np 

# função de ativação (degrau)
def ativacao(z):
    if z >= 0:
        return 1
    else:
        return 0


def neuronio(x, w, b):
    soma = np.dot(x, w) + b
    y = ativacao(soma)
    return y


x = np.array([1, 0])      
w = np.array([0.6, 0.2]) 
b = -0.5                  

saida = neuronio(x, w, b)

print("Saída do neurônio:", saida)