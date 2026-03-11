class Perceptron:

    def __init__(self, n_inputs, taxa_aprendizado=0.1):
        self.pesos = [0.0] * n_inputs
        self.bias = 0.0
        self.eta = taxa_aprendizado

    def ativacao(self, z):
        if z >= 0:
            return 1
        else:
            return 0

    def prever(self, x):
        soma = 0
        for i in range(len(x)):
            soma += x[i] * self.pesos[i]

        soma += self.bias
        return self.ativacao(soma)

    def treinar(self, X, d, epocas):

        for epoca in range(epocas):

            for i in range(len(X)):

                y = self.prever(X[i])
                erro = d[i] - y

                for j in range(len(self.pesos)):
                    self.pesos[j] += self.eta * erro * X[i][j]

                self.bias += self.eta * erro

            print("Época:", epoca+1, "Pesos:", self.pesos, "Bias:", self.bias)


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    d = [0, 0, 0, 1]

    perceptron = Perceptron(n_inputs=2, taxa_aprendizado=0.1)
    perceptron.treinar(X, d, epocas=10)

    for x in X:
        print("Entrada:", x, "Saída prevista:", perceptron.prever(x))