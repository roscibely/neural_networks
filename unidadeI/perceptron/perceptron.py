import numpy as np

class Perceptron(object):
    """Perceptron network"""

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        """
        Args:
            no_of_inputs (int): Número de entradas
            threshold (int): Número de iterações
            learning_rate (float): Taxa de aprendizado
        """
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        """
        Função para predizer a saída

        Args:
            inputs (array): Entradas

        Returns:
            activation (int): Saída
        """

        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        """
        Função para treinar a rede

        Args:
            training_inputs (array): Entradas
            labels (array): Saídas
        """

        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


if __name__ == "__main__":
    # Entradas
    training_inputs = []
    training_inputs.append(np.array([0, 0]))
    training_inputs.append(np.array([0, 1]))
    training_inputs.append(np.array([1, 0]))
    training_inputs.append(np.array([1, 1]))

    # Saídas
    labels = np.array([0, 0, 0, 1])

    # Criando a rede
    perceptron = Perceptron(2)

    # Treinando a rede
    perceptron.train(training_inputs, labels)

    # Testando a rede
    inputs = np.array([1, 1])
    print(perceptron.predict(inputs))

    inputs = np.array([0, 1])
    print(perceptron.predict(inputs))

    inputs = np.array([0, 0])
    print(perceptron.predict(inputs))

    inputs = np.array([1, 0])
    print(perceptron.predict(inputs))

    print(perceptron.weights)