import numpy as np
import matplotlib.pyplot as plt

class RadialBasisFunctionNetwork (object): 
    """ Radial Basis Function Network for function approximation"""

    def __init__(self, n_hidden, learning_rate, max_epochs, random_state):
        """ Constructor for RBFN class 
        Args: 
            n_hidden (int): number of hidden units
            learning_rate (float): learning rate
            max_epochs (int): maximum number of epochs
            random_state (int): random state
        """

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state

    def _gaussian_basis(self, c, d):
        """ Gaussian basis function 
        Args:
            
            c (float): center
            d (float): data point
        Returns:
            G (float): Gaussian basis function value
        """
        return np.exp(-np.linalg.norm(c - d)**2 / (2 * (self.sigma**2)))

    def _basis_matrix(self, X):
        """ Basis matrix
        Args:
            X (array): data matrix
        Returns:
            G (array): basis matrix
        """
        G = np.zeros((X.shape[0], self.n_hidden))
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._gaussian_basis(c, x)
        return G

    def fit(self, X, y):
        """ Fit the model to the data
        Args:
            X (array): data matrix
            y (array): target vector
        """
        rnd = np.random.RandomState(self.random_state)
        self.centers = rnd.permutation(X)[:self.n_hidden]
        self.sigma = np.linalg.norm(self.centers[0] - self.centers[1])

        G = self._basis_matrix(X)
        self.w = np.linalg.pinv(G.T @ G + 10**-6 * np.eye(self.n_hidden)) @ G.T @ y

    def predict(self, X):
        """ Predict the target vector
        Args:
            X (array): data matrix
        Returns:
            y (array): predicted target vector, which is the multiplication of the basis matrix and the weight vector
        """
        G = self._basis_matrix(X)
        return G @ self.w


if __name__ == '__main__':
    """
        Main function: Apply RBFN to a function approximation problem
        We will use the function f(x) = x * sin(x) as the function to approximate
    """
    # define the function to approximate
    f = lambda x: x * np.sin(x) # a função f(x) = x sen(x)

    # generate training set
    n_samples = 10
    X = np.sort(5 * np.random.rand(n_samples))
    y = f(X) + np.random.randn(n_samples) * 0.1

    # generate points used to plot
    X_plot = np.linspace(0, 5, 100)

    # plot training data
    plt.figure(figsize=(14, 5))
    plt.plot(X_plot, f(X_plot), label='f(x) = x * sin(x)')
    plt.plot(X, y, 'o', label='Training data')
    plt.legend(loc='lower left')

    # fit to RBF network
    rbf = RadialBasisFunctionNetwork(n_hidden=10, learning_rate=0.01, max_epochs=1000, random_state=0)
    rbf.fit(X, y)
    y_plot = rbf.predict(X_plot)
    plt.plot(X_plot, y_plot, label='RBF approximation')
    plt.legend(loc='lower left')
    plt.show()


