import numpy as np

class SVMKernel:
    def __init__(self, learningRate, C=1, gamma = None):
        self.learningRate = learningRate
        self.C = C
        self.weights = None
        self.bias = None
        self.X_train = None
        self.gamma = gamma

    def fit(self, X, y, epochs):
        self.X_train = X
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
              
        K = self.__RBF(X, X, self.gamma)

        self.weights = np.ones(K.shape[1])
        self.bias = 0

        for _ in range(epochs):
            gradientWeights, gradientBias = self.__gradients(K, y)
            self.weights -= self.learningRate * gradientWeights
            self.bias -= self.learningRate * gradientBias

    def predict(self, x):
        assert self.weights is not None, "needs fitting"
        
        if x.ndim == 1:
          x = x.reshape(1, -1)

        K = self.__RBF(x, self.X_train, self.gamma)
        return np.sign(K @ self.weights + self.bias)

    def __gradients(self, K, y):
        margins = y * (K @ self.weights + self.bias)
        mask = margins < 1

        gradientWeights = self.weights - self.C * np.sum(y[mask, np.newaxis] * K[mask], axis=0)
        gradientBias = -self.C * np.sum(y[mask])

        return gradientWeights, gradientBias

    def __RBF(self, X1, X2, gamma):
        
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * dist_sq)