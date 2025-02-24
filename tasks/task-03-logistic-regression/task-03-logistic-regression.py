import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_dim + 1) * 0.01
        # Including bias term
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def predict_proba(self, X):
        """
        Returns class prob. after prediction.
        Should return a value between 0 and 1.
        """
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        y_hat = self.sigmoid(z)
        return y_hat

    def predict(self, X):
        """
        Returns class label after prediction.
        Should return 0 or 1.
        """
        y_hat = self.predict_proba(X)
        return 1 if y_hat > 0.5 else 0

    def train(self, X, y):
        for ep in range(self.epochs):
            losses = []
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])

                # Preventing |infinite| loss
                if y_pred == 0:
                    y_pred = 0.0000000001
                elif y_pred == 1:
                    y_pred = 0.9999999999

                loss = (-y[i]*np.log(y_pred) -
                        (1-y[i])*np.log(1-y_pred))

                losses.append(loss)

                # Update weights
                self.weights[1:] = self.weights[1:] - self.learning_rate * (
                    y_pred - y[i]) * X[i]

                # Then update bias.
                # Using the same logic from perceptron weights array.
                self.weights[0] = self.weights[0] - self.learning_rate * (
                    y_pred - y[i])

            losses = np.array(losses)
            self.loss_history.append(np.mean(losses))


def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=42, cluster_std=2.0)
    return X, y


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Logistic Regression Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()


def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Training Iterations')
    plt.show()


# Generate dataset
X, y = generate_dataset()

# Train the model
neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
neuron.train(X, y)

# Plot decision boundary
plot_decision_boundary(neuron, X, y)

# Plot loss over training iterations
plot_loss(neuron)
