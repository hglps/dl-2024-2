import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras


def generate_linear_data(m, b, num_points=100, noise_std=5):
    """
    Generates random data points (x, y) based on the line y = mx + b with
    added Gaussian noise.
    """
    x = np.linspace(-10, 10, num_points)
    noise = np.random.normal(0, noise_std, num_points)
    y = m * x + b + noise
    return x, y


def plot_data(x, y, title="Generated Data"):
    """
    Plots the generated data points.
    """
    plt.scatter(x, y, label='Data Points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_regression_line(x, y, model, title="Regression Line"):
    """
    Plots the dataset along with the regression line.
    """
    plt.scatter(x, y, label='Data Points')
    x_range = np.linspace(min(x), max(x), 100)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color='red', label='Regression Line')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


def get_loss(y_real, y_pred):
    return np.mean((y_pred - y_real)**2)*0.5


def gradient_descent(m, b, x, y_real, y_pred):
    dm = np.mean((y_pred - y_real) * x)
    db = np.mean(y_pred - y_real)
    return dm, db


def solve_linear_regression_nn(X_train, y_train, epochs=100,
                               learning_rate=0.01):
    """
    Computes the linear regression coefficients using a neural
    network with a single neuron.
    """
    # m = np.random.normal(scale=10)
    # b = np.random.normal(scale=10)

    # losses = []
    # for i in range(epochs):
    #     y_pred = m*X_train + b
    #     curr_loss = get_loss(y_train, y_pred)

    #     losses.append(curr_loss)

    #     dm, db = gradient_descent(m, b, X_train, y_train, y_pred)
    #     m, b = m - learning_rate*dm, b - learning_rate*db

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,),
                              activation='linear'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                          momentum=0.0),
        loss='mean_squared_error'
    )

    model.fit(
        X_train, y_train, epochs=epochs, verbose=0
    )

    return model


# Example use case (to be replaced by your script
# when evaluating the students' code)
if __name__ == "__main__":
    # Generate synthetic data
    m_true, b_true = 3, -2
    x_data, y_data = generate_linear_data(m_true, b_true)

    # Split into training and testing sets
    indices = np.random.permutation(len(x_data))
    train_size = int(0.8 * len(x_data))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    X_train, y_train = x_data[train_indices], y_data[train_indices]
    X_test, y_test = x_data[test_indices], y_data[test_indices]

    # Train the neural network
    model = solve_linear_regression_nn(X_train, y_train)

    # Plot the results
    plot_data(x_data, y_data, "Generated Data")
    plot_regression_line(X_test, y_test, model, "Fitted Regression Line")

    # Print results
    weights, bias = model.layers[0].get_weights()
    print(f"Estimated parameters: m = {weights[0][0]:.4f}, b = {bias[0]:.4f}")
