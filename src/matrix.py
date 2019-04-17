import numpy as np
import activation_function


def matrix_test():
    X = np.array([1.0, 0.5])
    # layer 1
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    A1 = np.dot(X, W1) + B1
    Z1 = activation_function.sigmoid(A1)
    print(A1)
    print(Z1)
    # layer2
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = activation_function.sigmoid(A2)
    print(A2)
    print(Z2)


if __name__ == "__main__":
    matrix_test()
