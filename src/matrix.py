import numpy as np
import activation_function


def layer(X, W1, B1):
    A1 = np.dot(X, W1) + B1
    return activation_function.sigmoid(A1)


def matrix_test():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    Z1 = layer(X, W1, B1)
    Z2 = layer(Z1, W2, B2)
    print(Z2)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp = np.sum(exp_a)
    return exp_a / sum_exp


if __name__ == "__main__":
    a = softmax(np.array([1010, 1000, 990]))
    b = softmax(np.array([0.3, 2.9, 7.0]))
    print(a)
    print(b)
    print(np.sum(b))
    print(np.sum(a))
