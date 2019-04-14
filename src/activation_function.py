import numpy as np
import matplotlib.pyplot as plt


def step(x):
    return (x > 0).astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def activation_function_test():
    x = np.arange(-5.0, 5.0, 0.1)
    plt.plot(x, step(x))
    plt.plot(x, sigmoid(x))
    plt.plot(x, relu(x))
    plt.ylim(-0.1, 1.1)
    plt.show()


def main():
    activation_function_test()


if __name__ == "__main__":
    main()
