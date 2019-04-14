import numpy as np
import matplotlib.pyplot as plt


def perceptron(x, w, b):
    if np.sum(w * x) + b <= 0:
        return 0
    return 1


def AND(x1, x2):
    return perceptron(np.array([x1, x2]), np.array([0.5, 0.5]), -0.7)


def NAND(x1, x2):
    return perceptron(np.array([x1, x2]), np.array([-0.5, -0.5]), 0.7)


def OR(x1, x2):
    return perceptron(np.array([x1, x2]), np.array([0.5, 0.5]), -0.2)


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


def perceptron_test():
    print('AND---------')
    print('0, 0|' + str(AND(0, 0)))
    print('1, 0|' + str(AND(1, 0)))
    print('0, 1|' + str(AND(0, 1)))
    print('1, 1|' + str(AND(1, 1)))
    print('NAND---------')
    print('0, 0|' + str(NAND(0, 0)))
    print('1, 0|' + str(NAND(1, 0)))
    print('0, 1|' + str(NAND(0, 1)))
    print('1, 1|' + str(NAND(1, 1)))
    print('OR---------')
    print('0, 0|' + str(OR(0, 0)))
    print('1, 0|' + str(OR(1, 0)))
    print('0, 1|' + str(OR(0, 1)))
    print('1, 1|' + str(OR(1, 1)))
    print('XOR---------')
    print('0, 0|' + str(XOR(0, 0)))
    print('1, 0|' + str(XOR(1, 0)))
    print('0, 1|' + str(XOR(0, 1)))
    print('1, 1|' + str(XOR(1, 1)))


def mod_test():
    x = np.arange(0, 6, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()


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
