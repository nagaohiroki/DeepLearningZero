import numpy as np


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


def main():
    perceptron_test()


if __name__ == "__main__":
    main()
