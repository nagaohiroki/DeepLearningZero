import numpy as np


def matrix_test():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2], [3, 4]])
    print(np.dot(a, b))


def main():
    matrix_test()


if __name__ == "__main__":
    main()
