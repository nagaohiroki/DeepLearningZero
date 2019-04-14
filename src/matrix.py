import numpy as np


def matrix_test():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    print(a)
    print(a.shape)
    print(a.shape[0])
    print(np.ndim(a))


def main():
    matrix_test()


if __name__ == "__main__":
    main()
