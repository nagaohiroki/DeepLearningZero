import numpy as np


# TODO arguments (t, y) -> (y, t)
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(y * np.log(t + delta))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def main():
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    t = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(y.shape)
    print(t.shape)
    print('mean_squared_error = ', mean_squared_error(y, t))
    print('cross_entropy_error = ', cross_entropy_error(y, t))


if __name__ == '__main__':
    main()
