import numpy as np
import dataset.mnist


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # return -np.sum(y * np.log(t + 1e-7)) / batch_size
    return -np.sum(np.log(t[np.arange(batch_size), y] + 1e-7)) / batch_size


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def mini_bacth_test():
    data = dataset.mnist.load_mnist(flatten=True, one_hot_label=True)
    (x_train, t_train), (x_test, t_test) = data
    x_train_size = x_train.shape[0]
    bacth_size = 10
    bacth_mask = np.random.choice(bacth_size, x_train_size)
    x_batch = x_train[bacth_mask]
    t_batch = t_train[bacth_mask]
    print('x_train.shape', x_train.shape)
    print('t_train.shape', t_train.shape)
    print('x_train_size', x_train_size)
    print('bacth_mask.shape', bacth_mask.shape)
    print('x_batch.shape', x_batch.shape)
    print('t_batch.shape', t_batch.shape)


def error_test():
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    t = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(y.shape)
    print(t.shape)
    print('mean_squared_error', mean_squared_error(y, t))
    print('cross_entropy_error', cross_entropy_error(y, t))


def main():
    mini_bacth_test()
    error_test()


if __name__ == '__main__':
    main()
