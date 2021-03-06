import numpy as np
import matplotlib.pylab as plt
import dataset.mnist


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


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
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(y.shape)
    print(t.shape)
    print('mean_squared_error', mean_squared_error(y, t))
    print('cross_entropy_error', cross_entropy_error(y, t))


def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h


def function_2(x):
    return np.sum(x ** 2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def main():
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()

    print(numerical_gradient(function_2, X))
    # grad = numerical_gradient(function_2, np.array([X, Y]).T).T
    # plt.figure()
    # plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    # plt.grid()
    # plt.draw()
    # plt.show()


if __name__ == '__main__':
    main()
