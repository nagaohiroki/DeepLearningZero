import dataset.mnist


def main():
    data = dataset.mnist.load_mnist(flatten=True, normalize=False)
    (x_train, t_train), (x_test, t_test) = data
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)


if __name__ == "__main__":
    main()
