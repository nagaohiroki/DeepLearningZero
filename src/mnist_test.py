import dataset.mnist
import pickle
import numpy as np
import activation_function
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def mnist_test():
    data = dataset.mnist.load_mnist(flatten=True, normalize=False)
    (x_train, t_train), (x_test, t_test) = data
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    new_img = img.reshape(28, 28)
    img_show(new_img)


def init_network():
    with open('dataset/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def get_data():
    data = dataset.mnist.load_mnist(normalize=True,
                                    flatten=True,
                                    one_hot_label=False)
    (x_train, t_train), (x_test, t_test) = data
    return x_test, t_test


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = activation_function.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = activation_function.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return activation_function.softmax(a3)


def main():
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("accuracy_cnt" + str(float(accuracy_cnt) / len(x)))


if __name__ == "__main__":
    main()
