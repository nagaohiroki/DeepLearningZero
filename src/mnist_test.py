import dataset.mnist
import numpy as np
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def main():
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


if __name__ == "__main__":
    main()
