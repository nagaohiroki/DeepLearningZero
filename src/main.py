import numpy as np
import matplotlib.pyplot as plt


def mod_test():
    x = np.arange(0, 6, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()


def main():
    mod_test()


if __name__ == "__main__":
    main()
