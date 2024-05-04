import matplotlib.pyplot as plt
import numpy as np

def main():
    X, Y = grid()
    Z = f(X, Y)
    make_plot(X, Y, Z)

def grid():
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    return X, Y

def f(X, Y):
    return np.sqrt(X**2 + Y**2)

def make_plot(X, Y, Z):
    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z)
    fig.colorbar(CS)
    plt.show()

if __name__ == '__main__':
    main()