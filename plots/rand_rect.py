import matplotlib.pyplot as plt
import numpy as np

def main():
    x, y = coordinates()
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    make_plot(X, Y, Z)
    show_mesh(X, Y)

def coordinates():
    x = np.random.rand(100)
    x = np.sort(x)
    y = np.random.rand(100)
    y = np.sort(y)
    return x, y

def f(X, Y):
    return np.sqrt(X**2 + Y**2)

def make_plot(X, Y, Z):
    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z)
    fig.colorbar(CS)
    plt.show()

def show_mesh(X, Y):
    fig, ax = plt.subplots()
    ax.plot(X, Y, 'o')
    plt.show()

if __name__ == '__main__':
    main()