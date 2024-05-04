import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)

    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z)
    fig.colorbar(CS)
    plt.show()

if __name__ == '__main__':
    main()