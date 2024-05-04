import matplotlib.pyplot as plt
import matplotlib.tri as tri

def main():

    nodes_x = [0.000, 1.000, 2.000, 0.000, 1.000, 1.750, 1.000]
    nodes_y = [0.000, 0.000, 0.500, 1.000, 1.000, 1.300, 1.700]
    scalars = [1.000, 2.000, 1.000, 2.000, 7.000, 4.000, 5.000]
    elements = [
        [0, 1, 4],
        [4, 3, 0],
        [1, 2, 5],
        [5, 4, 1],
        [3, 4, 6],
        [4, 5, 6]
        ]

    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)
    plt.triplot(triangulation, '-k')
    plt.tricontourf(triangulation, scalars)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()